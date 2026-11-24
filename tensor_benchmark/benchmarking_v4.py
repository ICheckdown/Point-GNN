import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import copy
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from util.tf_compat import tf
from tqdm import tqdm
from collections import defaultdict

from dataset.kitti_dataset import KittiDataset
from models.graph_gen import get_graph_generate_fn
from models.models import get_model
from models.box_encoding import get_box_decoding_fn, get_encoding_len
from models import nms
from util.config_util import load_config
from models import graph_gen


# ── Argument parsing ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('checkpoint_path', type=str)
parser.add_argument('--dim', type=int, default=None,
                    help='Rewrite model hidden dimension before benchmarking '
                         '(e.g. 32, 64, 128). None = keep original config value.')
parser.add_argument('--test',           dest='test',           action='store_true', default=False)
parser.add_argument('--no-box-merge',   dest='use_box_merge',  action='store_false', default=True)
parser.add_argument('--no-box-score',   dest='use_box_score',  action='store_false', default=True)
parser.add_argument('--dataset_root_dir',   type=str, default='../dataset/kitti/')
parser.add_argument('--dataset_split_file', type=str, default='')
parser.add_argument('--output_dir',         type=str, default='')
parser.add_argument('--benchmark_frames',   type=int, default=100)
parser.add_argument('--all_frames_trace',   action='store_true', default=True,
                    help='Trace ALL benchmark frames (default True). '
                         'Set --no-all_frames_trace to trace only --trace_frames.')
parser.add_argument('--no-all_frames_trace', dest='all_frames_trace', action='store_false')
parser.add_argument('--trace_frames',       type=int, default=0,
                    help='Used only when --no-all_frames_trace is set.')
parser.add_argument('--skip_restore',   action='store_true', default=False,
                    help='Skip checkpoint restore (architecture-only timing).')
args = parser.parse_args()

# ── FIX 1: Define DIM_TAG immediately ──────────────────────────────────────────
DIM_TAG = str(args.dim) if args.dim is not None else 'orig'

IS_TEST       = args.test
USE_BOX_MERGE = args.use_box_merge
USE_BOX_SCORE = args.use_box_score
DATASET_DIR   = args.dataset_root_dir
BENCHMARK_FRAMES = args.benchmark_frames

DATASET_SPLIT_FILE = (
    args.dataset_split_file if args.dataset_split_file
    else os.path.join(DATASET_DIR, './3DOP_splits/val.txt'))

OUTPUT_DIR = (
    args.output_dir if args.output_dir
    else os.path.join(args.checkpoint_path, f'./eval_dim{DIM_TAG}/'))

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'data'), exist_ok=True)

CHECKPOINT_PATH = args.checkpoint_path
CONFIG_PATH     = os.path.join(CHECKPOINT_PATH, 'config')
assert os.path.isfile(CONFIG_PATH), f"Config not found: {CONFIG_PATH}"
config = load_config(CONFIG_PATH)


# ── FIX 2: make_config_for_dim — called below before model build ───────────────
def make_config_for_dim(base_cfg, D):
    """
    Rewrite all MLP depth lists to end in D.
    - scatter_max_point_set_pooling : output_MLP_depth_list → [D, D]
                                      point_MLP_depth_list  → [..., D]
    - scatter_max_graph_auto_center_net : edge_MLP_depth_list   → [D, D]
                                          update_MLP_depth_list → [D, D]
    - classaware_predictor : left unchanged (its input dim is whatever
                              the last GNN layer outputs, which is now D)
    """
    
    cfg = copy.deepcopy(base_cfg)
    for lc in cfg['model_kwargs']['layer_configs']:
        layer_type = lc['type']
        kw = lc['kwargs']
        if layer_type == 'scatter_max_point_set_pooling':
            if 'output_MLP_depth_list' in kw:
                kw['output_MLP_depth_list'] = [D, D]
            if 'point_MLP_depth_list' in kw:
                p = kw['point_MLP_depth_list']
                kw['point_MLP_depth_list'] = p[:-1] + [D]
        elif layer_type == 'scatter_max_graph_auto_center_net':
            if 'edge_MLP_depth_list' in kw:
                kw['edge_MLP_depth_list'] = [D, D]
            if 'update_MLP_depth_list' in kw:
                kw['update_MLP_depth_list'] = [D, D]
    return cfg


if args.dim is not None:
    '''
    print(f"\nRewriting config to dim={args.dim}")
    config = make_config_for_dim(config, args.dim)
    # Print the rewritten layer widths for verification
    print("Layer widths after rewrite:")
    for i, lc in enumerate(config['model_kwargs']['layer_configs']):
        print(f"  Layer {i}: {lc['scope']} ({lc['type']})")
        for k, v in lc['kwargs'].items():
            if 'depth_list' in k:
                print(f"    {k}: {v}")
else:
    print(f"\nUsing original config (dim={DIM_TAG})")
'''
    pass


TAXONOMY_OPS = {
    'sparse': {
        'GatherV2', 'GatherNd', 'ResourceGather', 'Gather',
        
        'UnsortedSegmentMax', 'UnsortedSegmentSum', 'UnsortedSegmentMean',
        'ScatterNd', 'TensorScatterUpdate', 'ScatterUpdate',
        
        'StridedSlice',
    },
    'dense': { 
        'MatMul', 'BatchMatMul', 'BatchMatMulV2',
        
        'BiasAdd', 'Relu', 'Relu6', 'LeakyRelu',
    
        'FusedBatchNorm', 'FusedBatchNormV3', 'BatchNormWithGlobalNormalization',
    },
    'memory': {
        'ConcatV2', 'Concat', 'Reshape', 'Transpose',
        'ExpandDims', 'Squeeze', 'Pack', 'Unpack', 'Stack',
        'Slice', 'Split', 'SplitV', 'Tile', 'BroadcastTo',

        'AddV2', 'Add', 'Sub', 'Mul', 'RealDiv',
        'Maximum', 'Minimum', 'Abs', 'Neg',

        'Shape', 'Size', 'Rank', 'Identity', 'StopGradient',
    },
    'other': {
        'Softmax', 'ArgMax', 'TopKV2', 'Where', 'BooleanMask',
        'Assert', 'Print', 'NoOp', 'VariableV2', 'Assign',
    },
}

TAXONOMY_PATTERNS = {
    'sparse':  ['gather', 'scatter', 'segment', 'stridedslice', 'strided_slice'],
    'dense':   ['matmul', 'biasadd', 'relu', 'conv', 'batchnorm', 'gemm'],
    'memory':  ['concat', 'reshape', 'transpose', 'expanddims', 'squeeze',
                'pack', 'unpack', 'identity', 'tile', 'slice', 'split',
                'sub', 'add', 'mul', 'cast', 'send', 'recv'],
    'other':   ['softmax', 'argmax', 'topk'],
}

TAXONOMY_COLORS = {
    'sparse':  '#FF8C00',
    'dense':   '#7B1FA2',
    'memory':  '#2196F3',
    'other':   '#607D8B',
}
TAXONOMY_LABELS = {
    'sparse':  'Sparse Ops\n(Gather+Scatter)',
    'dense':   'Dense Compute\n(MatMul+ReLU)',
    'memory':  'Memory Ops\n(concat+reshape+CPU)',
    'other':   'Other',
}


def _op_base(node_name: str) -> str:
    return node_name.split('/')[-1].split(':')[0].lower().replace('_', '')


def classify_op(node_name: str, device_name: str = '') -> str:
    base = _op_base(node_name)
    # CPU ops → memory (DMA / staging cost)
    if 'cpu' in (device_name or '').lower():
        return 'memory'
    # Exact-match lookup (fast path)
    for cat, op_set in TAXONOMY_OPS.items():
        if node_name.split('/')[-1].split(':')[0] in op_set:
            return cat

    for cat, patterns in TAXONOMY_PATTERNS.items():
        if any(p in base for p in patterns):
            return cat
    return 'other'


def extract_kernel_stats(run_metadata):
    stats = []
    for dev in run_metadata.step_stats.dev_stats:
        device_name = getattr(dev, 'device', '')
        for node in dev.node_stats:
            dur = node.op_end_rel_micros - node.op_start_rel_micros
            if dur > 0:
                stats.append({
                    'op_name':    node.node_name,
                    'device_name': device_name,
                    'duration_ms': dur / 1000.0,
                })
    return stats


def classify_frame_kernels(kernel_stats, wall_clock_ms: float):
    """
    Classify all kernel durations into four categories.
    Returns dict with 'times', 'pcts', 'total', 'coverage_ratio'.
    """
    cat_times = defaultdict(float)
    for k in kernel_stats:
        cat_times[classify_op(k['op_name'], k['device_name'])] += k['duration_ms']

    classified_total = sum(cat_times.values())
    # Guard against empty trace
    if classified_total == 0:
        return {
            'times':          {cat: 0.0 for cat in TAXONOMY_OPS},
            'pcts':           {cat: 0.0 for cat in TAXONOMY_OPS},
            'total':          0.0,
            'coverage_ratio': 0.0,
        }

    cat_pcts = {cat: cat_times[cat] / classified_total * 100 for cat in TAXONOMY_OPS}
    coverage = classified_total / wall_clock_ms if wall_clock_ms > 0 else 0.0
    return {
        'times':          dict(cat_times),
        'pcts':           cat_pcts,
        'total':          classified_total,
        'coverage_ratio': coverage,
    }


# ── Input feature helpers ───────────────────────────────────────────────────────
def get_input_features(cam_rgb_points, input_features_cfg):
    attr = cam_rgb_points.attr
    if input_features_cfg == 'irgb':   return attr
    if input_features_cfg == '0rgb':   return np.hstack([np.zeros((attr.shape[0], 1)), attr[:, 1:]])
    if input_features_cfg == '0000':   return np.zeros_like(attr)
    if input_features_cfg == 'i000':   return np.hstack([attr[:, [0]], np.zeros((attr.shape[0], 3))])
    if input_features_cfg == 'i':      return attr[:, [0]]
    return np.zeros((attr.shape[0], 1))


# ── Dataset ─────────────────────────────────────────────────────────────────────
if IS_TEST:
    dataset = KittiDataset(
        os.path.join(DATASET_DIR, 'image/testing/image_2'),
        os.path.join(DATASET_DIR, 'velodyne/testing/velodyne/'),
        os.path.join(DATASET_DIR, 'calib/testing/calib/'),
        '', num_classes=config['num_classes'], is_training=False)
else:
    dataset = KittiDataset(
        os.path.join(DATASET_DIR, 'image/training/image_2'),
        os.path.join(DATASET_DIR, 'velodyne/training/velodyne/'),
        os.path.join(DATASET_DIR, 'calib/training/calib/'),
        os.path.join(DATASET_DIR, 'labels/training/label_2'),
        DATASET_SPLIT_FILE,
        num_classes=config['num_classes'])

NUM_TEST_SAMPLE = min(BENCHMARK_FRAMES, dataset.num_files)
NUM_CLASSES     = dataset.num_classes
graph_generate_fn = get_graph_generate_fn(config['graph_gen_method'])

BOX_ENCODING_LEN = get_encoding_len(config['box_encoding_method'])
box_decoding_fn  = get_box_decoding_fn(config['box_encoding_method'])


# ── TF placeholders ─────────────────────────────────────────────────────────────
if config['input_features'] in ('irgb', '0000', 'i000', '0rgb'):
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 4])
elif config['input_features'] == 'rgb':
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 3])
else:
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 1])

num_levels = len(config['runtime_graph_gen_kwargs']['level_configs'])
t_vertex_coord_list      = [tf.placeholder(dtype=tf.float32, shape=[None, 3]) for _ in range(num_levels + 1)]
t_edges_list             = [tf.placeholder(dtype=tf.int32,   shape=[None, 2]) for _ in range(num_levels)]
t_keypoint_indices_list  = [tf.placeholder(dtype=tf.int32,   shape=[None, 1]) for _ in range(num_levels)]
t_is_training = tf.placeholder(dtype=tf.bool, shape=[])


# ── Build model (with rewritten config if --dim was given) ───────────────────
model = get_model(config['model_name'])(
    num_classes=NUM_CLASSES,
    box_encoding_len=BOX_ENCODING_LEN,
    mode='test',
    **config['model_kwargs']
)

t_logits, t_pred_box = model.predict(
    t_initial_vertex_features,
    t_vertex_coord_list,
    t_keypoint_indices_list,
    t_edges_list,
    t_is_training,
)

t_probs       = model.postprocess(t_logits)
t_predictions = tf.argmax(t_probs, axis=1, output_type=tf.int32)
global_step   = tf.Variable(0, dtype=tf.int32, trainable=False)

fetches = {
    'step':        global_step,
    'predictions': t_predictions,
    'probs':       t_probs,
    'pred_box':    t_pred_box,
}


def build_feed_dict(input_v, vertex_coord_list, keypoint_indices_list, edges_list):
    fd = {t_initial_vertex_features: input_v, t_is_training: False}
    fd.update(dict(zip(t_edges_list,            edges_list)))
    fd.update(dict(zip(t_keypoint_indices_list, keypoint_indices_list)))
    fd.update(dict(zip(t_vertex_coord_list,     vertex_coord_list)))
    return fd


# ── Session & restore ────────────────────────────────────────────────────────────
saver      = tf.train.Saver()
tf_graph   = tf.get_default_graph()
gpu_options = tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=0.80)

frame_records = []

with tf.Session(graph=tf_graph,
                config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    sess.run(tf.variables_initializer(tf.global_variables()))
    sess.run(tf.variables_initializer(tf.local_variables()))

    if not args.skip_restore:
        model_path = tf.train.latest_checkpoint(CHECKPOINT_PATH)
        assert model_path is not None, f"No checkpoint found in {CHECKPOINT_PATH}"
        print(f'Restoring checkpoint: {model_path}')
        saver.restore(sess, model_path)
    else:
        print('Skipping checkpoint restore (--skip_restore).')

    '''
    # Print actual variable shapes so you can verify the dim rewrite worked
    print(f"\nActual TF variable shapes (first 12):")
    for v in tf.trainable_variables()[:12]:
        print(f"  {v.name:<60} {str(v.shape):>20}")

    # Warm-up (no trace, not recorded)
    print("\nWarm-up pass...")
    cam_w = dataset.get_cam_points_in_image_with_rgb(0, config['downsample_by_voxel_size'])
    vcl_w, kil_w, el_w = graph_generate_fn(cam_w.xyz, **config['runtime_graph_gen_kwargs'])
    sess.run(fetches, feed_dict=build_feed_dict(
        get_input_features(cam_w, config['input_features']), vcl_w, kil_w, el_w))
    print("Warm-up done.\n")
    '''

    # ── Benchmark loop ───────────────────────────────────────────────────────
    for frame_idx in tqdm(range(NUM_TEST_SAMPLE), desc=f"Profiling dim={DIM_TAG}"):

        # FIX 3: decide whether to trace this frame
        if args.all_frames_trace:
            should_trace = True
        else:
            should_trace = frame_idx < args.trace_frames

        rec = {'frame': frame_idx, 'dim': DIM_TAG}

        # Data loading
        t0 = time.perf_counter()
        cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(
            frame_idx, config['downsample_by_voxel_size'])
        dataset.get_calib(frame_idx)
        dataset.get_image(frame_idx)
        if not IS_TEST:
            dataset.get_label(frame_idx)
        t1 = time.perf_counter()
        rec['t_data_loading_ms'] = (t1 - t0) * 1000

        # Graph construction
        t_gc0 = time.perf_counter()
        vertex_coord_list, keypoint_indices_list, edges_list = graph_generate_fn(
            cam_rgb_points.xyz, **config['runtime_graph_gen_kwargs'])
        t_gc1 = time.perf_counter()
        rec['t_graph_construction_ms'] = (t_gc1 - t_gc0) * 1000

        gc_downsample_ms = graph_gen._gc_timing.get('downsample_ms', 0.0)
        gc_edge_ms       = graph_gen._gc_timing.get('edge_build_ms', 0.0)
        gc_frnn_ms       = max(0.0, rec['t_graph_construction_ms'] - gc_downsample_ms - gc_edge_ms)
        rec['gc_sparse_ms']     = gc_frnn_ms
        rec['gc_memory_ms']     = gc_edge_ms
        rec['gc_dense_ms']      = gc_downsample_ms
        rec['gc_frnn_ms']       = gc_frnn_ms
        rec['gc_edge_build_ms'] = gc_edge_ms
        rec['gc_downsample_ms'] = gc_downsample_ms

        edges_f  = edges_list[1]
        rec['num_vertices'] = vertex_coord_list[1].shape[0]
        rec['num_edges']    = edges_f.shape[0]

        input_v   = get_input_features(cam_rgb_points, config['input_features'])
        feed_dict = build_feed_dict(input_v, vertex_coord_list, keypoint_indices_list, edges_list)

        llgl = config['model_kwargs']['layer_configs'][-1]['graph_level']
        last_layer_points_xyz = vertex_coord_list[llgl + 1]

        # GNN inference
        t4 = time.perf_counter()
        if should_trace:
            run_options  = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            results = sess.run(fetches, feed_dict=feed_dict,
                               options=run_options, run_metadata=run_metadata)
            t5 = time.perf_counter()
            wall_gnn_ms   = (t5 - t4) * 1000
            kernel_stats  = extract_kernel_stats(run_metadata)
            frame_cls     = classify_frame_kernels(kernel_stats, wall_gnn_ms)

            
            if frame_idx == 0:
                '''
                print(f"\nTop-20 kernel durations (frame 0, dim={DIM_TAG}):")
                top20 = sorted(kernel_stats, key=lambda x: x['duration_ms'], reverse=True)[:20]
                for k in top20:
                    cat = classify_op(k['op_name'], k['device_name'])
                    print(f"  [{cat:7s}] {k['op_name'].split('/')[-1]:<40} "
                          f"{k['device_name']:<30} {k['duration_ms']:>8.2f} ms")
                print(f"  Trace total: {frame_cls['total']:.1f} ms | "
                      f"Wall-clock: {wall_gnn_ms:.1f} ms | "
                      f"Coverage: {frame_cls['coverage_ratio']*100:.0f}%\n")
                '''
        else:
            results = sess.run(fetches, feed_dict=feed_dict)
            t5 = time.perf_counter()
            wall_gnn_ms = (t5 - t4) * 1000
            frame_cls = {
                'times':          {cat: float('nan') for cat in TAXONOMY_OPS},
                'pcts':           {cat: float('nan') for cat in TAXONOMY_OPS},
                'total':          float('nan'),
                'coverage_ratio': float('nan'),
            }

        rec['t_gnn_ms']               = wall_gnn_ms   # always reliable wall-clock
        rec['gnn_trace_total_ms']     = frame_cls['total']
        rec['gnn_coverage_ratio']     = frame_cls['coverage_ratio']
        for cat in TAXONOMY_OPS:
            rec[f'gnn_{cat}_ms']  = frame_cls['times'].get(cat, float('nan'))
            rec[f'gnn_pct_{cat}'] = frame_cls['pcts'].get(cat, float('nan'))

        # Post-processing
        if config['label_method'] == 'yaw':
            label_map = {'Background': 0, 'Car': 1, 'Pedestrian': 3, 'Cyclist': 5, 'DontCare': 7}
        elif config['label_method'] == 'Car':
            label_map = {'Background': 0, 'Car': 1, 'DontCare': 3}
        else:
            label_map = {'Background': 0, 'Pedestrian': 1, 'Cyclist': 3, 'DontCare': 5}

        t6 = time.perf_counter()
        box_probs  = results['probs']
        box_labels = np.tile(
            np.expand_dims(np.arange(NUM_CLASSES), axis=0),
            (box_probs.shape[0], 1)).reshape(-1)
        box_probs  = box_probs.reshape(-1)
        pred_boxes = results['pred_box'].reshape(-1, 1, BOX_ENCODING_LEN)
        last_xyz_tiled = np.tile(
            np.expand_dims(last_layer_points_xyz, axis=1),
            (1, NUM_CLASSES, 1)).reshape(-1, 3)

        t_bd0 = time.perf_counter()
        decoded_boxes = box_decoding_fn(
            np.expand_dims(box_labels, axis=1), last_xyz_tiled, pred_boxes, label_map)
        t_bd1 = time.perf_counter()
        rec['t_box_decode_ms'] = (t_bd1 - t_bd0) * 1000

        t_f0 = time.perf_counter()
        box_mask = ((box_labels > 0) * (box_labels < NUM_CLASSES - 1) *
                    (box_probs > 1. / NUM_CLASSES))
        box_indices = np.nonzero(box_mask)[0]
        t_f1 = time.perf_counter()
        rec['t_filter_ms'] = (t_f1 - t_f0) * 1000

        t_n0 = time.perf_counter()
        if box_indices.size != 0:
            box_labels_f = box_labels[box_indices].copy()
            box_probs_f  = box_probs[box_indices].copy()
            decoded_f    = decoded_boxes[box_indices, 0]
            box_labels_f[box_labels_f == 2] = 1
            box_labels_f[box_labels_f == 4] = 3
            box_labels_f[box_labels_f == 6] = 5
            if USE_BOX_MERGE and USE_BOX_SCORE:
                nms.nms_boxes_3d_uncertainty(
                    box_labels_f, decoded_f, box_probs_f,
                    overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                    overlapped_thres=config['nms_overlapped_thres'],
                    appr_factor=100.0, top_k=-1,
                    attributes=np.arange(len(box_indices)))
            else:
                nms.nms_boxes_3d(
                    box_labels_f, decoded_f, box_probs_f,
                    overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                    overlapped_thres=config['nms_overlapped_thres'],
                    appr_factor=100.0, top_k=-1,
                    attributes=np.arange(len(box_indices)))
        t_n1 = time.perf_counter()
        rec['t_nms_ms'] = (t_n1 - t_n0) * 1000

        t7 = time.perf_counter()
        rec['t_post_ms'] = (t7 - t6) * 1000
        rec['t_end_to_end_ms'] = (rec['t_data_loading_ms'] +
                                   rec['t_graph_construction_ms'] +
                                   rec['t_gnn_ms'] +
                                   rec['t_post_ms'])
        frame_records.append(rec)

        # Write empty det file so downstream eval doesn't crash
        filename = os.path.join(OUTPUT_DIR, 'data', dataset.get_filename(frame_idx) + '.txt')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.write('\n')

# ── Results ──────────────────────────────────────────────────────────────────────
df = pd.DataFrame(frame_records)


def nm(col):
    return float(np.nanmean(df[col]))
'''
print(f"\n{'=' * 65}")
print(f"BENCHMARK RESULTS  |  {NUM_TEST_SAMPLE} frames  |  dim={DIM_TAG}")
print(f"{'=' * 65}")

print(f"\nStage timing (wall-clock mean):")
print(f"{'Stage':<30} {'Mean (ms)':>10}")
print(f"{'─' * 42}")
for col, label in [
    ('t_data_loading_ms',     'Data Loading'),
    ('t_graph_construction_ms','Graph Construction'),
    ('t_gnn_ms',              'GNN Inference (wall)'),
    ('t_post_ms',             'Post-processing'),
    ('t_end_to_end_ms',       'End-to-End'),
]:
    print(f"{label:<30} {nm(col):>10.1f}")
'''
traced_frames = int(df['gnn_trace_total_ms'].notna().sum())
mean_coverage = float(np.nanmean(df['gnn_coverage_ratio'])) * 100

print(f"\nGNN Kernel Classification (nanmean % — traced frames only):")
print(f"{'Category':<15} {'Mean %':>10} {'Mean ms':>10}")
print(f"{'─' * 37}")
for cat in TAXONOMY_OPS:
    print(f"{cat:<15} {nm(f'gnn_pct_{cat}'):>9.1f}% {nm(f'gnn_{cat}_ms'):>10.1f}")


print(f"\nBOTTLENECK RANKING (wall-clock):")
total_ms = nm('t_end_to_end_ms')
for rank, (label, ms) in enumerate(sorted([
    ('Data Loading',      nm('t_data_loading_ms')),
    ('Graph Construction',nm('t_graph_construction_ms')),
    ('GNN Inference',     nm('t_gnn_ms')),
    ('Post-processing',   nm('t_post_ms')),
], key=lambda x: x[1], reverse=True), 1):
    print(f"#{rank} {label:<25} {ms:>8.1f} ms  ({ms/total_ms*100:.1f}%)")

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, f'benchmark_dim{DIM_TAG}.csv')
df.to_csv(csv_path, index=False)
print(f"\nCSV → {csv_path}")

# ── Plots ─────────────────────────────────────────────────────────────────────
cats = list(TAXONOMY_OPS.keys())
mean_pcts = [nm(f'gnn_pct_{cat}') for cat in cats]

fig1, ax1 = plt.subplots(figsize=(8, 6))
fig1.suptitle(
    f'GNN Inference — Kernel Classification\n'
    f'Mean across {traced_frames} traced frames | dim={DIM_TAG}',
    fontsize=13, fontweight='bold')
bars1 = ax1.bar(range(len(cats)), mean_pcts,
                color=[TAXONOMY_COLORS[c] for c in cats],
                edgecolor='black', linewidth=0.8, width=0.5)
for bar, pct in zip(bars1, mean_pcts):
    if not np.isnan(pct):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{pct:.1f}%', ha='center', va='bottom',
                 fontsize=12, fontweight='bold')
ax1.set_xticks(range(len(cats)))
ax1.set_xticklabels([TAXONOMY_LABELS[c] for c in cats], fontsize=10)
ax1.set_ylabel('Mean % of kernel time', fontsize=12)
valid_pcts = [p for p in mean_pcts if not np.isnan(p)]
ax1.set_ylim(0, (max(valid_pcts) * 1.25) if valid_pcts else 1.0)
ax1.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
p1 = os.path.join(OUTPUT_DIR, f'gnn_kernel_classification_dim{DIM_TAG}.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot → {p1}")

# System-level breakdown (wall-clock based, always reliable)
'''
sys_sparse  = df['gc_sparse_ms'] + df['gnn_sparse_ms'].fillna(0)
sys_dense   = df['gc_dense_ms']  + df['gnn_dense_ms'].fillna(0) + df['t_box_decode_ms']
sys_memory  = df['gc_memory_ms'] + df['gnn_memory_ms'].fillna(0) + df['t_filter_ms'] + df['t_data_loading_ms']
sys_other   = df['gnn_other_ms'].fillna(0) + df['t_nms_ms']
sys_total   = sys_sparse + sys_dense + sys_memory + sys_other
sys_pcts    = [(sys_sparse/sys_total*100).mean(),
               (sys_dense /sys_total*100).mean(),
               (sys_memory/sys_total*100).mean(),
               (sys_other /sys_total*100).mean()]

fig2, ax2 = plt.subplots(figsize=(8, 6))
fig2.suptitle(
    f'Full Pipeline Classification (wall-clock)\n'
    f'Mean across {NUM_TEST_SAMPLE} frames | dim={DIM_TAG}',
    fontsize=13, fontweight='bold')
bars2 = ax2.bar(range(4), sys_pcts,
                color=[TAXONOMY_COLORS[c] for c in cats],
                edgecolor='black', linewidth=0.8, width=0.5)
for bar, pct in zip(bars2, sys_pcts):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{pct:.1f}%', ha='center', va='bottom',
             fontsize=12, fontweight='bold')
ax2.set_xticks(range(4))
ax2.set_xticklabels([TAXONOMY_LABELS[c] for c in cats], fontsize=10)
ax2.set_ylabel('Mean % of pipeline time', fontsize=12)
ax2.set_ylim(0, max(sys_pcts) * 1.25 if sys_pcts else 1.0)
ax2.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
p2 = os.path.join(OUTPUT_DIR, f'system_classification_dim{DIM_TAG}.png')
plt.savefig(p2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot → {p2}")
'''
print("\nDone.\n")