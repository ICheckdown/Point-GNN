"""
Point-GNN Runtime Measurement
==============================
Measures per-frame wall-clock time across the full pipeline.
No tracing, no graphs, no taxonomy — just timing.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import argparse
import numpy as np
from util.tf_compat import tf
from tqdm import tqdm

from dataset.kitti_dataset import KittiDataset, Points
from models.graph_gen import get_graph_generate_fn
from models.models import get_model
from models.box_encoding import get_box_decoding_fn, get_encoding_len
from models import nms
from util.config_util import load_config

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint_path',    type=str)
parser.add_argument('--dataset_root_dir', type=str,
                    default='../dataset/kitti/')
parser.add_argument('--dataset_split_file', type=str, default='')
parser.add_argument('--n_frames',         type=int, default=100)
args = parser.parse_args()

DATASET_DIR     = args.dataset_root_dir
CHECKPOINT_PATH = args.checkpoint_path
CONFIG_PATH     = os.path.join(CHECKPOINT_PATH, 'config')
assert os.path.isfile(CONFIG_PATH)
config = load_config(CONFIG_PATH)

DATASET_SPLIT_FILE = (
    args.dataset_split_file if args.dataset_split_file
    else os.path.join(DATASET_DIR, './3DOP_splits/val.txt'))

dataset = KittiDataset(
    os.path.join(DATASET_DIR, 'image/training/image_2'),
    os.path.join(DATASET_DIR, 'velodyne/training/velodyne/'),
    os.path.join(DATASET_DIR, 'calib/training/calib/'),
    os.path.join(DATASET_DIR, 'labels/training/label_2'),
    DATASET_SPLIT_FILE,
    num_classes=config['num_classes'])

N_FRAMES          = min(args.n_frames, dataset.num_files)
graph_generate_fn = get_graph_generate_fn(config['graph_gen_method'])
BOX_ENCODING_LEN  = get_encoding_len(config['box_encoding_method'])
box_decoding_fn   = get_box_decoding_fn(config['box_encoding_method'])
NUM_CLASSES       = dataset.num_classes

# ── Model ─────────────────────────────────────────────────────────────────────
if config['input_features'] in ('irgb', '0000', 'i000', '0rgb'):
    t_ivf = tf.placeholder(dtype=tf.float32, shape=[None, 4])
elif config['input_features'] == 'rgb':
    t_ivf = tf.placeholder(dtype=tf.float32, shape=[None, 3])
else:
    t_ivf = tf.placeholder(dtype=tf.float32, shape=[None, 1])

num_levels = len(config['runtime_graph_gen_kwargs']['level_configs'])
t_vcl  = [tf.placeholder(dtype=tf.float32, shape=[None, 3])
           for _ in range(num_levels + 1)]
t_el   = [tf.placeholder(dtype=tf.int32,   shape=[None, 2])
           for _ in range(num_levels)]
t_kil  = [tf.placeholder(dtype=tf.int32,   shape=[None, 1])
           for _ in range(num_levels)]
t_istr = tf.placeholder(dtype=tf.bool, shape=[])

model = get_model(config['model_name'])(
    num_classes=NUM_CLASSES,
    box_encoding_len=BOX_ENCODING_LEN,
    mode='test',
    **config['model_kwargs'])

t_logits, t_pred_box = model.predict(
    t_ivf, t_vcl, t_kil, t_el, t_istr)
t_probs     = model.postprocess(t_logits)
global_step = tf.Variable(0, dtype=tf.int32, trainable=False)

fetches = {
    'step':     global_step,
    'probs':    t_probs,
    'pred_box': t_pred_box,
}

def build_feed(iv, vcl, kil, el):
    fd = {t_ivf: iv, t_istr: False}
    fd.update(dict(zip(t_el,  el)))
    fd.update(dict(zip(t_kil, kil)))
    fd.update(dict(zip(t_vcl, vcl)))
    return fd

def get_input_v(cam, cfg):
    attr = cam.attr
    if cfg == 'irgb':   return attr
    if cfg == '0rgb':   return np.hstack([np.zeros((attr.shape[0],1)), attr[:,1:]])
    if cfg == '0000':   return np.zeros_like(attr)
    if cfg == 'i000':   return np.hstack([attr[:,[0]], np.zeros((attr.shape[0],3))])
    if cfg == 'i':      return attr[:,[0]]
    return np.zeros((attr.shape[0], 1))

# ── Session ───────────────────────────────────────────────────────────────────
saver = tf.train.Saver()
gpu_options = tf.GPUOptions(
    allow_growth=True,
    per_process_gpu_memory_fraction=0.80)

# Per-stage timings
t_data   = []
t_graph  = []
t_gnn    = []
t_post   = []
t_total  = []

with tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True)) as sess:

    sess.run(tf.variables_initializer(tf.global_variables()))
    sess.run(tf.variables_initializer(tf.local_variables()))

    model_path = tf.train.latest_checkpoint(CHECKPOINT_PATH)
    print(f'Restoring: {model_path}')
    saver.restore(sess, model_path)

    # Warm-up — not recorded
    print('Warm-up...')
    cam_w = dataset.get_cam_points_in_image_with_rgb(
        0, config['downsample_by_voxel_size'])
    vcl_w, kil_w, el_w = graph_generate_fn(
        cam_w.xyz, **config['runtime_graph_gen_kwargs'])
    iv_w = get_input_v(cam_w, config['input_features'])
    for _ in range(3):
        sess.run(fetches, feed_dict=build_feed(iv_w, vcl_w, kil_w, el_w))
    print(f'Warm-up done. Running {N_FRAMES} frames...\n')

    for frame_idx in tqdm(range(N_FRAMES)):

        t0 = time.perf_counter()

        # Stage 1: Data loading
        cam = dataset.get_cam_points_in_image_with_rgb(
            frame_idx, config['downsample_by_voxel_size'])
        dataset.get_calib(frame_idx)
        dataset.get_image(frame_idx)
        dataset.get_label(frame_idx)
        t1 = time.perf_counter()

        # Stage 2: Graph construction
        vcl, kil, el = graph_generate_fn(
            cam.xyz, **config['runtime_graph_gen_kwargs'])
        t2 = time.perf_counter()

        # Stage 3: GNN inference
        iv = get_input_v(cam, config['input_features'])
        results = sess.run(fetches,
                           feed_dict=build_feed(iv, vcl, kil, el))
        t3 = time.perf_counter()

        # Stage 4: Post-processing
        llgl = config['model_kwargs']['layer_configs'][-1]['graph_level']
        last_xyz = vcl[llgl + 1]

        if config['label_method'] == 'Car':
            label_map = {'Background':0,'Car':1,'DontCare':3}
        elif config['label_method'] == 'yaw':
            label_map = {'Background':0,'Car':1,'Pedestrian':3,
                         'Cyclist':5,'DontCare':7}
        else:
            label_map = {'Background':0,'Pedestrian':1,
                         'Cyclist':3,'DontCare':5}

        box_probs  = results['probs']
        box_labels = np.tile(
            np.expand_dims(np.arange(NUM_CLASSES), axis=0),
            (box_probs.shape[0], 1)).reshape(-1)
        box_probs  = box_probs.reshape(-1)
        pred_boxes = results['pred_box'].reshape(-1, 1, BOX_ENCODING_LEN)
        centers    = np.tile(
            np.expand_dims(last_xyz, axis=1),
            (1, NUM_CLASSES, 1)).reshape(-1, 3)

        box_decoding_fn(
            np.expand_dims(box_labels, axis=1),
            centers, pred_boxes, label_map)

        box_mask    = ((box_labels > 0)
                       * (box_labels < NUM_CLASSES - 1)
                       * (box_probs > 1. / NUM_CLASSES))
        box_indices = np.nonzero(box_mask)[0]

        if box_indices.size > 0:
            bl = box_labels[box_indices].copy()
            bp = box_probs[box_indices].copy()
            db = box_decoding_fn(
                np.expand_dims(box_labels, axis=1),
                centers, pred_boxes, label_map)[box_indices, 0]
            bl[bl==2]=1; bl[bl==4]=3; bl[bl==6]=5
            nms.nms_boxes_3d_uncertainty(
                bl, db, bp,
                overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                overlapped_thres=config['nms_overlapped_thres'],
                appr_factor=100.0, top_k=-1,
                attributes=np.arange(len(box_indices)))

        t4 = time.perf_counter()

        t_data.append( (t1-t0)*1000)
        t_graph.append((t2-t1)*1000)
        t_gnn.append(  (t3-t2)*1000)
        t_post.append( (t4-t3)*1000)
        t_total.append((t4-t0)*1000)

# ── Results ───────────────────────────────────────────────────────────────────
t_data  = np.array(t_data)
t_graph = np.array(t_graph)
t_gnn   = np.array(t_gnn)
t_post  = np.array(t_post)
t_total = np.array(t_total)

print(f"\n{'='*55}")
print(f"  RUNTIME RESULTS — {N_FRAMES} frames  (Jetson Orin)")
print(f"{'='*55}")
print(f"  {'Stage':<25} {'Mean':>8} {'Min':>8} {'Max':>8} {'Std':>8}")
print(f"  {'─'*53}")
for label, arr in [
    ('Data Loading',       t_data),
    ('Graph Construction', t_graph),
    ('GNN Inference',      t_gnn),
    ('Post-processing',    t_post),
    ('End-to-End',         t_total),
]:
    print(f"  {label:<25} "
          f"{arr.mean():>7.1f}ms "
          f"{arr.min():>7.1f}ms "
          f"{arr.max():>7.1f}ms "
          f"{arr.std():>7.1f}ms")

print(f"\n  Mean FPS:  {1000/t_total.mean():.3f}")
print(f"  Mean time: {t_total.mean()/1000:.3f}s per frame")
print(f"  Range:     {t_total.min()/1000:.3f}s – {t_total.max()/1000:.3f}s")
print(f"{'='*55}\n")