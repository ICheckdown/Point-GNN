"""This file implements an inference pipeline for Point-GNN on KITTI dataset."""

import os
import time
import argparse
import numpy as np
from util.tf_compat import tf
from tqdm import tqdm

from dataset.kitti_dataset import KittiDataset, Points
from models.graph_gen import get_graph_generate_fn
from models.models import get_model
from models.box_encoding import (
    get_box_decoding_fn,
    get_box_encoding_fn,
    get_encoding_len,
)
from models import nms
from util.config_util import load_config


# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Point-GNN inference on KITTI')
parser.add_argument('checkpoint_path', type=str)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--no-box-merge', dest='use_box_merge',
                    action='store_false', default=True)
parser.add_argument('--no-box-score', dest='use_box_score',
                    action='store_false', default=True)
parser.add_argument('--dataset_root_dir', type=str,
                    default='../dataset/kitti/')
parser.add_argument('--dataset_split_file', type=str, default='')
parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--max_frames', type=int, default=None,
                    help='Maximum number of frames to run (for speed / memory).')
args = parser.parse_args()

MAX_FRAMES    = args.max_frames
IS_TEST       = args.test
USE_BOX_MERGE = args.use_box_merge
USE_BOX_SCORE = args.use_box_score
DATASET_DIR   = args.dataset_root_dir
print("DATASET_DIR =", DATASET_DIR)

DATASET_SPLIT_FILE = (
    args.dataset_split_file if args.dataset_split_file
    else os.path.join(DATASET_DIR, '3DOP_splits/val.txt')
)

OUTPUT_DIR = (
    args.output_dir if args.output_dir
    else os.path.join(args.checkpoint_path, 'eval')
)

CHECKPOINT_PATH = args.checkpoint_path
CONFIG_PATH     = os.path.join(CHECKPOINT_PATH, 'config')
assert os.path.isfile(CONFIG_PATH), f'Config not found: {CONFIG_PATH}'
config = load_config(CONFIG_PATH)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'data'), exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────
if IS_TEST:
    dataset = KittiDataset(
        os.path.join(DATASET_DIR, 'image/testing/image_2'),
        os.path.join(DATASET_DIR, 'velodyne/testing/velodyne/'),
        os.path.join(DATASET_DIR, 'calib/testing/calib/'),
        '',
        num_classes=config['num_classes'],
        is_training=False
    )
else:
    dataset = KittiDataset(
        os.path.join(DATASET_DIR, 'image/training/image_2'),
        os.path.join(DATASET_DIR, 'velodyne/training/velodyne/'),
        os.path.join(DATASET_DIR, 'calib/training/calib/'),
        os.path.join(DATASET_DIR, 'labels/training/label_2'),
        DATASET_SPLIT_FILE,
        num_classes=config['num_classes']
    )

NUM_TEST_SAMPLE   = dataset.num_files
NUM_CLASSES       = dataset.num_classes
graph_generate_fn = get_graph_generate_fn(config['graph_gen_method'])


# ── Occlusion helper ──────────────────────────────────────────────────────────
def occlusion(label, xyz):
    if xyz.shape[0] == 0:
        return 0.0
    normals, lower, upper = dataset.box3d_to_normals(label)
    projected    = np.matmul(xyz, np.transpose(normals))
    x_cover_rate = (np.max(projected[:, 0]) - np.min(projected[:, 0])) / (upper[0] - lower[0])
    y_cover_rate = (np.max(projected[:, 1]) - np.min(projected[:, 1])) / (upper[1] - lower[1])
    z_cover_rate = (np.max(projected[:, 2]) - np.min(projected[:, 2])) / (upper[2] - lower[2])
    return x_cover_rate * y_cover_rate * z_cover_rate


# ── Model setup ───────────────────────────────────────────────────────────────
BOX_ENCODING_LEN = get_encoding_len(config['box_encoding_method'])
box_encoding_fn  = get_box_encoding_fn(config['box_encoding_method'])
box_decoding_fn  = get_box_decoding_fn(config['box_encoding_method'])

if config['input_features'] in ('irgb', '0000', 'i000', '0rgb'):
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 4])
elif config['input_features'] == 'rgb':
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 3])
else:
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 1])

num_levels = len(config['runtime_graph_gen_kwargs']['level_configs'])
t_vertex_coord_list     = [tf.placeholder(dtype=tf.float32, shape=[None, 3])
                           for _ in range(num_levels + 1)]
t_edges_list            = [tf.placeholder(dtype=tf.int32,   shape=[None, 2])
                           for _ in range(num_levels)]
t_keypoint_indices_list = [tf.placeholder(dtype=tf.int32,   shape=[None, 1])
                           for _ in range(num_levels)]
t_is_training = tf.placeholder(dtype=tf.bool, shape=[])

model = get_model(config['model_name'])(
    num_classes=NUM_CLASSES,
    box_encoding_len=BOX_ENCODING_LEN,
    mode='test',
    **config['model_kwargs']
)

t_logits, t_pred_box = model.predict(
    t_initial_vertex_features, t_vertex_coord_list,
    t_keypoint_indices_list, t_edges_list, t_is_training
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


# ── Label map ─────────────────────────────────────────────────────────────────
if config['label_method'] == 'yaw':
    label_map      = {'Background':0,'Car':1,'Pedestrian':3,
                      'Cyclist':5,'DontCare':7}
    all_class_name = ['Background','Car','Car',
                      'Pedestrian','Pedestrian',
                      'Cyclist','Cyclist','DontCare']
elif config['label_method'] == 'Car':
    label_map      = {'Background':0,'Car':1,'DontCare':3}
    all_class_name = ['Background','Car','Car','DontCare']
else:
    label_map      = {'Background':0,'Pedestrian':1,
                      'Cyclist':3,'DontCare':5}
    all_class_name = ['Background','Pedestrian','Pedestrian',
                      'Cyclist','Cyclist','DontCare']


# ── Session ───────────────────────────────────────────────────────────────────
time_dict   = {}
saver       = tf.train.Saver()
graph       = tf.get_default_graph()
gpu_options = tf.GPUOptions(
    allow_growth=True,
    per_process_gpu_memory_fraction=0.80
)

with tf.Session(
    graph=graph,
    config=tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True
    )
) as sess:

    sess.run(tf.variables_initializer(tf.global_variables()))
    sess.run(tf.variables_initializer(tf.local_variables()))

    model_path = tf.train.latest_checkpoint(CHECKPOINT_PATH)
    print(f'Restoring from checkpoint: {model_path}')
    saver.restore(sess, model_path)

    frames_to_run = NUM_TEST_SAMPLE
    if MAX_FRAMES is not None:
        frames_to_run = min(NUM_TEST_SAMPLE, MAX_FRAMES)

    print(f'Running inference on {frames_to_run} frames...\n')

    for frame_idx in tqdm(range(frames_to_run)):

        start_time = time.time()

        # Data loading
        cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(
            frame_idx, config['downsample_by_voxel_size'])
        calib = dataset.get_calib(frame_idx)
        if not IS_TEST:
            box_label_list = dataset.get_label(frame_idx)
        input_time = time.time()
        time_dict['fetch input'] = time_dict.get('fetch input', 0) + (input_time - start_time)

        # Graph construction
        (vertex_coord_list,
         keypoint_indices_list,
         edges_list) = graph_generate_fn(
            cam_rgb_points.xyz,
            **config['runtime_graph_gen_kwargs'])
        graph_time = time.time()
        time_dict['gen graph'] = time_dict.get('gen graph', 0) + (graph_time - input_time)

        # Input features
        if config['input_features'] == 'irgb':
            input_v = cam_rgb_points.attr
        elif config['input_features'] == '0rgb':
            input_v = np.hstack([np.zeros((cam_rgb_points.attr.shape[0], 1)),
                                 cam_rgb_points.attr[:, 1:]])
        elif config['input_features'] == '0000':
            input_v = np.zeros_like(cam_rgb_points.attr)
        elif config['input_features'] == 'i000':
            input_v = np.hstack([cam_rgb_points.attr[:, [0]],
                                 np.zeros((cam_rgb_points.attr.shape[0], 3))])
        elif config['input_features'] == 'i':
            input_v = cam_rgb_points.attr[:, [0]]
        else:
            input_v = np.zeros((cam_rgb_points.attr.shape[0], 1))

        last_layer_graph_level = config['model_kwargs']['layer_configs'][-1]['graph_level']
        last_layer_points_xyz = vertex_coord_list[last_layer_graph_level + 1]

        feed_dict = {t_initial_vertex_features: input_v, t_is_training: False}
        feed_dict.update(dict(zip(t_edges_list,            edges_list)))
        feed_dict.update(dict(zip(t_keypoint_indices_list, keypoint_indices_list)))
        feed_dict.update(dict(zip(t_vertex_coord_list,     vertex_coord_list)))

        # GNN inference
        results   = sess.run(fetches, feed_dict=feed_dict)
        gnn_time  = time.time()
        time_dict['gnn inference'] = time_dict.get('gnn inference', 0) + (gnn_time - graph_time)

        # Box decoding
        box_probs  = results['probs']
        box_labels = np.tile(
            np.expand_dims(np.arange(NUM_CLASSES), axis=0),
            (box_probs.shape[0], 1)
        ).reshape(-1)
        raw_box_labels = box_labels.copy()
        box_probs      = box_probs.reshape(-1)
        pred_boxes     = results['pred_box'].reshape(-1, 1, BOX_ENCODING_LEN)

        boxes_centers = np.tile(
            np.expand_dims(last_layer_points_xyz, axis=1),
            (1, NUM_CLASSES, 1)
        ).reshape(-1, 3)

        decoded_boxes = box_decoding_fn(
            np.expand_dims(box_labels, axis=1),
            boxes_centers, pred_boxes, label_map
        )

        box_mask    = ((box_labels > 0)
                       * (box_labels < NUM_CLASSES - 1)
                       * (box_probs > 1. / NUM_CLASSES))
        box_indices = np.nonzero(box_mask)[0]

        decode_time = time.time()
        time_dict['decode box'] = time_dict.get('decode box', 0) + (decode_time - gnn_time)

        # NMS and write detections
        pred_labels = []
        filename = os.path.join(
            OUTPUT_DIR, 'data', dataset.get_filename(frame_idx) + '.txt'
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if box_indices.size != 0:
            box_labels_f    = box_labels[box_indices].copy()
            box_probs_f     = box_probs[box_indices].copy()
            decoded_boxes_f = decoded_boxes[box_indices, 0]

            # Merge duplicated class IDs
            box_labels_f[box_labels_f == 2] = 1
            box_labels_f[box_labels_f == 4] = 3
            box_labels_f[box_labels_f == 6] = 5

            # Run NMS
            if USE_BOX_MERGE and USE_BOX_SCORE:
                (class_labels, detection_boxes_3d,
                 detection_scores, nms_indices) = nms.nms_boxes_3d_uncertainty(
                    box_labels_f, decoded_boxes_f, box_probs_f,
                    overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                    overlapped_thres=config['nms_overlapped_thres'],
                    appr_factor=100.0, top_k=-1,
                    attributes=np.arange(len(box_indices))
                )
            else:
                (class_labels, detection_boxes_3d,
                 detection_scores, nms_indices) = nms.nms_boxes_3d(
                    box_labels_f, decoded_boxes_f, box_probs_f,
                    overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                    overlapped_thres=config['nms_overlapped_thres'],
                    appr_factor=100.0, top_k=-1,
                    attributes=np.arange(len(box_indices))
                )

            box_probs_f = detection_scores

            # Map flattened box indices back to point indices to avoid IndexError
            point_indices_all = box_indices // NUM_CLASSES
            # Optional: unique to reduce redundant indexing
            point_indices_all = np.clip(
                point_indices_all, 0, last_layer_points_xyz.shape[0] - 1
            )
            points_for_scoring = last_layer_points_xyz[point_indices_all]

            detection_boxes_3d_corners = nms.boxes_3d_to_corners(
                detection_boxes_3d
            )

            for i in range(len(detection_boxes_3d_corners)):
                corners_cam = Points(
                    xyz=detection_boxes_3d_corners[i], attr=None)
                corners_img = dataset.cam_points_to_image(corners_cam, calib)
                corners_xy  = corners_img.xyz[:, :2]

                class_name = all_class_name[class_labels[i]]
                xmin, ymin = np.amin(corners_xy, axis=0)
                xmax, ymax = np.amax(corners_xy, axis=0)
                clip_xmin  = max(xmin, 0.0)
                clip_ymin  = max(ymin, 0.0)
                clip_xmax  = min(xmax, 1242.0)
                clip_ymax  = min(ymax, 375.0)

                trunc = 1.0 - (
                    (clip_ymax - clip_ymin) * (clip_xmax - clip_xmin)
                    / max((ymax - ymin) * (xmax - xmin), 1e-6)
                )
                if trunc > 0.4:
                    continue

                x3d, y3d, z3d, l, h, w, yaw = detection_boxes_3d[i]
                if l <= 0:
                    continue
                score = box_probs_f[i]

                if USE_BOX_SCORE:
                    tmp_label = {
                        'x3d': x3d, 'y3d': y3d, 'z3d': z3d,
                        'yaw': yaw, 'height': h,
                        'width': w, 'length': l
                    }
                    inside_mask = dataset.sel_xyz_in_box3d(
                        tmp_label,
                        points_for_scoring
                    )
                    points_inside = points_for_scoring[inside_mask]
                    score = (1.0 + occlusion(tmp_label, points_inside)) * score

                pred_labels.append((
                    class_name, -1, -1, 0,
                    clip_xmin, clip_ymin, clip_xmax, clip_ymax,
                    h, w, l, x3d, y3d, z3d, yaw, score
                ))

        nms_time = time.time()
        time_dict['nms'] = time_dict.get('nms', 0) + (nms_time - decode_time)

        with open(filename, 'w') as f:
            for pred_label in pred_labels:
                f.write(' '.join(str(x) for x in pred_label) + '\n')
            if not pred_labels:
                f.write('\n')

        time_dict['total'] = time_dict.get('total', 0) + (time.time() - start_time)

    # ── Timing summary ────────────────────────────────────────────────────────

    avg_data_loading = (
        time_dict.get('fetch input', 0.0) / frames_to_run
    )

    avg_graph_construction = (
        time_dict.get('gen graph', 0.0) / frames_to_run
    )

    avg_gnn_inference = (
        time_dict.get('gnn inference', 0.0) / frames_to_run
    )

    # Post-processing = box decoding + NMS / scoring / detection processing
    avg_post_processing = (
        time_dict.get('decode box', 0.0)
        + time_dict.get('nms', 0.0)
    ) / frames_to_run

    avg_end_to_end = (
    time_dict.get('total', 0.0) / frames_to_run
    )

    print("\n" + "=" * 60)
    print(f"  AVERAGE TIMING SUMMARY — {frames_to_run} frames")
    print("=" * 60)

    print(f"  Data Loading:        {avg_data_loading * 1000:>10.2f} ms")
    print(f"  Graph Construction:  {avg_graph_construction * 1000:>10.2f} ms")
    print(f"  GNN Inference:       {avg_gnn_inference * 1000:>10.2f} ms")
    print(f"  Post Processing:     {avg_post_processing * 1000:>10.2f} ms")
    print(f"  End-to-End:          {avg_end_to_end * 1000:>10.2f} ms")

    print("=" * 60)

    print(f"\n  Detections written to: {OUTPUT_DIR}/data/")
    print("  Done.\n")