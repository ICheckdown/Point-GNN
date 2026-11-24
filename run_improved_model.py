"""
run_triton.py
End-to-end Point-GNN timing with TF32 + Triton gather + expandable CUDA segments.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time, argparse
import numpy as np
import torch
from tqdm import tqdm

from dataset.kitti_dataset import KittiDataset, Points
from models.graph_gen import get_graph_generate_fn
from models.box_encoding import get_box_decoding_fn, get_encoding_len
from models import nms
from util.config_util import load_config
from pointgnn_torch import PointGNN

parser = argparse.ArgumentParser(description="Run Point-GNN with PyTorch TF32 + Triton gather.")
parser.add_argument("checkpoint_path", type=str)
parser.add_argument("--dataset_root_dir", type=str, default="../kitti/")
parser.add_argument("--dataset_split_file", type=str, default="")
parser.add_argument("--n_frames", type=int, default=100)
parser.add_argument("--weights_dir", type=str, default="triton_weights")
parser.add_argument("--output_dir", type=str, default="pytorch_results")
parser.add_argument("--no-box-merge", dest="use_box_merge", action="store_false", default=True)
parser.add_argument("--no-box-score", dest="use_box_score", action="store_false", default=True)
args = parser.parse_args()

SKIP_FRAMES = set()

CHECKPOINT_PATH = args.checkpoint_path
CONFIG_PATH = os.path.join(CHECKPOINT_PATH, "config")
if not os.path.isfile(CONFIG_PATH): raise FileNotFoundError(f"Point-GNN config not found:\n{CONFIG_PATH}")
config = load_config(CONFIG_PATH)

DATASET_SPLIT_FILE = args.dataset_split_file if args.dataset_split_file else os.path.join(args.dataset_root_dir, "3DOP_splits/val.txt")
OUTPUT_DIR = args.output_dir
OUTPUT_DATA_DIR = os.path.join(OUTPUT_DIR, "data")
os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

USE_BOX_MERGE = args.use_box_merge
USE_BOX_SCORE = args.use_box_score

dataset = KittiDataset(os.path.join(args.dataset_root_dir, "image/training/image_2"), os.path.join(args.dataset_root_dir, "velodyne/training/velodyne/"), os.path.join(args.dataset_root_dir, "calib/training/calib/"), os.path.join(args.dataset_root_dir, "labels/training/label_2"), DATASET_SPLIT_FILE, num_classes=config["num_classes"])

NUM_CLASSES = dataset.num_classes
N_FRAMES = min(args.n_frames, dataset.num_files)
graph_generate_fn = get_graph_generate_fn(config["graph_gen_method"])
BOX_ENCODING_LEN = get_encoding_len(config["box_encoding_method"])
box_decoding_fn = get_box_decoding_fn(config["box_encoding_method"])

if config["label_method"] == "yaw":
    label_map = {"Background": 0, "Car": 1, "Pedestrian": 3, "Cyclist": 5, "DontCare": 7}
    all_class_name = ["Background", "Car", "Car", "Pedestrian", "Pedestrian", "Cyclist", "Cyclist", "DontCare"]
elif config["label_method"] == "Car":
    label_map = {"Background": 0, "Car": 1, "DontCare": 3}
    all_class_name = ["Background", "Car", "Car", "DontCare"]
else:
    label_map = {"Background": 0, "Pedestrian": 1, "Cyclist": 3, "DontCare": 5}
    all_class_name = ["Background", "Pedestrian", "Pedestrian", "Cyclist", "Cyclist", "DontCare"]

def get_input_features(cam, input_type):
    attr = cam.attr
    if input_type == "irgb": return attr
    if input_type == "0rgb": return np.hstack([np.zeros((attr.shape[0], 1), dtype=attr.dtype), attr[:, 1:]])
    if input_type == "0000": return np.zeros_like(attr)
    if input_type == "i000": return np.hstack([attr[:, [0]], np.zeros((attr.shape[0], 3), dtype=attr.dtype)])
    if input_type == "i": return attr[:, [0]]
    if input_type == "rgb": return attr[:, 1:4]
    return np.zeros((attr.shape[0], 1), dtype=np.float32)

def occlusion(label, xyz):
    if xyz.shape[0] == 0: return 0.0
    normals, lower, upper = dataset.box3d_to_normals(label)
    projected = np.matmul(xyz, np.transpose(normals))
    x_cover = (np.max(projected[:, 0]) - np.min(projected[:, 0])) / (upper[0] - lower[0])
    y_cover = (np.max(projected[:, 1]) - np.min(projected[:, 1])) / (upper[1] - lower[1])
    z_cover = (np.max(projected[:, 2]) - np.min(projected[:, 2])) / (upper[2] - lower[2])
    return x_cover * y_cover * z_cover

'''
print("\n" + "=" * 72)
print("  INITIALIZING Point-GNN PYTORCH")
print("=" * 72)
'''

model = PointGNN(weights_dir=args.weights_dir, num_classes=NUM_CLASSES, device="cuda")

print("\nWarming up Point-GNN...")
cam0 = dataset.get_cam_points_in_image_with_rgb(0, config["downsample_by_voxel_size"])
vcl0, kil0, el0 = graph_generate_fn(cam0.xyz, **config["runtime_graph_gen_kwargs"])
raw_features0 = get_input_features(cam0, config["input_features"])
for _ in range(3):
    _logits, _boxes = model.forward(vcl0[0], raw_features0, vcl0[1], el0[0], el0[1])
    del _logits, _boxes
if torch.cuda.is_available(): torch.cuda.synchronize()
del cam0, vcl0, kil0, el0, raw_features0
if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
print("Warmup done.\n")

t_data_list, t_graph_list, t_gnn_list, t_post_list, t_total_list = [], [], [], [], []

for frame_idx in tqdm(range(N_FRAMES)):
    if frame_idx in SKIP_FRAMES: continue

    t0 = time.perf_counter()
    cam = dataset.get_cam_points_in_image_with_rgb(frame_idx, config["downsample_by_voxel_size"])
    calib = dataset.get_calib(frame_idx)
    box_label_list = dataset.get_label(frame_idx)
    t1 = time.perf_counter()

    vertex_coord_list, keypoint_indices_list, edges_list = graph_generate_fn(cam.xyz, **config["runtime_graph_gen_kwargs"])
    t2 = time.perf_counter()

    raw_features = get_input_features(cam, config["input_features"])
    raw_xyz, keypoint_xyz = vertex_coord_list[0], vertex_coord_list[1]
    logits, pred_box = model.forward(raw_xyz, raw_features, keypoint_xyz, edges_list[0], edges_list[1])
    t3 = time.perf_counter()

    last_layer_graph_level = config["model_kwargs"]["layer_configs"][-1]["graph_level"]
    last_layer_points_xyz = vertex_coord_list[last_layer_graph_level + 1]

    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits_shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    del logits_shifted

    box_labels = np.tile(np.expand_dims(np.arange(NUM_CLASSES), axis=0), (probs.shape[0], 1)).reshape(-1)
    box_probs = probs.reshape(-1)
    expected_shape = (probs.shape[0], NUM_CLASSES, BOX_ENCODING_LEN)
    if pred_box.shape != expected_shape: raise RuntimeError(f"Unexpected pred_box shape. Expected {expected_shape}, received {pred_box.shape}.")

    pred_boxes = pred_box.reshape(-1, 1, BOX_ENCODING_LEN)
    boxes_centers = np.tile(np.expand_dims(last_layer_points_xyz, axis=1), (1, NUM_CLASSES, 1)).reshape(-1, 3)
    decoded_boxes = box_decoding_fn(np.expand_dims(box_labels, axis=1), boxes_centers, pred_boxes, label_map)

    box_mask = (box_labels > 0) * (box_labels < NUM_CLASSES - 1) * (box_probs > 1.0 / NUM_CLASSES)
    box_indices = np.nonzero(box_mask)[0]
    pred_labels = []

    if box_indices.size != 0:
        box_labels_f = box_labels[box_indices].copy()
        box_probs_f = box_probs[box_indices].copy()
        decoded_boxes_f = decoded_boxes[box_indices, 0]
        box_labels_f[box_labels_f == 2] = 1
        box_labels_f[box_labels_f == 4] = 3
        box_labels_f[box_labels_f == 6] = 5

        if USE_BOX_MERGE and USE_BOX_SCORE:
            class_labels, detection_boxes_3d, detection_scores, nms_indices = nms.nms_boxes_3d_uncertainty(box_labels_f, decoded_boxes_f, box_probs_f, overlapped_fn=nms.overlapped_boxes_3d_fast_poly, overlapped_thres=config["nms_overlapped_thres"], appr_factor=100.0, top_k=-1, attributes=np.arange(len(box_indices)))
        else:
            class_labels, detection_boxes_3d, detection_scores, nms_indices = nms.nms_boxes_3d(box_labels_f, decoded_boxes_f, box_probs_f, overlapped_fn=nms.overlapped_boxes_3d_fast_poly, overlapped_thres=config["nms_overlapped_thres"], appr_factor=100.0, top_k=-1, attributes=np.arange(len(box_indices)))

        box_probs_f = detection_scores
        point_indices_all = np.clip(box_indices // NUM_CLASSES, 0, last_layer_points_xyz.shape[0] - 1)
        points_for_scoring = last_layer_points_xyz[point_indices_all]
        detection_boxes_3d_corners = nms.boxes_3d_to_corners(detection_boxes_3d)

        for i in range(len(detection_boxes_3d_corners)):
            corners_cam = Points(xyz=detection_boxes_3d_corners[i], attr=None)
            corners_img = dataset.cam_points_to_image(corners_cam, calib)
            corners_xy = corners_img.xyz[:, :2]
            class_name = all_class_name[class_labels[i]]
            xmin, ymin = np.amin(corners_xy, axis=0)
            xmax, ymax = np.amax(corners_xy, axis=0)
            clip_xmin, clip_ymin = max(xmin, 0.0), max(ymin, 0.0)
            clip_xmax, clip_ymax = min(xmax, 1242.0), min(ymax, 375.0)
            trunc = 1.0 - (clip_ymax - clip_ymin) * (clip_xmax - clip_xmin) / max((ymax - ymin) * (xmax - xmin), 1e-6)
            if trunc > 0.4: continue

            x3d, y3d, z3d, length, height, width, yaw = detection_boxes_3d[i]
            if length <= 0: continue

            score = box_probs_f[i]
            if USE_BOX_SCORE:
                tmp_label = {"x3d": x3d, "y3d": y3d, "z3d": z3d, "yaw": yaw, "height": height, "width": width, "length": length}
                inside_mask = dataset.sel_xyz_in_box3d(tmp_label, points_for_scoring)
                score = (1.0 + occlusion(tmp_label, points_for_scoring[inside_mask])) * score

            pred_labels.append((class_name, -1, -1, 0, clip_xmin, clip_ymin, clip_xmax, clip_ymax, height, width, length, x3d, y3d, z3d, yaw, float(score)))

    t4 = time.perf_counter()

    filename = os.path.join(OUTPUT_DATA_DIR, dataset.get_filename(frame_idx) + ".txt")
    with open(filename, "w") as f:
        for pred_label in pred_labels: f.write(" ".join(str(x) for x in pred_label) + "\n")
        if not pred_labels: f.write("\n")
    t5 = time.perf_counter()

    t_data_list.append((t1 - t0) * 1000.0)
    t_graph_list.append((t2 - t1) * 1000.0)
    t_gnn_list.append((t3 - t2) * 1000.0)
    t_post_list.append((t4 - t3) * 1000.0)
    t_total_list.append((t5 - t0) * 1000.0)

    del logits, pred_box, probs, box_labels, box_probs, pred_boxes, boxes_centers, decoded_boxes
    del vertex_coord_list, keypoint_indices_list, edges_list, raw_features, raw_xyz, keypoint_xyz
    del cam, calib, box_label_list

arrays = {
    "Data Loading": np.asarray(t_data_list, dtype=np.float64),
    "Graph Construction": np.asarray(t_graph_list, dtype=np.float64),
    "GNN Inference": np.asarray(t_gnn_list, dtype=np.float64),
    "Post Processing": np.asarray(t_post_list, dtype=np.float64),
    "End-to-End": np.asarray(t_total_list, dtype=np.float64)
}

measured = len(t_total_list)
print("\n" + "=" * 72)
print(f"  Point-GNN PYTORCH | {measured} measured frames")
print("=" * 72)
print(f"  {'Stage':<25}{'Mean':>11}{'Min':>11}{'Max':>11}{'Std':>11}")
print("  " + "─" * 70)
for label, arr in arrays.items(): print(f"  {label:<25}{arr.mean():>9.2f}ms{arr.min():>9.2f}ms{arr.max():>9.2f}ms{arr.std():>9.2f}ms")
print("  " + "─" * 70)

gnn, total = arrays["GNN Inference"], arrays["End-to-End"]
print(f"\n  Mean GNN inference: {gnn.mean():.2f} ms")
print(f"  Mean End-to-End:    {total.mean():.2f} ms")
print(f"  Mean time/frame:    {total.mean()/1000.0:.3f} s")
print(f"  Mean FPS:           {1000.0/total.mean():.3f}")

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(f"  CUDA memory: allocated={allocated:.2f} GB | reserved={reserved:.2f} GB | peak_alloc={peak_alloc:.2f} GB | peak_reserved={peak_reserved:.2f} GB")

print("=" * 72)
print(f"Detections written to: {OUTPUT_DATA_DIR}")
