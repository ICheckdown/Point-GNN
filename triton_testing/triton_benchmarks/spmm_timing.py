"""
benchmark_gather_scatter.py
Profiles gather vs scatter-max inside the real PyTorch Point-GNN.
Designed for Jetson Orin memory constraints.
"""

import os, gc, argparse
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from dataset.kitti_dataset import KittiDataset
from models.graph_gen import get_graph_generate_fn
from util.config_util import load_config
import pointgnn_torch as pgnn
from pointgnn_torch import PointGNN


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(description="Profile Point-GNN gather vs scatter-max.")
parser.add_argument("checkpoint_path", type=str)
parser.add_argument("--dataset_root_dir", type=str, default="../kitti/")
parser.add_argument("--dataset_split_file", type=str, default="")
parser.add_argument("--n_frames", type=int, default=100)
parser.add_argument("--weights_dir", type=str, default="triton_weights")
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--triton", action="store_true", default=False)
args = parser.parse_args()

SKIP_FRAMES = {13}


# ============================================================
# CONFIG
# ============================================================

config_path = os.path.join(args.checkpoint_path, "config")
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"Point-GNN config not found:\n{config_path}")

config = load_config(config_path)

split_file = (
    args.dataset_split_file
    if args.dataset_split_file
    else os.path.join(args.dataset_root_dir, "3DOP_splits/val.txt")
)


# ============================================================
# DATASET
# ============================================================

root = args.dataset_root_dir
dataset = KittiDataset(
    os.path.join(root, "image/training/image_2"),
    os.path.join(root, "velodyne/training/velodyne/"),
    os.path.join(root, "calib/training/calib/"),
    os.path.join(root, "labels/training/label_2"),
    split_file,
    num_classes=config["num_classes"],
)

graph_generate_fn = get_graph_generate_fn(config["graph_gen_method"])
N_FRAMES = min(args.n_frames, dataset.num_files)


# ============================================================
# INPUT FEATURES
# ============================================================

def get_input_features(cam, input_type):
    attr = cam.attr
    n = attr.shape[0]

    if input_type == "irgb":
        return attr
    if input_type == "0rgb":
        return np.hstack([np.zeros((n, 1), dtype=attr.dtype), attr[:, 1:]])
    if input_type == "0000":
        return np.zeros_like(attr)
    if input_type == "i000":
        return np.hstack([attr[:, [0]], np.zeros((n, 3), dtype=attr.dtype)])
    if input_type == "i":
        return attr[:, [0]]
    if input_type == "rgb":
        return attr[:, 1:4]
    return np.zeros((n, 1), dtype=np.float32)


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# INITIALIZE MODEL
# ============================================================

print("\n" + "=" * 72)
print("POINT-GNN GATHER / SCATTER PROFILER")
print("=" * 72)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this benchmark.")

print("GPU:", torch.cuda.get_device_name(0))
print("Mode:", "PyTorch + Triton" if args.triton else "PyTorch F.linear + TF32")
print("Requested frames:", N_FRAMES)
print("Skipped frames:", sorted(SKIP_FRAMES))

model = PointGNN(
    weights_dir=args.weights_dir,
    num_classes=dataset.num_classes,
    device="cuda",
    release_cache_each_frame=False,
    use_triton=args.triton,
)


# ============================================================
# WARMUP
# ============================================================

pgnn.GRAPH_PROFILE = False  # disable profiling during warmup

warm_idx = next(i for i in range(N_FRAMES) if i not in SKIP_FRAMES)
cam = dataset.get_cam_points_in_image_with_rgb(warm_idx, config["downsample_by_voxel_size"])
vertex_coord_list, keypoint_indices_list, edges_list = graph_generate_fn(
    cam.xyz, **config["runtime_graph_gen_kwargs"]
)
raw_features = get_input_features(cam, config["input_features"])

print(f"\nWarming up {args.warmup} iteration(s)...")

for _ in range(args.warmup):
    logits, pred_box = model.forward(
        vertex_coord_list[0], raw_features, vertex_coord_list[1],
        edges_list[0], edges_list[1],
    )
    del logits, pred_box

torch.cuda.synchronize()
del cam, vertex_coord_list, keypoint_indices_list, edges_list, raw_features
cleanup()

print("Warmup done.\n")


# ============================================================
# PROFILING
# ============================================================

pgnn.GRAPH_PROFILE = True

totals = defaultdict(float)
frame_gather, frame_scatter, frame_total = [], [], []
profiled_frames = 0

for frame_idx in tqdm(range(N_FRAMES), desc="Profiling gather/scatter"):
    if frame_idx in SKIP_FRAMES:
        continue

    # Clean before frame
    pgnn.reset_graph_profile()
    cleanup()

    # Load data
    cam = dataset.get_cam_points_in_image_with_rgb(frame_idx, config["downsample_by_voxel_size"])

    # Graph generation
    vertex_coord_list, keypoint_indices_list, edges_list = graph_generate_fn(
        cam.xyz, **config["runtime_graph_gen_kwargs"]
    )
    raw_features = get_input_features(cam, config["input_features"])
    raw_xyz = vertex_coord_list[0]
    keypoint_xyz = vertex_coord_list[1]

    # Model forward
    logits, pred_box = model.forward(
        raw_xyz, raw_features, keypoint_xyz, edges_list[0], edges_list[1]
    )

    # Resolve CUDA events
    frame_times = pgnn.get_graph_profile()

    # Accumulate
    for name, ms in frame_times.items():
        totals[name] += ms

    gather_ms = sum(ms for name, ms in frame_times.items() if "gather" in name)
    scatter_ms = sum(ms for name, ms in frame_times.items() if "scatter" in name)

    frame_gather.append(gather_ms)
    frame_scatter.append(scatter_ms)
    frame_total.append(gather_ms + scatter_ms)
    profiled_frames += 1

    # Delete frame outputs and CUDA events, then force memory cleanup
    del logits, pred_box, frame_times
    del raw_features, raw_xyz, keypoint_xyz
    del vertex_coord_list, keypoint_indices_list, edges_list, cam
    pgnn.reset_graph_profile()
    cleanup()


# ============================================================
# DISABLE PROFILING
# ============================================================

pgnn.GRAPH_PROFILE = False
torch.cuda.synchronize()
cleanup()

if profiled_frames == 0:
    raise RuntimeError("No frames were profiled.")


# ============================================================
# MEAN PER OPERATION / CATEGORY TOTALS
# ============================================================

means = {name: total / profiled_frames for name, total in totals.items()}

feature_gather = sum(ms for n, ms in means.items() if "gather_features" in n)
coordinate_gather = sum(
    ms for n, ms in means.items() if "gather_src_xyz" in n or "gather_dst_xyz" in n
)
all_gather = sum(ms for n, ms in means.items() if "gather" in n)
all_scatter = sum(ms for n, ms in means.items() if "scatter" in n)
graph_total = all_gather + all_scatter


def pct(ms):
    return 100.0 * ms / graph_total if graph_total > 0 else 0.0


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 78)
print(f"POINT-GNN GATHER vs SCATTER — {profiled_frames} PROFILED FRAMES")
print("=" * 78)
print(f"{'Operation':<32}{'Mean ms/frame':>16}{'% graph ops':>16}")
print("-" * 78)

for name, ms in sorted(means.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:<32}{ms:>14.3f} ms{pct(ms):>14.2f}%")

print("-" * 78)

categories = [
    ("Feature gathers", feature_gather),
    ("Coordinate gathers", coordinate_gather),
    ("ALL GATHER", all_gather),
    ("ALL SCATTER-MAX", all_scatter),
]
for name, ms in categories:
    print(f"{name:<32}{ms:>14.3f} ms{pct(ms):>14.2f}%")

print("-" * 78)
print(f"{'TOTAL GRAPH OPS':<32}{graph_total:>14.3f} ms{100.0:>14.2f}%")

if all_scatter > 0:
    print(f"\nGather : Scatter = {all_gather / all_scatter:.3f} : 1")


# ============================================================
# PER-FRAME VARIABILITY
# ============================================================

print("\nPer-frame variability:")

for label, data in [
    ("Gather     ", frame_gather),
    ("Scatter-max", frame_scatter),
    ("Combined   ", frame_total),
]:
    arr = np.asarray(data, dtype=np.float64)
    print(
        f"  {label} mean={arr.mean():.3f} ms  min={arr.min():.3f}  "
        f"max={arr.max():.3f}  std={arr.std():.3f}"
    )


# ============================================================
# INTERPRETATION
# ============================================================

print("\nLargest graph-operation target:")

if all_gather > all_scatter:
    print("  GATHER")
elif all_scatter > all_gather:
    print("  SCATTER-MAX")
else:
    print("  Gather and scatter-max are equal.")

print("\n" + "=" * 78)