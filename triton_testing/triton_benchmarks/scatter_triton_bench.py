"""Benchmark real Point-GNN scatter_max calls over KITTI frames.

Requires pointgnn_torch.py and scatter_csr.py in the same directory.
The model and its weights are never modified.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse, statistics, time
import numpy as np
import torch
from dataset.kitti_dataset import KittiDataset
from models.graph_gen import get_graph_generate_fn
from util.config_util import load_config
import pointgnn_torch as pg
import scatter_csr as sc

torch.set_float32_matmul_precision("high")

def get_input_features(cam, kind):
    a = cam.attr
    if kind == "irgb": return a
    if kind == "0rgb": return np.hstack((np.zeros((len(a), 1), dtype=a.dtype), a[:, 1:]))
    if kind == "0000": return np.zeros_like(a)
    if kind == "i000": return np.hstack((a[:, [0]], np.zeros((len(a), 3), dtype=a.dtype)))
    if kind == "i": return a[:, [0]]
    if kind == "rgb": return a[:, 1:4]
    return np.zeros((len(a), 1), dtype=np.float32)

class ScatterTest:
    def __init__(self, fn, capture=False):
        self.fn, self.capture = fn, capture
        self.events, self.meta, self.outputs = [], [], []

    def __call__(self, features, dst, V):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = self.fn(features, dst, V)
        end.record()
        self.events.append((start, end))
        self.meta.append((features.shape[0], V, features.shape[1]))
        if self.capture: self.outputs.append(result.detach())
        return result

    def elapsed(self):
        return [a.elapsed_time(b) for a, b in self.events]

def forward(model, inputs, reducer, capture=False):
    test = ScatterTest(reducer, capture)
    old = pg.scatter_max
    pg.scatter_max = test
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    try:
        torch.cuda.synchronize()
        begin.record()
        logits, boxes = model.forward(*inputs)
        end.record()
        end.synchronize()
        ms = begin.elapsed_time(end)
    finally:
        pg.scatter_max = old
    return (logits, boxes), ms, test

def validate(a, b, label):
    if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
        diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        raise AssertionError(f"{label} mismatch: max absolute diff={diff}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint_path")
    p.add_argument("--dataset_root_dir", default="../kitti/")
    p.add_argument("--dataset_split_file", default="")
    p.add_argument("--weights_dir", default="triton_weights")
    p.add_argument("--n_frames", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--skip_frames", default="13")
    args = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    if args.n_frames < 1: raise ValueError("--n_frames must be positive")

    config = load_config(os.path.join(args.checkpoint_path, "config"))
    split = args.dataset_split_file or os.path.join(args.dataset_root_dir, "3DOP_splits/val.txt")
    dataset = KittiDataset(
        os.path.join(args.dataset_root_dir, "image/training/image_2"),
        os.path.join(args.dataset_root_dir, "velodyne/training/velodyne/"),
        os.path.join(args.dataset_root_dir, "calib/training/calib/"),
        os.path.join(args.dataset_root_dir, "labels/training/label_2"),
        split, num_classes=config["num_classes"])
    graph_generate_fn = get_graph_generate_fn(config["graph_gen_method"])

    def make_inputs(i):
        cam = dataset.get_cam_points_in_image_with_rgb(i, config["downsample_by_voxel_size"])
        verts, _, edges = graph_generate_fn(cam.xyz, **config["runtime_graph_gen_kwargs"])
        return (verts[0], get_input_features(cam, config["input_features"]),
                verts[1], edges[0], edges[1])

    skip = {int(x) for x in args.skip_frames.split(",") if x.strip()}
    frames = [i for i in range(dataset.num_files) if i not in skip][:args.n_frames]
    if not frames: raise RuntimeError("No valid frames selected")

    # Constructor and weight path come directly from your run_triton.py.
    model = pg.PointGNN(weights_dir=args.weights_dir, num_classes=dataset.num_classes, device="cuda")
    pytorch_fn = pg.scatter_max
    triton_fn = sc.scatter_max_csr

    print(f"GPU: {torch.cuda.get_device_name(0)} | frames: {frames}")
    print(f"Weights: {args.weights_dir} | TF32 + existing Triton gather unchanged")
    warm = make_inputs(frames[0])
    for _ in range(args.warmup):
        forward(model, warm, pytorch_fn)
        forward(model, warm, triton_fn)
    del warm
    print("Warmup complete.\\n")

    results = []
    for idx, frame in enumerate(frames):
        inputs = make_inputs(frame)
        order = ("pytorch", "triton") if idx % 2 == 0 else ("triton", "pytorch")
        runs = {}
        for mode in order:
            fn = pytorch_fn if mode == "pytorch" else triton_fn
            output, forward_ms, test = forward(model, inputs, fn, capture=True)
            runs[mode] = (output, forward_ms, test)

        (p_logits, p_boxes), pt_forward, pt = runs["pytorch"]
        (t_logits, t_boxes), tr_forward, tr = runs["triton"]
        if pt.meta != tr.meta: raise AssertionError(f"Frame {frame}: different scatter shapes")
        if len(pt.outputs) != len(tr.outputs): raise AssertionError("Different scatter call counts")

        for k, (p_out, t_out) in enumerate(zip(pt.outputs, tr.outputs), 1):
            if not torch.equal(p_out, t_out): 
                diff = (p_out - t_out).abs().max().item()
                raise AssertionError(f"Frame {frame}, scatter call {k}: max diff={diff}")
        validate(p_logits, t_logits, f"Frame {frame} logits")
        validate(p_boxes, t_boxes, f"Frame {frame} boxes")

        pt_times, tr_times = pt.elapsed(), tr.elapsed()
        pt_total, tr_total = sum(pt_times), sum(tr_times)
        results.append((pt_forward, tr_forward, pt_total, tr_total))
        print(f"Frame {frame}: PASS ({len(pt_times)} scatter calls)")
        print(f"  GNN forward: {pt_forward:.2f} -> {tr_forward:.2f} ms | {pt_forward / tr_forward:.2f}x")
        print(f"  Scatter sum: {pt_total:.2f} -> {tr_total:.2f} ms | {pt_total / tr_total:.2f}x")
        for k, (meta, a, b) in enumerate(zip(pt.meta, pt_times, tr_times), 1):
            E, V, F = meta
            print(f"    Scatter {k}: E={E:,}, V={V:,}, F={F} | {a:.2f} -> {b:.2f} ms | {a / b:.2f}x")
        del inputs, runs, pt, tr

    means = [statistics.mean(row[i] for row in results) for i in range(4)]
    pf, tf, ps, ts = means
    print(f"\\n{'=' * 66}\\nMEAN OVER {len(results)} REAL FRAMES")
    print(f"GNN forward:   {pf:.2f} -> {tf:.2f} ms | {pf / tf:.2f}x | saved {pf - tf:.2f} ms")
    print(f"Scatter total: {ps:.2f} -> {ts:.2f} ms | {ps / ts:.2f}x | saved {ps - ts:.2f} ms")
    print("Correctness: PASS, all scatter outputs and final logits/boxes exact")

if __name__ == "__main__": main()
