"""Point-GNN neighbour-cap experiment: full detections + end-to-end latency.

Independent experiment file. Does not modify Point-GNN source, checkpoints,
model weights, configuration, or the existing run_improved_model.py.

Run from the Point-GNN repository root. Uses the same PyTorch/Triton model and
KITTI postprocessing as run_improved_model.py, with only GNN graph neighbour cap changed.
"""

import argparse
import copy
import csv
import json
import os
import random
import subprocess
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np


def get_input_features(cam, input_type):
    # Mirrors run_improved_model.py; not a different feature pipeline.
    attr = cam.attr
    if input_type == "irgb":
        return attr
    if input_type == "0rgb":
        return np.hstack((np.zeros((attr.shape[0], 1), dtype=attr.dtype), attr[:, 1:]))
    if input_type == "0000":
        return np.zeros_like(attr)
    if input_type == "i000":
        return np.hstack((attr[:, [0]], np.zeros((attr.shape[0], 3), dtype=attr.dtype)))
    if input_type == "i":
        return attr[:, [0]]
    if input_type == "rgb":
        return attr[:, 1:4]
    return np.zeros((attr.shape[0], 1), dtype=np.float32)


def make_neighbour_cases(config, neighbour_caps):
    """Limit ONLY GNN graph (edges_list[1]); raw pooling graph is unchanged."""
    if config.get("graph_gen_method") != "multi_level_local_graph_v3":
        raise ValueError("Expected multi_level_local_graph_v3 for this PyTorch Point-GNN model")
    graph_cfg = config["runtime_graph_gen_kwargs"]
    levels = graph_cfg["level_configs"]
    if len(levels) != 2 or [x["graph_level"] for x in levels] != [0, 1]:
        raise ValueError("Expected two graph levels: raw pooling (0) and GNN (1)")
    for level in levels:
        if level["graph_gen_method"] != "disjointed_rnn_local_graph_v3":
            raise ValueError("Unexpected graph generation method")
    if "num_neighbors" not in levels[1]["graph_gen_kwargs"]:
        raise ValueError("Checkpoint graph configuration has no num_neighbors")

    # No uncapped condition is inserted. Only caps explicitly requested by user.
    caps = []
    for cap in neighbour_caps:
        if cap <= 0:
            raise ValueError("Neighbour caps must be positive integers")
        if cap not in caps:
            caps.append(cap)
    if not caps:
        raise ValueError("Provide at least one neighbour cap")

    cases = []
    for cap in caps:
        kwargs = copy.deepcopy(graph_cfg)
        kwargs["level_configs"][1]["graph_gen_kwargs"]["num_neighbors"] = cap
        cases.append({"label": f"K{cap}", "neighbour_cap": cap,
                      "folder": f"neighbours_{cap}", "graph_kwargs": kwargs})
    # Largest requested cap is a comparison REFERENCE, not the original
    # full-graph inference baseline (which may have no cap).
    reference = max(caps)
    return cases, f"K{reference}", int(levels[1]["graph_gen_kwargs"]["num_neighbors"])


def seed_for_frame(seed, frame_idx):
    # Stable across neighbour-cap conditions; protects reproducibility if sampling is random.
    value = (int(seed) + 1000003 * int(frame_idx)) % (2 ** 32)
    random.seed(value)
    np.random.seed(value)


def occlusion(dataset, label, xyz):
    # Exactly the occlusion-factor calculation in run_improved_model.py.
    if xyz.shape[0] == 0:
        return 0.0
    normals, lower, upper = dataset.box3d_to_normals(label)
    projected = np.matmul(xyz, np.transpose(normals))
    x_cover = (np.max(projected[:, 0]) - np.min(projected[:, 0])) / (upper[0] - lower[0])
    y_cover = (np.max(projected[:, 1]) - np.min(projected[:, 1])) / (upper[1] - lower[1])
    z_cover = (np.max(projected[:, 2]) - np.min(projected[:, 2])) / (upper[2] - lower[2])
    return x_cover * y_cover * z_cover


def class_maps(label_method):
    # Mirrors the supplied run_improved_model.py.
    if label_method == "yaw":
        return ({"Background": 0, "Car": 1, "Pedestrian": 3, "Cyclist": 5, "DontCare": 7},
                ["Background", "Car", "Car", "Pedestrian", "Pedestrian", "Cyclist", "Cyclist", "DontCare"])
    if label_method == "Car":
        return ({"Background": 0, "Car": 1, "DontCare": 3},
                ["Background", "Car", "Car", "DontCare"])
    return ({"Background": 0, "Pedestrian": 1, "Cyclist": 3, "DontCare": 5},
            ["Background", "Pedestrian", "Pedestrian", "Cyclist", "Cyclist", "DontCare"])


def postprocess_detections(logits, pred_box, last_layer_points_xyz, calib,
                           dataset, config, box_decoding_fn, box_encoding_len,
                           nms, Points, use_box_merge, use_box_score):
    """Same decoding, filtering, NMS and KITTI rows as run_improved_model.py."""
    num_classes = dataset.num_classes
    label_map, all_class_name = class_maps(config["label_method"])

    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits_shifted)
    probs /= probs.sum(axis=1, keepdims=True)

    box_labels = np.tile(np.expand_dims(np.arange(num_classes), axis=0),
                         (probs.shape[0], 1)).reshape(-1)
    box_probs = probs.reshape(-1)
    expected_shape = (probs.shape[0], num_classes, box_encoding_len)
    if pred_box.shape != expected_shape:
        raise RuntimeError(f"Expected box shape {expected_shape}, got {pred_box.shape}")
    pred_boxes = pred_box.reshape(-1, 1, box_encoding_len)
    boxes_centers = np.tile(np.expand_dims(last_layer_points_xyz, axis=1),
                            (1, num_classes, 1)).reshape(-1, 3)
    decoded_boxes = box_decoding_fn(np.expand_dims(box_labels, axis=1),
                                    boxes_centers, pred_boxes, label_map)

    box_mask = ((box_labels > 0) * (box_labels < num_classes - 1) *
                (box_probs > 1.0 / num_classes))
    box_indices = np.nonzero(box_mask)[0]
    pred_labels = []
    if box_indices.size != 0:
        box_labels_f = box_labels[box_indices].copy()
        box_probs_f = box_probs[box_indices].copy()
        decoded_boxes_f = decoded_boxes[box_indices, 0]
        box_labels_f[box_labels_f == 2] = 1
        box_labels_f[box_labels_f == 4] = 3
        box_labels_f[box_labels_f == 6] = 5

        if use_box_merge and use_box_score:
            class_labels, detection_boxes_3d, detection_scores, nms_indices = (
                nms.nms_boxes_3d_uncertainty(
                    box_labels_f, decoded_boxes_f, box_probs_f,
                    overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                    overlapped_thres=config["nms_overlapped_thres"],
                    appr_factor=100.0, top_k=-1,
                    attributes=np.arange(len(box_indices))))
        else:
            class_labels, detection_boxes_3d, detection_scores, nms_indices = (
                nms.nms_boxes_3d(
                    box_labels_f, decoded_boxes_f, box_probs_f,
                    overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                    overlapped_thres=config["nms_overlapped_thres"],
                    appr_factor=100.0, top_k=-1,
                    attributes=np.arange(len(box_indices))))

        box_probs_f = detection_scores
        point_indices_all = np.clip(box_indices // num_classes, 0,
                                    last_layer_points_xyz.shape[0] - 1)
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
            trunc = 1.0 - (clip_ymax - clip_ymin) * (clip_xmax - clip_xmin) / max(
                (ymax - ymin) * (xmax - xmin), 1e-6)
            if trunc > 0.4:
                continue

            x3d, y3d, z3d, length, height, width, yaw = detection_boxes_3d[i]
            if length <= 0:
                continue
            score = box_probs_f[i]
            if use_box_score:
                tmp_label = {"x3d": x3d, "y3d": y3d, "z3d": z3d,
                             "yaw": yaw, "height": height, "width": width,
                             "length": length}
                inside_mask = dataset.sel_xyz_in_box3d(tmp_label, points_for_scoring)
                score = (1.0 + occlusion(dataset, tmp_label,
                                         points_for_scoring[inside_mask])) * score

            pred_labels.append((class_name, -1, -1, 0,
                                clip_xmin, clip_ymin, clip_xmax, clip_ymax,
                                height, width, length, x3d, y3d, z3d,
                                yaw, float(score)))
    return pred_labels


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)), "min": float(x.min()),
            "max": float(x.max()), "std": float(x.std())}


def write_csv(filename, rows):
    if not rows:
        return
    with filename.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Neighbour-cap sweep with KITTI detections and full inference timing")
    p.add_argument("checkpoint_path", help="Checkpoint directory containing the 'config' file")
    p.add_argument("--dataset_root_dir", default="../kitti/")
    p.add_argument("--dataset_split_file", default="")
    p.add_argument("--weights_dir", default="triton_testing/torch_weights")
    p.add_argument("--n_frames", type=int, default=100)
    p.add_argument("--all-frames", action="store_true", help="Use the entire KITTI validation split")
    p.add_argument("--frame_start", type=int, default=0)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--neighbours", "--neighbors", dest="neighbours", type=int, nargs="+",
                   default=[64, 96, 128, 192, 256],
                   help="Maximum incoming GNN neighbours per vertex; positive integers ONLY")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results",
                   help="Parent directory; this script writes experiment2_neighbours_detections/ beneath it")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite this experiment's existing results (never affects other runs)")
    p.add_argument("--no-box-merge", dest="use_box_merge", action="store_false", default=True)
    p.add_argument("--no-box-score", dest="use_box_score", action="store_false", default=True)
    p.add_argument("--evaluate", action="store_true",
                   help="Run native KITTI evaluator after each condition; requires FULL val split")
    p.add_argument("--evaluator", default="kitti_native_evaluation/evaluate_object_offline")
    return p.parse_args()


def main():
    args = parse_args()
    if args.n_frames < 1 or args.frame_start < 0 or args.frame_stride < 1 or args.warmup < 0:
        raise ValueError("Invalid frame selection or warmup")

    import torch
    from dataset.kitti_dataset import KittiDataset, Points
    from models.graph_gen import get_graph_generate_fn
    from models.box_encoding import get_box_decoding_fn, get_encoding_len
    from models import nms
    from util.config_util import load_config
    from pointgnn_torch import PointGNN

    config_file = Path(args.checkpoint_path) / "config"
    if not config_file.is_file():
        raise FileNotFoundError(config_file)
    config = load_config(str(config_file))
    cases, reference_label, original_cap = make_neighbour_cases(config, args.neighbours)

    root = Path(args.dataset_root_dir)
    split_file = Path(args.dataset_split_file) if args.dataset_split_file else root / "3DOP_splits/val.txt"
    dataset = KittiDataset(str(root / "image/training/image_2"),
                           str(root / "velodyne/training/velodyne"),
                           str(root / "calib/training/calib"),
                           str(root / "labels/training/label_2"),
                           str(split_file), num_classes=config["num_classes"])
    if args.all_frames:
        if args.frame_start != 0 or args.frame_stride != 1:
            raise ValueError("--all-frames requires default frame_start=0 and frame_stride=1")
        frame_indices = list(range(dataset.num_files))
    else:
        frame_indices = list(range(args.frame_start, dataset.num_files,
                                   args.frame_stride))[:args.n_frames]
    if not frame_indices:
        raise ValueError("No dataset frames selected")

    full_validation = (len(frame_indices) == dataset.num_files and
                       frame_indices[0] == 0 and frame_indices[-1] == dataset.num_files - 1 and
                       args.frame_stride == 1)
    if args.evaluate and not full_validation:
        raise ValueError("Native KITTI AP evaluation requires the full validation split. "
                         "Use --all-frames --evaluate; --n_frames 100 is only a sample.")
    evaluator = Path(args.evaluator)
    if args.evaluate and not evaluator.is_file():
        raise FileNotFoundError(f"Native KITTI evaluator not found: {evaluator}. Build it first.")

    if not torch.cuda.is_available():
        raise RuntimeError("The supplied pointgnn_torch.py uses CUDA/Triton; run on your Jetson")
    torch.set_float32_matmul_precision("high")
    graph_generate_fn = get_graph_generate_fn(config["graph_gen_method"])
    box_decoding_fn = get_box_decoding_fn(config["box_encoding_method"])
    box_encoding_len = get_encoding_len(config["box_encoding_method"])
    model = PointGNN(weights_dir=args.weights_dir,
                     num_classes=dataset.num_classes, device="cuda")

    out = Path(args.output_dir) / "experiment2_neighbours_detections"
    if out.exists() and (out / "metadata.json").exists() and not args.overwrite:
        raise FileExistsError(f"Prior run exists in {out}. Choose another --output_dir or use --overwrite.")
    out.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for old in (out / "summary.csv", out / "per_frame.csv"):
            if old.exists():
                old.unlink()
    for case in cases:
        data_dir = out / case["folder"] / "data"
        if args.overwrite:
            for old in (data_dir.parent / "per_frame.csv",
                        data_dir.parent / "evaluation.txt"):
                if old.exists():
                    old.unlink()
        if data_dir.exists() and list(data_dir.glob("*.txt")):
            if not args.overwrite:
                raise FileExistsError(f"Existing detection files: {data_dir}; use --overwrite")
            # Only remove files from this run's explicitly scoped condition directory.
            for txt in data_dir.glob("*.txt"):
                txt.unlink()
        data_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "description": "KITTI detections + full inference timing for GNN neighbour-cap sweep",
        "checkpoint_path": str(args.checkpoint_path),
        "weights_dir": str(args.weights_dir),
        "dataset_split_file": str(split_file),
        "dataset_num_files": dataset.num_files,
        "selected_frame_indices": frame_indices,
        "full_validation_split": full_validation,
        "reference_condition": reference_label,
        "reference_note": "Largest requested cap; NOT original uncapped inference baseline",
        "original_checkpoint_gnn_neighbour_cap": original_cap,
        "gpu": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "warmup": args.warmup,
        "use_box_merge": args.use_box_merge,
        "use_box_score": args.use_box_score,
        "downsample_by_voxel_size_unchanged": config["downsample_by_voxel_size"],
        "graph_base_voxel_size_unchanged": config["runtime_graph_gen_kwargs"]["base_voxel_size"],
        "raw_pooling_num_neighbors_unchanged": config["runtime_graph_gen_kwargs"]["level_configs"][0]["graph_gen_kwargs"].get("num_neighbors"),
        "original_runtime_graph_gen_kwargs": config["runtime_graph_gen_kwargs"],
        "conditions": [{"label": c["label"], "folder": c["folder"],
                        "gnn_neighbour_cap": c["neighbour_cap"], "graph_kwargs": c["graph_kwargs"]}
                       for c in cases],
        "timing_note": "Each condition runs sequentially with its own data load and output writes. "
                       "End-to-end includes data load, graph generation, torch model.forward, "
                       "full detection postprocessing, and writing KITTI txt files.",
    }
    with (out / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2,
                  default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))

    print(f"\nNeighbour experiment — detections + full inference, on {metadata['gpu']}")
    print(f"Frames per condition: {len(frame_indices)} / {dataset.num_files}")
    print("Conditions: " + ", ".join(c["label"] for c in cases))
    print(f"Reference for speedup: {reference_label} (not the original uncapped baseline)")
    print(f"Output: {out.resolve()}", flush=True)
    if not full_validation:
        print("NOTE: Predictions from a partial split do not constitute a valid full KITTI AP score.")

    summaries, all_rows = [], []
    for case in cases:
        label = case["label"]
        condition_dir = out / case["folder"]
        data_dir = condition_dir / "data"
        print(f"\n--- GNN neighbour cap {label} ---", flush=True)

        # Warm up only the current condition; do not build every graph up front.
        if args.warmup:
            first_idx = frame_indices[0]
            cam0 = dataset.get_cam_points_in_image_with_rgb(
                first_idx, config["downsample_by_voxel_size"])
            f0 = get_input_features(cam0, config["input_features"])
            for _ in range(args.warmup):
                seed_for_frame(args.seed, first_idx)
                vv0, _, ee0 = graph_generate_fn(cam0.xyz, **case["graph_kwargs"])
                logits0, box0 = model.forward(vv0[0], f0, vv0[1], ee0[0], ee0[1])
                torch.cuda.synchronize()
                del vv0, ee0, logits0, box0
            del cam0, f0
            print(f"Warmed: {label}", flush=True)

        rows = []
        for pos, frame_idx in enumerate(frame_indices):
            seed_for_frame(args.seed, frame_idx)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            cam = dataset.get_cam_points_in_image_with_rgb(
                frame_idx, config["downsample_by_voxel_size"])
            calib = dataset.get_calib(frame_idx)
            # The original inference runner reads labels even though it does
            # not use them, so preserve that in the timing comparison.
            box_label_list = dataset.get_label(frame_idx)
            t1 = time.perf_counter()

            vertex_coord_list, keypoint_indices_list, edges_list = graph_generate_fn(
                cam.xyz, **case["graph_kwargs"])
            t2 = time.perf_counter()
            raw_features = get_input_features(cam, config["input_features"])
            raw_xyz, keypoint_xyz = vertex_coord_list[0], vertex_coord_list[1]
            torch.cuda.reset_peak_memory_stats()
            logits, pred_box = model.forward(raw_xyz, raw_features,
                                             keypoint_xyz, edges_list[0], edges_list[1])
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

            last_level = config["model_kwargs"]["layer_configs"][-1]["graph_level"]
            last_layer_points_xyz = vertex_coord_list[last_level + 1]
            pred_labels = postprocess_detections(
                logits, pred_box, last_layer_points_xyz, calib, dataset, config,
                box_decoding_fn, box_encoding_len, nms, Points,
                args.use_box_merge, args.use_box_score)
            t4 = time.perf_counter()

            name = dataset.get_filename(frame_idx)
            with (data_dir / f"{name}.txt").open("w") as f:
                for pred_label in pred_labels:
                    f.write(" ".join(str(x) for x in pred_label) + "\n")
                if not pred_labels:
                    f.write("\n")
            t5 = time.perf_counter()

            row = {
                "condition": label,
                "frame_idx": frame_idx,
                "frame_id": name,
                "data_load_ms": (t1 - t0) * 1000.0,
                "graph_ms": (t2 - t1) * 1000.0,
                "gnn_ms": (t3 - t2) * 1000.0,
                "postprocess_ms": (t4 - t3) * 1000.0,
                "write_ms": (t5 - t4) * 1000.0,
                "graph_plus_gnn_ms": (t3 - t1) * 1000.0,
                "end_to_end_ms": (t5 - t0) * 1000.0,
                "raw_points": len(vertex_coord_list[0]),
                "graph_vertices": len(keypoint_xyz),
                "pooling_edges_e0": len(edges_list[0]),
                "gnn_edges_e1": len(edges_list[1]),
                "gnn_mean_in_degree": len(edges_list[1]) / max(1, len(keypoint_xyz)),
                "detections": len(pred_labels),
                "cuda_peak_allocated_mb": peak_mb,
            }
            rows.append(row)
            del cam, calib, box_label_list, vertex_coord_list, keypoint_indices_list
            del edges_list, raw_features, raw_xyz, keypoint_xyz, logits, pred_box, pred_labels
            if (pos + 1) % 10 == 0 or pos + 1 == len(frame_indices):
                print(f"  {label}: {pos + 1}/{len(frame_indices)} frames", flush=True)

        write_csv(condition_dir / "per_frame.csv", rows)
        all_rows.extend(rows)
        summary = {"condition": label, "gnn_neighbour_cap": case["neighbour_cap"],
                   "frames": len(rows), "detection_files": len(rows),
                   "detections_total": sum(row["detections"] for row in rows)}
        for key in ("data_load_ms", "graph_ms", "gnn_ms", "postprocess_ms",
                    "write_ms", "graph_plus_gnn_ms", "end_to_end_ms",
                    "raw_points", "graph_vertices", "pooling_edges_e0",
                    "gnn_edges_e1", "gnn_mean_in_degree", "detections", "cuda_peak_allocated_mb"):
            for stat, value in stats([r[key] for r in rows]).items():
                summary[f"{key}_{stat}"] = value
        summaries.append(summary)

        if args.evaluate:
            gt_dir = root / "labels/training/label_2"
            command = [str(evaluator.resolve()), str(gt_dir.resolve()),
                       str(condition_dir.resolve())]
            print("Evaluating full validation split: " + " ".join(command), flush=True)
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, check=False)
            (condition_dir / "evaluation.txt").write_text(result.stdout)
            print(result.stdout[-4000:], flush=True)
            if result.returncode != 0:
                raise RuntimeError(f"KITTI evaluation failed for {label}; "
                                   f"see {condition_dir / 'evaluation.txt'}")

        # Save after every condition; compute reference speedup once its cap
        # has been measured. No original/uncapped condition is run.
        ref_rows = [entry for entry in summaries if entry["condition"] == reference_label]
        if ref_rows:
            reference = ref_rows[0]
            for entry in summaries:
                for key in ("graph_plus_gnn_ms_mean", "end_to_end_ms_mean"):
                    if entry[key] > 0:
                        entry[key.replace("_ms_mean", "_speedup_vs_reference")] = (
                            reference[key] / entry[key])
        write_csv(out / "summary.csv", summaries)
        write_csv(out / "per_frame.csv", all_rows)
        print(f"  Finished {label}: graph+GNN {summary['graph_plus_gnn_ms_mean']:.1f} ms | "
              f"end-to-end {summary['end_to_end_ms_mean']:.1f} ms | "
              f"mean detections/frame {summary['detections_mean']:.1f}", flush=True)
        torch.cuda.empty_cache()

    print("\n" + "=" * 108)
    print(f"{'Cap':>7} {'Vertices':>10} {'Edges':>11} {'Graph ms':>10} {'GNN ms':>10} "
          f"{'Post ms':>10} {'Total ms':>10} {'vs ref':>11} {'Boxes/frame':>12}")
    print("-" * 108)
    for s in summaries:
        print(f"{s['condition']:>7} {s['graph_vertices_mean']:>10.1f} "
              f"{s['gnn_edges_e1_mean']:>11.1f} {s['graph_ms_mean']:>10.1f} "
              f"{s['gnn_ms_mean']:>10.1f} {s['postprocess_ms_mean']:>10.1f} "
              f"{s['end_to_end_ms_mean']:>10.1f} "
              f"{s['end_to_end_speedup_vs_reference']:>10.2f}x "
              f"{s['detections_mean']:>12.1f}")
    print(f"\nDetections: {out}/neighbours_<K>/data/*.txt")
    print(f"CSV summaries: {out}/summary.csv and {out}/per_frame.csv")
    if not args.evaluate:
        print("AP is NOT computed from raw detection count; use --all-frames --evaluate ")
        print("after building the repository's kitti_native_evaluation executable.")


if __name__ == "__main__":
    main()
