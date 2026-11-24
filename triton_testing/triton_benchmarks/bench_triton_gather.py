"""
Real Point-GNN gather benchmark.
Compares current index_select, specialized Triton gather, and explicit CSR SpMM.
CSR construction time is excluded from SpMM kernel timing.
"""
import os, gc, argparse
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from dataset.kitti_dataset import KittiDataset
from models.graph_gen import get_graph_generate_fn
from util.config_util import load_config
from triton_gather import triton_gather

SKIP_FRAMES = {13}
CONFIGS = CONFIGS = [(1,32),(1,64),(1,128),(1,256),(2,32),(2,64),(4,32),(4,64),(8,32)]

def cuda_ms(fn, warmup=2, repeat=5):
    for _ in range(warmup):
        y = fn(); del y
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); y = fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e)); del y, s, e
    return float(np.median(times))

def make_csr(src, V, dtype):
    E = src.numel()
    crow = torch.arange(E + 1, device=src.device, dtype=torch.int64)
    vals = torch.ones(E, device=src.device, dtype=dtype)
    return torch.sparse_csr_tensor(crow, src, vals, size=(E,V), device=src.device)


def row(name, xs):
    a = np.asarray(xs, dtype=np.float64)
    return f"{name:<23}{a.mean():>11.3f}{np.median(a):>11.3f}{a.min():>11.3f}{a.max():>11.3f}"

p = argparse.ArgumentParser(description="Point-GNN real-graph gather benchmark")
p.add_argument("checkpoint_path")
p.add_argument("--dataset_root_dir", default=os.path.expanduser("~/kitti/"))
p.add_argument("--dataset_split_file", default="")
p.add_argument("--n_frames", type=int, default=100)
p.add_argument("--feature_dim", type=int, default=300)
p.add_argument("--repeat", type=int, default=5)
p.add_argument("--dtype", choices=["fp32","fp16"], default="fp32")
p.add_argument("--no_csr", action="store_true")
args = p.parse_args()

if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
config = load_config(os.path.join(args.checkpoint_path, "config"))
split_file = args.dataset_split_file or os.path.join(args.dataset_root_dir, "3DOP_splits/val.txt")
dataset = KittiDataset(
    os.path.join(args.dataset_root_dir, "image/training/image_2"),
    os.path.join(args.dataset_root_dir, "velodyne/training/velodyne/"),
    os.path.join(args.dataset_root_dir, "calib/training/calib/"),
    os.path.join(args.dataset_root_dir, "labels/training/label_2"),
    split_file, num_classes=config["num_classes"])
graph_generate_fn = get_graph_generate_fn(config["graph_gen_method"])
dtype = torch.float32 if args.dtype == "fp32" else torch.float16
N = min(args.n_frames, dataset.num_files)

results, shapes = defaultdict(list), []
checked, csr_supported = False, not args.no_csr
print("="*78)
print("POINT-GNN FEATURE GATHER: index_select vs Triton vs explicit CSR SpMM")
print("="*78)
print("GPU:", torch.cuda.get_device_name(0))
print(f"dtype={args.dtype}, F={args.feature_dim}, frames={N}, skip={sorted(SKIP_FRAMES)}\n")

for frame_idx in tqdm(range(N), desc="Gather benchmark"):
    if frame_idx in SKIP_FRAMES: continue
    cam = dataset.get_cam_points_in_image_with_rgb(frame_idx, config["downsample_by_voxel_size"])
    vertex_coord_list, _, edges_list = graph_generate_fn(cam.xyz, **config["runtime_graph_gen_kwargs"])
    V = int(vertex_coord_list[1].shape[0])
    edges = torch.as_tensor(edges_list[1], device="cuda", dtype=torch.int64)
    src = edges[:,0].contiguous()
    E = src.numel()
    x = torch.randn((V,args.feature_dim), device="cuda", dtype=dtype)
    shapes.append((V,E))

    if not checked:
        ref = x.index_select(0,src)
        print(f"Example graph: V={V:,}, E={E:,}, X={tuple(x.shape)}, output={tuple(ref.shape)}")
        print(f"Implicit A=[{E:,},{V:,}], nnz={E:,}, sparsity={(1-1/V)*100:.5f}%")
        for be,bf in CONFIGS:
            tri = triton_gather(x,src,bf,be)
            print(f"  correctness BE={be:<2} BF={bf:<3}: max_diff={(tri-ref).abs().max().item():g}")
            del tri
        del ref
        checked = True

    results["PyTorch index_select"].append(cuda_ms(lambda: x.index_select(0,src), repeat=args.repeat))
    for be,bf in CONFIGS:
        results[f"Triton BE{be} BF{bf}"].append(
            cuda_ms(lambda be=be,bf=bf: triton_gather(x,src,bf,be), repeat=args.repeat))

    if csr_supported:
        try:
            A = make_csr(src,V,dtype)
            csr_out, ref = torch.sparse.mm(A,x), x.index_select(0,src)
            diff = (csr_out-ref).abs().max().item()
            if diff != 0: print(f"\nWARNING: CSR max_diff={diff:g}")
            del csr_out, ref
            results["CSR torch.sparse.mm"].append(cuda_ms(lambda: torch.sparse.mm(A,x), repeat=args.repeat))
            del A
        except Exception as exc:
            print(f"\nCSR SpMM unavailable: {type(exc).__name__}: {exc}")
            print("Continuing with index_select and Triton.")
            csr_supported = False

    del x, src, edges, vertex_coord_list, cam
    gc.collect(); torch.cuda.empty_cache()

print("\n"+"="*78)
print("RESULTS — isolated feature gather on real Point-GNN graph indices")
print("="*78)
print(f"{'Method':<23}{'Mean ms':>11}{'Median':>11}{'Min':>11}{'Max':>11}")
print("-"*78)
for name,xs in sorted(results.items(), key=lambda kv: np.mean(kv[1])): print(row(name,xs))

base = np.mean(results["PyTorch index_select"])
names = [k for k in results if k.startswith("Triton")]
best = min(names, key=lambda k: np.mean(results[k]))
best_ms = np.mean(results[best])
vs, es = np.array([v for v,_ in shapes]), np.array([e for _,e in shapes])
print("-"*78)
print(f"V mean/min/max: {vs.mean():,.0f} / {vs.min():,} / {vs.max():,}")
print(f"E mean/min/max: {es.mean():,.0f} / {es.min():,} / {es.max():,}")
print(f"Best Triton: {best} = {best_ms:.3f} ms")
print(f"Triton speedup vs index_select: {base/best_ms:.3f}x")
if "CSR torch.sparse.mm" in results:
    c = np.mean(results["CSR torch.sparse.mm"])
    print(f"CSR SpMM: {c:.3f} ms ({base/c:.3f}x vs index_select)")
print("\nindex_select = current Point-GNN call")
print("Triton       = specialized one-nonzero-per-row gather")
print("CSR SpMM     = same gather explicitly represented as A @ X")
