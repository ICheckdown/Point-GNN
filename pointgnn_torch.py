"""Point-GNN inference: tuned Triton GEMM + original Triton gather/CSR scatter.

Replace pointgnn_torch.py with this file, keeping your existing driver unchanged.
The tuned Triton GEMMs are used automatically; no environment toggle is needed.
The six tuned GEMM configurations cover four L1 extraction FC layers and
both extraction FC operations in each of L2, L3 and L4.

Warning: the benchmark used ReLU=False on L1_fc0 and L1_fc1, but your real
model calls fc_relu() for both. This version retains ReLU on all four L1
extraction layers to preserve the model's semantics.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

torch.set_float32_matmul_precision("high")

@triton.jit
def _gather_kernel(x, src, out, E, F: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_F: tl.constexpr):
    e = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    f = tl.program_id(1) * BLOCK_F + tl.arange(0, BLOCK_F)
    mask_e, mask_f = e < E, f < F
    s = tl.load(src + e, mask=mask_e, other=0)
    mask = mask_e[:, None] & mask_f[None, :]
    values = tl.load(x + s[:, None] * F + f[None, :], mask=mask)
    tl.store(out + e[:, None] * F + f[None, :], values, mask=mask)

def triton_gather(x, src):
    E, Fdim = src.numel(), x.shape[1]
    if not x.is_contiguous(): x = x.contiguous()
    if not src.is_contiguous(): src = src.contiguous()
    out = torch.empty((E, Fdim), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(E, 4), triton.cdiv(Fdim, 64))
    _gather_kernel[grid](x, src, out, E, Fdim, BLOCK_E=4, BLOCK_F=64)
    return out

def load_fc(weights_dir, scope, device="cuda", dtype=torch.float32):
    base = os.path.join(weights_dir, scope.replace("/", "_"))
    wp, bp = base + "_weights.npy", base + "_biases.npy"
    if not os.path.isfile(wp): raise FileNotFoundError(wp)
    if not os.path.isfile(bp): raise FileNotFoundError(bp)
    W = torch.as_tensor(np.load(wp), dtype=dtype, device=device).T.contiguous()
    b = torch.as_tensor(np.load(bp), dtype=dtype, device=device).contiguous()
    return W, b

# Kernels are adapted directly from the tested fused_linear_kernel in gemm.py.
# Shape convention: x[M,K], W[N,K] (as produced by load_fc), b[N].
# TF32 precision and grouped tile scheduling match the GEMM benchmark.
# Keep M dynamic: different frames have different edge counts.
@triton.jit(do_not_specialize=["M"])
def _pointgnn_fused_linear(X, WT, Bias, Y,
                           M, N: tl.constexpr, K: tl.constexpr,
                           SX0: tl.constexpr, SX1: tl.constexpr,
                           SW0: tl.constexpr, SW1: tl.constexpr,
                           SY0: tl.constexpr, SY1: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                           GROUP_M: tl.constexpr, RELU: tl.constexpr,
                           PRECISION: tl.constexpr):
    pid = tl.program_id(0)
    n_m = tl.cdiv(M, BM)
    n_n = tl.cdiv(N, BN)
    n_group = GROUP_M * n_n
    group_id = pid // n_group
    first_m = group_id * GROUP_M
    group_m = tl.minimum(n_m - first_m, GROUP_M)
    pid_m = first_m + (pid % n_group) % group_m
    pid_n = (pid % n_group) // group_m

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0., dtype=tl.float32)
    for ki in range(tl.cdiv(K, BK)):
        kk = ki * BK + rk
        xx = tl.load(X + rm[:, None] * SX0 + kk[None, :] * SX1,
                     mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.)
        ww = tl.load(WT + kk[:, None] * SW0 + rn[None, :] * SW1,
                     mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.)
        acc = tl.dot(xx, ww, acc, input_precision=PRECISION)
    bias = tl.load(Bias + rn, mask=rn < N, other=0.).to(tl.float32)
    val = acc + bias[None, :]
    if RELU:
        val = tl.maximum(val, 0.)
    tl.store(Y + rm[:, None] * SY0 + rn[None, :] * SY1,
             val, mask=(rm[:, None] < M) & (rn[None, :] < N))


# (K, N, BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, GROUP_M)
# Tuned on Jetson Orin with TF32 using the provided 523,000/696,000-row shapes.
_GEMM_CONFIGS = {
    "l1_fc0": (4, 32, 128, 32, 32, 4, 8),
    "l1_fc1": (32, 64, 32, 64, 32, 4, 8),
    "l1_fc2": (64, 128, 128, 128, 32, 4, 8),
    "l1_fc3": (128, 300, 128, 128, 32, 4, 8),
    "e0": (303, 300, 128, 64, 32, 4, 8),
    "e1": (300, 300, 128, 128, 32, 4, 8),
}

def fc_relu(x, W, b, op=None):
    """Run tuned Triton GEMMs on selected layers, PyTorch on other layers."""
    key = op if op and op.startswith("l1_") else (
        "e0" if op and op.endswith("_e0") else
        "e1" if op and op.endswith("_e1") else None
    )
    cfg = _GEMM_CONFIGS.get(key)
    if cfg is None or not x.is_cuda or x.dtype != torch.float32 or W.dtype != torch.float32 or b.dtype != torch.float32:
        return F.relu_(F.linear(x, W, b))

    k, n, bm, bn, bk, warps, gm = cfg
    if x.ndim != 2 or W.ndim != 2 or b.ndim != 1 or x.shape[1] != k or W.shape != (n, k) or b.numel() != n:
        raise ValueError(
            f"Unexpected GEMM dimensions for {op}: "
            f"x={tuple(x.shape)}, W={tuple(W.shape)}, b={tuple(b.shape)}; "
            f"expected K={k}, N={n}"
        )
    m = x.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if m == 0:
        return out
    wt = W.T
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    _pointgnn_fused_linear[grid](
        x, wt, b, out, m, n, k,
        *x.stride(), *wt.stride(), *out.stride(),
        bm, bn, bk, gm, True, "tf32",
        num_warps=warps,
    )
    return out


def fc_no_relu(x, W, b):
    return F.linear(x, W, b)

@triton.jit
def _segmented_max_kernel(x, row_ptr, out, F: tl.constexpr, NEG: tl.constexpr,
                          BLOCK_E: tl.constexpr, BLOCK_F: tl.constexpr):
    v = tl.program_id(0)
    f = tl.program_id(1) * BLOCK_F + tl.arange(0, BLOCK_F)
    r = tl.arange(0, BLOCK_E)
    start = tl.load(row_ptr + v)
    end = tl.load(row_ptr + v + 1)
    acc = tl.full((BLOCK_F,), NEG, tl.float32)
    for base in range(start, end, BLOCK_E):
        e = base + r
        vals = tl.load(x + e[:, None] * F + f[None, :],
                       mask=(e[:, None] < end) & (f[None, :] < F), other=NEG)
        acc = tl.maximum(acc, tl.max(vals.to(tl.float32), axis=0))
    tl.store(out + v * F + f, acc, mask=f < F)

def scatter_max(edge_features, dst, num_vertices):
    """CSR segmented max for destination-sorted Point-GNN edges.

    Exact replacement for index_reduce_ in the tested KITTI configuration.
    Empty vertices retain torch.finfo(dtype).min, as before.
    """
    if not edge_features.is_contiguous(): edge_features = edge_features.contiguous()
    if not dst.is_contiguous(): dst = dst.contiguous()
    Fdim = edge_features.shape[1]
    vertices = torch.arange(num_vertices + 1, device=dst.device, dtype=dst.dtype)
    row_ptr = torch.searchsorted(dst, vertices).to(torch.int64)
    out = torch.empty((num_vertices, Fdim), dtype=edge_features.dtype, device=edge_features.device)
    if num_vertices:
        neg = torch.finfo(edge_features.dtype).min
        grid = (num_vertices, triton.cdiv(Fdim, 64))
        _segmented_max_kernel[grid](edge_features, row_ptr, out, Fdim, neg, 32, 64, num_warps=4)
    return out

def layer1_pointset_pooling(raw_xyz, raw_features, keypoint_xyz, edges, W):
    src, dst = edges[:, 0], edges[:, 1]
    V = keypoint_xyz.shape[0]

    intensity = raw_features.index_select(0, src)[:, :1]
    rel_xyz = raw_xyz.index_select(0, src)
    dst_xyz = keypoint_xyz.index_select(0, dst)
    rel_xyz.sub_(dst_xyz)
    del dst_xyz

    x = torch.cat((intensity, rel_xyz), dim=1)
    del intensity, rel_xyz, src

    x = fc_relu(x, *W["l1_fc0"], op="l1_fc0")
    x = fc_relu(x, *W["l1_fc1"], op="l1_fc1")
    x = fc_relu(x, *W["l1_fc2"], op="l1_fc2")
    x = fc_relu(x, *W["l1_fc3"], op="l1_fc3")

    x = scatter_max(x, dst, V)
    del dst

    x = fc_relu(x, *W["l1_c0"])
    x = fc_relu(x, *W["l1_c1"])
    return x

def layer_graph_net(vertex_features, keypoint_xyz, edges, W, L):
    V = vertex_features.shape[0]
    src, dst = edges[:, 0], edges[:, 1]

    source_xyz = keypoint_xyz.index_select(0, src)

    offset = fc_relu(vertex_features, *W[f"{L}_ao0"])
    offset = fc_no_relu(offset, *W[f"{L}_ao1"])
    shifted_xyz = keypoint_xyz + offset
    del offset

    destination_xyz = shifted_xyz.index_select(0, dst)
    del shifted_xyz
    source_xyz.sub_(destination_xyz)
    del destination_xyz

    source_features = triton_gather(vertex_features, src)
    del src

    edge_input = torch.cat((source_features, source_xyz), dim=1)
    del source_features, source_xyz

    edge_feat = fc_relu(edge_input, *W[f"{L}_e0"], op=f"{L}_e0")
    del edge_input
    edge_next = fc_relu(edge_feat, *W[f"{L}_e1"], op=f"{L}_e1")
    del edge_feat
    edge_feat = edge_next

    aggregated = scatter_max(edge_feat, dst, V)
    del edge_feat, dst

    combined = fc_relu(aggregated, *W[f"{L}_c0"])
    del aggregated
    combined = fc_no_relu(combined, *W[f"{L}_c1"])

    out = combined + vertex_features
    del combined
    return out

def detection_head(vertex_features, W, num_classes):
    cls = fc_relu(vertex_features, *W["cls_fc0"])
    cls = fc_no_relu(cls, *W["cls_fc1"])

    boxes = []
    for c in range(num_classes):
        x = fc_relu(vertex_features, *W[f"loc{c}_fc0"])
        x = fc_relu(x, *W[f"loc{c}_fc1"])
        x = fc_no_relu(x, *W[f"loc{c}_fc2"])
        boxes.append(x)

    pred_box = torch.stack(boxes, dim=1)
    del boxes
    return cls, pred_box

class PointGNN:
    def __init__(self, weights_dir, num_classes=4, device="cuda", dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_classes = num_classes
        self.W = self._load_weights(weights_dir)

    def _load_weights(self, wd):
        W = {}
        def load(name, scope): W[name] = load_fc(wd, scope, self.device, self.dtype)

        load("l1_fc0", "layer1/extract_vertex_features/fully_connected")
        load("l1_fc1", "layer1/extract_vertex_features/fully_connected_1")
        load("l1_fc2", "layer1/extract_vertex_features/fully_connected_2")
        load("l1_fc3", "layer1/extract_vertex_features/fully_connected_3")
        load("l1_c0", "layer1/combined_features/fully_connected")
        load("l1_c1", "layer1/combined_features/fully_connected_1")

        for L, layer in (("l2", "layer2"), ("l3", "layer3"), ("l4", "layer4")):
            load(f"{L}_ao0", f"{layer}/fully_connected")
            load(f"{L}_ao1", f"{layer}/fully_connected_1")
            load(f"{L}_e0", f"{layer}/extract_vertex_features/fully_connected")
            load(f"{L}_e1", f"{layer}/extract_vertex_features/fully_connected_1")
            load(f"{L}_c0", f"{layer}/combined_features/fully_connected")
            load(f"{L}_c1", f"{layer}/combined_features/fully_connected_1")

        load("cls_fc0", "output/predictor/cls/fully_connected")
        load("cls_fc1", "output/predictor/cls/fully_connected_1")

        for c in range(self.num_classes):
            base = f"output/predictor/loc/cls_{c}"
            load(f"loc{c}_fc0", f"{base}/fully_connected")
            load(f"loc{c}_fc1", f"{base}/fully_connected_1")
            load(f"loc{c}_fc2", f"{base}/fully_connected_2")

        return W

    @torch.inference_mode()
    def forward(self, raw_xyz_np, raw_features_np, keypoint_xyz_np, edges_e0_np, edges_e1_np):
        d = self.device

        raw_xyz = torch.as_tensor(raw_xyz_np, dtype=self.dtype, device=d)
        raw_features = torch.as_tensor(raw_features_np, dtype=self.dtype, device=d)
        keypoint_xyz = torch.as_tensor(keypoint_xyz_np, dtype=self.dtype, device=d)
        edges_e0 = torch.as_tensor(edges_e0_np, dtype=torch.long, device=d)
        edges_e1 = torch.as_tensor(edges_e1_np, dtype=torch.long, device=d)

        vertex_features = layer1_pointset_pooling(raw_xyz, raw_features, keypoint_xyz, edges_e0, self.W)
        del raw_xyz, raw_features, edges_e0

        for L in ("l2", "l3", "l4"):
            next_features = layer_graph_net(vertex_features, keypoint_xyz, edges_e1, self.W, L)
            del vertex_features
            vertex_features = next_features

        del keypoint_xyz, edges_e1

        logits, pred_box = detection_head(vertex_features, self.W, self.num_classes)
        del vertex_features

        logits_np = logits.cpu().numpy()
        pred_box_np = pred_box.cpu().numpy()
        del logits, pred_box
        return logits_np, pred_box_np

    def reset_peak_memory_stats(self):
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

