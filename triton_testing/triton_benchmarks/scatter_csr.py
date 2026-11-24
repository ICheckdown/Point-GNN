"""CSR segmented-max for Point-GNN: Triton kernel + correctness/speed benchmark.

Run on Jetson Orin (PyTorch + Triton):
    python3 triton_scatter_max_csr.py
    python3 triton_scatter_max_csr.py --edges 696654 --vertices 2094 --features 300

Or use real Point-GNN tensors:
    from triton_scatter_max_csr import benchmark_scatter, scatter_max_csr
    benchmark_scatter(edge_features, dst, num_vertices)
    output = scatter_max_csr(edge_features, dst, num_vertices)

Requires destination-grouped/sorted dst, with edge_features in the same row order.
"""

import argparse
import statistics
import torch
import triton
import triton.language as tl


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


def dst_to_rowptr(dst, num_vertices):
    """CSR boundaries for sorted dst. No sorting or copying edge features.

    Example: dst=[0,0,2,2,2], V=4 -> row_ptr=[0,2,2,5,5].
    searchsorted exploits sorted dst and avoids a full E-element bincount.
    """
    vertices = torch.arange(num_vertices + 1, device=dst.device, dtype=dst.dtype)
    return torch.searchsorted(dst.contiguous(), vertices).to(torch.int64)


def scatter_max_torch(edge_features, dst, num_vertices):
    """Original Point-GNN reduction, used as the reference."""
    out = torch.full((num_vertices, edge_features.shape[1]),
                     torch.finfo(edge_features.dtype).min,
                     device=edge_features.device, dtype=edge_features.dtype)
    out.index_reduce_(0, dst, edge_features, reduce="amax", include_self=True)
    return out


def scatter_max_from_rowptr(edge_features, row_ptr, num_vertices, block_e=32, block_f=64):
    """Triton segmented max, excluding CSR construction time."""
    if not edge_features.is_contiguous(): edge_features = edge_features.contiguous()
    E, F = edge_features.shape
    out = torch.empty((num_vertices, F), device=edge_features.device, dtype=edge_features.dtype)
    if num_vertices == 0: return out
    neg = torch.finfo(edge_features.dtype).min
    grid = (num_vertices, triton.cdiv(F, block_f))
    _segmented_max_kernel[grid](edge_features, row_ptr, out, F, neg, block_e, block_f, num_warps=4)
    return out


def scatter_max_csr(edge_features, dst, num_vertices, block_e=32, block_f=64):
    """Complete replacement: construct CSR and run Triton segmented max."""
    row_ptr = dst_to_rowptr(dst, num_vertices)
    return scatter_max_from_rowptr(edge_features, row_ptr, num_vertices, block_e, block_f)


def _cuda_ms(fn, repeats=15, warmup=4):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def benchmark_scatter(edge_features, dst, num_vertices, repeats=15, block_e=32, block_f=64):
    """Measure the exact PyTorch operation against Triton, including CSR overhead."""
    if not edge_features.is_cuda: raise ValueError("CUDA tensors required for Triton benchmark")
    if dst.device != edge_features.device: raise ValueError("dst and features must share a device")
    if dst.numel() != edge_features.shape[0]: raise ValueError("dst length must equal edge count")
    if dst.numel() and (bool((dst[1:] < dst[:-1]).any()) or
                        bool((dst < 0).any()) or bool((dst >= num_vertices).any())):
        raise ValueError("dst must be sorted and all vertex IDs must be in range")
    if edge_features.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise ValueError("Use float16, float32 or bfloat16 features")
    if not edge_features.is_contiguous(): edge_features = edge_features.contiguous()
    if not dst.is_contiguous(): dst = dst.contiguous()

    row_ptr = dst_to_rowptr(dst, num_vertices)
    expected = scatter_max_torch(edge_features, dst, num_vertices)
    actual = scatter_max_from_rowptr(edge_features, row_ptr, num_vertices, block_e, block_f)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    reconstructed = torch.repeat_interleave(torch.arange(num_vertices, device=dst.device),
                                           row_ptr[1:] - row_ptr[:-1])
    assert torch.equal(reconstructed.to(dst.dtype), dst), "CSR row boundaries do not match dst"

    baseline_ms = _cuda_ms(lambda: scatter_max_torch(edge_features, dst, num_vertices), repeats)
    csr_ms = _cuda_ms(lambda: dst_to_rowptr(dst, num_vertices), repeats)
    kernel_ms = _cuda_ms(lambda: scatter_max_from_rowptr(edge_features, row_ptr, num_vertices,
                                                          block_e, block_f), repeats)
    total_ms = _cuda_ms(lambda: scatter_max_csr(edge_features, dst, num_vertices,
                                                block_e, block_f), repeats)
    print(f"\nPoint-GNN scatter_max | E={len(dst):,}, V={num_vertices:,}, F={edge_features.shape[1]}, dtype={edge_features.dtype}")
    print("PASS: Triton output exactly matches PyTorch; CSR matches sorted dst")
    print(f"PyTorch index_reduce_ : {baseline_ms:9.3f} ms")
    print(f"CSR row_ptr creation  : {csr_ms:9.3f} ms")
    print(f"Triton kernel only    : {kernel_ms:9.3f} ms   speedup {baseline_ms/kernel_ms:.3f}x")
    print(f"CSR + Triton total    : {total_ms:9.3f} ms   speedup {baseline_ms/total_ms:.3f}x")
    return {"pytorch_ms": baseline_ms, "csr_ms": csr_ms, "triton_ms": kernel_ms,
            "total_ms": total_ms, "speedup": baseline_ms / total_ms}


def _small_correctness_test(device):
    dst = torch.tensor([0, 0, 2, 2, 2], dtype=torch.int64, device=device)
    x = torch.tensor([[2., 8., 1.], [5., 3., 4.], [1., 7., 9.],
                      [6., 2., 3.], [4., 9., 5.]], device=device)
    row_ptr = dst_to_rowptr(dst, 4)
    assert row_ptr.tolist() == [0, 2, 2, 5, 5]
    expected = scatter_max_torch(x, dst, 4)
    actual = scatter_max_csr(x, dst, 4)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    print("Small test PASS | row_ptr:", row_ptr.tolist())
    print("Output:\n", actual.cpu())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edges", type=int, default=300000)
    p.add_argument("--vertices", type=int, default=2094)
    p.add_argument("--features", type=int, default=300)
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--block-e", type=int, default=32)
    p.add_argument("--block-f", type=int, default=64)
    args = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("Run on a CUDA GPU with Triton installed")
    if args.edges < 0 or args.vertices < 1 or args.features < 1: raise ValueError("Invalid sizes")
    torch.manual_seed(123)
    _small_correctness_test("cuda")
    # Generate a sorted destination list; use real Point-GNN tensors for representative timings.
    dst = torch.randint(args.vertices, (args.edges,), dtype=torch.int64).sort().values.cuda()
    features = torch.randn((args.edges, args.features), device="cuda", dtype=torch.float32)
    benchmark_scatter(features, dst, args.vertices, args.repeats, args.block_e, args.block_f)


if __name__ == "__main__": main()
