#!/usr/bin/env python3
"""Point-GNN GEMM: current PyTorch F.linear(+inplace ReLU) vs tiled Triton.

Your ACTUAL current implementation in pointgnn_torch.py is:
    torch.set_float32_matmul_precision('high')
    fc_relu(x,W,b)    = torch.nn.functional.relu_(torch.nn.functional.linear(x,W,b))
    fc_no_relu(x,W,b) = torch.nn.functional.linear(x,W,b)
where W has shape [N,K] as returned by load_fc(). There is no Triton GEMM
in that file: Triton is used for gather and segmented scatter max.

Requires: PyTorch with CUDA, Triton, and adjacent gemm.json.
Outputs: TERMINAL ONLY. No CSV or result JSON is generated.

Usage:
    python3 gemm.py
    python3 gemm.py --max-configs 16 --repeats 30
    python3 gemm.py   # default: L2 extract operations ONLY
    python3 gemm.py --all   # entire model
    python3 gemm.py --names L1_extract_fc3 L2_extract_fc1
"""
import argparse
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_linear_kernel(X, WT, Bias, Y,
                        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                        SX0: tl.constexpr, SX1: tl.constexpr,
                        SW0: tl.constexpr, SW1: tl.constexpr,
                        SY0: tl.constexpr, SY1: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        GROUP_M: tl.constexpr, RELU: tl.constexpr,
                        PRECISION: tl.constexpr):
    """X[M,K] @ WT[K,N] + b[N], with optional fused ReLU.

    WT is W.transpose(0,1); no explicit weight-copy is needed.
    """
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
        x = tl.load(X + rm[:, None] * SX0 + kk[None, :] * SX1,
                    mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.)
        w = tl.load(WT + kk[:, None] * SW0 + rn[None, :] * SW1,
                    mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.)
        acc = tl.dot(x, w, acc, input_precision=PRECISION)
    b = tl.load(Bias + rn, rn < N, other=0.).to(tl.float32)
    y = acc + b[None, :]
    if RELU:
        y = tl.maximum(y, 0.)
    tl.store(Y + rm[:, None] * SY0 + rn[None, :] * SY1,
             y, mask=(rm[:, None] < M) & (rn[None, :] < N))


# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, group_m)
CONFIGS = [
    (32, 32, 32, 4, 4),
    (64, 32, 32, 4, 4),
    (64, 64, 32, 4, 8),
    (128, 32, 32, 4, 8),
    (128, 64, 32, 4, 8),
    (64, 128, 32, 4, 8),
    (128, 128, 32, 4, 8),
    (64, 64, 64, 4, 8),
    (128, 64, 64, 4, 8),
    (128, 128, 64, 4, 8),
    (32, 64, 32, 4, 8),
    (32, 128, 32, 4, 8),
    (64, 256, 32, 8, 8),
    (128, 256, 32, 8, 8),
    (256, 64, 32, 8, 8),
    (64, 64, 32, 8, 8),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shapes-file', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'gemm.json'))
    p.add_argument('--names', nargs='*', default=None, help='Run only layers containing any of these strings')
    p.add_argument('--all', action='store_true', help='Benchmark every layer (default only L2_extract layers)')
    p.add_argument('--max-configs', type=int, default=12)
    p.add_argument('--warmup', type=int, default=8, help='Warmup calls after compilation')
    p.add_argument('--repeats', type=int, default=25, help='Number of timed calls per trial')
    p.add_argument('--trials', type=int, default=3, help='Independent trials; report median')
    p.add_argument('--precision', choices=['tf32', 'tf32x3', 'ieee'], default='tf32',
                   help='Triton FP32 dot precision; current PyTorch uses matmul_precision=high')
    p.add_argument('--threshold', type=float, default=1.10)
    p.add_argument('--atol', type=float, default=0.03)
    p.add_argument('--rtol', type=float, default=0.03)
    p.add_argument('--check-rows', type=int, default=512,
                   help='Number of evenly spaced output rows to check')
    p.add_argument('--max-memory-mb', type=float, default=3200,
                   help='Skip shapes whose input/weight/two outputs exceed this size')
    p.add_argument('--seed', type=int, default=7)
    return p.parse_args()


def time_gpu(fn, *, warmup, repeats, trials):
    """CUDA-event timings of actual calls on the current stream, including allocations.

    Synchronize between independent trials, and report median ms/call.
    This is not end-to-end model latency, and doesn't include CPU preprocessing.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) / repeats)
    return statistics.median(samples)


def baseline_fc(x, weight, bias, relu):
    # Exactly the two functions in the Point-GNN code supplied by the user.
    if relu:
        return F.relu_(F.linear(x, weight, bias))
    return F.linear(x, weight, bias)


def compare_rows(a, b, sample_count, atol, rtol):
    m = a.shape[0]
    rows = torch.linspace(0, m-1, steps=min(m, sample_count), device=a.device).long()
    aa, bb = a.index_select(0, rows), b.index_select(0, rows)
    diff = (aa - bb).abs()
    max_abs = float(diff.max().item())
    ok = bool(torch.allclose(aa, bb, atol=atol, rtol=rtol))
    return ok, max_abs


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit('CUDA GPU not detected. Run this on your Jetson Orin.')
    if min(args.max_configs, args.repeats, args.trials, args.check_rows) < 1 or args.warmup < 0:
        sys.exit('Invalid benchmark arguments.')
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    with open(args.shapes_file, encoding='utf-8') as file:
        layers = json.load(file)
    if not isinstance(layers, list):
        sys.exit('gemm.json must contain a list of layer configurations.')

    print('='*100)
    print('Point-GNN ACTUAL GEMM baseline vs optimized Triton GEMM')
    print(f'GPU: {torch.cuda.get_device_name(0)} | PyTorch: {torch.__version__} | Triton: {triton.__version__}')
    print('CURRENT: F.linear(x,W,b) + optional F.relu_(...) | torch.set_float32_matmul_precision("high")')
    print(f'TEST:    Triton fused GEMM + bias + optional ReLU, precision={args.precision}')
    print('Timing: CUDA events, batch 1 frame shapes, individual GEMM calls, allocations included.')
    print('All accuracy tests use sampled rows; final end-to-end KITTI accuracy validation is still needed.')
    print('DEFAULT FOCUS: L2_extract_fc0 and L2_extract_fc1, 696000 rows per call × 3 calls.')
    print('Override with --all or --names. No result files written.\n')

    sum_baseline = 0.0
    sum_candidate = 0.0
    successes = 0
    faster = 0
    for layer in layers:
        name = str(layer['name'])
        if args.names:
            if not any(needle.lower() in name.lower() for needle in args.names):
                continue
        elif not args.all and not name.startswith('L2_extract_'):
            continue
        m, k, n = (int(layer[key]) for key in ('M','K','N'))
        calls = int(layer.get('weight', 1))
        relu = bool(layer.get('bias_relu', False))
        if min(m, k, n, calls) < 1:
            raise ValueError(f'Invalid layer spec {layer}')
        tensor_mib = 4*(m*k + n*k + n + 2*m*n)/1024**2
        free_mib = torch.cuda.mem_get_info()[0]/1024**2
        if tensor_mib > args.max_memory_mb or tensor_mib > free_mib*0.7:
            print(f'{name:<29} SKIP: estimated {tensor_mib:.0f} MiB; free {free_mib:.0f} MiB')
            continue
        print(f'[{name}] M={m:,}, K={k}, N={n}, calls={calls}, ReLU={relu}, approx={tensor_mib:.0f} MiB',flush=True)
        x = torch.randn((m,k),device='cuda',dtype=torch.float32)*0.1
        w = torch.randn((n,k),device='cuda',dtype=torch.float32)*0.1  # same layout as Point-GNN load_fc
        b = torch.randn((n,),device='cuda',dtype=torch.float32)*0.01
        wt = w.T
        o = torch.empty((m,n),device='cuda',dtype=torch.float32)
        try:
            reference = baseline_fc(x,w,b,relu)
            torch.cuda.synchronize()
            baseline_ms = time_gpu(lambda: baseline_fc(x,w,b,relu),warmup=args.warmup,
                                   repeats=args.repeats,trials=args.trials)
        except Exception as ex:
            print(f'  BASELINE FAILED: {ex}\n')
            del x,w,b,wt,o
            torch.cuda.empty_cache()
            continue
        best = None
        for bm,bn,bk,warps,gm in CONFIGS[:args.max_configs]:
            grid = (triton.cdiv(m,bm)*triton.cdiv(n,bn),)
            def candidate(bm=bm,bn=bn,bk=bk,warps=warps,gm=gm,grid=grid):
                # Matches the current F.linear output allocation semantics.
                y = torch.empty_like(o)
                fused_linear_kernel[grid](x,wt,b,y,m,n,k,*x.stride(),*wt.stride(),*y.stride(),
                                          bm,bn,bk,gm,relu,args.precision,num_warps=warps)
                return y
            try:
                y = candidate()
                torch.cuda.synchronize()
                ok, err = compare_rows(y,reference,args.check_rows,args.atol,args.rtol)
                if not ok:
                    print(f'  REJECT {bm}x{bn}x{bk} w{warps}: numerically different; max_abs={err:.5g}')
                    continue
                ms = time_gpu(candidate,warmup=args.warmup,repeats=args.repeats,trials=args.trials)
                if best is None or ms < best[0]:
                    best = (ms,bm,bn,bk,warps,gm,err)
            except Exception as ex:
                print(f'  SKIP {bm}x{bn}x{bk} w{warps}: {str(ex).splitlines()[0][:120]}')
        if best is None:
            print(f'  No numerically valid Triton configs. Baseline={baseline_ms:.3f} ms\n')
        else:
            ms,bm,bn,bk,warps,gm,err=best
            gain=baseline_ms/ms
            successes+=1
            faster+=int(gain>=args.threshold)
            sum_baseline+=calls*baseline_ms
            sum_candidate+=calls*ms
            print(f'  CURRENT F.linear: {baseline_ms:9.3f} ms | BEST Triton: {ms:9.3f} ms | SPEEDUP: {gain:6.3f}x')
            print(f'  Config: BM={bm}, BN={bn}, BK={bk}, warps={warps}, group={gm}; max sampled |diff|={err:.5g}')
            print(f'  {"MEETS" if gain>=args.threshold else "BELOW"} {args.threshold:.2f}x speedup target\n')
        del x,w,b,wt,o,reference
        torch.cuda.empty_cache()

    if not successes:
        sys.exit('No valid GEMM comparisons completed. Check errors and numerical tolerances.')
    gain=sum_baseline/sum_candidate
    print('='*100)
    print('WEIGHTED GEMM SUMMARY (frequency-weighted across valid model layers)')
    print(f'Current PyTorch GEMM + activations: {sum_baseline:9.3f} ms/frame')
    print(f'Best Triton GEMM + activations:    {sum_candidate:9.3f} ms/frame')
    print(f'WEIGHTED SPEEDUP:                  {gain:9.3f}x')
    print(f'Layers achieving {args.threshold:.2f}x: {faster}/{successes}')
    print(f'RECOMMENDATION: {"Worth investigating integration" if gain>=args.threshold else "Below threshold: prioritize other bottlenecks"}')
    print('NOTE: best-per-layer configs selected from same benchmark data; retest winners before deployment.')
    print('NOTE: actual whole-model latency depends on memory traffic, allocation, and inter-operator effects.')
    print('Nothing was saved to disk.')


if __name__ == '__main__':
    main()
