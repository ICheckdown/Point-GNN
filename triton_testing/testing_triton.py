import torch
import torch.nn.functional as F
import triton
import triton.language as tl

'''
CONFIGS = [
    triton.Config({"BLOCK_M":128,"BLOCK_N":64,"BLOCK_K":32,"GROUP_M":8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M":64, "BLOCK_N":64,"BLOCK_K":32,"GROUP_M":8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M":128,"BLOCK_N":64,"BLOCK_K":16,"GROUP_M":8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M":64, "BLOCK_N":64,"BLOCK_K":16,"GROUP_M":8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M":256,"BLOCK_N":64,"BLOCK_K":16,"GROUP_M":8}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M":128,"BLOCK_N":128,"BLOCK_K":32,"GROUP_M":8}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M":64, "BLOCK_N":128,"BLOCK_K":32,"GROUP_M":8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M":128,"BLOCK_N":256,"BLOCK_K":64,"GROUP_M":8}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M":64, "BLOCK_N":256,"BLOCK_K":32,"GROUP_M":8}, num_stages=4, num_warps=4),
]
'''

@triton.jit
def fused_matmul_bias_relu_kernel(
    A, B, bias, C,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
    APPLY_RELU: tl.constexpr,
):
    pid = tl.program_id(0)

    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)

    group_width = GROUP_M * num_n
    group = pid // group_width
    first_m = group * GROUP_M
    group_size = tl.minimum(num_m - first_m, GROUP_M)

    pid_group = pid % group_width
    pid_m = first_m + pid_group % group_size
    pid_n = pid_group // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:,None] * stride_am + offs_k[None,:] * stride_ak
    b_ptrs = B + offs_k[:,None] * stride_bk + offs_n[None,:] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        remaining = K - k * BLOCK_K

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:,None] < M) & (offs_k[None,:] < remaining),
            other=0.0
        )

        b = tl.load(
            b_ptrs,
            mask=(offs_k[:,None] < remaining) & (offs_n[None,:] < N),
            other=0.0
        )

        acc = tl.dot(a, b, acc=acc, allow_tf32=False)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc += tl.load(
        bias + offs_n,
        mask=offs_n < N,
        other=0.0
    )[None,:]

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    c_ptrs = C + offs_m[:,None] * stride_cm + offs_n[None,:] * stride_cn

    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:,None] < M) & (offs_n[None,:] < N)
    )


def fused_matmul_bias_relu(A, B, bias, apply_relu=True):
    if not A.is_contiguous(): A = A.contiguous()
    if not B.is_contiguous(): B = B.contiguous()
    if not bias.is_contiguous(): bias = bias.contiguous()

    M, K = A.shape
    K2, N = B.shape

    if K != K2:
        raise ValueError(f"K mismatch: {K} != {K2}")

    BM, BN, BK, GM, warps, stages = get_config(K, N)

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    grid = (
        triton.cdiv(M, BM) *
        triton.cdiv(N, BN),
    )

    fused_matmul_bias_relu_kernel[grid](
        A, B, bias, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        GROUP_M=GM,
        APPLY_RELU=apply_relu,
        num_warps=warps,
        num_stages=stages,
    )

    return C

def get_config(K, N):
    if K == 64 and N == 128:
        return 64, 64, 16, 8, 4, 4

    if K == 128 and N == 300:
        return 128, 64, 16, 8, 4, 4

    if K == 303 and N == 300:
        return 64, 64, 16, 8, 4, 4

    if K == 300 and N == 300:
        return 64, 64, 16, 8, 4, 4

    raise ValueError(f"No Triton config for K={K}, N={N}")


def torch_fc(A, W, bias):
    return F.relu_(F.linear(A, W, bias))

'''
def benchmark(M, K, N, name):
    A    = torch.randn((M,K), device="cuda")
    W    = torch.randn((N,K), device="cuda").contiguous()
    B    = W.T.contiguous()
    bias = torch.randn(N, device="cuda")

    # Warmup
    fused_matmul_bias_relu(A, B, bias)
    torch.cuda.synchronize()

    # Baseline 1: unfused (what TF does)
    ms_unfused = triton.testing.do_bench(
        lambda: F.relu(torch.matmul(A, B) + bias),
        warmup=20, rep=100)

    # Baseline 2: PyTorch fused F.linear+relu_ (what you had)
    ms_flinear = triton.testing.do_bench(
        lambda: F.relu_(F.linear(A, W, bias)),
        warmup=20, rep=100)

    # Triton
    ms_triton = triton.testing.do_bench(
        lambda: fused_matmul_bias_relu(A, B, bias),
        warmup=20, rep=100)

    cfg = fused_matmul_bias_relu_kernel.best_config
    
    print(f"\n{name}: [{M:,},{K}] -> {N}")
    print(f"  Unfused (matmul+add+relu):  {ms_unfused:.3f}ms")
    print(f"  F.linear + relu_:           {ms_flinear:.3f}ms")
    print(f"  Triton fused:               {ms_triton:.3f}ms")
    print(f"  Speedup vs unfused:         {ms_unfused/ms_triton:.3f}×")
    print(f"  Speedup vs F.linear:        {ms_flinear/ms_triton:.3f}×")


    print(
        f"Best config: BM={cfg.kwargs['BLOCK_M']} "
        f"BN={cfg.kwargs['BLOCK_N']} "
        f"BK={cfg.kwargs['BLOCK_K']} "
        f"GM={cfg.kwargs['GROUP_M']} "
        f"warps={cfg.num_warps} stages={cfg.num_stages}"
    )


if __name__ == "__main__":
    shapes = [
        (523615,64,128,"L1_extract_fc2"),
        (523615,128,300,"L1_extract_fc3"),
        (696654,303,300,"Graph_extract_fc0"),
        (696654,300,300,"Graph_extract_fc1"),
    ]

    print("=" * 65)
    print("POINT-GNN: PYTORCH vs TRITON")
    print("M runtime | K,N compile-time constants")
    print("=" * 65)

    for shape in shapes:
        benchmark(*shape)
'''