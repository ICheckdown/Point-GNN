import torch
import triton
import triton.language as tl

@triton.jit
def _gather_1d_kernel(x, src, out, E: tl.constexpr, F: tl.constexpr, BLOCK_F: tl.constexpr):
    e = tl.program_id(0)
    f = tl.program_id(1) * BLOCK_F + tl.arange(0, BLOCK_F)
    s = tl.load(src + e)
    mask = f < F
    vals = tl.load(x + s * F + f, mask=mask)
    tl.store(out + e * F + f, vals, mask=mask)

@triton.jit
def _gather_2d_kernel(x, src, out, E: tl.constexpr, F: tl.constexpr,
                      BLOCK_E: tl.constexpr, BLOCK_F: tl.constexpr):
    e = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    f = tl.program_id(1) * BLOCK_F + tl.arange(0, BLOCK_F)
    emask = e < E
    s = tl.load(src + e, mask=emask, other=0)
    offs = s[:, None] * F + f[None, :]
    mask = emask[:, None] & (f[None, :] < F)
    vals = tl.load(x + offs, mask=mask)
    tl.store(out + e[:, None] * F + f[None, :], vals, mask=mask)

def triton_gather(x, src, block_f=128, block_e=1):
    if not x.is_cuda or not src.is_cuda: raise ValueError("x and src must be CUDA tensors")
    if x.ndim != 2 or src.ndim != 1: raise ValueError("Expected x=[V,F] and src=[E]")
    if not x.is_contiguous(): x = x.contiguous()
    if src.dtype != torch.int64: src = src.long()
    if not src.is_contiguous(): src = src.contiguous()
    E, F = src.numel(), x.shape[1]
    out = torch.empty((E, F), device=x.device, dtype=x.dtype)
    if block_e == 1:
        _gather_1d_kernel[(E, triton.cdiv(F, block_f))](x, src, out, E=E, F=F, BLOCK_F=block_f)
    else:
        grid = (triton.cdiv(E, block_e), triton.cdiv(F, block_f))
        _gather_2d_kernel[grid](x, src, out, E=E, F=F, BLOCK_E=block_e, BLOCK_F=block_f)
    return out
