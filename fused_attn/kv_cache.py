"""INT8 KV cache: quantisation, paged append and decode attention.

Reference implementations in PyTorch (CPU-testable, used in CI) with the same
semantics as the CUDA kernels in ``csrc/kv_cache.cu``
on a GPU with the
extension built the CUDA path is used.

Scheme: symmetric int8 with one fp32 scale per (token, head) row
(``scale = max|x| / 127``).  Per-row scales track the dynamic range of each
token, which is what keeps attention logits accurate when a few tokens have
much larger key norms than the rest (the "attention sink" tokens).
"""
from __future__ import annotations

import math
import os

import torch

try:
    from . import _C as _cuda_ext  # type: ignore
except Exception:  # pragma: no cover
    _cuda_ext = None


def _use_cuda(t: torch.Tensor) -> bool:
    return _cuda_ext is not None and t.is_cuda and os.environ.get("FUSED_ATTN_KV_BACKEND", "auto") != "reference" and hasattr(_cuda_ext, "quantize_kv")


# ---- quantisation ------------------------------------------------------------
def quantize_kv_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x: [..., D] -> (int8 [..., D], scale fp32 [...])."""
    if _use_cuda(x):
        return _cuda_ext.quantize_kv(x)
    xf = x.float()
    amax = xf.abs().amax(dim=-1).clamp_min(1e-8)
    scale = amax / 127.0
    xq = torch.round(xf / scale.unsqueeze(-1)).clamp_(-127, 127).to(torch.int8)
    return xq, scale


def dequantize_kv_int8(xq: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return (xq.float() * scale.unsqueeze(-1).float()).to(dtype)


# ---- paged cache ----------------------------------------------------------------
class PagedKvCacheInt8:
    """Paged int8 K/V cache: [num_blocks, block_size, H_kv, D] + per-token scales."""

    def __init__(self, num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int, device="cpu"):
        shape = (num_blocks, block_size, num_kv_heads, head_dim)
        self.block_size, self.num_kv_heads, self.head_dim = block_size, num_kv_heads, head_dim
        self.k = torch.zeros(shape, dtype=torch.int8, device=device)
        self.v = torch.zeros(shape, dtype=torch.int8, device=device)
        self.k_scale = torch.zeros(shape[:3], dtype=torch.float32, device=device)
        self.v_scale = torch.zeros(shape[:3], dtype=torch.float32, device=device)

    @property
    def bytes_per_token(self) -> int:
        """Bytes streamed per token per KV head at decode (K + V + scales)."""
        return 2 * self.head_dim + 2 * 4

    def append(self, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        """k, v: [T, H_kv, D]
        slot_mapping: int32 [T], slot = block * block_size + offset (-1 = skip)."""
        if _use_cuda(k) and hasattr(_cuda_ext, "paged_append_kv8"):
            _cuda_ext.paged_append_kv8(k, v, self.k, self.v, self.k_scale, self.v_scale, slot_mapping.to(torch.int32))
            return
        keep = slot_mapping >= 0
        slots = slot_mapping[keep].long()
        blk, off = slots // self.block_size, slots % self.block_size
        kq, ks = quantize_kv_int8(k[keep])
        vq, vs = quantize_kv_int8(v[keep])
        self.k[blk, off] = kq
        self.v[blk, off] = vq
        self.k_scale[blk, off] = ks
        self.v_scale[blk, off] = vs

    def gather(self, block_table: torch.Tensor, ctx_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantised K, V for one sequence: [ctx_len, H_kv, D] fp32 (reference / tests)."""
        nblk = (ctx_len + self.block_size - 1) // self.block_size
        ids = block_table[:nblk].long()
        k = dequantize_kv_int8(self.k[ids], self.k_scale[ids]).reshape(-1, self.num_kv_heads, self.head_dim)[:ctx_len]
        v = dequantize_kv_int8(self.v[ids], self.v_scale[ids]).reshape(-1, self.num_kv_heads, self.head_dim)[:ctx_len]
        return k, v


def decode_attention_kv8_reference(q: torch.Tensor, cache: PagedKvCacheInt8, block_tables: torch.Tensor, context_lens: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    """q: [B, H_q, D] -> [B, H_q, D]
    attends over the dequantised paged cache."""
    B, Hq, D = q.shape
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    out = torch.empty_like(q)
    rep = Hq // cache.num_kv_heads
    for b in range(B):
        ctx = int(context_lens[b])
        k, v = cache.gather(block_tables[b], ctx)                       # [ctx, H_kv, D]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        s = torch.einsum("hd,khd->hk", q[b].float(), k) * scale         # [H_q, ctx]
        p = torch.softmax(s, dim=-1)
        out[b] = torch.einsum("hk,khd->hd", p, v).to(q.dtype)
    return out


def decode_attention_kv8(q: torch.Tensor, cache: PagedKvCacheInt8, block_tables: torch.Tensor, context_lens: torch.Tensor, scale: float | None = None, num_splits: int = 0) -> torch.Tensor:
    """Dispatch: CUDA split-K kernel on GPU, torch reference elsewhere."""
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    if _use_cuda(q) and hasattr(_cuda_ext, "decode_attention_kv8"):
        return _cuda_ext.decode_attention_kv8(q.contiguous(), cache.k, cache.v, cache.k_scale, cache.v_scale,
                                              block_tables.to(torch.int32), context_lens.to(torch.int32), scale, num_splits)
    return decode_attention_kv8_reference(q, cache, block_tables, context_lens, scale)


def kv_bytes_per_token(head_dim: int, dtype_bytes: int, num_kv_heads: int) -> dict[str, int]:
    """Bytes per token (K+V, all KV heads) for the memory-footprint comparison in the README."""
    return {
        "fp16": 2 * head_dim * dtype_bytes * num_kv_heads,
        "int8": (2 * head_dim + 8) * num_kv_heads,
        "fp8": (2 * head_dim + 2 * 4 / 64) * num_kv_heads,  # per-64-key block scales (fused_attn.fp8)
    }
