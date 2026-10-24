"""INT8 KV cache: quantisation error, paged append, decode attention (reference on CPU, CUDA kernel on GPU)."""
import pytest
import torch

from fused_attn.kv_cache import (
    PagedKvCacheInt8,
    decode_attention_kv8,
    decode_attention_kv8_reference,
    dequantize_kv_int8,
    kv_bytes_per_token,
    quantize_kv_int8,
)
from fused_attn.reference import attention_reference

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_int8_roundtrip_is_tight_per_token():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 300, 128) * torch.rand(4, 8, 300, 1) * 5   # per-token dynamic range varies a lot
    xq, s = quantize_kv_int8(x)
    assert xq.dtype == torch.int8 and s.shape == (4, 8, 300)
    back = dequantize_kv_int8(xq, s)
    rel = ((back - x).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-6)).max()
    assert rel < 0.012, rel                       # per-row scale: < ~1.2% error on every token, even small-norm ones
    assert torch.equal(xq.abs().amax(dim=-1), torch.full_like(s, 127, dtype=torch.int8))


def _build(B=3, Hq=8, Hkv=2, D=128, block_size=16, ctx=(37, 160, 5), dtype=torch.float32):
    torch.manual_seed(1)
    num_blocks = sum((c + block_size - 1) // block_size for c in ctx) + 1
    cache = PagedKvCacheInt8(num_blocks, block_size, Hkv, D, device=DEV)
    max_blocks = max((c + block_size - 1) // block_size for c in ctx)
    bt = torch.zeros(B, max_blocks, dtype=torch.int32, device=DEV)
    k_dense = torch.zeros(B, Hkv, max(ctx), D, device=DEV)
    v_dense = torch.zeros_like(k_dense)
    nxt = 1
    for b, c in enumerate(ctx):
        n = (c + block_size - 1) // block_size
        bt[b, :n] = torch.arange(nxt, nxt + n, dtype=torch.int32, device=DEV)
        nxt += n
        k = torch.randn(c, Hkv, D, device=DEV, dtype=dtype)
        v = torch.randn(c, Hkv, D, device=DEV, dtype=dtype)
        slots = (bt[b, torch.arange(c, device=DEV) // block_size].long() * block_size + torch.arange(c, device=DEV) % block_size).to(torch.int32)
        cache.append(k, v, slots)
        k_dense[b, :, :c] = k.transpose(0, 1).float()
        v_dense[b, :, :c] = v.transpose(0, 1).float()
    q = torch.randn(B, Hq, D, device=DEV, dtype=dtype)
    return cache, bt, torch.tensor(ctx, dtype=torch.int32, device=DEV), q, k_dense, v_dense


def test_paged_append_and_decode_match_dense_attention():
    cache, bt, ctx, q, k_dense, v_dense = _build()
    out = decode_attention_kv8_reference(q, cache, bt, ctx)
    for b in range(q.shape[0]):
        c = int(ctx[b])
        ref = attention_reference(q[b][None, :, None], k_dense[b:b + 1, :, :c], v_dense[b:b + 1, :, :c])[0, :, 0]
        assert torch.allclose(out[b], ref, atol=3e-2, rtol=3e-2), (out[b] - ref).abs().max()


def test_append_skips_negative_slots_and_overwrites():
    cache = PagedKvCacheInt8(2, 4, 1, 64, device=DEV)
    k = torch.randn(3, 1, 64, device=DEV)
    cache.append(k, k, torch.tensor([5, -1, 5], dtype=torch.int32, device=DEV))   # token 2 overwrites slot 5
    kq, _ = quantize_kv_int8(k[2])
    assert torch.equal(cache.k[1, 1], kq)
    assert cache.k_scale[1, 1] > 0 and cache.k_scale[0, 0] == 0


def test_bytes_per_token_table():
    t = kv_bytes_per_token(128, 2, 8)
    assert t["fp16"] == 4096 and t["int8"] == (256 + 8) * 8
    assert t["int8"] < t["fp16"] / 1.9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block_size", [16, 32])
def test_cuda_decode_kv8_matches_reference(dtype, block_size):
    from fused_attn.kv_cache import _cuda_ext
    if _cuda_ext is None or not hasattr(_cuda_ext, "decode_attention_kv8"):
        pytest.skip("extension not built")
    cache, bt, ctx, q, _, _ = _build(ctx=(1000, 33, 2050), block_size=block_size, dtype=dtype)
    out = decode_attention_kv8(q, cache, bt, ctx)                    # CUDA
    ref = decode_attention_kv8_reference(q, cache, bt, ctx)          # torch, same cache
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2), (out.float() - ref.float()).abs().max()
    out2 = decode_attention_kv8(q, cache, bt, ctx, num_splits=7)
    assert torch.allclose(out2.float(), ref.float(), atol=2e-2, rtol=2e-2)
