"""Decode attention: fp16 KV (split-K Triton) vs int8 paged KV (CUDA) across context lengths.
Reports latency, effective KV bandwidth and bytes per token."""
import argparse

import torch

from fused_attn.kv_cache import PagedKvCacheInt8, decode_attention_kv8, kv_bytes_per_token
from fused_attn.triton_splitk import attention_splitk_triton


def timeit(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, nargs="+", default=[2048, 8192, 32768])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--block-size", type=int, default=16)
    args = p.parse_args()
    dt = torch.float16
    print(f"{'ctx':>7} {'fp16 splitK ms':>15} {'int8 paged ms':>14} {'fp16 GB/s':>10} {'int8 GB/s':>10} {'bytes/tok fp16':>15} {'int8':>8}")
    for ctx in args.ctx:
        B, H, Hkv, D, bs = args.batch, args.heads, args.kv_heads, args.head_dim, args.block_size
        q = torch.randn(B, H, 1, D, device="cuda", dtype=dt)
        k = torch.randn(B, Hkv, ctx, D, device="cuda", dtype=dt)
        v = torch.randn_like(k)
        t16 = timeit(lambda: attention_splitk_triton(q, k, v))
        nblk = B * (ctx // bs) + 1
        cache = PagedKvCacheInt8(nblk, bs, Hkv, D, device="cuda")
        slots = torch.arange(bs, bs + B * ctx, dtype=torch.int32, device="cuda")
        cache.append(k.permute(0, 2, 1, 3).reshape(-1, Hkv, D), v.permute(0, 2, 1, 3).reshape(-1, Hkv, D), slots)
        bt = (torch.arange(B * (ctx // bs), device="cuda").view(B, -1) + 1).int()
        cl = torch.full((B,), ctx, dtype=torch.int32, device="cuda")
        t8 = timeit(lambda: decode_attention_kv8(q[:, :, 0], cache, bt, cl))
        bytes16 = 2 * B * Hkv * ctx * D * 2
        bytes8 = B * Hkv * ctx * (2 * D + 8)
        bpt = kv_bytes_per_token(D, 2, Hkv)
        print(f"{ctx:>7} {t16:>15.3f} {t8:>14.3f} {bytes16 / t16 / 1e6:>10.0f} {bytes8 / t8 / 1e6:>10.0f} {bpt['fp16']:>15} {bpt['int8']:>8}")


if __name__ == "__main__":
    main()
