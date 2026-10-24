"""Benchmark every backend against the PyTorch baselines across batch sizes and
sequence lengths, the way a kernel is actually judged: speed-up over
F.scaled_dot_product_attention (flash / mem-efficient / math) at each shape,
plus achieved TFLOP/s and the memory traffic the fused kernel avoids.

    python benchmarks/bench_vs_torch.py --batches 1 4 16 --seqlens 512 1024 2048 4096 8192 \
        --heads 32 --kv-heads 8 --head-dim 128 --dtype bf16 --causal --out results/vs_torch.csv
    python benchmarks/plot_vs_torch.py results/vs_torch.csv      # speed-up heatmaps per backend
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import torch
import torch.nn.functional as F

from fused_attn import attention, available_backends


def timeit(fn, iters: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def sdpa(kind: str):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    be = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION, "math": SDPBackend.MATH}[kind]

    def run(q, k, v, causal):
        with sdpa_kernel(be):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)

    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--seqlens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    p.add_argument("--causal", action="store_true")
    p.add_argument("--backends", nargs="+", default=None)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--out", default="results/vs_torch.csv")
    args = p.parse_args()
    assert torch.cuda.is_available()
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    backends = args.backends or [b for b in available_backends() if b != "reference"]
    baselines = ["torch_flash", "torch_efficient", "torch_math"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    print(f"device={torch.cuda.get_device_name()} dtype={args.dtype} H={args.heads}/{args.kv_heads} D={args.head_dim} causal={args.causal}")
    print(f"{'B':>3} {'N':>6} | " + " ".join(f"{b:>14}" for b in backends + baselines) + " | speedup vs flash")
    for B in args.batches:
        for N in args.seqlens:
            if B * args.heads * N * args.head_dim * 2 * 3 > 0.6 * torch.cuda.get_device_properties(0).total_memory:
                continue
            q = torch.randn(B, args.heads, N, args.head_dim, device="cuda", dtype=dt)
            k = torch.randn(B, args.kv_heads, N, args.head_dim, device="cuda", dtype=dt)
            v = torch.randn_like(k)
            flops = 4.0 * B * args.heads * N * N * args.head_dim * (0.5 if args.causal else 1.0)
            res: dict[str, float] = {}
            for b in backends:
                os.environ["FUSED_ATTN_BACKEND"] = b
                try:
                    res[b] = timeit(lambda: attention(q, k, v, causal=args.causal), args.iters)
                except Exception as e:  # unsupported shape for this backend
                    res[b] = float("nan")
                    print(f"  {b} failed at B={B} N={N}: {str(e)[:60]}")
            os.environ.pop("FUSED_ATTN_BACKEND", None)
            for name in baselines:
                try:
                    res[name] = timeit(lambda: sdpa(name.split("_")[1])(q, k, v, args.causal), args.iters if name != "torch_math" else 3)
                except Exception:
                    res[name] = float("nan")
            for name, ms in res.items():
                rows.append({"batch": B, "seqlen": N, "backend": name, "ms": ms, "tflops": flops / (ms * 1e-3) / 1e12 if ms == ms else float("nan"),
                             "speedup_vs_flash": res["torch_flash"] / ms if ms == ms and res["torch_flash"] == res["torch_flash"] else float("nan")})
            best = min((b for b in backends if res[b] == res[b]), key=lambda b: res[b], default=None)
            print(f"{B:>3} {N:>6} | " + " ".join(f"{res[b]:>11.3f}ms" if res[b] == res[b] else f"{'n/a':>14}" for b in backends + baselines)
                  + (f" | {best}: {res['torch_flash'] / res[best]:.2f}x" if best and res["torch_flash"] == res["torch_flash"] else ""))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)  {time.strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
