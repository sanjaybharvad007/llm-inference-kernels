"""Launch-configuration tuner: measures every SIMT tile configuration (block_m,
block_n, threads, shared memory) and every Triton autotune config across a grid
of shapes, and writes the winners to fused_attn/tuned_configs.json, which
ops.py consults at dispatch time.

    python benchmarks/tune.py --dtype bf16 --head-dims 64 128 --seqlens 256 1024 4096 16384 --batch 4 --heads 16

Prints a table of (config -> ms, TFLOP/s, achieved occupancy proxy) so the
effect of each knob is visible, not just the winner.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from fused_attn import ops

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "fused_attn", "tuned_configs.json")


def bench(fn, iters=30, warmup=5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def tflops(ms: float, b: int, h: int, n: int, d: int, causal: bool) -> float:
    flops = 4.0 * b * h * n * n * d * (0.5 if causal else 1.0)
    return flops / (ms * 1e-3) / 1e12


def tune_simt(args, table: dict) -> None:
    ext = ops._cuda_ext
    if ext is None or not hasattr(ext, "simt_configs"):
        print("CUDA extension not built; skipping SIMT tuning")
        return
    dt = getattr(torch, args.dtype)
    props = torch.cuda.get_device_properties(0)
    for d in args.head_dims:
        cfgs = ext.simt_configs(d, torch.tensor([], dtype=dt).element_size())
        key = f"{args.dtype}/{d}"
        table.setdefault(key, {})
        print(f"\n== SIMT  dtype={args.dtype} head_dim={d}  ({len(cfgs)} configs)")
        print(f"{'seqlen':>7} " + " ".join(f"cfg{c[0]}:{c[1]}x{c[2]}/{c[3]}t/{c[4] // 1024}KB".rjust(22) for c in cfgs) + "   best")
        for n in args.seqlens:
            q = torch.randn(args.batch, args.heads, n, d, device="cuda", dtype=dt)
            k = torch.randn(args.batch, args.kv_heads, n, d, device="cuda", dtype=dt)
            v = torch.randn_like(k)
            row, best = [], (None, 1e9)
            for c in cfgs:
                cid, bm = c[0], c[1]
                blocks = ((n + bm - 1) // bm) * args.batch * args.heads
                waves = blocks / (props.multi_processor_count * max(1, 2048 // c[3]))
                ms = bench(lambda: ext.attention_fwd(q, k, v, args.causal, d ** -0.5, cid))
                row.append(f"{ms:7.3f}ms {tflops(ms, args.batch, args.heads, n, d, args.causal):5.1f}T w{waves:4.1f}")
                if ms < best[1]:
                    best = (cid, ms)
            table[key][str(n)] = best[0]
            print(f"{n:>7} " + " ".join(x.rjust(22) for x in row) + f"   cfg{best[0]}")


def tune_triton(args, table: dict) -> None:
    try:
        from fused_attn.triton_fwd import attention_triton
    except Exception as e:  # pragma: no cover
        print("Triton unavailable:", e)
        return
    dt = getattr(torch, args.dtype)
    print("\n== Triton autotune (best config per shape is cached by Triton; recorded here for the README)")
    for d in args.head_dims:
        for n in args.seqlens:
            q = torch.randn(args.batch, args.heads, n, d, device="cuda", dtype=dt)
            k = torch.randn(args.batch, args.kv_heads, n, d, device="cuda", dtype=dt)
            v = torch.randn_like(k)
            ms = bench(lambda: attention_triton(q, k, v, causal=args.causal))
            kern = getattr(attention_triton, "_kernel", None) or getattr(__import__("fused_attn.triton_fwd", fromlist=["_attn_fwd_kernel"]), "_attn_fwd_kernel", None)
            best = None
            if kern is not None and hasattr(kern, "best_config") and kern.best_config is not None:
                bc = kern.best_config
                best = {**bc.kwargs, "num_warps": bc.num_warps, "num_stages": bc.num_stages}
            table[f"{args.dtype}/{d}/{n}"] = {"ms": round(ms, 4), "tflops": round(tflops(ms, args.batch, args.heads, n, d, args.causal), 2), "config": best}
            print(f"  head_dim={d:4d} seqlen={n:6d}  {ms:8.3f} ms  {tflops(ms, args.batch, args.heads, n, d, args.causal):6.1f} TFLOP/s  {best}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--head-dims", type=int, nargs="+", default=[64, 128])
    p.add_argument("--seqlens", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--kv-heads", type=int, default=4)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args()
    assert torch.cuda.is_available(), "tuning needs a GPU"
    with open(OUT) as f:
        tuned = json.load(f)
    tune_simt(args, tuned.setdefault("simt", {}))
    tune_triton(args, tuned.setdefault("triton", {}))
    tuned["device"] = torch.cuda.get_device_name(0)
    tuned["measured_at"] = time.strftime("%Y-%m-%d")
    if not args.no_write:
        with open(OUT, "w") as f:
            json.dump(tuned, f, indent=2)
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
