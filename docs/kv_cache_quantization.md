# KV-cache optimisation and quantisation

Decode attention is memory-bound: for every generated token, each layer reads
its whole K and V history once and does ~4 FLOPs per element read. The
arithmetic intensity (~1 FLOP/byte for fp16) is two orders of magnitude below
the ridge point of any modern GPU, so the only lever is **bytes**.

| Format | bytes / token / KV head (D=128) | notes |
|---|---:|---|
| fp16 / bf16 | 512 | baseline |
| **int8, per-token scale** (`csrc/kv_cache.cu`) | 264 | one fp32 scale per (token, head) for K and V |
| fp8 E4M3, per-64-key block scale (`fused_attn/fp8.py`) | ~258 | sm_89+, Triton |

## INT8 with per-token scales

`quantize_kv_int8`: symmetric, `scale = max|x| / 127` per (token, head) row.
Per-row scales matter for attention: a few tokens (BOS, punctuation "sinks")
have key norms many times larger than the rest; a per-tensor or per-channel
scale would flatten every other token into a handful of integer levels and
distort the softmax. Per-row scales keep the relative error under ~1% for
every token (`tests/test_kv_cache.py::test_int8_roundtrip_is_tight_per_token`).

### Kernels

- **`quantize_kv_kernel`** — one warp per row; lanes stride over D for
  coalesced loads, warp-shuffle max, quantise, store. 4 rows per 128-thread block.
- **`paged_append_kernel`** — grid `(num_tokens, H_kv)`, two warps per block:
  warp 0 quantises K, warp 1 V, written straight into the paged cache at
  `slot_mapping[t]` (`block * block_size + offset`). Quantisation is fused into
  the cache write, so the fp16 K/V are never stored.
- **`decode_kv8_kernel`** — split-K decode over the int8 cache. Grid
  `(splits, H_q, B)`, 4 warps; warps interleave over key positions, each lane
  owns `D/32` elements of every row (4 consecutive int8 = one 32-bit load per
  lane per row, fully coalesced). The dot product is computed on raw int8
  values and multiplied by the row scale once (`k_scale * Σ q_i k_i`), the
  probability is folded into `v_scale` before the FMA into the fp32
  accumulator — dequantisation costs one multiply per row, not per element.
  Each warp keeps its own online-softmax `(m, l, acc)`; the four are merged in
  shared memory; the splits are merged by `combine_splits_kernel` with the
  standard `exp2(m_s − m)` rescale. `decode_kv8_num_splits` picks splits so
  the grid fills the machine ~2x with ≥256 keys per split.

### Mixed precision

Storage int8, dot products in fp32 (`fmaf` on converted int8), softmax in
fp32 with `exp2` and `log2(e)` folded into the Q scale, output rounded to the
query dtype (fp16/bf16/fp32). The error budget is dominated by the int8
storage (≤ ~1% per element), not by the arithmetic.

## Measuring it

```bash
python benchmarks/bench_kv8.py --ctx 2048 8192 32768 --batch 8 --heads 32 --kv-heads 8
```

prints latency and achieved GB/s for fp16 split-K vs int8 paged decode. The
int8 kernel moves ~1.9x fewer bytes; on bandwidth-bound shapes the speed-up
approaches that ratio, and the achieved GB/s column tells you how close each
kernel is to the device's HBM peak (`scripts/ncu_summary.py` prints the same
as `dram__throughput`).
