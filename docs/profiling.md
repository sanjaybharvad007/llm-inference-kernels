# Nsight-driven optimisation

The kernels in this repository were tuned by measuring, not by guessing. This
is the loop, with the metrics that decided each change.

```bash
bash scripts/profile_ncu.sh 4096 cuda_mma        # Nsight Compute, full section set, 3 launches
python scripts/ncu_summary.py ncu_cuda_mma_4096.ncu-rep
bash scripts/profile_nsys.sh cuda_mma 4096        # Nsight Systems timeline (launch gaps, overlap)
python benchmarks/tune.py --dtype bf16 --causal   # sweep tile configs, write fused_attn/tuned_configs.json
```

## The metrics that matter

| Metric (ncu) | Question it answers | Action when bad |
|---|---|---|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | Are we HBM-bound? | reduce bytes: fused attention (no N×N round-trip), int8 KV, bigger tiles for reuse |
| `sm__throughput`, `sm__pipe_tensor_cycles_active` | Are we compute-bound, and on the right pipe? | move QKᵀ/PV to `mma.sync`; vectorise the softmax epilogue |
| `sm__warps_active` + `launch__occupancy_limit_*` | Enough warps to hide latency, and what caps them? | registers: smaller `BLOCK_N` (fewer `s[]`), `__launch_bounds__`; smem: smaller tiles or dynamic smem opt-in; blocks: smaller `BLOCK_M` |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | Do `ldmatrix`/smem reads serialise? | pad rows (`LDS = HEAD_DIM + 8`) or XOR-swizzle |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` | Are global loads coalesced? | 16-byte vector loads, lanes along the contiguous dim |
| stall reasons (`smsp__pcsamp_warps_issue_stalled_*`) | long scoreboard = memory latency; barrier = too many `__syncthreads`; mio throttle = smem pressure | double-buffer / `cp.async`; wider `BLOCK_N`; fewer smem round-trips |

`scripts/ncu_summary.py` extracts exactly these and prints a one-line
diagnosis per kernel (memory-bound / compute-bound / latency-bound + the
limiter), so a tuning iteration is: change one knob → re-profile → read the
diagnosis → keep or revert.

## What the measurements drove

1. **SIMT kernel, first version**: `dram` 25%, `sm` 30%, occupancy capped by
   shared memory at 2 blocks/SM; `long_scoreboard` dominant. → double-buffered
   K/V tiles (prefetch the next tile while computing the current one), 16-byte
   vector loads. Occupancy limiter moved to registers; DRAM utilisation up.
2. **Two threads per query row** (`THREADS_PER_ROW=2`, each owning `D/2`):
   halves `acc[]`/`q[]` registers per thread, one `__shfl_xor_sync` to combine
   the dot product. Registers/thread dropped below the 128 cliff for `D=128`,
   doubling resident warps.
3. **Tile configurations** (`BLOCK_M × BLOCK_N` = 64×32 default; 128×32,
   64×64, 32×32, 32×64 selectable): `benchmarks/tune.py` measures them per
   shape. Short sequences and small batches want more blocks (32×32); long
   sequences want 128×32 (each K/V tile amortised over more rows). The winner
   per `(dtype, head_dim, seqlen bucket)` is written to
   `fused_attn/tuned_configs.json` and used at dispatch.
4. **Tensor-Core kernel**: `ldmatrix` from unpadded rows showed
   `bank_conflicts` in the tens of thousands per launch → padded row stride
   (`HEAD_DIM + 8` halves) → zero conflicts; `cp.async` two-stage ring to
   overlap the next tile's global→smem copy with the current `mma.sync`s.
5. **Decode**: single-pass kernels at long contexts left most SMs idle
   (`grid` of `B·H` blocks) → split-K with a log-sum-exp combine; then int8 KV
   to cut DRAM bytes ~1.9x (`docs/kv_cache_quantization.md`).

## Roofline

`ncu_summary.py` prints arithmetic intensity (FLOP/byte from FMA counts and
DRAM bytes). Compare it with the device ridge point
(peak FLOP/s ÷ peak bytes/s; ~300 FLOP/B for an A100 in bf16 Tensor-Core
terms): prefill attention sits right of the ridge (compute-bound → Tensor
Cores and softmax overlap matter), decode sits far left (memory-bound → bytes
matter, nothing else does).
