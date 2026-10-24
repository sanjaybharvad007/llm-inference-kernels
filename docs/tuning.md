# Block / grid / shared-memory tuning

The SIMT kernel (`csrc/flash_attn.cu`) is templated on the tile shape and
exposes five configurations through `AttnParams::config`
(`_C.simt_configs(head_dim, elem_size)` lists the ones valid for a given
head size):

| id | BLOCK_M (rows/block) | BLOCK_N (keys/tile) | threads | static smem (D=128, bf16) | when it wins |
|---:|---:|---:|---:|---:|---|
| 0 | 64 | 32 | 128 | 32 KB | default; balanced |
| 1 | 128 | 32 | 256 | 32 KB | long sequences: each K/V tile load serves 128 rows |
| 2 | 64 | 64 | 128 | D=64 only | half the `__syncthreads` per key; needs `s[64]` registers |
| 3 | 32 | 32 | 64 | 32 KB | short sequences / small batch: 2x more blocks to fill SMs |
| 4 | 32 | 64 | 64 | D=64 only | |

The trade-offs are the classic ones:

- **Occupancy** = resident warps / max warps, capped by the smallest of
  registers, shared memory and blocks per SM. `BLOCK_N` sets the `s[BLOCK_N]`
  score array (registers); the double-buffered K and V tiles set static smem
  (`2 × 2 × BLOCK_N × D × sizeof(T)`). Configurations that would exceed 48 KB
  static smem or the register budget are not instantiated (`if constexpr`).
- **Grid size** = `ceil(N_q / BLOCK_M) × B × H`. Below ~2 waves of blocks the
  machine is under-filled, which is why small shapes prefer `BLOCK_M = 32`.
- **Reuse**: every K/V tile is read from global memory once per block, so
  bytes moved ∝ `N_k × N_q / BLOCK_M`; larger `BLOCK_M` cuts traffic.
- **Warp-level execution**: two lanes per query row split `D`, combined with
  one `__shfl_xor_sync`; the Tensor-Core kernel instead assigns 16 rows per
  warp and uses `mma.sync.m16n8k16` fragments with `ldmatrix` loads.

```bash
python benchmarks/tune.py --dtype bf16 --head-dims 64 128 --seqlens 256 1024 4096 16384 --causal
```

prints a `seqlen × config` table of ms / TFLOP/s / waves and writes the winners
to `fused_attn/tuned_configs.json`; `ops.simt_config_for()` reads it at
dispatch time. The Triton kernel's `@triton.autotune` covers the same space
(`BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages`) and the script records its
chosen configuration per shape for the README.
