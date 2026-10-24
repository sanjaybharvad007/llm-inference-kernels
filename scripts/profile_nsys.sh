#!/usr/bin/env bash
# Nsight Systems timeline of a benchmark run: kernel durations, launch gaps, memcpy overlap.
#   bash scripts/profile_nsys.sh [backend] [seqlen]
set -euo pipefail
BACKEND="${1:-cuda_mma}"; SEQ="${2:-4096}"
FUSED_ATTN_BACKEND="$BACKEND" nsys profile -t cuda,nvtx,osrt --cuda-graph-trace=node --stats=true -o "nsys_${BACKEND}_${SEQ}" --force-overwrite true \
  python benchmarks/bench.py --seqlens "$SEQ" --heads 32 --kv-heads 8 --head-dim 128 --causal --dtype bf16 --out /tmp/bench.csv
nsys stats --report cuda_gpu_kern_sum --format csv "nsys_${BACKEND}_${SEQ}.nsys-rep" | head -20
