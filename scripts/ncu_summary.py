"""Turn an Nsight Compute report into the handful of numbers that drive kernel
tuning decisions, with a plain-language diagnosis.

    ncu --set full -o attn python benchmarks/bench.py --seqlens 4096 ...
    python scripts/ncu_summary.py attn.ncu-rep            # or an exported CSV: ncu --import attn.ncu-rep --csv --page raw > attn.csv

For each kernel launch it prints: duration, SM and DRAM throughput as % of
peak, Tensor-pipe utilisation, achieved occupancy and its limiter (registers /
shared memory / blocks), shared-memory bank conflicts, global load sector
efficiency, and the roofline position (arithmetic intensity vs ridge point).
"""
from __future__ import annotations

import csv
import io
import subprocess
import sys
from collections import defaultdict

METRICS = {
    "gpu__time_duration.sum": "duration_us",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": "tensor_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
    "launch__occupancy_limit_registers": "occ_limit_regs",
    "launch__occupancy_limit_shared_mem": "occ_limit_smem",
    "launch__occupancy_limit_blocks": "occ_limit_blocks",
    "launch__registers_per_thread": "regs",
    "launch__shared_mem_per_block_static": "smem_static",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum": "bank_conflicts",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct": "ld_sector_eff",
    "dram__bytes.sum": "dram_bytes",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum": "ffma",
    "sm__sass_thread_inst_executed_op_hfma_pred_on.sum": "hfma",
    "launch__grid_size": "grid",
    "launch__block_size": "block",
}


def load(path: str) -> list[dict]:
    if path.endswith(".ncu-rep"):
        out = subprocess.run(["ncu", "--import", path, "--csv", "--page", "raw"], check=True, capture_output=True, text=True).stdout
    else:
        out = open(path).read()
    rows = list(csv.DictReader(io.StringIO(out)))
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """ncu raw CSV: one row per metric per launch (ID, Kernel Name, Metric Name, Metric Unit, Metric Value)."""
    per_launch: dict[str, dict] = defaultdict(dict)
    for r in rows:
        key = r.get("ID") or r.get("Kernel Name")
        name = r.get("Metric Name", "")
        if name in METRICS:
            try:
                val = float(str(r["Metric Value"]).replace(",", ""))
            except ValueError:
                continue
            unit = r.get("Metric Unit", "")
            if name == "gpu__time_duration.sum":
                val = val / 1000 if unit == "nsecond" else val * 1000 if unit == "msecond" else val
            per_launch[key][METRICS[name]] = val
            per_launch[key]["kernel"] = r.get("Kernel Name", "?")
    return list(per_launch.values())


def diagnose(m: dict) -> list[str]:
    notes = []
    sm, dram, occ = m.get("sm_pct", 0), m.get("dram_pct", 0), m.get("occupancy_pct", 0)
    if dram > 70:
        notes.append("memory-bound: DRAM near peak -> reduce bytes (quantised KV, larger tiles for reuse)")
    elif sm > 70:
        notes.append("compute-bound: SM near peak -> Tensor Cores / fewer instructions per element")
    elif occ < 30:
        lim = min(((m.get("occ_limit_regs", 99), "registers"), (m.get("occ_limit_smem", 99), "shared memory"), (m.get("occ_limit_blocks", 99), "block count")))
        notes.append(f"latency-bound: low occupancy limited by {lim[1]} -> smaller tiles / fewer registers / more blocks")
    else:
        notes.append("neither pipe saturated: check stall reasons (long scoreboard = memory latency; barrier = too many __syncthreads)")
    if m.get("bank_conflicts", 0) > 1000:
        notes.append(f"{int(m['bank_conflicts'])} shared-memory bank conflicts -> pad rows / swizzle")
    if 0 < m.get("ld_sector_eff", 100) < 80:
        notes.append(f"global loads use {m['ld_sector_eff']:.0f}% of fetched sectors -> vectorise / coalesce")
    if m.get("tensor_pct", 0) > 0 and m["tensor_pct"] < 40 and sm > 40:
        notes.append("Tensor pipe under-used while SM busy -> softmax/epilogue dominates; overlap or vectorise it")
    return notes


def main(path: str):
    launches = summarize(load(path))
    if not launches:
        print("no recognised metrics; export with: ncu --import <rep> --csv --page raw")
        return
    for m in launches:
        flops = 2 * (m.get("ffma", 0) + m.get("hfma", 0))
        ai = flops / m["dram_bytes"] if m.get("dram_bytes") else float("nan")
        print(f"\n{m['kernel'][:90]}")
        print(f"  {m.get('duration_us', 0):8.1f} us   grid {int(m.get('grid', 0))} x block {int(m.get('block', 0))}   regs/thread {int(m.get('regs', 0))}   static smem {int(m.get('smem_static', 0))} B")
        print(f"  SM {m.get('sm_pct', 0):5.1f}%   DRAM {m.get('dram_pct', 0):5.1f}%   Tensor pipe {m.get('tensor_pct', 0):5.1f}%   achieved occupancy {m.get('occupancy_pct', 0):5.1f}%")
        print(f"  bank conflicts {int(m.get('bank_conflicts', 0))}   global load sector efficiency {m.get('ld_sector_eff', 0):5.1f}%   arithmetic intensity {ai:.1f} FLOP/B (FMA-based)")
        for n in diagnose(m):
            print("  ->", n)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "attn.ncu-rep")
