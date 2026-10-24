"""The Nsight Compute summariser works on an exported raw CSV (no GPU needed)."""
import csv
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import ncu_summary  # noqa: E402


def _csv(metrics: dict[str, tuple[float, str]], kernel="attn_fwd_kernel", id_="0") -> list[dict]:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
    w.writeheader()
    for name, (val, unit) in metrics.items():
        w.writerow({"ID": id_, "Kernel Name": kernel, "Metric Name": name, "Metric Unit": unit, "Metric Value": f"{val:,}"})
    return list(csv.DictReader(io.StringIO(buf.getvalue())))


def test_memory_bound_diagnosis():
    rows = _csv({"gpu__time_duration.sum": (1_234_000, "nsecond"), "dram__throughput.avg.pct_of_peak_sustained_elapsed": (82.5, "%"),
                 "sm__throughput.avg.pct_of_peak_sustained_elapsed": (35, "%"), "sm__warps_active.avg.pct_of_peak_sustained_active": (60, "%"),
                 "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum": (0, ""), "launch__registers_per_thread": (96, "register/thread")})
    m = ncu_summary.summarize(rows)[0]
    assert abs(m["duration_us"] - 1234) < 1e-6 and m["regs"] == 96
    assert any("memory-bound" in n for n in ncu_summary.diagnose(m))


def test_occupancy_limiter_and_bank_conflicts():
    rows = _csv({"sm__throughput.avg.pct_of_peak_sustained_elapsed": (20, "%"), "dram__throughput.avg.pct_of_peak_sustained_elapsed": (15, "%"),
                 "sm__warps_active.avg.pct_of_peak_sustained_active": (18, "%"), "launch__occupancy_limit_registers": (2, "block"),
                 "launch__occupancy_limit_shared_mem": (8, "block"), "launch__occupancy_limit_blocks": (16, "block"),
                 "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum": (48_000, "")})
    notes = ncu_summary.diagnose(ncu_summary.summarize(rows)[0])
    assert any("limited by registers" in n for n in notes)
    assert any("bank conflicts" in n for n in notes)
