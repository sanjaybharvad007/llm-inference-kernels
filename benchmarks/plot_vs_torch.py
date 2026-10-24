"""Speed-up heatmaps (batch x seqlen) per backend vs torch flash, from bench_vs_torch.py output."""
import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(path: str):
    rows = list(csv.DictReader(open(path)))
    backends = sorted({r["backend"] for r in rows if not r["backend"].startswith("torch_")})
    batches = sorted({int(r["batch"]) for r in rows})
    seqs = sorted({int(r["seqlen"]) for r in rows})
    fig, axes = plt.subplots(1, len(backends), figsize=(4.2 * len(backends), 3.8), squeeze=False)
    for ax, b in zip(axes[0], backends):
        grid = [[float("nan")] * len(seqs) for _ in batches]
        for r in rows:
            if r["backend"] == b and r["speedup_vs_flash"] not in ("nan", ""):
                grid[batches.index(int(r["batch"]))][seqs.index(int(r["seqlen"]))] = float(r["speedup_vs_flash"])
        im = ax.imshow(grid, cmap="RdYlGn", vmin=0.5, vmax=1.5, aspect="auto")
        ax.set_xticks(range(len(seqs)), seqs, rotation=45)
        ax.set_yticks(range(len(batches)), batches)
        ax.set_xlabel("seqlen")
        ax.set_ylabel("batch")
        ax.set_title(f"{b} / torch flash")
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                if val == val:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[0].tolist(), label="speed-up (>1 = faster than torch flash)")
    out = path.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)
    agg = defaultdict(list)
    for r in rows:
        if r["speedup_vs_flash"] not in ("nan", ""):
            agg[r["backend"]].append(float(r["speedup_vs_flash"]))
    for b, v in sorted(agg.items()):
        print(f"{b:>16}: geomean speed-up vs torch flash = {(__import__('math').prod(v)) ** (1 / len(v)):.2f}x over {len(v)} shapes")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/vs_torch.csv")
