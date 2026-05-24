# RQ2 - Scatter plot of all tested configurations
# Each point = one (nlist, nprobe) or (M, efSearch) combination
# Highlights the best config per index family at >= TARGET_RECALL

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

IN_NPZ        = Path(__file__).parent / "rq2_results.npz"
OUT_PNG       = Path(__file__).parent / "rq2_plot_v4.png"
TARGET_RECALL = 0.95

IVF_COLOR  = "#1E88E5"
HNSW_COLOR = "#EF6C00"
FLAT_COLOR = "#757575"


def best_above_target(rows):
    """Highest-QPS row at >= TARGET_RECALL, or None if none reach it."""
    above = rows[rows[:, 2] >= TARGET_RECALL]
    if not len(above):
        return None
    return above[above[:, 3].argmax()]


def main():
    if not IN_NPZ.exists():
        raise FileNotFoundError(f"{IN_NPZ} not found — run rq2_benchmark.py first.")

    data     = np.load(IN_NPZ)
    ivf      = data["ivf"]
    hnsw     = data["hnsw"]
    flat_qps = float(data["flat_qps"])
    K        = int(data["K"])

    ivf_best  = best_above_target(ivf)
    hnsw_best = best_above_target(hnsw)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # ── all IVF configs ───────────────────────────────────────────────────────
    ax.scatter(
        ivf[:, 3], ivf[:, 2],
        marker="o", s=30, color=IVF_COLOR, alpha=0.45, zorder=2,
        label="IVF configurations",
    )

    # ── all HNSW configs ──────────────────────────────────────────────────────
    ax.scatter(
        hnsw[:, 3], hnsw[:, 2],
        marker="^", s=35, color=HNSW_COLOR, alpha=0.45, zorder=2,
        label="HNSW configurations",
    )

    # ── highlight selected configs ────────────────────────────────────────────
    if ivf_best is not None:
        ax.scatter(
            ivf_best[3], ivf_best[2],
            marker="o", s=120, color=IVF_COLOR, edgecolors="white",
            linewidths=1.5, zorder=5,
        )
        ax.annotate(
            f"IVF  nlist={int(ivf_best[0])}, nprobe={int(ivf_best[1])}\n"
            f"Recall={ivf_best[2]:.3f},  {ivf_best[3]:,.0f} QPS",
            xy=(ivf_best[3], ivf_best[2]),
            xytext=(15, -30), textcoords="offset points",
            fontsize=8, color=IVF_COLOR,
            arrowprops=dict(arrowstyle="-", color=IVF_COLOR, lw=0.9),
        )

    if hnsw_best is not None:
        ax.scatter(
            hnsw_best[3], hnsw_best[2],
            marker="^", s=130, color=HNSW_COLOR, edgecolors="white",
            linewidths=1.5, zorder=5,
        )
        ax.annotate(
            f"HNSW  M={int(hnsw_best[0])}, efSearch={int(hnsw_best[1])}\n"
            f"Recall={hnsw_best[2]:.3f},  {hnsw_best[3]:,.0f} QPS",
            xy=(hnsw_best[3], hnsw_best[2]),
            xytext=(15, 12), textcoords="offset points",
            fontsize=8, color=HNSW_COLOR,
            arrowprops=dict(arrowstyle="-", color=HNSW_COLOR, lw=0.9),
        )

    # ── reference lines ───────────────────────────────────────────────────────
    ax.axhline(TARGET_RECALL, color="#E53935", linestyle="--", linewidth=1.2,
               label=f"{int(TARGET_RECALL*100)}% recall target", zorder=1)
    ax.axvline(flat_qps, color=FLAT_COLOR, linestyle="--", linewidth=1.2,
               label="Flat / exact baseline", zorder=1)
    ax.text(flat_qps * 1.03, 0.47, "Flat (exact)", fontsize=8,
            color=FLAT_COLOR, va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("Queries per second (QPS)  [log scale]", fontsize=11)
    ax.set_ylabel(f"Recall@{K}", fontsize=11)
    ax.set_ylim(0.44, 1.06)
    ax.set_title(
        f"RQ2 — All Tested FAISS Configurations  "
        f"({len(ivf)} IVF  ·  {len(hnsw)} HNSW)\n"
        f"Highlighted: best config per family at Recall@{K} ≥ {int(TARGET_RECALL*100)}%",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
