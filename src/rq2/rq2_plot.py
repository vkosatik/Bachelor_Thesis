# RQ2 - Recall-Latency Plot
# Loads rq2_results.npz and produces the recall@K vs QPS frontier figure

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

IN_NPZ  = Path(__file__).parent / "rq2_results.npz"
OUT_PNG = Path(__file__).parent / "rq2_plot.png"

# IVF: shades of blue;  HNSW: shades of orange/red
IVF_COLORS  = ["#90CAF9", "#42A5F5", "#1E88E5", "#1565C0", "#0D47A1"]
HNSW_COLORS = ["#FFCC80", "#FFA726", "#EF6C00", "#BF360C"]
FLAT_COLOR  = "#757575"

TARGET_RECALL = 0.95


def load(path: Path):
    d        = np.load(path)
    ivf      = d["ivf"]        # cols: nlist, nprobe, recall, qps
    hnsw     = d["hnsw"]       # cols: M, efSearch, recall, qps
    flat_qps = float(d["flat_qps"])
    K        = int(d["K"])
    return ivf, hnsw, flat_qps, K


def plot_frontier(ax, rows, param_col, param_vals, colors, label_prefix, param_name):
    """One curve per unique value of param_col (e.g. nlist or M)."""
    for color, val in zip(colors, param_vals):
        mask = rows[:, param_col] == val
        subset = rows[mask]
        subset = subset[subset[:, 3].argsort()]     # sort by QPS ascending
        ax.plot(
            subset[:, 3], subset[:, 2],
            "o-", color=color, linewidth=1.8, markersize=5,
            label=f"{label_prefix} {param_name}={int(val)}",
        )


def main():
    if not IN_NPZ.exists():
        raise FileNotFoundError(f"{IN_NPZ} not found — run rq2_benchmark.py first.")

    ivf, hnsw, flat_qps, K = load(IN_NPZ)

    ivf_nlists  = sorted(np.unique(ivf[:, 0]).tolist())
    hnsw_ms     = sorted(np.unique(hnsw[:, 0]).tolist())

    fig, ax = plt.subplots(figsize=(10, 6))

    plot_frontier(ax, ivf,  param_col=0, param_vals=ivf_nlists,
                  colors=IVF_COLORS,  label_prefix="IVF",  param_name="nlist")
    plot_frontier(ax, hnsw, param_col=0, param_vals=hnsw_ms,
                  colors=HNSW_COLORS, label_prefix="HNSW", param_name="M")

    ax.axvline(flat_qps, color=FLAT_COLOR, linestyle="--", linewidth=1.5,
               zorder=5, label=f"Flat (exact)")

    ax.axhline(TARGET_RECALL, color="#E53935", linestyle="--", linewidth=1.2,
               label=f"{int(TARGET_RECALL*100)}% recall target")

    ax.set_xscale("log")
    ax.set_xlabel("Queries per second (QPS)  ←  slower  |  faster  →", fontsize=11)
    ax.set_ylabel(f"Recall@{K}", fontsize=11)
    ax.set_ylim(0.45, 1.05)
    ax.set_title(
        f"RQ2 — Recall@{K} vs. Query Throughput\n"
        f"FAISS Index Comparison on FediData CLIP User Embeddings",
        fontsize=12,
    )
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PNG}")

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n── RQ2 Summary ({'─'*60})")
    print(f"{'Index':<18} {'Config':<20} {'Recall@'+str(K):>10} {'QPS':>10}")
    print("-" * 62)

    print(f"  {'Flat (exact)':<16} {'—':<20} {'1.000':>10} {flat_qps:>10,.0f}")

    # best IVF config per nlist at >= TARGET_RECALL
    for nlist in ivf_nlists:
        subset = ivf[ivf[:, 0] == nlist]
        above  = subset[subset[:, 2] >= TARGET_RECALL]
        if len(above):
            best = above[above[:, 3].argmax()]
            print(f"  {'IVF':<16} nlist={int(nlist):<5} nprobe={int(best[1]):<5}"
                  f"  {best[2]:>8.3f}  {best[3]:>10,.0f}")
        else:
            worst = subset[subset[:, 2].argmax()]
            print(f"  {'IVF':<16} nlist={int(nlist):<5} (max recall={worst[2]:.3f}, "
                  f"never reaches {TARGET_RECALL})")

    for M in hnsw_ms:
        subset = hnsw[hnsw[:, 0] == M]
        above  = subset[subset[:, 2] >= TARGET_RECALL]
        if len(above):
            best = above[above[:, 3].argmax()]
            print(f"  {'HNSW':<16} M={int(M):<8} efSearch={int(best[1]):<5}"
                  f"  {best[2]:>8.3f}  {best[3]:>10,.0f}")
        else:
            worst = subset[subset[:, 2].argmax()]
            print(f"  {'HNSW':<16} M={int(M):<8} (max recall={worst[2]:.3f}, "
                  f"never reaches {TARGET_RECALL})")

    plt.show()


if __name__ == "__main__":
    main()
