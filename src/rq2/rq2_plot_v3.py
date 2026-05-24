# RQ2 - Bar chart + summary table
# Picks the best config per index at >= TARGET_RECALL and shows QPS vs Flat

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

IN_NPZ        = Path(__file__).parent / "rq2_results.npz"
OUT_PNG       = Path(__file__).parent / "rq2_plot_v3.png"
TARGET_RECALL = 0.95


def best_config(rows, param_col, param_name, sweep_name):
    """Return (label, recall, qps) for the config with highest QPS at >= TARGET_RECALL."""
    above = rows[rows[:, 2] >= TARGET_RECALL]
    if not len(above):
        # fallback: highest recall achieved
        row = rows[rows[:, 2].argmax()]
        label = f"{param_name}={int(row[param_col])}, {sweep_name}={int(row[1])}"
        return label, float(row[2]), float(row[3]), False
    row = above[above[:, 3].argmax()]
    label = f"{param_name}={int(row[param_col])}, {sweep_name}={int(row[1])}"
    return label, float(row[2]), float(row[3]), True


def main():
    if not IN_NPZ.exists():
        raise FileNotFoundError(f"{IN_NPZ} not found — run rq2_benchmark.py first.")

    data     = np.load(IN_NPZ)
    ivf      = data["ivf"]
    hnsw     = data["hnsw"]
    flat_qps = float(data["flat_qps"])
    K        = int(data["K"])

    ivf_label,  ivf_recall,  ivf_qps,  ivf_ok  = best_config(ivf,  0, "nlist", "nprobe")
    hnsw_label, hnsw_recall, hnsw_qps, hnsw_ok = best_config(hnsw, 0, "M",     "efSearch")

    entries = [
        ("Flat\n(exact)",  flat_qps,  1.0,         "#9E9E9E", "—"),
        (f"IVF\n({ivf_label})",   ivf_qps,   ivf_recall,  "#1E88E5", f"{ivf_qps/flat_qps:.1f}×"),
        (f"HNSW\n({hnsw_label})", hnsw_qps,  hnsw_recall, "#EF6C00", f"{hnsw_qps/flat_qps:.1f}×"),
    ]

    labels    = [e[0] for e in entries]
    qps_vals  = [e[1] for e in entries]
    recalls   = [e[2] for e in entries]
    colors    = [e[3] for e in entries]
    speedups  = [e[4] for e in entries]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(labels, qps_vals, color=colors, width=0.5, edgecolor="white", linewidth=1.2)

    for bar, recall, speedup in zip(bars, recalls, speedups):
        h = bar.get_height()
        # recall inside the bar
        ax.text(
            bar.get_x() + bar.get_width() / 2, h * 0.5,
            f"Recall@{K}\n{recall:.3f}",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold",
        )
        # speedup label above bar
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + max(qps_vals) * 0.01,
            speedup,
            ha="center", va="bottom", fontsize=9, color="#333333",
        )

    ax.axhline(flat_qps, color="#9E9E9E", linestyle="--", linewidth=1.0, alpha=0.5)

    ax.set_ylabel("Queries per second (QPS)", fontsize=11)
    ax.set_title(
        #f"RQ2 — Query Throughput of Selected FAISS Configurations\n"
        f"(best config at Recall@{K} ≥ {int(TARGET_RECALL*100)}%,  speedup relative to exact Flat search)",
        fontsize=11,
    )
    ax.set_ylim(0, max(qps_vals) * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PNG}")

    # ── summary table (copy-paste into LaTeX) ────────────────────────────────
    print(f"\n── RQ2 Summary Table {'─'*50}")
    print(f"{'Index':<8}  {'Configuration':<30}  {'Recall@'+str(K):>10}  {'QPS':>10}  {'Speedup':>8}")
    print("─" * 74)
    print(f"  {'Flat':<6}  {'—':<30}  {'1.000':>10}  {flat_qps:>10,.0f}  {'1.0×':>8}")
    print(f"  {'IVF':<6}  {ivf_label:<30}  {ivf_recall:>10.3f}  {ivf_qps:>10,.0f}  {ivf_qps/flat_qps:>7.1f}×")
    print(f"  {'HNSW':<6}  {hnsw_label:<30}  {hnsw_recall:>10.3f}  {hnsw_qps:>10,.0f}  {hnsw_qps/flat_qps:>7.1f}×")

    plt.show()


if __name__ == "__main__":
    main()
