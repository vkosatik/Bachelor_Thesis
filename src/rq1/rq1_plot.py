# creates plot for rq1 visualizing the distribution shifts of similarity between the 3 tested techniques
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

IN_NPZ  = Path(__file__).parent / "rq1_results.npz"
OUT_PNG = Path(__file__).parent / "rq1_plot.png"

CONN_COLOR = "#2196F3"   # blue  — connected
DISC_COLOR = "#FF9800"   # orange — disconnected
BINS = 60

data = np.load(IN_NPZ)

methods = [
    ("Mean pooling",   data["conn_mean_pool"],     data["disc_mean_pool"]),
    ("Pairwise mean",  data["conn_pairwise_mean"], data["disc_pairwise_mean"]),
    ("Top-3 pairwise", data["conn_top_k"],         data["disc_top_k"]),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("RQ1 — Visual Homophily: Cosine Similarity Distribution\n"
             "Connected vs. Disconnected User Pairs (n = 10,000 each)",
             fontsize=13)

for ax, (name, conn, disc) in zip(axes, methods):
    lo = min(conn.min(), disc.min())
    hi = max(conn.max(), disc.max())
    bins = np.linspace(lo, hi, BINS)

    ax.hist(disc, bins=bins, color=DISC_COLOR, alpha=0.6,
            label=f"Disconnected  μ={disc.mean():.3f}")
    ax.hist(conn, bins=bins, color=CONN_COLOR, alpha=0.6,
            label=f"Connected     μ={conn.mean():.3f}")

    ax.axvline(disc.mean(), color=DISC_COLOR, linewidth=1.5, linestyle="--")
    ax.axvline(conn.mean(), color=CONN_COLOR, linewidth=1.5, linestyle="--")

    _, p = stats.mannwhitneyu(conn, disc, alternative="greater")
    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.set_title(f"{name}\n{p_str}", fontsize=11)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Number of pairs")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved to {OUT_PNG}")

print("\n── RQ1 Statistical Results ───────────────────────────────────────────")
print(f"{'Method':<20} {'Conn μ':>8} {'Disc μ':>8} {'Δ':>8} {'U':>12} {'p':>12} {'r (effect)':>12}")
print("-" * 84)
for name, conn, disc in methods:
    U, p = stats.mannwhitneyu(conn, disc, alternative="greater")
    # rank-biserial correlation: r = (2U / (n1*n2)) - 1
    n = len(conn) * len(disc)
    r = (2 * U / n) - 1
    p_str = "< 0.001" if p < 0.001 else f"{p:.4f}"
    print(f"{name:<20} {conn.mean():>8.4f} {disc.mean():>8.4f} "
          f"{conn.mean()-disc.mean():>+8.4f} {U:>12.0f} {p_str:>12} {r:>12.4f}")

plt.show()
