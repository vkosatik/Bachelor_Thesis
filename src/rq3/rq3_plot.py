# RQ3 results plot.
#
# Three panels (one per follow rate). Within each panel:
#   - x-axis: n_accounts
#   - grouped bars: n_images (5, 10, 20)
#   - CLIP recall only (pHash recall = 0 across all conditions, reported in text)
# Saves: src/rq3/rq3_plot.png

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_CSV = Path(__file__).parent / "rq3_results.csv"
OUT_PNG     = Path(__file__).parent / "rq3_plot.png"

FOLLOW_RATES  = [0.0, 0.3, 0.6]
ACCOUNTS_GRID = [20, 30, 50, 100]
IMAGES_GRID   = [5, 10, 20]

CLIP_COLOR = ["#2166ac", "#4393c3", "#92c5de"]   # blue shades per n_images


def main():
    df = pd.read_csv(RESULTS_CSV)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.suptitle("RQ3: CLIP Recall on Synthetic IO Accounts", fontsize=13, fontweight="bold")

    x       = np.arange(len(ACCOUNTS_GRID))
    n_img   = len(IMAGES_GRID)
    width   = 0.18
    offsets = np.linspace(-(n_img - 1) * width / 2, (n_img - 1) * width / 2, n_img)

    for ax_idx, follow_rate in enumerate(FOLLOW_RATES):
        ax  = axes[ax_idx]
        sub = df[df["follow_rate"] == follow_rate]

        for i, n_images in enumerate(IMAGES_GRID):
            row = sub[sub["n_images"] == n_images].set_index("n_accounts")
            clip_vals = [row.loc[n, "clip_recall"] if n in row.index else 0 for n in ACCOUNTS_GRID]

            ax.bar(x + offsets[i], clip_vals, width, label=f"n_images={n_images}",
                   color=CLIP_COLOR[i], edgecolor="white")

        ax.set_title(f"Follow rate = {int(follow_rate*100)}%", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in ACCOUNTS_GRID])
        ax.set_xlabel("Synthetic accounts (n)")
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("CLIP Recall")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
