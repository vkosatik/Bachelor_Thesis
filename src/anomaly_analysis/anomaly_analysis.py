    # Anomaly analysis: real disconnected user pairs flagged by CLIP+FAISS.
#
# Loads the 10k original RQ1 disconnected pairs alongside their mean-pool
# cosine similarities, annotates each pair as same-instance or cross-instance,
# then characterises the 2040 pairs (20.4%) that exceed the RQ1 detection
# threshold (0.787).
#
# Outputs:
#   src/anomaly_analysis/anomaly_results.csv   — per-pair table with sim + flags
#   src/anomaly_analysis/anomaly_plot.png      — 3-panel figure

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT            = Path(__file__).parent.parent.parent
DISC_PAIRS_CSV  = ROOT / "dataset" / "rq1_pairs" / "rq1_disconnected_pairs.csv"
ACCOUNTS_CSV    = ROOT / "dataset" / "anonymous_60k_user_info" / "accounts_info_60k_anonymized.csv"
RQ1_RESULTS     = ROOT / "src" / "rq1" / "rq1_results.npz"
OUT_CSV         = Path(__file__).parent / "anomaly_results.csv"
OUT_PNG         = Path(__file__).parent / "anomaly_plot.png"

CLIP_THRESH = 0.787   # RQ1 μ_conn (mean pooling)


# ── load ──────────────────────────────────────────────────────────────────────

def load_instance_map():
    """row_index → instance string. Pair CSVs use row indices into accounts CSV."""
    accounts = pd.read_csv(ACCOUNTS_CSV, usecols=["uid"])
    imap = {}
    for idx, uid_field in enumerate(accounts["uid"]):
        if not isinstance(uid_field, str):
            continue
        # uid format: "handle@instance#numeric_id"
        handle_part = uid_field.split("#")[0]   # "handle@instance"
        if "@" in handle_part:
            imap[idx] = handle_part.split("@")[-1]
    return imap


def main():
    print("Loading data...")
    disc_sims = np.load(RQ1_RESULTS)["disc_mean_pool"]          # shape (10000,)
    pairs     = pd.read_csv(DISC_PAIRS_CSV)                      # user_a, user_b
    assert len(pairs) == len(disc_sims), "Pair count mismatch with npz"

    imap = load_instance_map()
    print(f"  Instance map: {len(imap):,} users")

    # annotate pairs
    pairs["sim"]            = disc_sims
    pairs["flagged"]        = pairs["sim"] > CLIP_THRESH
    pairs["inst_a"]         = pairs["user_a"].map(imap)
    pairs["inst_b"]         = pairs["user_b"].map(imap)
    pairs["same_instance"]  = pairs["inst_a"] == pairs["inst_b"]

    # drop pairs where instance is unknown for both
    known_mask = pairs["inst_a"].notna() & pairs["inst_b"].notna()
    pairs_k = pairs[known_mask].copy()
    print(f"  Pairs with known instance: {len(pairs_k):,} / {len(pairs):,}")

    # ── summary stats ─────────────────────────────────────────────────────────

    n_flagged      = pairs_k["flagged"].sum()
    fp_overall     = pairs_k["flagged"].mean()
    same           = pairs_k[pairs_k["same_instance"]]
    cross          = pairs_k[~pairs_k["same_instance"]]
    fp_same        = same["flagged"].mean()
    fp_cross       = cross["flagged"].mean()

    flagged_same   = pairs_k[pairs_k["flagged"] & pairs_k["same_instance"]]
    flagged_cross  = pairs_k[pairs_k["flagged"] & ~pairs_k["same_instance"]]
    pct_same_in_flagged  = len(flagged_same) / n_flagged
    pct_cross_in_flagged = len(flagged_cross) / n_flagged

    print(f"\nFlagged disconnected pairs: {n_flagged} / {len(pairs_k)} = {fp_overall:.4f}")
    print(f"  Same-instance FP rate : {fp_same:.4f}  (n={len(same):,} pairs)")
    print(f"  Cross-instance FP rate: {fp_cross:.4f}  (n={len(cross):,} pairs)")
    print(f"  Flagged pairs that are same-instance : {len(flagged_same)} ({pct_same_in_flagged:.1%})")
    print(f"  Flagged pairs that are cross-instance: {len(flagged_cross)} ({pct_cross_in_flagged:.1%})")

    # top-10 most similar flagged disconnected pairs
    top10 = (pairs_k[pairs_k["flagged"]]
             .sort_values("sim", ascending=False)
             .head(10)[["user_a", "user_b", "sim", "inst_a", "inst_b", "same_instance"]])
    print("\nTop-10 most similar flagged disconnected pairs:")
    print(top10.to_string(index=False))

    # save CSV
    pairs_k.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")

    # ── plot ──────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("False Positives Analysis: Real Disconnected Pairs Flagged",
                 fontsize=12, fontweight="bold")

    # Panel 1: similarity distribution (all disconnected, flagged highlighted)
    ax = axes[0]
    bins = np.linspace(0.15, 1.0, 50)
    ax.hist(pairs_k["sim"], bins=bins, color="#aec6e8", edgecolor="white",
            label="not flagged")
    ax.hist(pairs_k.loc[pairs_k["flagged"], "sim"], bins=bins,
            color="#2166ac", edgecolor="white", label="flagged (sim > 0.787)")
    ax.axvline(CLIP_THRESH, color="red", linewidth=1.2, linestyle="--",
               label=f"threshold = {CLIP_THRESH}")
    ax.set_xlabel("Cosine similarity (mean pool)")
    ax.set_ylabel("Count")
    ax.set_title("Similarity distribution\n(disconnected pairs)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: FP rate by instance type
    ax = axes[1]
    labels  = ["Same-instance", "Cross-instance", "Overall"]
    rates   = [fp_same, fp_cross, fp_overall]
    colors  = ["#d6604d", "#4393c3", "#888888"]
    bars = ax.bar(labels, rates, color=colors, edgecolor="white", width=0.5)
    ax.axhline(fp_overall, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{rate:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("False-positive rate")
    ax.set_title("FP rate by instance relationship\n(disconnected pairs above threshold)")
    ax.set_ylim(0, max(rates) + 0.08)
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: composition of flagged pairs (same vs cross instance)
    ax = axes[2]
    n_same_flag  = len(flagged_same)
    n_cross_flag = len(flagged_cross)
    ax.bar(["Flagged pairs\ncomposition"], [n_same_flag],
           color="#d6604d", edgecolor="white", label="Same-instance")
    ax.bar(["Flagged pairs\ncomposition"], [n_cross_flag],
           bottom=[n_same_flag], color="#4393c3", edgecolor="white",
           label="Cross-instance")
    ax.text(0, n_same_flag / 2, f"{pct_same_in_flagged:.0%}\nsame-inst",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.text(0, n_same_flag + n_cross_flag / 2,
            f"{pct_cross_in_flagged:.0%}\ncross-inst",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.set_ylabel("Number of flagged pairs")
    ax.set_title("Composition of flagged\ndisconnected pairs")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
