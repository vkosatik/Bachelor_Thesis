# Robustness check: repeat the RQ1 analysis using only pairs where
# both users appear exactly once in the connected set.
# High-degree nodes dominate the full connected sample (one user appears 45×);
# this sub-analysis confirms the result does not depend on them.
# Reads rq1_results.npz produced by rq1_homophily.py — no re-computation needed.

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

ROOT     = Path(__file__).parent.parent.parent
CONN_CSV = ROOT / "dataset" / "rq1_pairs" / "rq1_connected_pairs.csv"
DISC_CSV = ROOT / "dataset" / "rq1_pairs" / "rq1_disconnected_pairs.csv"
IN_NPZ   = Path(__file__).parent / "rq1_results.npz"


def load_pairs(path):
    pairs = []
    with open(path) as f:
        for row in csv.DictReader(f):
            pairs.append((int(row["user_a"]), int(row["user_b"])))
    return pairs


def main():
    conn_pairs = load_pairs(CONN_CSV)
    disc_pairs = load_pairs(DISC_CSV)

    # find users who appear exactly once in the connected set
    conn_user_counts = Counter()
    for a, b in conn_pairs:
        conn_user_counts[a] += 1
        conn_user_counts[b] += 1
    one_timers = {u for u, c in conn_user_counts.items() if c == 1}

    # keep only connected pairs where both users are one-timers
    keep_conn = [i for i, (a, b) in enumerate(conn_pairs) if a in one_timers and b in one_timers]

    # use the same number of disconnected pairs for a fair comparison
    n = len(keep_conn)
    rng = np.random.default_rng(42)
    keep_disc = rng.choice(len(disc_pairs), size=n, replace=False).tolist()

    print(f"Full connected pairs:    {len(conn_pairs):,}")
    print(f"One-timer pairs kept:    {n:,}  ({100*n/len(conn_pairs):.1f}% of original)")
    print(f"Disconnected pairs used: {n:,}\n")

    data = np.load(IN_NPZ)

    methods = [
        ("Mean pooling",   data["conn_mean_pool"],     data["disc_mean_pool"]),
        ("Pairwise mean",  data["conn_pairwise_mean"], data["disc_pairwise_mean"]),
        ("Top-3 pairwise", data["conn_top_k"],         data["disc_top_k"]),
    ]

    print("── Robustness check (one-timer pairs only) ──────────────────────────────────")
    print(f"{'Method':<20} {'Conn μ':>8} {'Disc μ':>8} {'Δ':>8} {'p':>12} {'r':>8}")
    print("-" * 70)
    for name, conn_full, disc_full in methods:
        conn = conn_full[keep_conn]
        disc = disc_full[keep_disc]
        U, p = stats.mannwhitneyu(conn, disc, alternative="greater")
        r = (2 * U / (len(conn) * len(disc))) - 1
        p_str = "< 0.001" if p < 0.001 else f"{p:.4f}"
        print(f"{name:<20} {conn.mean():>8.4f} {disc.mean():>8.4f} "
              f"{conn.mean()-disc.mean():>+8.4f} {p_str:>12} {r:>8.4f}")

    print("\n── Full-sample results (for comparison) ─────────────────────────────────────")
    print(f"{'Method':<20} {'Conn μ':>8} {'Disc μ':>8} {'Δ':>8} {'p':>12} {'r':>8}")
    print("-" * 70)
    for name, conn_full, disc_full in methods:
        U, p = stats.mannwhitneyu(conn_full, disc_full, alternative="greater")
        r = (2 * U / (len(conn_full) * len(disc_full))) - 1
        p_str = "< 0.001" if p < 0.001 else f"{p:.4f}"
        print(f"{name:<20} {conn_full.mean():>8.4f} {disc_full.mean():>8.4f} "
              f"{conn_full.mean()-disc_full.mean():>+8.4f} {p_str:>12} {r:>8.4f}")


if __name__ == "__main__":
    main()
