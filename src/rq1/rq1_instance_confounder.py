# RQ1 confounder analysis: does visual homophily persist when controlling for Mastodon instance?
#
# Mastodon users from the same instance share a community/visual culture independent of
# social ties. If the RQ1 signal is driven by instance membership rather than individual
# follow relationships, it should vanish or weaken in cross-instance connected pairs.
#
# Uses stratified pairs from sample_pairs_instance.py: 10k same-instance connected,
# 10k same-instance disconnected, 10k cross-instance connected, 10k cross-instance
# disconnected — equal sample sizes for all four groups.

import csv
from pathlib import Path

import numpy as np
from scipy import stats

ROOT       = Path(__file__).parent.parent.parent
PAIRS_DIR  = ROOT / "dataset" / "rq1_pairs"
EMBEDDINGS = Path("/Volumes/Samsung PSSD T7 Media/FediData/embeddings/embeddings_60k.npz")
OUT_NPZ    = Path(__file__).parent / "rq1_instance_results.npz"


def load_pairs(path):
    pairs = []
    with open(path) as f:
        for row in csv.DictReader(f):
            pairs.append((int(row["user_a"]), int(row["user_b"])))
    return pairs


def build_user_index(user_ids, embeddings):
    index = {}
    for uid in np.unique(user_ids):
        mask = user_ids == uid
        index[int(uid)] = embeddings[mask]
    return index


def sim_mean_pool(ea, eb):
    va = ea.mean(axis=0); va /= np.linalg.norm(va)
    vb = eb.mean(axis=0); vb /= np.linalg.norm(vb)
    return float(va @ vb)


def compute_sims(pairs, user_index):
    sims = []
    missing = 0
    for a, b in pairs:
        if a not in user_index or b not in user_index:
            missing += 1
            continue
        sims.append(sim_mean_pool(user_index[a], user_index[b]))
    if missing:
        print(f"  Skipped {missing} pairs (user not in embeddings)")
    return np.array(sims)


def mw(conn, disc, label):
    u, p = stats.mannwhitneyu(conn, disc, alternative="greater")
    print(f"  {label}")
    print(f"    n_conn={len(conn):,}  μ={conn.mean():.4f}  |  n_disc={len(disc):,}  μ={disc.mean():.4f}  |  Δ={conn.mean()-disc.mean():+.4f}  U={u:.0f}  p={p:.2e}")


def main():
    print("Loading embeddings...")
    data = np.load(EMBEDDINGS)
    user_index = build_user_index(data["user_ids"], data["embeddings"])
    print(f"  {len(user_index):,} users indexed")

    print("Loading stratified pairs...")
    same_conn  = load_pairs(PAIRS_DIR / "rq1_same_instance_connected.csv")
    same_disc  = load_pairs(PAIRS_DIR / "rq1_same_instance_disconnected.csv")
    cross_conn = load_pairs(PAIRS_DIR / "rq1_cross_instance_connected.csv")
    cross_disc = load_pairs(PAIRS_DIR / "rq1_cross_instance_disconnected.csv")
    print(f"  same-instance:  {len(same_conn):,} connected  {len(same_disc):,} disconnected")
    print(f"  cross-instance: {len(cross_conn):,} connected  {len(cross_disc):,} disconnected")

    print("Computing similarities...")
    same_conn_sims  = compute_sims(same_conn,  user_index)
    same_disc_sims  = compute_sims(same_disc,  user_index)
    cross_conn_sims = compute_sims(cross_conn, user_index)
    cross_disc_sims = compute_sims(cross_disc, user_index)

    print("\n=== Mann-Whitney U (connected > disconnected) — mean pooling ===")
    mw(same_conn_sims,  same_disc_sims,  "same-instance  connected vs disconnected")
    mw(cross_conn_sims, cross_disc_sims, "cross-instance connected vs disconnected")

    print("\n=== Within connected: same vs cross instance ===")
    u, p = stats.mannwhitneyu(same_conn_sims, cross_conn_sims, alternative="greater")
    print(f"  same-instance connected μ={same_conn_sims.mean():.4f}  cross-instance μ={cross_conn_sims.mean():.4f}")
    print(f"  U={u:.0f}  p={p:.2e}  (H: same > cross)")

    np.savez(
        OUT_NPZ,
        same_conn=same_conn_sims,  same_disc=same_disc_sims,
        cross_conn=cross_conn_sims, cross_disc=cross_disc_sims,
    )
    print(f"\nResults saved to {OUT_NPZ}")


if __name__ == "__main__":
    main()
