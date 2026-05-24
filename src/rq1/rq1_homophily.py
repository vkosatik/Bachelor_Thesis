# RQ1 - Visual Homophily
# computes cosine similarity for connected vs disconnected user pairs
# using three methods: mean pooling, pairwise mean, top-3 pairwise

import csv
from pathlib import Path

import numpy as np
from scipy import stats

ROOT       = Path(__file__).parent.parent.parent
CONN_CSV   = ROOT / "dataset" / "rq1_pairs" / "rq1_connected_pairs.csv"
DISC_CSV   = ROOT / "dataset" / "rq1_pairs" / "rq1_disconnected_pairs.csv"
EMBEDDINGS = Path("/Volumes/Samsung PSSD T7 Media/FediData/embeddings/embeddings_60k.npz")
OUT_NPZ    = Path(__file__).parent / "rq1_results.npz"

TOP_K = 3


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


def sim_pairwise_mean(ea, eb):
    return float((ea @ eb.T).mean())


def sim_top_k(ea, eb, k=TOP_K):
    sims = (ea @ eb.T).ravel()
    k = min(k, len(sims))
    return float(np.partition(sims, -k)[-k:].mean())


def compute_similarities(pairs, user_index):
    mean_pool, pw_mean, top_k = [], [], []
    missing = 0
    for a, b in pairs:
        if a not in user_index or b not in user_index:
            missing += 1
            continue
        ea, eb = user_index[a], user_index[b]
        mean_pool.append(sim_mean_pool(ea, eb))
        pw_mean.append(sim_pairwise_mean(ea, eb))
        top_k.append(sim_top_k(ea, eb))
    if missing:
        print(f"  Skipped {missing} pairs (user not in embeddings)")
    return np.array(mean_pool), np.array(pw_mean), np.array(top_k)


def main():
    print("Loading pairs...")
    conn_pairs = load_pairs(CONN_CSV)
    disc_pairs = load_pairs(DISC_CSV)
    print(f"  Connected: {len(conn_pairs):,}  |  Disconnected: {len(disc_pairs):,}")

    print("Loading embeddings...")
    data = np.load(EMBEDDINGS)
    user_index = build_user_index(data["user_ids"], data["embeddings"])
    print(f"  {len(user_index):,} users indexed")

    print("Computing similarities — connected pairs...")
    conn_mp, conn_pw, conn_tk = compute_similarities(conn_pairs, user_index)

    print("Computing similarities — disconnected pairs...")
    disc_mp, disc_pw, disc_tk = compute_similarities(disc_pairs, user_index)

    np.savez(
        OUT_NPZ,
        conn_mean_pool=conn_mp, disc_mean_pool=disc_mp,
        conn_pairwise_mean=conn_pw, disc_pairwise_mean=disc_pw,
        conn_top_k=conn_tk, disc_top_k=disc_tk,
    )
    print(f"Results saved to {OUT_NPZ}")


if __name__ == "__main__":
    main()
