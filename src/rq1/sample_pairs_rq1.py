# Sample 10k connected and 10k disconnected user pairs for RQ1
# both sets only include users with at least MIN_IMAGES images
# connected = actual edges from the graph, disconnected = random non-edge pairs
# disconnected pool is restricted to users who appear in at least one valid edge
# (same population as connected pairs, just not paired with each other)

import csv
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

ROOT         = Path(__file__).parent.parent.parent
ACCOUNTS_CSV = ROOT / "dataset" / "anonymous_60k_user_info" / "accounts_info_60k_anonymized.csv"
EDGES_CSV    = ROOT / "dataset" / "anonymous_60k_user_info" / "edge_60k_anonymized.csv"
EMBEDDINGS   = Path("/Volumes/Samsung PSSD T7 Media/FediData/embeddings/embeddings_60k.npz")
OUT_CONNECTED    = ROOT / "dataset" / "rq1_pairs" / "rq1_connected_pairs.csv"
OUT_DISCONNECTED = ROOT / "dataset" / "rq1_pairs" / "rq1_disconnected_pairs.csv"

N_PAIRS    = 10_000
MIN_IMAGES = 5
SEED       = 42


def main():
    random.seed(SEED)

    # acct handle -> row index (row index matches the images folder number)
    print("Loading accounts...")
    acct_to_idx = {}
    with open(ACCOUNTS_CSV) as f:
        for idx, row in enumerate(csv.DictReader(f)):
            acct_to_idx[row["acct"]] = idx
    print(f"  {len(acct_to_idx):,} accounts loaded")

    # only keep users with enough images
    print(f"Counting images per user (min={MIN_IMAGES})...")
    data = np.load(EMBEDDINGS)
    img_counts = Counter(data["user_ids"].tolist())
    eligible_users = set(uid for uid, count in img_counts.items() if count >= MIN_IMAGES)
    print(f"  {len(eligible_users):,} users with ≥{MIN_IMAGES} images")

    # load edges, keep only pairs where both users are eligible
    # store as sorted tuples so (a,b) and (b,a) are treated as the same pair
    print("Loading edges...")
    valid_edges = set()
    total = 0
    with open(EDGES_CSV) as f:
        for row in csv.DictReader(f):
            total += 1
            src = acct_to_idx.get(row["source"])
            tgt = acct_to_idx.get(row["target"])
            if src is None or tgt is None:
                continue
            if src in eligible_users and tgt in eligible_users:
                valid_edges.add((min(src, tgt), max(src, tgt)))
    print(f"  {total:,} total edges → {len(valid_edges):,} valid (both users have ≥{MIN_IMAGES} images)")

    # users who appear in at least one valid edge — the same population as connected pairs
    users_in_edges = set()
    for a, b in valid_edges:
        users_in_edges.add(a)
        users_in_edges.add(b)
    print(f"  {len(users_in_edges):,} distinct users in valid edges")

    print(f"\nSampling {N_PAIRS:,} connected pairs...")
    connected = random.sample(sorted(valid_edges), N_PAIRS)

    print(f"Sampling {N_PAIRS:,} disconnected pairs (pool = users in valid edges only)...")
    users_list = sorted(users_in_edges)
    disconnected = []
    disconnected_set = set()

    while len(disconnected) < N_PAIRS:
        a, b = random.sample(users_list, 2)
        pair = (min(a, b), max(a, b))
        if pair not in valid_edges and pair not in disconnected_set:
            disconnected.append(pair)
            disconnected_set.add(pair)

    def write_pairs(path, pairs):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_a", "user_b"])
            writer.writerows(pairs)

    write_pairs(OUT_CONNECTED, connected)
    write_pairs(OUT_DISCONNECTED, disconnected)

    print(f"\nSaved:")
    print(f"  {OUT_CONNECTED}")
    print(f"  {OUT_DISCONNECTED}")
    print(f"\nEligible users:       {len(eligible_users):,}  (≥{MIN_IMAGES} images)")
    print(f"Users in valid edges: {len(users_in_edges):,}  (disconnected pool)")
    print(f"Valid edges:          {len(valid_edges):,}")


if __name__ == "__main__":
    main()
