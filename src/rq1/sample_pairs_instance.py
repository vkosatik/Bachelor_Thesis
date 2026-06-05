# Stratified pair sampling for instance-specific analysis.
# Produces 4 files of N_PAIRS each:
#   rq1_same_instance_connected.csv
#   rq1_same_instance_disconnected.csv
#   rq1_cross_instance_connected.csv
#   rq1_cross_instance_disconnected.csv
#
# All constraints from the original sampling apply (MIN_IMAGES=5, users must
# appear in at least one valid edge). Pairs in each file are disjoint from
# the original rq1_connected_pairs.csv / rq1_disconnected_pairs.csv.

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT         = Path(__file__).parent.parent.parent
ACCOUNTS_CSV = ROOT / "dataset" / "anonymous_60k_user_info" / "accounts_info_60k_anonymized.csv"
EDGES_CSV    = ROOT / "dataset" / "anonymous_60k_user_info" / "edge_60k_anonymized.csv"
EMBEDDINGS   = Path("/Volumes/Samsung PSSD T7 Media/FediData/embeddings/embeddings_60k.npz")
OUT_DIR      = ROOT / "dataset" / "rq1_pairs"

ORIG_CONN = OUT_DIR / "rq1_connected_pairs.csv"
ORIG_DISC = OUT_DIR / "rq1_disconnected_pairs.csv"

N_PAIRS    = 10_000
MIN_IMAGES = 5
SEED       = 42


def load_accounts():
    idx_to_instance = {}
    acct_to_idx = {}
    with open(ACCOUNTS_CSV) as f:
        for idx, row in enumerate(csv.DictReader(f)):
            acct = row["acct"]
            instance = acct.split("@")[1] if "@" in acct else "unknown"
            idx_to_instance[idx] = instance
            acct_to_idx[acct] = idx
    return idx_to_instance, acct_to_idx


def load_original_pairs(path):
    pairs = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            a, b = int(row["user_a"]), int(row["user_b"])
            pairs.add((min(a, b), max(a, b)))
    return pairs


def write_pairs(path, pairs):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_a", "user_b"])
        writer.writerows(pairs)
    print(f"  Saved {len(pairs):,} pairs → {path.name}")


def main():
    random.seed(SEED)

    print("Loading accounts...")
    idx_to_instance, acct_to_idx = load_accounts()

    print("Loading embeddings (eligible users)...")
    data = np.load(EMBEDDINGS)
    from collections import Counter
    img_counts = Counter(data["user_ids"].tolist())
    eligible = set(uid for uid, n in img_counts.items() if n >= MIN_IMAGES)
    print(f"  {len(eligible):,} users with ≥{MIN_IMAGES} images")

    print("Loading original pairs (to exclude from new samples)...")
    orig_conn = load_original_pairs(ORIG_CONN)
    orig_disc = load_original_pairs(ORIG_DISC)
    exclude = orig_conn | orig_disc
    print(f"  Excluding {len(exclude):,} already-sampled pairs")

    print("Loading edges...")
    valid_edges_same  = set()
    valid_edges_cross = set()
    total = 0
    with open(EDGES_CSV) as f:
        for row in csv.DictReader(f):
            total += 1
            src = acct_to_idx.get(row["source"])
            tgt = acct_to_idx.get(row["target"])
            if src is None or tgt is None:
                continue
            if src not in eligible or tgt not in eligible:
                continue
            pair = (min(src, tgt), max(src, tgt))
            if idx_to_instance[src] == idx_to_instance[tgt]:
                valid_edges_same.add(pair)
            else:
                valid_edges_cross.add(pair)

    valid_edges_same  -= exclude
    valid_edges_cross -= exclude

    print(f"  Same-instance edges available:  {len(valid_edges_same):,}")
    print(f"  Cross-instance edges available: {len(valid_edges_cross):,}")

    # users in valid edges, split by whether they have same-instance partners
    users_in_same  = set(u for pair in valid_edges_same  for u in pair)
    users_in_cross = set(u for pair in valid_edges_cross for u in pair)
    all_edge_users = users_in_same | users_in_cross

    # group eligible edge-users by instance (for same-instance disconnected sampling)
    instance_to_users = defaultdict(list)
    for uid in all_edge_users:
        instance_to_users[idx_to_instance[uid]].append(uid)

    # ── connected pairs ──────────────────────────────────────────────────────

    n_same = min(N_PAIRS, len(valid_edges_same))
    n_cross = min(N_PAIRS, len(valid_edges_cross))

    if n_same < N_PAIRS:
        print(f"  WARNING: only {n_same:,} same-instance connected pairs available (need {N_PAIRS:,})")
    if n_cross < N_PAIRS:
        print(f"  WARNING: only {n_cross:,} cross-instance connected pairs available (need {N_PAIRS:,})")

    print(f"\nSampling {n_same:,} same-instance connected pairs...")
    same_conn = random.sample(sorted(valid_edges_same), n_same)

    print(f"Sampling {n_cross:,} cross-instance connected pairs...")
    cross_conn = random.sample(sorted(valid_edges_cross), n_cross)

    all_valid_edges = valid_edges_same | valid_edges_cross

    # ── same-instance disconnected ───────────────────────────────────────────
    # pick two users from the same instance who are NOT connected and NOT in exclude

    print(f"Sampling {N_PAIRS:,} same-instance disconnected pairs...")
    same_disc = []
    same_disc_set = set()

    # build pool of instances with ≥2 eligible edge-users
    large_instances = [inst for inst, users in instance_to_users.items() if len(users) >= 2]
    large_instances_users = {inst: instance_to_users[inst] for inst in large_instances}

    attempts = 0
    max_attempts = N_PAIRS * 200
    while len(same_disc) < N_PAIRS and attempts < max_attempts:
        attempts += 1
        inst = random.choice(large_instances)
        users = large_instances_users[inst]
        a, b = random.sample(users, 2)
        pair = (min(a, b), max(a, b))
        if pair in all_valid_edges or pair in exclude or pair in same_disc_set:
            continue
        same_disc.append(pair)
        same_disc_set.add(pair)

    if len(same_disc) < N_PAIRS:
        print(f"  WARNING: only sampled {len(same_disc):,} same-instance disconnected pairs (need {N_PAIRS:,})")
    else:
        print(f"  OK: {len(same_disc):,} pairs")

    # ── cross-instance disconnected ──────────────────────────────────────────

    print(f"Sampling {N_PAIRS:,} cross-instance disconnected pairs...")
    all_edge_users_list = sorted(all_edge_users)
    cross_disc = []
    cross_disc_set = set()

    attempts = 0
    max_attempts = N_PAIRS * 20
    while len(cross_disc) < N_PAIRS and attempts < max_attempts:
        attempts += 1
        a, b = random.sample(all_edge_users_list, 2)
        if idx_to_instance[a] == idx_to_instance[b]:
            continue
        pair = (min(a, b), max(a, b))
        if pair in all_valid_edges or pair in exclude or pair in cross_disc_set:
            continue
        cross_disc.append(pair)
        cross_disc_set.add(pair)

    if len(cross_disc) < N_PAIRS:
        print(f"  WARNING: only sampled {len(cross_disc):,} cross-instance disconnected pairs")
    else:
        print(f"  OK: {len(cross_disc):,} pairs")

    # ── save ─────────────────────────────────────────────────────────────────

    print("\nSaving...")
    write_pairs(OUT_DIR / "rq1_same_instance_connected.csv",    same_conn)
    write_pairs(OUT_DIR / "rq1_same_instance_disconnected.csv", same_disc)
    write_pairs(OUT_DIR / "rq1_cross_instance_connected.csv",   cross_conn)
    write_pairs(OUT_DIR / "rq1_cross_instance_disconnected.csv", cross_disc)


if __name__ == "__main__":
    main()
