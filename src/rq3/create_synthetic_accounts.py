# Create synthetic IO accounts and edges from the deduped staging pool.
#
# Phase 1 — accounts: sample images from staging, distribute to per-account dirs.
# Phase 2 — edges: intra-group follow edges at given rate + inter-group follows
#            to real FediData accounts (makes synthetic accounts look embedded).
#
# Run once for the maximum combo (100 accounts × 20 images). Smaller combos
# in rq3_detection.py just slice this set — no re-running needed.
#
# Output:
#   dataset/rq3_synthetic/synthetic_accounts.csv   (synthetic_id, n_images)
#   dataset/rq3_synthetic/synthetic_edges.csv      (source, target, follow_rate)
#   /Volumes/.../RQ3_Synthetic/images/{synthetic_id}/   (images)

import csv
import random
import shutil
from pathlib import Path

import numpy as np

ROOT        = Path(__file__).parent.parent.parent
STAGING_DIR = Path("/Volumes/Samsung PSSD T7 Media/RQ3_Synthetic/images/staging")
OUT_IMAGES  = Path("/Volumes/Samsung PSSD T7 Media/RQ3_Synthetic/images")
OUT_DATA    = ROOT / "dataset" / "rq3_synthetic"
ACCOUNTS_CSV = ROOT / "dataset" / "anonymous_60k_user_info" / "accounts_info_60k_anonymized.csv"

SYNTHETIC_ID_START = 100_000
MAX_ACCOUNTS       = 100
MAX_IMAGES         = 20
FOLLOW_RATES       = [0.0, 0.3, 0.6]
INTER_GROUP_N      = 15   # real FediData accounts each synthetic account follows
SEED               = 42


def load_real_user_ids():
    ids = []
    with open(ACCOUNTS_CSV) as f:
        for idx, _ in enumerate(csv.DictReader(f)):
            ids.append(idx)
    return ids


def main():
    random.seed(SEED)
    rng = np.random.default_rng(SEED)

    OUT_DATA.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: assign images to accounts ───────────────────────────────────

    staging_images = sorted(STAGING_DIR.glob("*.jpg")) + sorted(STAGING_DIR.glob("*.jpeg")) + sorted(STAGING_DIR.glob("*.png"))
    needed = MAX_ACCOUNTS * MAX_IMAGES
    print(f"Staging pool: {len(staging_images):,} images  |  Need: {needed:,}")
    if len(staging_images) < needed:
        raise RuntimeError(f"Not enough images in staging: {len(staging_images)} < {needed}")

    selected = random.sample(staging_images, needed)

    synthetic_ids = list(range(SYNTHETIC_ID_START, SYNTHETIC_ID_START + MAX_ACCOUNTS))
    account_rows = []

    print(f"Distributing images to {MAX_ACCOUNTS} synthetic accounts...")
    for i, sid in enumerate(synthetic_ids):
        acct_dir = OUT_IMAGES / str(sid)
        acct_dir.mkdir(parents=True, exist_ok=True)
        imgs = selected[i * MAX_IMAGES : (i + 1) * MAX_IMAGES]
        for post_idx, src in enumerate(imgs):
            dst = acct_dir / f"{sid}_{post_idx}_0{src.suffix.lower()}"
            shutil.copy2(src, dst)
        account_rows.append({"synthetic_id": sid, "n_images": MAX_IMAGES})

    accounts_path = OUT_DATA / "synthetic_accounts.csv"
    with open(accounts_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["synthetic_id", "n_images"])
        writer.writeheader()
        writer.writerows(account_rows)
    print(f"  Saved {accounts_path}")

    # ── Phase 2: create edges ─────────────────────────────────────────────────

    real_user_ids = load_real_user_ids()
    edge_rows = []

    for follow_rate in FOLLOW_RATES:
        print(f"Creating edges at follow_rate={follow_rate}...")
        # intra-group: directed pairs, each kept with probability = follow_rate
        for src in synthetic_ids:
            for tgt in synthetic_ids:
                if src == tgt:
                    continue
                if follow_rate > 0 and rng.random() < follow_rate:
                    edge_rows.append({"source": src, "target": tgt, "follow_rate": follow_rate})

        # inter-group: each synthetic account follows INTER_GROUP_N real accounts
        for src in synthetic_ids:
            targets = rng.choice(real_user_ids, size=INTER_GROUP_N, replace=False).tolist()
            for tgt in targets:
                edge_rows.append({"source": src, "target": tgt, "follow_rate": follow_rate})

    edges_path = OUT_DATA / "synthetic_edges.csv"
    with open(edges_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "follow_rate"])
        writer.writeheader()
        writer.writerows(edge_rows)

    total_intra = sum(1 for r in edge_rows if int(str(r["source"]).split(".")[0]) >= SYNTHETIC_ID_START
                      and int(str(r["target"]).split(".")[0]) >= SYNTHETIC_ID_START)
    print(f"  Saved {edges_path}  ({len(edge_rows):,} total edges)")
    print(f"Done.")


if __name__ == "__main__":
    main()
