# RQ3 detection: CLIP vs pHash across 36 parameter combinations.
#
# For each combo (n_accounts, n_images, follow_rate):
#   - Select first n_accounts synthetic accounts
#   - Mean-pool first n_images per account → one 512d vector per account
#   - Build merged FAISS HNSW index (all real users + synthetic accounts)
#   - For each synthetic account query top-K neighbours
#   - Flag synthetic-synthetic pairs where sim > CLIP_THRESH AND no follow edge
#   - Compute CLIP recall: flagged / total unconnected synthetic-synthetic pairs
#   - Compute pHash recall: account pairs with ≥1 image pair Hamming ≤ 10
#   - Record FP rate: fraction of real disconnected pairs above threshold (from RQ1)
#
# Output: src/rq3/rq3_results.csv

import csv
import itertools
from collections import defaultdict
from pathlib import Path

import faiss
import imagehash
import numpy as np
from PIL import Image

ROOT               = Path(__file__).parent.parent.parent
REAL_EMBEDDINGS    = Path("/Volumes/Samsung PSSD T7 Media/FediData/embeddings/user_vectors_60k.npz")
SYNTH_EMBEDDINGS   = Path("/Volumes/Samsung PSSD T7 Media/RQ3_Synthetic/embeddings/rq3_synthetic_embeddings.npz")
SYNTH_EDGES_CSV    = ROOT / "dataset" / "rq3_synthetic" / "synthetic_edges.csv"
SYNTH_PHASHES_CSV  = ROOT / "dataset" / "rq3_synthetic" / "synthetic_phashes.csv"
REAL_EDGES_CSV     = ROOT / "dataset" / "anonymous_60k_user_info" / "edge_60k_anonymized.csv"
ACCOUNTS_CSV       = ROOT / "dataset" / "anonymous_60k_user_info" / "accounts_info_60k_anonymized.csv"
RQ1_RESULTS        = ROOT / "src" / "rq1" / "rq1_results.npz"
OUT_CSV            = Path(__file__).parent / "rq3_results.csv"

SYNTHETIC_ID_START = 100_000
CLIP_THRESH        = 0.787   # RQ1 μ_conn (mean pooling)
PHASH_THRESH       = 10      # Hamming bits
K_NEIGHBOURS       = 500     # wide enough to catch synthetic peers despite real-user displacement
HNSW_M             = 16
HNSW_EF_SEARCH     = 32

ACCOUNTS_GRID = [20, 30, 50, 100]
IMAGES_GRID   = [5, 10, 20]
FOLLOW_RATES  = [0.0, 0.3, 0.6]


# ── data loading ──────────────────────────────────────────────────────────────

def load_real_vectors():
    print("Loading pre-computed real user vectors...")
    data = np.load(REAL_EMBEDDINGS)
    uids, embs = data["user_ids"], data["embeddings"]
    vectors = {int(uid): embs[i] for i, uid in enumerate(uids)}
    print(f"  {len(vectors):,} real users")
    return vectors


def load_synthetic_data():
    data = np.load(SYNTH_EMBEDDINGS)
    return data["embeddings"], data["user_ids"].astype(int), data["img_ids"].astype(int)


def synth_mean_pool(synth_embs, synth_uids, synth_img_ids, account_ids, n_images):
    vectors = {}
    for sid in account_ids:
        mask = (synth_uids == sid) & (synth_img_ids < n_images)
        imgs = synth_embs[mask]
        if len(imgs) == 0:
            continue
        v = imgs.mean(axis=0)
        v /= np.linalg.norm(v)
        vectors[sid] = v
    return vectors


def load_synth_edges():
    edges_by_rate = defaultdict(set)
    with open(SYNTH_EDGES_CSV) as f:
        for row in csv.DictReader(f):
            rate = float(row["follow_rate"])
            src, tgt = int(row["source"]), int(row["target"])
            pair = (min(src, tgt), max(src, tgt))
            edges_by_rate[rate].add(pair)
    return edges_by_rate


def load_phashes():
    hashes = defaultdict(dict)  # sid -> {img_idx: hash}
    with open(SYNTH_PHASHES_CSV) as f:
        for row in csv.DictReader(f):
            sid = int(row["synthetic_id"])
            img_path = Path(row["img_path"])
            # img_idx from filename: {sid}_{post_idx}_0.jpg
            img_idx = int(img_path.stem.split("_")[1])
            hashes[sid][img_idx] = imagehash.hex_to_hash(row["phash_hex"])
    return hashes


def fp_rate_from_rq1():
    data = np.load(RQ1_RESULTS)
    disc = data["disc_mean_pool"]
    return float((disc > CLIP_THRESH).mean())


# ── FAISS ─────────────────────────────────────────────────────────────────────

def build_index(vectors_dict):
    id_list = sorted(vectors_dict.keys())
    matrix  = np.stack([vectors_dict[i] for i in id_list]).astype(np.float32)
    index   = faiss.IndexHNSWFlat(512, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(matrix)
    return index, id_list


# ── detection ─────────────────────────────────────────────────────────────────

def clip_recall(synth_ids, synth_vectors, all_vectors, edge_set):
    index, id_list = build_index(all_vectors)
    id_arr = np.array(id_list)

    flagged = set()
    synth_set = set(synth_ids)

    query_matrix = np.stack([synth_vectors[sid] for sid in synth_ids]).astype(np.float32)
    sims, indices = index.search(query_matrix, K_NEIGHBOURS)

    for q_idx, sid in enumerate(synth_ids):
        for rank, (nb_idx, sim) in enumerate(zip(indices[q_idx], sims[q_idx])):
            if nb_idx < 0:
                continue
            nb_id = int(id_arr[nb_idx])
            if nb_id == sid:
                continue
            if nb_id not in synth_set:
                continue
            if sim <= CLIP_THRESH:
                continue
            pair = (min(sid, nb_id), max(sid, nb_id))
            if pair in edge_set:
                continue
            flagged.add(pair)

    # total unconnected synthetic-synthetic pairs
    all_synth_pairs = set(
        (min(a, b), max(a, b))
        for a, b in itertools.combinations(synth_ids, 2)
    )
    unconnected = all_synth_pairs - edge_set
    recall = len(flagged) / len(unconnected) if unconnected else 0.0
    return recall, len(flagged), len(unconnected)


def phash_recall(synth_ids, n_images, phashes):
    flagged_pairs = 0
    total_pairs   = 0
    ids = list(synth_ids)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            total_pairs += 1
            a_hashes = [phashes[a][k] for k in range(n_images) if k in phashes[a]]
            b_hashes = [phashes[b][k] for k in range(n_images) if k in phashes[b]]
            caught = any(
                (ha - hb) <= PHASH_THRESH
                for ha in a_hashes
                for hb in b_hashes
            )
            if caught:
                flagged_pairs += 1
    return flagged_pairs / total_pairs if total_pairs else 0.0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    real_vectors                        = load_real_vectors()
    synth_embs, synth_uids, synth_imgs  = load_synthetic_data()
    edges_by_rate                       = load_synth_edges()
    phashes                             = load_phashes()
    fp_rate                             = fp_rate_from_rq1()
    all_synth_ids = sorted(set(synth_uids.tolist()))

    print(f"\nFP rate from RQ1 disconnected pairs: {fp_rate:.4f}")
    print(f"Running {len(ACCOUNTS_GRID)*len(IMAGES_GRID)*len(FOLLOW_RATES)} parameter combinations...\n")

    rows = []
    combo_n = 0
    for n_accounts, n_images, follow_rate in itertools.product(ACCOUNTS_GRID, IMAGES_GRID, FOLLOW_RATES):
        combo_n += 1
        synth_ids    = all_synth_ids[:n_accounts]
        synth_vectors = synth_mean_pool(synth_embs, synth_uids, synth_imgs, synth_ids, n_images)
        edge_set     = edges_by_rate[follow_rate]

        all_vectors  = {**real_vectors, **synth_vectors}

        cr, n_flagged, n_unconnected = clip_recall(synth_ids, synth_vectors, all_vectors, edge_set)
        pr = phash_recall(synth_ids, n_images, phashes)

        print(f"[{combo_n:2d}/36] n_accounts={n_accounts:3d}  n_images={n_images:2d}  follow_rate={follow_rate:.1f}"
              f"  CLIP_recall={cr:.3f}  pHash_recall={pr:.3f}  flagged={n_flagged}/{n_unconnected}")

        rows.append({
            "n_accounts":    n_accounts,
            "n_images":      n_images,
            "follow_rate":   follow_rate,
            "clip_recall":   round(cr, 4),
            "phash_recall":  round(pr, 4),
            "fp_rate":       round(fp_rate, 4),
            "n_flagged":     n_flagged,
            "n_unconnected": n_unconnected,
        })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {OUT_CSV}")


if __name__ == "__main__":
    main()
