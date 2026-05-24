# RQ2 - FAISS Benchmarking
# Sweeps IVF (nlist x nprobe) and HNSW (M x efSearch) on FediData CLIP user embeddings
# Ground truth: IndexFlatIP (exact cosine search after L2 normalisation)
# Outputs: rq2_results.npz with recall@K and QPS for every config

import time
from pathlib import Path

import faiss
import numpy as np

EMBEDDINGS = Path("/Volumes/Samsung PSSD T7 Media/FediData/embeddings/embeddings_60k.npz")
OUT_NPZ    = Path(__file__).parent / "rq2_results.npz"

K             = 10
N_QUERIES     = 1_000
SEED          = 42
N_TIMING_RUNS = 5   # repeat each search, keep best time to reduce jitter

NLIST_VALUES  = [64, 128, 256, 512, 1024]
NPROBE_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]

M_VALUES         = [8, 16, 32, 64]
EF_CONSTRUCTION  = 200
EF_SEARCH_VALUES = [16, 32, 64, 128, 256, 512]


# ── data helpers ──────────────────────────────────────────────────────────────

def build_user_vectors(embeddings: np.ndarray, user_ids: np.ndarray):
    unique_users = np.unique(user_ids)
    vecs = []
    for uid in unique_users:
        mean_vec = embeddings[user_ids == uid].mean(axis=0)
        vecs.append(mean_vec)
    matrix = np.stack(vecs).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.maximum(norms, 1e-8)
    return matrix, unique_users


def exclude_self(ids: np.ndarray, query_indices: np.ndarray, k: int) -> np.ndarray:
    """Remove the query's own index from each result row, return top-k remainder."""
    out = np.full((len(ids), k), -1, dtype=np.int64)
    for i, (row, qidx) in enumerate(zip(ids, query_indices)):
        filtered = row[row != qidx]
        n = min(k, len(filtered))
        out[i, :n] = filtered[:n]
    return out


def recall_at_k(approx_ids: np.ndarray, gt_ids: np.ndarray) -> float:
    recalls = []
    for approx, gt in zip(approx_ids, gt_ids):
        gt_set = set(gt[gt >= 0].tolist())
        if not gt_set:
            recalls.append(1.0)
            continue
        found = sum(1 for idx in approx if idx >= 0 and idx in gt_set)
        recalls.append(found / len(gt_set))
    return float(np.mean(recalls))


# ── timing ────────────────────────────────────────────────────────────────────

def timed_search(index, queries: np.ndarray, k: int, n_runs: int = N_TIMING_RUNS):
    best_s = float("inf")
    ids_out = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _, ids = index.search(queries, k + 1)   # +1 so we can strip self
        elapsed = time.perf_counter() - t0
        if elapsed < best_s:
            best_s = elapsed
            ids_out = ids
    return ids_out, len(queries) / best_s        # (ids, QPS)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not EMBEDDINGS.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {EMBEDDINGS}\n"
            "Make sure the Samsung T7 drive is mounted and embed_clip_60k.py has finished."
        )

    print("Loading embeddings...")
    data       = np.load(EMBEDDINGS)
    raw_embs   = data["embeddings"]
    user_ids   = data["user_ids"]
    print(f"  {len(raw_embs):,} image embeddings  |  {len(np.unique(user_ids)):,} users")

    print("Mean-pooling per user and re-normalising...")
    user_matrix, unique_users = build_user_vectors(raw_embs, user_ids)
    n_users, d = user_matrix.shape
    print(f"  User matrix: {user_matrix.shape}  (d={d})")

    rng         = np.random.default_rng(SEED)
    query_idx   = rng.choice(n_users, size=min(N_QUERIES, n_users), replace=False)
    queries     = user_matrix[query_idx]
    n_q         = len(queries)
    print(f"  Query set: {n_q:,} users (seed={SEED})")

    # ── ground truth ──────────────────────────────────────────────────────────
    print("\nBuilding Flat index and computing ground truth...")
    flat = faiss.IndexFlatIP(d)
    flat.add(user_matrix)

    flat_ids, flat_qps = timed_search(flat, queries, K)
    gt_ids = flat_ids[:, 1:K + 1]          # position 0 is always self for Flat
    print(f"  Flat:  recall@{K} = 1.000  |  {flat_qps:,.0f} QPS")

    ivf_rows  = []    # (nlist, nprobe, recall, qps)
    hnsw_rows = []    # (M, efSearch, recall, qps)

    # ── IVF sweep ─────────────────────────────────────────────────────────────
    print("\nBenchmarking IVF...")
    for nlist in NLIST_VALUES:
        print(f"  nlist={nlist}")
        quantizer = faiss.IndexFlatIP(d)
        ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        ivf.train(user_matrix)
        ivf.add(user_matrix)

        for nprobe in NPROBE_VALUES:
            if nprobe > nlist:
                break
            ivf.nprobe = nprobe
            ids, qps   = timed_search(ivf, queries, K)
            approx     = exclude_self(ids, query_idx, K)
            rec        = recall_at_k(approx, gt_ids)
            ivf_rows.append((nlist, nprobe, rec, qps))
            print(f"    nprobe={nprobe:3d}  recall@{K}={rec:.3f}  {qps:,.0f} QPS")

    # ── HNSW sweep ────────────────────────────────────────────────────────────
    print("\nBenchmarking HNSW...")
    for M in M_VALUES:
        print(f"  M={M}  (efConstruction={EF_CONSTRUCTION})")
        hnsw = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = EF_CONSTRUCTION
        hnsw.add(user_matrix)

        for ef in EF_SEARCH_VALUES:
            hnsw.hnsw.efSearch = ef
            ids, qps           = timed_search(hnsw, queries, K)
            approx             = exclude_self(ids, query_idx, K)
            rec                = recall_at_k(approx, gt_ids)
            hnsw_rows.append((M, ef, rec, qps))
            print(f"    efSearch={ef:3d}  recall@{K}={rec:.3f}  {qps:,.0f} QPS")

    # ── save ──────────────────────────────────────────────────────────────────
    ivf_arr  = np.array(ivf_rows,  dtype=np.float32)   # (N, 4)
    hnsw_arr = np.array(hnsw_rows, dtype=np.float32)   # (N, 4)

    np.savez(
        OUT_NPZ,
        ivf       = ivf_arr,
        hnsw      = hnsw_arr,
        flat_qps  = np.float32(flat_qps),
        K         = np.int32(K),
        n_users   = np.int32(n_users),
        n_queries = np.int32(n_q),
    )
    print(f"\nResults saved → {OUT_NPZ}")
    print(f"  IVF  rows: {len(ivf_rows)}")
    print(f"  HNSW rows: {len(hnsw_rows)}")


if __name__ == "__main__":
    main()
