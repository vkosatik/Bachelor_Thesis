# RQ3 Implementation Plan
**Question:** Can CLIP-based visual similarity detect semantically coordinated accounts
that pixel-level (pHash) methods miss?

---

## Why RQ3 is the Novel Contribution

Existing CIB/IO detection signals in the literature:
- Text similarity (URLs, hashtags, captions)
- Behavioural timing (synchronised posting patterns)
- Network structure (follow/retweet graph)
- Pixel-level image reuse (pHash — Pacheco et al. 2021, Ng et al. 2022)

**What does not exist:** semantic account-level visual similarity as a coordination signal.
Ng et al. and Pacheco et al. operate at the image-to-image level — they detect copy/reuse.
This thesis introduces mean-pooled CLIP account vectors as a signal that captures
*thematic* visual coordination: accounts that post visually similar content without
reusing the same images. No prior work has applied this to Fediverse/Mastodon.

The concept is the contribution. The synthetic injection is a feasibility demonstration —
a workaround for the absence of labeled IO datasets on decentralized platforms, which
do not yet exist. This absence is itself a limitation of the field, not of the thesis,
and should be framed explicitly in the thesis limitations section.

**Thesis framing to use:**
> "We identify semantic visual similarity as a novel CIB detection signal and provide
> a controlled feasibility demonstration. Real-world validation is limited by the absence
> of labeled information operation datasets for decentralized platforms — we flag this
> as a priority for future work."

---

## Narrative Chain (connects all three RQs)

- RQ1 established μ_conn = 0.787 (mean pooling) as the baseline similarity of connected pairs.
  Visual similarity is a meaningful signal — connected users are more similar than disconnected.
- RQ2 identified HNSW M=16 efSearch=32 as the optimal FAISS config (recall=0.968, 1.8× speedup).
  The pipeline can compute this signal at scale efficiently.
- RQ3 inverts RQ1's finding: unconnected accounts with similarity *above* μ_conn are anomalous —
  more similar than typical friends, yet no social relationship. Flag them as potentially
  coordinated. Use RQ2's FAISS config to find them. Compare against pHash to show
  CLIP catches semantic coordination that pixel-level methods miss entirely.

**Detection logic:** a pair of accounts is flagged if:
1. Their cosine similarity (CLIP, mean-pooled) > 0.787 (RQ1 μ_conn), AND
2. They have no follow edge between them in the graph

Both CLIP and pHash use the same candidate pool (unconnected pairs only) for a fair comparison.
CLIP uses the 0.787 threshold from RQ1. pHash uses Hamming ≤ 10 (standard in literature).
These are different metrics at different granularity levels — the comparison is not
"which is better overall" but "which catches semantic coordination specifically."

---

## Parameter Grid (full sensitivity analysis)

| Parameter             | Values to test     |
|-----------------------|--------------------|
| Synthetic accounts    | 20, 30, 50, 100    |
| Images per account    | 5, 10, 20          |
| Intra-group follow %  | 0%, 30%, 60%       |

**Total combinations:** 4 × 3 × 3 = 36 trials

**Max images needed:** 100 accounts × 20 images = **2,000 unique images**
(each image assigned to exactly one account — no cross-account reuse, otherwise pHash catches it)

---

## Data Organisation (no duplication of FediData)

**Do NOT modify or duplicate the original FediData files.**

Synthetic data lives entirely in its own directory:
```
dataset/rq3_synthetic/
    synthetic_accounts.csv       # cols: synthetic_id, n_images
    synthetic_edges.csv          # cols: source, target, follow_rate_param
    synthetic_account_ids.txt    # plain list of IDs for fast lookup
    images/                      # one subdir per synthetic account
        100000/
            100000_0_0.jpg       # same naming convention as FediData (uid_postid_imgid)
            100000_1_0.jpg
            ...
        100001/
            ...
```

**Synthetic account IDs start at 100,000** — FediData goes up to ~64,345, so no collision.

At runtime the RQ3 script loads FediData + synthetic data and merges them in memory.
A simple `id >= 100000` check distinguishes synthetic from real at any point.
Real embeddings on Samsung T7 stay untouched.
Synthetic embeddings saved separately:
`/Volumes/Samsung PSSD T7 Media/FediData/embeddings/rq3_synthetic_embeddings.npz`

---

## Step 1 — Download Forest Images

**Tool:** `fiftyone` (already imported in `synthetic_images.py`)
**Dataset:** Google Open Images v7, label `Forest` (and/or `Tree`)
**Target:** 2,100 images (2,000 needed + 10% buffer for quality filtering)

```python
# in synthetic_images.py
import fiftyone.zoo as foz
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=["Forest"],
    max_samples=2100,
)
```

**After download — deduplicate:**
Run pHash on all downloaded images and remove any pairs with Hamming distance < 10.
Goal: the downloaded set should be pHash-diverse so pHash cannot catch them.
This is critical — if any two images are pHash-similar, and they end up in different
synthetic accounts, pHash would catch that cross-account pair, contaminating results.

Save cleaned images to `dataset/rq3_synthetic/images/staging/` before distributing to accounts.

---

## Step 2 — Create Synthetic Accounts

Script: `src/rq3/create_synthetic_accounts.py`

For each combination of (n_accounts, n_images_per_account):
- Sample n_accounts × n_images_per_account images from the cleaned staging pool (no repeats)
- Distribute to per-account subdirs under `dataset/rq3_synthetic/images/`
- Use naming convention: `{synthetic_id}_{post_idx}_0.jpg`
- Write `dataset/rq3_synthetic/synthetic_accounts.csv`

Run this once for the maximum (100 accounts × 20 images = 2,000 images).
Smaller parameter combos just use a subset of the same accounts/images — no re-download needed.

---

## Step 3 — Create Synthetic Edges

Script: `src/rq3/create_synthetic_accounts.py` (same script, second phase)

For each intra-group follow rate (0%, 30%, 60%):
- Sample follow edges between synthetic accounts at the given rate
  (from all N×(N-1) directed pairs, keep each with probability = follow_rate)
- Add inter-group edges: each synthetic account follows 15 randomly sampled real FediData accounts
  (makes synthetic accounts look embedded in the real graph, not isolated islands)
- Write to `dataset/rq3_synthetic/synthetic_edges.csv` with a `follow_rate` column

The original `dataset/anonymous_60k_user_info/edge_60k_anonymized.csv` is never touched.
At runtime, merge both edge files in memory.

---

## Step 4 — Embed Synthetic Accounts with CLIP

Script: `src/rq3/embed_synthetic.py`

- Re-use the same CLIP pipeline as `embed_clip_60k.py` (ViT-B/32, MPS, batch=64)
- Input: `dataset/rq3_synthetic/images/{synthetic_id}/`
- Output: `/Volumes/Samsung PSSD T7 Media/FediData/embeddings/rq3_synthetic_embeddings.npz`
  - Arrays: `embeddings`, `user_ids`, `post_ids`, `img_ids` (same schema as real embeddings)
- Mean-pool and L2-normalise per account to get one 512d vector per synthetic account

---

## Step 5 — Compute pHash for All Images

Script: `src/rq3/compute_phash.py`

**Synthetic images:**
- Compute pHash for every image in `dataset/rq3_synthetic/images/`
- Save: `dataset/rq3_synthetic/synthetic_phashes.csv` (cols: synthetic_id, img_path, phash_hex)

**Real FediData images (needed for false positive rate):**
- pHash the full FediData image set on the Samsung T7
- Save: `/Volumes/Samsung PSSD T7 Media/FediData/embeddings/fedidata_phashes.csv`
- This may take time — checkpoint every 1,000 users

**pHash comparison logic at detection time:**
For a pair of accounts (A, B): they are pHash-similar if ANY image from A has
Hamming distance ≤ 10 bits with ANY image from B.

---

## Step 6 — Run Detection and Collect Results

Script: `src/rq3/rq3_detection.py`

For each of the 36 parameter combinations:

```
n_accounts ∈ {20, 30, 50, 100}
n_images   ∈ {5, 10, 20}
follow_rate ∈ {0.0, 0.3, 0.6}
```

**Per trial:**
1. Load real user vectors + synthetic user vectors (subset to n_accounts accounts)
2. Build merged FAISS index (HNSW M=16 efSearch=32 — from RQ2)
3. Load merged edge list (real edges + synthetic edges at this follow_rate)
4. For each synthetic account: query top-K=10 neighbours from FAISS
5. Filter to pairs where: similarity > 0.787 AND no edge exists → CLIP-flagged pairs
6. Count: how many synthetic-synthetic pairs are in the flagged set → **CLIP recall**
7. For pHash: check all synthetic-synthetic image pairs for Hamming ≤ 10 → **pHash recall**
8. Also record: how many real-user pairs are flagged (false positive rate)

**Save results:**
`src/rq3/rq3_results.csv`
cols: n_accounts, n_images, follow_rate, clip_recall, phash_recall, fp_rate, n_flagged_pairs

---

## Step 7 — Plot Results

Script: `src/rq3/rq3_plot.py`

- Heatmap or line plot: CLIP recall vs (n_accounts, n_images) for each follow_rate
- One panel per follow_rate showing CLIP recall (blue) vs pHash recall (orange, near-zero)
- Table: summary of CLIP recall across all 36 combinations
- Key message: CLIP recall high across all parameter combos; pHash recall near zero throughout

---

## Key Design Decisions to Justify in Thesis

| Decision | Justification |
|---|---|
| Forest images | Supervisor suggestion; plausible IO visual theme; CLIP captures semantic coherence; varied enough to defeat pHash |
| ID space 100,000+ | No collision with FediData (max ~64,345); trivial to filter in code |
| Similarity threshold 0.787 | Derived from RQ1 μ_conn (mean pooling) — not arbitrary |
| FAISS config from RQ2 | Explicit pipeline chain: RQ2 optimises, RQ3 applies |
| 0/30/60% follow rates | Based on Luceri et al. (minimally overlapping subnetworks) + Ng et al. (densely connected Account-Account networks); tests the full evasion spectrum |
| 15 inter-group follows | Makes synthetic accounts look embedded in real graph; consistent with IO accounts following legitimate users (Luceri et al.) |
| pHash threshold δ=10 | Standard in literature (TODO: cite) |
| No image reuse across accounts | Ensures pHash cannot detect cross-account coordination by design |
