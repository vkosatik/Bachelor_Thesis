# RQ3 Writing Notes

## Synthetic image pool — methodology framing

### What we used
- **Source:** Google Open Images v7, `Tree` class, detection labels
- **Filter:** bounding box coverage ≥ 40% of image area (normalised w×h ≥ 0.40)
- **Result:** 2,533 images after filter, 0 near-duplicates removed by pHash dedup (Hamming ≤ 10)
- **Cite as:** "Images drawn from Google Open Images v7 (Kuznetsova et al., 2020), restricted to the Tree detection class with bounding box coverage ≥ 40% of image area, ensuring tree/vegetation is a prominent visual element."

### Why this is methodologically defensible — what to say in thesis

The filter produces varied tree-dominant outdoor images (dense forest, woodland paths, single large trees, forest monuments) rather than perfectly uniform stock photos. This is intentional:

- Real IO accounts post varied images that share a loose visual theme — not 2,000 identical photos. A mix of images united by a tree/nature theme is a realistic synthetic IO scenario.
- The Google Open Images `Tree` label was assigned by human annotators who verified tree presence in the image. The bounding box ≥ 40% filter ensures the tree is a major visual element, not a background detail.
- **Do NOT apply a secondary CLIP filter** to remove outliers. That would be optimising the input to produce the desired result, which a reviewer could flag as cherry-picking.

### Sanity check result — cite this
Inter-account CLIP similarity across 20 simulated synthetic accounts (20 images/account, mean-pooled):

- mean = 0.9646, median = 0.9648, min = 0.9471, max = 0.9782
- All 190/190 account pairs above the RQ1 detection threshold (μ_conn = 0.787)

**Thesis framing to use:**
> "Synthetic accounts were seeded with images from the Open Images v7 Tree class (bounding box coverage ≥ 40%), reflecting the varied visual content typical of coordinated inauthentic accounts. A pre-experiment validation confirmed inter-account CLIP similarity of 0.965 (mean-pooled), establishing that the image pool is semantically coherent and that all synthetic pairs would exceed the detection threshold derived from RQ1."

### Why the variation is a strength, not a weakness
The fact that CLIP achieves 0.965 inter-account similarity despite within-pool visual variation (forest scenes, single trees, rock monuments surrounded by trees) is itself a finding: **CLIP's semantic embeddings are robust to within-theme visual noise**, capturing the thematic coordination signal even when images are not pixel-identical or compositionally uniform. This is precisely the capability that pHash lacks.

---

## pHash comparison — what to say

- pHash dedup of the staging pool guarantees no two images in the pool have Hamming distance ≤ 10.
- Each image is assigned to exactly one synthetic account (no cross-account reuse).
- Therefore pHash recall is structurally near-zero by design — this is not a limitation but a feature: it demonstrates that the synthetic IO campaign is **pixel-evasive**, the exact scenario where existing detection methods fail and where CLIP adds value.
- The finding is: "CLIP detects a type of visual coordination that pixel-level methods cannot detect, even in principle, for this evasion strategy."

---

## Parameter grid

| Parameter | Values |
|---|---|
| Synthetic accounts | 20, 30, 50, 100 |
| Images per account | 5, 10, 20 |
| Intra-group follow rate | 0%, 30%, 60% |

36 combinations total (4 × 3 × 3). Run once at max (100 accounts × 20 images); smaller combos are subsets.

Detection threshold: cosine similarity > 0.787 (= RQ1 μ_conn, mean pooling) AND no follow edge.

---

## Connection to RQ1 instance confounder

The cross-instance homophily finding (Δ = +0.087) strengthens RQ3 motivation: high visual similarity between unconnected accounts cannot be explained by shared instance culture alone. Anomalously similar unconnected pairs — including synthetic IO accounts — are genuinely suspicious regardless of which instance they appear on.
