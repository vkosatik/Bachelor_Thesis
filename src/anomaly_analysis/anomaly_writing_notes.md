# Anomaly Analysis — Writing Notes

## What this analysis is

After establishing the 20.4% false-positive rate in RQ1 (disconnected real-user pairs
that CLIP+FAISS flags above threshold = 0.787), this analysis asks: *who* are those
false-positive pairs, and is the error rate uniform across the network?

The answer: it is not uniform. FPs cluster among same-instance users who share
visual community norms without having a follow edge.

---

## Key numbers to cite

| Group | n pairs | FP rate |
|---|---|---|
| Same-instance disconnected | 453 | **46.8%** |
| Cross-instance disconnected | 9,547 | **19.1%** |
| Overall disconnected | 10,000 | **20.4%** |

- Same-instance pairs are **2.4× more likely** to be falsely flagged.
- 90% of all flagged pairs are cross-instance — not because the cross-instance FP rate
  is high, but because the dataset is dominated by cross-instance pairs (~95%).
- Top anomalous pairs come from semantically adjacent communities:
  `infosec.exchange ↔ mastodon.social`, `floss.social ↔ tech.lgbt`, same-server
  `baraag.net` pairs — users who share thematic interests without following each other.

---

## What to say in the thesis

### Framing (results section)
> To characterise the pipeline's false positives, we annotated each of the 10,000
> disconnected pairs with same- vs cross-instance membership and compared FP rates.
> Same-instance pairs were flagged at 46.8% vs 19.1% for cross-instance pairs,
> a 2.4× difference (Figure X). This confirms that the overall 20.4% FP rate is not
> uniformly distributed: it is inflated by pairs that share instance-level visual
> norms without direct social ties.

### Interpretation
- The result is a direct consequence of the instance confounder established in the
  RQ1 stratified analysis (same-instance Δ = +0.039 over cross-instance).
  Users on the same server post into the same cultural space, so their mean-pooled
  CLIP vectors are closer — not because they coordinated, but because of shared
  community aesthetics (e.g., art servers, tech servers, NSFW servers).
- Cross-instance FP rate (19.1%) is very close to the overall rate (20.4%) because
  cross-instance pairs dominate the dataset. The overall FP rate is essentially the
  cross-instance rate.
- This is a meaningful limitation of a globally calibrated threshold: the cutoff 0.787
  was set at μ_conn (overall connected-pair mean), which does not account for the
  elevated within-instance baseline.

### Connection to RQ3 (IO detection)
- The synthetic IO accounts in RQ3 are all from the same visual "campaign" (Open
  Images Tree class) but are injected as cross-instance accounts (synthetic IDs
  starting at 100,000, not on any real instance).
- The 20.4% FP rate used as the RQ3 baseline is the overall/cross-instance rate,
  which is the realistic operating point for a detector scanning the full network.
- If the detector were deployed within a single instance, the effective FP rate would
  be closer to 46.8% — nearly half of all unconnected same-instance pairs would be
  incorrectly flagged. This motivates instance-aware thresholding as future work.

---

## What NOT to overclaim

- The 453 same-instance disconnected pairs are a small subsample (the original
  sampling was not stratified for this). The 46.8% rate is directionally correct
  and corroborated by the RQ1 instance confounder (which used a dedicated 10k
  stratified sample), but treat the exact figure as illustrative rather than precise.
- "Anomalous" pairs are not necessarily suspicious: they are users who happen to
  post visually similar content without following each other. Many are plausibly
  interested in the same topics on different servers.

---

## Suggested thesis placement

- **Results section**: as a sub-section of RQ1 or as a bridging analysis between
  RQ1 and RQ3 ("Characterising False Positives").
- **Discussion**: use it to motivate instance-aware thresholding as a limitation and
  future direction.
- **Limitations**: cite the 20.4% global FP rate and note it rises to ~47% within
  single instances, so deployment context matters.

---

## Figures

- `anomaly_plot.png`: 3-panel figure
  - Panel 1: Similarity distribution of all disconnected pairs with flagged region highlighted
  - Panel 2: FP rate by same-instance vs cross-instance vs overall
  - Panel 3: Composition of the 2040 flagged pairs (10% same-inst, 90% cross-inst)

---

## One-sentence summary for abstract/conclusion

> False positives from the CLIP+FAISS pipeline are not random: same-instance
> disconnected pairs are flagged at 2.4× the rate of cross-instance pairs,
> reflecting shared visual community norms rather than coordinated behaviour.
