# RQ1 Instance Confounder — Writing Notes

## What this analysis is and why it matters

- The original RQ1 result (Δ=+0.101, p<0.001) shows connected pairs are more visually similar than disconnected pairs.
- Potential confounder: Mastodon is federated — users on the same instance (e.g. mastodon.social, pawoo.net) share a community, norms, and likely visual culture, independent of whether they follow each other.
- If the signal is only driven by instance membership, it would vanish in cross-instance pairs.
- This analysis controls for that by splitting pairs into same-instance and cross-instance groups, 10k each, balanced.

---

## Key numbers to cite

| Comparison | n_conn | μ_conn | n_disc | μ_disc | Δ | p |
|---|---|---|---|---|---|---|
| Same-instance connected vs disconnected | 10,000 | 0.8038 | 10,000 | 0.7651 | +0.039 | 2.34e-181 |
| Cross-instance connected vs disconnected | 10,000 | 0.7690 | 10,000 | 0.6819 | +0.087 | ≈0 |
| Same-instance connected vs cross-instance connected | 10,000 | 0.8038 | 10,000 | 0.7690 | +0.035 | 5.93e-174 |

---

## What to say

**Homophily survives the confounder test.**
The cross-instance result (Δ=+0.087) is the cleanest: two users on different Mastodon servers who follow each other are still significantly more visually similar than random strangers across servers. Instance culture cannot explain this — there is no shared community pulling them together.

**Instance culture is a real but partial effect.**
Same-instance disconnected pairs (μ=0.765) score higher than cross-instance disconnected pairs (μ=0.682). This confirms that shared instance membership does inflate visual similarity — users on the same server share a visual culture.

**The social tie signal is *on top of* the instance effect.**
Same-instance connected (μ=0.804) > same-instance disconnected (μ=0.765) — homophily persists even after the instance baseline is accounted for (Δ=+0.039).

**Cross-instance gap is larger than same-instance gap.**
Δ_cross=+0.087 > Δ_same=+0.039. This might seem counterintuitive — you'd expect same-instance pairs to show stronger homophily. One interpretation: within an instance, the community floor is already high (shared visual culture), so social ties add less on top. Across instances, the baseline is lower, so the tie signal is more visible.

**Connected users are more similar than what instance explains.**
Same-instance connected (μ=0.804) > cross-instance connected (μ=0.769) — friends on the same server are more visually aligned than friends across servers. Some of this is the instance culture floor, but the homophily holds in both groups.

---

## Framing for the thesis

- Present this as a robustness check / confounder analysis for RQ1, not a separate RQ.
- Lead with the cross-instance result as the strongest evidence that the signal is about individual social ties, not just community membership.
- Acknowledge same-instance effect as a legitimate partial explanation, and say the analysis disentangles the two.
- The two-component framing works well: *community-level visual culture* (instance effect) + *individual tie formation* (homophily on top of that).

---

## Connection to RQ3

- The cross-instance finding strengthens the RQ3 motivation: if high visual similarity between unconnected accounts cannot be explained by instance membership alone, then anomalously similar unconnected cross-instance pairs are genuinely suspicious.
- When RQ3 flags synthetic IO accounts (or real anomalous pairs), you can note that the threshold (0.787 = μ_conn from RQ1) is derived from the overall connected-pair mean, and the cross-instance confounder analysis shows the signal is robust.
