import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from datetime import datetime
from collections import Counter

DATA_DIR = "/Users/vojtechkosatik/Downloads/anonymous_60k_user_info"

# ── Load human accounts only ───────────────────────────────────────────────────
print("Loading accounts...")
humans = []
with open(f"{DATA_DIR}/accounts_info_60k_anonymized.csv") as f:
    for row in csv.DictReader(f):
        if row["bot"].strip().upper() == "TRUE":
            continue
        try:
            created = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
        except Exception:
            created = None
        humans.append({
            "followers":    int(row["followers_count"] or 0),
            "following":    int(row["following_count"] or 0),
            "statuses":     int(row["statuses_count"] or 0),
            "locked":       row["locked"].strip().upper() == "TRUE",
            "discoverable": row["discoverable"].strip().upper() == "TRUE",
            "islocal":      row["islocal"].strip().upper() == "TRUE",
            "ismastodon":   row["ismastodon"].strip().upper() == "TRUE",
            "has_note":     len(row["note"].strip()) > 0,
            "fields_count": row["fields"].strip().count('"name"'),
            "created":      created,
        })

n = len(humans)
print(f"Human accounts: {n:,}")

followers  = np.array([h["followers"]  for h in humans])
following  = np.array([h["following"]  for h in humans])
statuses   = np.array([h["statuses"]   for h in humans])
ratio      = np.where(following > 0, followers / following, np.nan)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.suptitle(f"Human Account Analysis — {n:,} Non-Bot Accounts", fontsize=16, fontweight="bold", y=0.99)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)

C1 = "#5c8de0"   # main blue
C2 = "#e0a85c"   # amber
C3 = "#5ce08d"   # green
C4 = "#e05c8d"   # pink

# ── 1. Account creation over time ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
years = [h["created"].year for h in humans if h["created"]]
year_counts = Counter(years)
yr_sorted = sorted(year_counts.keys())
ax1.bar(yr_sorted, [year_counts[y] for y in yr_sorted], color=C1, edgecolor="white", linewidth=0.5)
ax1.set_xlabel("Year", fontsize=9)
ax1.set_ylabel("Accounts Created", fontsize=9)
ax1.set_title("Account Creation Over Time", fontsize=11, fontweight="bold")
ax1.tick_params(labelsize=8)
for x, y in zip(yr_sorted, [year_counts[yr] for yr in yr_sorted]):
    if y > 500:
        ax1.text(x, y + 100, f"{y:,}", ha="center", va="bottom", fontsize=7)

# ── 2. Local vs Remote + Mastodon vs Other ────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
n_local    = sum(1 for h in humans if h["islocal"])
n_remote   = n - n_local
n_masto    = sum(1 for h in humans if h["ismastodon"])
n_other    = n - n_masto
categories = ["Local", "Remote", "Mastodon\nprotocol", "Other\nprotocol"]
values     = [n_local, n_remote, n_masto, n_other]
colors     = [C1, C2, C3, C4]
bars = ax2.bar(categories, values, color=colors, edgecolor="white")
ax2.set_title("Local/Remote & Protocol", fontsize=11, fontweight="bold")
ax2.set_ylabel("Count", fontsize=9)
ax2.tick_params(labelsize=8)
for bar in bars:
    pct = 100 * bar.get_height() / n
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

# ── 3. Statuses (posts) distribution — log scale ──────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
nonzero_statuses = statuses[statuses > 0]
ax3.hist(np.log10(nonzero_statuses), bins=50, color=C1, edgecolor="white", linewidth=0.3)
ax3.set_xlabel("log₁₀(Post Count)", fontsize=9)
ax3.set_ylabel("Number of Accounts", fontsize=9)
ax3.set_title(f"Posts per Account (log scale)\nMedian: {int(np.median(statuses)):,}  Max: {int(statuses.max()):,}", fontsize=11, fontweight="bold")
ax3.tick_params(labelsize=8)
xticks = [0, 1, 2, 3, 4, 5]
ax3.set_xticks(xticks)
ax3.set_xticklabels([f"10^{x}" for x in xticks], fontsize=7)

# ── 4. Followers distribution — log scale ─────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
nonzero_followers = followers[followers > 0]
ax4.hist(np.log10(nonzero_followers), bins=50, color=C2, edgecolor="white", linewidth=0.3)
ax4.set_xlabel("log₁₀(Followers)", fontsize=9)
ax4.set_ylabel("Number of Accounts", fontsize=9)
ax4.set_title(f"Followers per Account (log scale)\nMedian: {int(np.median(followers)):,}  Max: {int(followers.max()):,}", fontsize=11, fontweight="bold")
ax4.tick_params(labelsize=8)
ax4.set_xticks(xticks)
ax4.set_xticklabels([f"10^{x}" for x in xticks], fontsize=7)

# ── 5. Following distribution — log scale ─────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
nonzero_following = following[following > 0]
ax5.hist(np.log10(nonzero_following), bins=50, color=C3, edgecolor="white", linewidth=0.3)
ax5.set_xlabel("log₁₀(Following)", fontsize=9)
ax5.set_ylabel("Number of Accounts", fontsize=9)
ax5.set_title(f"Following per Account (log scale)\nMedian: {int(np.median(following)):,}  Max: {int(following.max()):,}", fontsize=11, fontweight="bold")
ax5.tick_params(labelsize=8)
ax5.set_xticks(xticks)
ax5.set_xticklabels([f"10^{x}" for x in xticks], fontsize=7)

# ── 6. Followers vs Following scatter ─────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0])
# Sample 5000 for readability
idx = np.random.choice(n, min(5000, n), replace=False)
f_sample  = followers[idx]
fg_sample = following[idx]
ax6.scatter(
    np.log10(fg_sample + 1), np.log10(f_sample + 1),
    alpha=0.15, s=6, color=C1
)
# diagonal = equal followers/following
lim = max(np.log10(following.max() + 1), np.log10(followers.max() + 1))
ax6.plot([0, lim], [0, lim], "r--", linewidth=1, label="Equal ratio", alpha=0.7)
ax6.set_xlabel("log₁₀(Following + 1)", fontsize=9)
ax6.set_ylabel("log₁₀(Followers + 1)", fontsize=9)
ax6.set_title("Followers vs Following\n(5k sample)", fontsize=11, fontweight="bold")
ax6.legend(fontsize=8)
ax6.tick_params(labelsize=8)

# ── 7. Follower/Following ratio ────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 1])
valid_ratio = ratio[~np.isnan(ratio) & (ratio < 50)]  # clip extreme outliers
ax7.hist(valid_ratio, bins=60, color=C4, edgecolor="white", linewidth=0.3)
ax7.axvline(1.0, color="red", linestyle="--", linewidth=1.2, label="Ratio = 1")
ax7.set_xlabel("Followers / Following", fontsize=9)
ax7.set_ylabel("Number of Accounts", fontsize=9)
ax7.set_title(f"Follower/Following Ratio\n(clipped at 50, median: {np.nanmedian(ratio):.2f})", fontsize=11, fontweight="bold")
ax7.legend(fontsize=8)
ax7.tick_params(labelsize=8)

# ── 8. Profile completeness ────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
n_locked      = sum(1 for h in humans if h["locked"])
n_discoverable= sum(1 for h in humans if h["discoverable"])
n_has_bio     = sum(1 for h in humans if h["has_note"])
n_has_fields  = sum(1 for h in humans if h["fields_count"] > 0)

labels  = ["Locked\naccount", "Discoverable", "Has bio", "Has profile\nfields"]
vals    = [n_locked, n_discoverable, n_has_bio, n_has_fields]
colors  = [C4, C3, C1, C2]
bars    = ax8.bar(labels, vals, color=colors, edgecolor="white")
ax8.set_title("Profile Attributes", fontsize=11, fontweight="bold")
ax8.set_ylabel("Count", fontsize=9)
ax8.tick_params(labelsize=8)
for bar in bars:
    pct = 100 * bar.get_height() / n
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = Path(__file__).parent / "human_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")

# ── Summary stats ──────────────────────────────────────────────────────────────
print(f"\n── Human Account Summary ({'─'*30})")
print(f"Total                : {n:,}")
print(f"Local accounts       : {n_local:,} ({100*n_local/n:.1f}%)")
print(f"Mastodon protocol    : {n_masto:,} ({100*n_masto/n:.1f}%)")
print(f"Locked               : {n_locked:,} ({100*n_locked/n:.1f}%)")
print(f"Discoverable         : {n_discoverable:,} ({100*n_discoverable/n:.1f}%)")
print(f"Has bio              : {n_has_bio:,} ({100*n_has_bio/n:.1f}%)")
print(f"Has profile fields   : {n_has_fields:,} ({100*n_has_fields/n:.1f}%)")
print(f"\nPost counts  — median: {int(np.median(statuses)):,}  mean: {int(np.mean(statuses)):,}  max: {int(statuses.max()):,}")
print(f"Followers    — median: {int(np.median(followers)):,}  mean: {int(np.mean(followers)):,}  max: {int(followers.max()):,}")
print(f"Following    — median: {int(np.median(following)):,}  mean: {int(np.mean(following)):,}  max: {int(following.max()):,}")
print(f"Follow ratio — median: {np.nanmedian(ratio):.2f}  (>1 means more followers than following)")

plt.show()
