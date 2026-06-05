# makes the bot_analysis plot focusing on analysis of how many bots we have and what's their behavior
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent.parent / "dataset" / "anonymous_60k_user_info"

# ── Load accounts ──────────────────────────────────────────────────────────────
print("Loading accounts...")
accounts = {}
with open(f"{DATA_DIR}/accounts_info_60k_anonymized.csv") as f:
    for row in csv.DictReader(f):
        accounts[row["acct"]] = {
            "bot": row["bot"].strip().upper() == "TRUE",
            "followers": int(row["followers_count"] or 0),
            "following": int(row["following_count"] or 0),
            "statuses": int(row["statuses_count"] or 0),
        }

bot_accts  = {a for a, d in accounts.items() if d["bot"]}
human_accts = {a for a, d in accounts.items() if not d["bot"]}

# ── Load edges ─────────────────────────────────────────────────────────────────
print("Loading edges...")
# Use 'following' edges only to avoid double-counting
bot_to_bot = 0
bot_to_human = 0
human_to_bot = 0
human_to_human = 0

with open(f"{DATA_DIR}/edge_60k_anonymized.csv") as f:
    for row in csv.DictReader(f):
        if row["relation"] != "following":
            continue
        src_is_bot = row["source"] in bot_accts
        tgt_is_bot = row["target"] in bot_accts
        if src_is_bot and tgt_is_bot:
            bot_to_bot += 1
        elif src_is_bot and not tgt_is_bot:
            bot_to_human += 1
        elif not src_is_bot and tgt_is_bot:
            human_to_bot += 1
        else:
            human_to_human += 1

# ── Prepare activity arrays ────────────────────────────────────────────────────
def clip99(arr):
    """Clip to 99th percentile to leave out extreme outliers for display."""
    p99 = np.percentile(arr, 99)
    return np.clip(arr, 0, p99)

bot_stats    = [d for d in accounts.values() if d["bot"]]
human_stats  = [d for d in accounts.values() if not d["bot"]]

bot_statuses   = clip99([d["statuses"]  for d in bot_stats])
human_statuses = clip99([d["statuses"]  for d in human_stats])
bot_followers  = clip99([d["followers"] for d in bot_stats])
human_followers= clip99([d["followers"] for d in human_stats])
bot_following  = clip99([d["following"] for d in bot_stats])
human_following= clip99([d["following"] for d in human_stats])

# ── Figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Mastodon Bot Analysis — 64 k Accounts", fontsize=16, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

COLOR_BOT   = "#e05c5c"
COLOR_HUMAN = "#5c8de0"
COLOR_MIX   = "#8e5ce0"

# ── 1. Pie chart: bot vs human ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
n_bots   = len(bot_accts)
n_humans = len(human_accts)
total    = n_bots + n_humans

wedges, texts, autotexts = ax1.pie(
    [n_bots, n_humans],
    labels=["Bot\n(self-reported)", "Human"],
    colors=[COLOR_BOT, COLOR_HUMAN],
    autopct="%1.2f%%",
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2),
    textprops=dict(fontsize=10),
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight("bold")
ax1.set_title(f"Bot vs Human Accounts\n(n = {total:,})", fontsize=11, fontweight="bold")

# ── 2. Posts (statuses) distribution ──────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
bins = np.linspace(0, max(max(bot_statuses), max(human_statuses)), 60)
ax2.hist(human_statuses, bins=bins, color=COLOR_HUMAN, alpha=0.65, label="Human", density=True)
ax2.hist(bot_statuses,   bins=bins, color=COLOR_BOT,   alpha=0.75, label="Bot",   density=True)
ax2.set_xlabel("Number of Posts (clipped at 99th pct)", fontsize=9)
ax2.set_ylabel("Density", fontsize=9)
ax2.set_title("Post Count Distribution\nBots vs Humans", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.tick_params(labelsize=8)

# ── 3. Followers distribution ──────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
bins_f = np.linspace(0, max(max(bot_followers), max(human_followers)), 60)
ax3.hist(human_followers, bins=bins_f, color=COLOR_HUMAN, alpha=0.65, label="Human", density=True)
ax3.hist(bot_followers,   bins=bins_f, color=COLOR_BOT,   alpha=0.75, label="Bot",   density=True)
ax3.set_xlabel("Followers Count (clipped at 99th pct)", fontsize=9)
ax3.set_ylabel("Density", fontsize=9)
ax3.set_title("Followers Distribution\nBots vs Humans", fontsize=11, fontweight="bold")
ax3.legend(fontsize=9)
ax3.tick_params(labelsize=8)

# ── 4. Following distribution ──────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
bins_fg = np.linspace(0, max(max(bot_following), max(human_following)), 60)
ax4.hist(human_following, bins=bins_fg, color=COLOR_HUMAN, alpha=0.65, label="Human", density=True)
ax4.hist(bot_following,   bins=bins_fg, color=COLOR_BOT,   alpha=0.75, label="Bot",   density=True)
ax4.set_xlabel("Following Count (clipped at 99th pct)", fontsize=9)
ax4.set_ylabel("Density", fontsize=9)
ax4.set_title("Following Count Distribution\nBots vs Humans", fontsize=11, fontweight="bold")
ax4.legend(fontsize=9)
ax4.tick_params(labelsize=8)

# ── 5. Median activity bar chart ───────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
metrics = ["Posts", "Followers", "Following"]
bot_medians   = [
    np.median([d["statuses"]  for d in bot_stats]),
    np.median([d["followers"] for d in bot_stats]),
    np.median([d["following"] for d in bot_stats]),
]
human_medians = [
    np.median([d["statuses"]  for d in human_stats]),
    np.median([d["followers"] for d in human_stats]),
    np.median([d["following"] for d in human_stats]),
]
x = np.arange(len(metrics))
w = 0.35
bars1 = ax5.bar(x - w/2, bot_medians,   w, label="Bot",   color=COLOR_BOT,   edgecolor="white")
bars2 = ax5.bar(x + w/2, human_medians, w, label="Human", color=COLOR_HUMAN, edgecolor="white")
ax5.set_xticks(x)
ax5.set_xticklabels(metrics, fontsize=10)
ax5.set_ylabel("Median Count", fontsize=9)
ax5.set_title("Median Activity\nBots vs Humans", fontsize=11, fontweight="bold")
ax5.legend(fontsize=9)
ax5.tick_params(labelsize=8)
for bar in bars1:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)

# ── 6. Bot interconnectedness stacked bar ─────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])

bot_out_total   = bot_to_bot + bot_to_human
human_out_total = human_to_bot + human_to_human

pct_bot_to_bot    = 100 * bot_to_bot    / bot_out_total   if bot_out_total   else 0
pct_bot_to_human  = 100 * bot_to_human  / bot_out_total   if bot_out_total   else 0
pct_hum_to_bot    = 100 * human_to_bot  / human_out_total if human_out_total else 0
pct_hum_to_human  = 100 * human_to_human/ human_out_total if human_out_total else 0

categories = ["Bot accounts\nfollowing", "Human accounts\nfollowing"]
vals_to_bot   = [pct_bot_to_bot,  pct_hum_to_bot]
vals_to_human = [pct_bot_to_human, pct_hum_to_human]

b1 = ax6.bar(categories, vals_to_bot,   color=COLOR_BOT,   label="→ Bot",   edgecolor="white")
b2 = ax6.bar(categories, vals_to_human, bottom=vals_to_bot, color=COLOR_HUMAN, label="→ Human", edgecolor="white")

for bar, val in zip(b1, vals_to_bot):
    if val > 3:
        ax6.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height()/2,
                 f"{val:.1f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
for bar, base, val in zip(b2, vals_to_bot, vals_to_human):
    if val > 3:
        ax6.text(bar.get_x() + bar.get_width()/2,
                 base + val/2,
                 f"{val:.1f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

ax6.set_ylabel("% of outgoing follows", fontsize=9)
ax6.set_title("Bot Interconnectedness\n(Who do bots follow?)", fontsize=11, fontweight="bold")
ax6.legend(fontsize=9, loc="upper right")
ax6.set_ylim(0, 110)
ax6.tick_params(labelsize=8)

# ── Footnote ──────────────────────────────────────────────────────────────────
fig.text(0.5, 0.01,
         "Note: 'bot' is a self-reported flag set by the account owner on Mastodon. "
         "No likes/favourites data available in this dataset.",
         ha="center", fontsize=8, color="gray")

# ── Save & show ───────────────────────────────────────────────────────────────
out_path = Path(__file__).parent / "bot_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")

# Print summary stats
print("\n── Summary ──────────────────────────────────────")
print(f"Total accounts : {total:,}")
print(f"  Bots         : {n_bots:,}  ({100*n_bots/total:.2f}%)")
print(f"  Humans       : {n_humans:,}  ({100*n_humans/total:.2f}%)")
print(f"\nFollowing edges (deduped): {bot_to_bot+bot_to_human+human_to_bot+human_to_human:,}")
print(f"  Bot  → Bot   : {bot_to_bot:,}  ({pct_bot_to_bot:.1f}% of bot outgoing)")
print(f"  Bot  → Human : {bot_to_human:,}  ({pct_bot_to_human:.1f}% of bot outgoing)")
print(f"  Human→ Bot   : {human_to_bot:,}  ({pct_hum_to_bot:.1f}% of human outgoing)")
print(f"  Human→ Human : {human_to_human:,}  ({pct_hum_to_human:.1f}% of human outgoing)")
print(f"\nMedian posts   — Bot: {np.median([d['statuses'] for d in bot_stats]):.0f}  Human: {np.median([d['statuses'] for d in human_stats]):.0f}")
print(f"Median followers — Bot: {np.median([d['followers'] for d in bot_stats]):.0f}  Human: {np.median([d['followers'] for d in human_stats]):.0f}")
print(f"Median following — Bot: {np.median([d['following'] for d in bot_stats]):.0f}  Human: {np.median([d['following'] for d in human_stats]):.0f}")

plt.show()
