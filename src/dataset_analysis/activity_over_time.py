import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from datetime import datetime
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent.parent / "dataset" / "anonymous_60k_user_info"

# ── Load bot flags ─────────────────────────────────────────────────────────────
print("Loading accounts...")
bots = set()
with open(f"{DATA_DIR}/accounts_info_60k_anonymized.csv") as f:
    for row in csv.DictReader(f):
        if row["bot"].strip().upper() == "TRUE":
            bots.add(row["acct"])

# ── Load posts and bucket by year-month ───────────────────────────────────────
print("Loading posts...")
bot_monthly   = defaultdict(int)
human_monthly = defaultdict(int)
bot_hourly    = defaultdict(int)
human_hourly  = defaultdict(int)

with open(f"{DATA_DIR}/user_posts_60k_anonymized.json") as f:
    posts_data = json.load(f)

for acct, posts in posts_data.items():
    is_bot = acct in bots
    for post in posts:
        raw = post.get("created_at", "")
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            continue
        ym = (dt.year, dt.month)
        hr = dt.hour
        if is_bot:
            bot_monthly[ym] += 1
            bot_hourly[hr]  += 1
        else:
            human_monthly[ym] += 1
            human_hourly[hr]  += 1

# ── Build sorted month axis ────────────────────────────────────────────────────
all_months = sorted(set(bot_monthly) | set(human_monthly))
# Filter to reasonable range
all_months = [m for m in all_months if 2008 <= m[0] <= 2024]

labels      = [f"{y}-{mo:02d}" for y, mo in all_months]
bot_counts  = [bot_monthly.get(m, 0)   for m in all_months]
human_counts= [human_monthly.get(m, 0) for m in all_months]

# Normalise to posts per account so scale is comparable
n_bots   = len(bots)
n_humans = 63744  # from earlier analysis
bot_norm   = [c / n_bots   for c in bot_counts]
human_norm = [c / n_humans for c in human_counts]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Posting Activity Over Time — Bots vs Humans", fontsize=15, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)

C_BOT   = "#e05c5c"
C_HUMAN = "#5c8de0"
x = np.arange(len(labels))

# ── 1. Raw post volume over time ───────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
# Tick every 6 months
tick_every = 6
tick_idx = list(range(0, len(labels), tick_every))

ax1.fill_between(x, human_counts, alpha=0.4, color=C_HUMAN, label="Human (raw)")
ax1.fill_between(x, bot_counts,   alpha=0.7, color=C_BOT,   label="Bot (raw)")
ax1.plot(x, human_counts, color=C_HUMAN, linewidth=0.8)
ax1.plot(x, bot_counts,   color=C_BOT,   linewidth=0.8)
ax1.set_xticks(tick_idx)
ax1.set_xticklabels([labels[i] for i in tick_idx], rotation=45, ha="right", fontsize=7)
ax1.set_ylabel("Total Posts", fontsize=9)
ax1.set_title("Raw Post Volume per Month", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9)
ax1.tick_params(axis="y", labelsize=8)

# ── 2. Normalised: posts per account per month ────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.fill_between(x, human_norm, alpha=0.4, color=C_HUMAN, label="Human")
ax2.fill_between(x, bot_norm,   alpha=0.7, color=C_BOT,   label="Bot")
ax2.plot(x, human_norm, color=C_HUMAN, linewidth=0.8)
ax2.plot(x, bot_norm,   color=C_BOT,   linewidth=0.8)
ax2.set_xticks(tick_idx)
ax2.set_xticklabels([labels[i] for i in tick_idx], rotation=45, ha="right", fontsize=7)
ax2.set_ylabel("Posts per Account", fontsize=9)
ax2.set_title("Normalised Activity\n(posts per account per month)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.tick_params(axis="y", labelsize=8)

# ── 3. Posting hour of day ────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
hours = np.arange(24)
bot_hr_norm   = [bot_hourly.get(h, 0)   / n_bots   for h in hours]
human_hr_norm = [human_hourly.get(h, 0) / n_humans for h in hours]

width = 0.4
ax3.bar(hours - width/2, bot_hr_norm,   width, color=C_BOT,   alpha=0.85, label="Bot",   edgecolor="white")
ax3.bar(hours + width/2, human_hr_norm, width, color=C_HUMAN, alpha=0.85, label="Human", edgecolor="white")
ax3.set_xlabel("Hour of Day (UTC)", fontsize=9)
ax3.set_ylabel("Posts per Account", fontsize=9)
ax3.set_title("Posting Hour Distribution\n(normalised per account)", fontsize=11, fontweight="bold")
ax3.set_xticks(hours[::2])
ax3.legend(fontsize=9)
ax3.tick_params(labelsize=8)

# ── Save ───────────────────────────────────────────────────────────────────────
out = Path(__file__).parent / "activity_over_time.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")

# ── Summary ────────────────────────────────────────────────────────────────────
peak_bot_m   = all_months[np.argmax(bot_counts)]
peak_human_m = all_months[np.argmax(human_counts)]
print(f"\nPeak bot posting month   : {peak_bot_m[0]}-{peak_bot_m[1]:02d} ({max(bot_counts):,} posts)")
print(f"Peak human posting month : {peak_human_m[0]}-{peak_human_m[1]:02d} ({max(human_counts):,} posts)")
print(f"Peak bot hour   : {max(bot_hourly, key=bot_hourly.get):02d}:00 UTC")
print(f"Peak human hour : {max(human_hourly, key=human_hourly.get):02d}:00 UTC")

plt.show()
