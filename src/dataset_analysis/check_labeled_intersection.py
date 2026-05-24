import csv
from collections import Counter
from pathlib import Path

ROOT         = Path(__file__).parent.parent
ACCOUNTS_CSV = ROOT / "dataset" / "anonymous_60k_user_info" / "accounts_info_60k_anonymized.csv"
LABELED_CSV  = ROOT / "dataset" / "anonymous_60k_user_info" / "12k_labeled_anonymized.csv"
IMAGES_DIR   = Path("/Volumes/Samsung PSSD T7 Media/FediData/raw_images")

# acct -> row index (= user_id in embeddings)
print("Loading accounts...")
acct_to_id = {}
with open(ACCOUNTS_CSV) as f:
    for idx, row in enumerate(csv.DictReader(f)):
        acct_to_id[row["acct"]] = idx
print(f"  {len(acct_to_id):,} total accounts")

# user_ids that have at least one image on disk (= were embedded)
print("Scanning image folders...")
embedded_ids = set()
img_counts = {}
for folder in IMAGES_DIR.iterdir():
    if folder.is_dir() and folder.name.isdigit():
        uid = int(folder.name)
        count = sum(1 for f in folder.iterdir() if f.suffix == ".jpg")
        if count > 0:
            embedded_ids.add(uid)
            img_counts[uid] = count
print(f"  {len(embedded_ids):,} users with ≥1 image on disk")

# load labels
print("Loading labels...")
label_map = {}
with open(LABELED_CSV) as f:
    for row in csv.DictReader(f):
        uid = acct_to_id.get(row["acct"])
        if uid is not None:
            label_map[uid] = int(row["label"])
        # else: labeled user not in accounts CSV (remote account)

label_counts = Counter(label_map.values())
print(f"  {len(label_map):,} labeled users mapped to user_ids")
print(f"  Label distribution:")
label_names = {0: "human", 1: "bot", 2: "unknown/spam", 3: "high-reach"}
for lbl in sorted(label_counts):
    print(f"    {lbl} ({label_names.get(lbl, '?')}): {label_counts[lbl]:,}")

# intersection: labeled AND embedded
print("\nIntersection (labeled + has images on disk):")
both = {uid: label_map[uid] for uid in label_map if uid in embedded_ids}
both_counts = Counter(both.values())
print(f"  {len(both):,} users are both labeled and embedded")
print(f"  Label distribution in intersection:")
for lbl in sorted(both_counts):
    total_lbl = label_counts[lbl]
    n = both_counts[lbl]
    print(f"    {lbl} ({label_names.get(lbl, '?')}): {n:,} / {total_lbl:,} ({100*n/total_lbl:.1f}% of that label have images)")

print(f"\nSummary for RQ3:")
print(f"  Labeled bots with images:      {both_counts.get(1, 0):,}")
print(f"  Labeled humans with images:    {both_counts.get(0, 0):,}")
print(f"  High-reach with images:        {both_counts.get(3, 0):,}")
print(f"  Unknown/spam with images:      {both_counts.get(2, 0):,}")
