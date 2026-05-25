# Compute pHash for all synthetic images.
# Saves a CSV mapping (synthetic_id, img_path, phash_hex) for use in rq3_detection.py.
#
# Note: real FediData images are NOT pHashed here — detection uses CLIP similarity
# for real accounts and only checks pHash within synthetic-synthetic pairs.
#
# Output:
#   dataset/rq3_synthetic/synthetic_phashes.csv   (synthetic_id, img_path, phash_hex)

import csv
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm

IMAGES_DIR         = Path("/Volumes/Samsung PSSD T7 Media/RQ3_Synthetic/images")
ROOT               = Path(__file__).parent.parent.parent
OUT_CSV            = ROOT / "dataset" / "rq3_synthetic" / "synthetic_phashes.csv"
SYNTHETIC_ID_START = 100_000


def main():
    acct_dirs = sorted(
        d for d in IMAGES_DIR.iterdir()
        if d.is_dir() and d.name != "staging" and int(d.name) >= SYNTHETIC_ID_START
    )
    print(f"Computing pHashes for {len(acct_dirs)} accounts...")

    rows = []
    for acct_dir in tqdm(acct_dirs, unit="account"):
        sid = int(acct_dir.name)
        for img_path in sorted(acct_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                h = imagehash.phash(Image.open(img_path).convert("RGB"))
                rows.append({
                    "synthetic_id": sid,
                    "img_path": str(img_path),
                    "phash_hex": str(h),
                })
            except Exception:
                pass

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["synthetic_id", "img_path", "phash_hex"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows):,} image hashes → {OUT_CSV}")


if __name__ == "__main__":
    main()
