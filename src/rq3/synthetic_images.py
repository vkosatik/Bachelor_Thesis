# Download forest images from Open Images v7, pHash-deduplicate, save to staging dir.
# Target: 2100 images downloaded → ≥2000 after dedup (need 100 accounts × 20 images max).
#
# Dedup rationale: images are distributed across different synthetic accounts. Any two
# images within Hamming ≤ 10 that land in different accounts would be caught by pHash
# detection, contaminating the CLIP-vs-pHash comparison. We remove those pairs here so
# that pHash recall near-zero is a genuine finding, not an artefact of lucky image choice.
# Threshold 10 is the strict end of the 6-10 range used in the literature.
#
# Output: /Volumes/Samsung PSSD T7 Media/RQ3_Synthetic/images/staging/

import shutil
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm

DRIVE          = Path("/Volumes/Samsung PSSD T7 Media/RQ3_Synthetic")
STAGING_DIR    = DRIVE / "images" / "staging"
MAX_SAMPLES    = 2100
HAMMING_THRESH = 10   # strict end of 6-10 range used in literature


def download_images():
    import fiftyone.zoo as foz

    print(f"Downloading up to {MAX_SAMPLES} Forest images from Open Images v7...")
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=["Forest"],
        max_samples=MAX_SAMPLES,
        dataset_name="rq3_forest_raw",
        overwrite=True,
    )
    return dataset


def collect_image_paths(dataset):
    paths = []
    for sample in dataset:
        p = Path(sample.filepath)
        if p.exists() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            paths.append(p)
    print(f"Collected {len(paths)} valid images")
    return paths


def compute_phashes(paths):
    hashes = {}
    print("Computing pHashes...")
    for p in tqdm(paths, unit="img"):
        try:
            hashes[p] = imagehash.phash(Image.open(p).convert("RGB"))
        except Exception:
            pass  # skip corrupt images
    return hashes


def deduplicate(hashes):
    paths = list(hashes.keys())
    to_remove = set()
    print(f"Deduplicating (Hamming ≤ {HAMMING_THRESH})...")
    for i in tqdm(range(len(paths)), unit="img"):
        if paths[i] in to_remove:
            continue
        for j in range(i + 1, len(paths)):
            if paths[j] in to_remove:
                continue
            if hashes[paths[i]] - hashes[paths[j]] <= HAMMING_THRESH:
                to_remove.add(paths[j])
    kept = [p for p in paths if p not in to_remove]
    print(f"Removed {len(to_remove)} near-duplicates → {len(kept)} images kept")
    return kept


def copy_to_staging(paths):
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Copying {len(paths)} images to {STAGING_DIR} ...")
    for i, src in enumerate(tqdm(paths, unit="img")):
        dst = STAGING_DIR / f"forest_{i:04d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
    print(f"Done. {len(paths)} images in {STAGING_DIR}")


def main():
    dataset = download_images()
    paths = collect_image_paths(dataset)
    hashes = compute_phashes(paths)
    kept = deduplicate(hashes)

    if len(kept) < 2000:
        print(f"WARNING: only {len(kept)} images after dedup — need ≥2000 for full parameter grid.")
        print("Consider increasing MAX_SAMPLES or adding 'Tree' as a second class.")
    else:
        print(f"OK: {len(kept)} images available (need 2000 max).")

    copy_to_staging(kept)


if __name__ == "__main__":
    main()
