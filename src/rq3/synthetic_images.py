# Download forest images from Open Images v7, save to staging dir (no deduplication).
# Target: 2100 images (need 100 accounts × 20 images max).
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
    import fiftyone as fo
    import fiftyone.zoo as foz
    from fiftyone import ViewField as F

    print(f"Downloading up to {MAX_SAMPLES * 3} Tree images (will filter to tree-dominant)...")
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=["Tree"],
        max_samples=MAX_SAMPLES * 5,   # oversample before filtering so we're sure have enough
        dataset_name="rq3_forest_raw",
        overwrite=True,
    )

    # Keep only images where at least one tree box covers ≥40% of image area.
    # Bounding boxes in Open Images are normalised [x, y, w, h] relative to image dims,
    # so w*h gives the fraction of image area covered by the box.
    dominant_tree_view = dataset.filter_labels(
        "ground_truth",
        (F("label") == "Tree") & ((F("bounding_box")[2] * F("bounding_box")[3]) >= 0.60),
    ).match(F("ground_truth.detections").length() > 0)

    n_after = len(dominant_tree_view)
    print(f"  {n_after} images with tree covering ≥60% of image area")

    if n_after < MAX_SAMPLES:
        raise RuntimeError(f"Only {n_after} images at ≥60% threshold — need {MAX_SAMPLES}. Increase MAX_SAMPLES.")

    return dominant_tree_view


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

    if len(paths) < 2000:
        print(f"WARNING: only {len(paths)} images — need ≥2000 for full parameter grid.")
        print("Consider increasing MAX_SAMPLES or adding 'Tree' as a second class.")
    else:
        print(f"OK: {len(paths)} images available (need 2000 max).")

    copy_to_staging(paths)


if __name__ == "__main__":
    main()
