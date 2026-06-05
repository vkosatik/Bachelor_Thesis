# Reorganize flat 60k_jpg dir into per-user subdirs - ran only once when I pulled from fedidata
# all images and got a single flat dir
# images named a_b_c.jpg where a=user index, b=post index, c=image index

import os
import sys
import argparse
from pathlib import Path

SRC = Path("/Volumes/Samsung PSSD T7 Media/FediData/60k_jpg")
DST = Path("/Volumes/Samsung PSSD T7 Media/FediData/raw_images")


def reorganize(src, dst, dry_run=False):
    try:
        from tqdm import tqdm
        progress = tqdm
    except ImportError:
        def progress(iterable, **kwargs):
            print("tqdm not installed, running silently...")
            return iterable

    all_files = [f for f in src.iterdir() if f.is_file()]
    print(f"Found {len(all_files):,} files in {src}")
    if dry_run:
        print("DRY RUN — no files will be moved\n")

    moved = 0
    skipped = 0
    errors = []

    for f in progress(all_files, total=len(all_files), desc="Reorganizing", unit="file"):
        parts = f.stem.split("_")

        if len(parts) < 3 or not parts[0].isdigit():
            errors.append((f.name, "unexpected filename format"))
            continue

        user_dir = dst / parts[0]
        dest = user_dir / f.name

        if dest.exists():
            skipped += 1
            continue

        if not dry_run:
            user_dir.mkdir(parents=True, exist_ok=True)
            f.rename(dest)

        moved += 1

    print(f"\nDone.")
    print(f"  Moved:   {moved:,}")
    print(f"  Skipped: {skipped:,}  (already at destination)")
    print(f"  Errors:  {len(errors):,}")

    if errors:
        print("\nCouldn't process:")
        for name, reason in errors[:20]:
            print(f"  {name} — {reason}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--src", type=Path, default=SRC)
    parser.add_argument("--dst", type=Path, default=DST)
    args = parser.parse_args()

    if not args.src.exists():
        print(f"Source not found: {args.src}")
        sys.exit(1)

    reorganize(args.src, args.dst, dry_run=args.dry_run)
