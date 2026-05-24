# CLIP ViT-B/32 embeddings for the full 60k FediData dataset
# stores per-image embeddings (not per-user) so we keep full granularity
# checkpoints every 500 users so interrupted runs can resume

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

DRIVE      = Path("/Volumes/Samsung PSSD T7 Media/FediData")
IMG_DIR    = DRIVE / "images"
OUT_DIR    = DRIVE / "embeddings"
OUT_NPZ    = OUT_DIR / "embeddings_60k.npz"
CHECKPOINT = OUT_DIR / "embeddings_60k_checkpoint.npz"

MODEL_NAME       = "openai/clip-vit-base-patch32"
BATCH_SIZE       = 64
CHECKPOINT_EVERY = 500


def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


def parse_filename(stem):
    parts = stem.split("_")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1]), int(parts[2])


def load_user_images(user_dir):
    entries = []
    for p in sorted(user_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        parsed = parse_filename(p.stem)
        if parsed is not None:
            entries.append((p, *parsed))
    return entries


def embed_batch(images, processor, model, device):
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        pooled = model.vision_model(pixel_values=pixel_values).pooler_output  # (N, 768)
        feats = model.visual_projection(pooled)                                # (N, 512)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


def load_checkpoint():
    if not CHECKPOINT.exists():
        return set(), [], [], [], []
    data = np.load(CHECKPOINT)
    done = set(data["done_user_ids"].tolist())
    print(f"Resuming from checkpoint: {len(done):,} users done, {len(data['embeddings']):,} images stored.")
    return done, list(data["embeddings"]), data["user_ids"].tolist(), data["post_ids"].tolist(), data["img_ids"].tolist()


def save_checkpoint(done_user_ids, embeddings, user_ids, post_ids, img_ids):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        CHECKPOINT,
        embeddings=np.stack(embeddings).astype(np.float32),
        user_ids=np.array(user_ids, dtype=np.int32),
        post_ids=np.array(post_ids, dtype=np.int32),
        img_ids=np.array(img_ids, dtype=np.int32),
        done_user_ids=np.array(sorted(done_user_ids), dtype=np.int32),
    )


def main(dry_run=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    user_dirs = sorted(
        [d for d in IMG_DIR.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    print(f"Found {len(user_dirs):,} user directories")

    if dry_run:
        print("DRY RUN — showing first 5 users\n")
        for d in user_dirs[:5]:
            entries = load_user_images(d)
            print(f"  user {d.name}: {len(entries)} images")
            for path, uid, pid, iid in entries[:3]:
                print(f"    {path.name}  →  user={uid} post={pid} img={iid}")
        return

    done_users, embeddings, user_ids, post_ids, img_ids = load_checkpoint()
    remaining = [d for d in user_dirs if int(d.name) not in done_users]
    print(f"To process: {len(remaining):,}  |  Already done: {len(done_users):,}")

    device = get_device()
    print(f"Loading {MODEL_NAME}...")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME, ignore_mismatched_sizes=True).to(device)
    model.eval()

    skipped_users = 0
    skipped_images = 0
    since_checkpoint = 0

    for user_dir in tqdm(remaining, desc="Embedding users", unit="user"):
        entries = load_user_images(user_dir)

        if not entries:
            skipped_users += 1
            done_users.add(int(user_dir.name))
            continue

        pil_images, valid_entries = [], []
        for path, uid, pid, iid in entries:
            try:
                pil_images.append(Image.open(path).convert("RGB"))
                valid_entries.append((uid, pid, iid))
            except (UnidentifiedImageError, OSError):
                skipped_images += 1

        if not pil_images:
            skipped_users += 1
            done_users.add(int(user_dir.name))
            continue

        all_feats = []
        for i in range(0, len(pil_images), BATCH_SIZE):
            all_feats.append(embed_batch(pil_images[i:i + BATCH_SIZE], processor, model, device))

        feats = np.concatenate(all_feats, axis=0)

        for feat, (uid, pid, iid) in zip(feats, valid_entries):
            embeddings.append(feat)
            user_ids.append(uid)
            post_ids.append(pid)
            img_ids.append(iid)

        done_users.add(int(user_dir.name))
        since_checkpoint += 1

        if since_checkpoint >= CHECKPOINT_EVERY:
            save_checkpoint(done_users, embeddings, user_ids, post_ids, img_ids)
            since_checkpoint = 0

    embeddings_matrix = np.stack(embeddings).astype(np.float32)
    np.savez(
        OUT_NPZ,
        embeddings=embeddings_matrix,
        user_ids=np.array(user_ids, dtype=np.int32),
        post_ids=np.array(post_ids, dtype=np.int32),
        img_ids=np.array(img_ids, dtype=np.int32),
    )

    print(f"\nEmbeddings shape: {embeddings_matrix.shape}")
    print(f"Images embedded: {len(embeddings):,}")
    print(f"Users skipped (no images): {skipped_users:,}")
    print(f"Images skipped (corrupt): {skipped_images:,}")
    print(f"Saved to {OUT_NPZ}")

    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print("Checkpoint removed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
