# Embed synthetic IO accounts with CLIP (ViT-B/32), same pipeline as embed_clip_60k.py.
# Mean-pools per-image embeddings to one L2-normalised 512d vector per account.
#
# Output:
#   /Volumes/.../RQ3_Synthetic/embeddings/rq3_synthetic_embeddings.npz
#     arrays: embeddings (N_accounts × 512), user_ids (N_accounts,)

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

IMAGES_DIR   = Path("/Volumes/Samsung PSSD T7 Media/RQ3_Synthetic/images")
OUT_NPZ      = Path("/Volumes/Samsung PSSD T7 Media/RQ3_Synthetic/embeddings/rq3_synthetic_embeddings.npz")
SYNTHETIC_ID_START = 100_000
BATCH_SIZE   = 64


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_account_images(acct_dir):
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in acct_dir.iterdir() if p.suffix.lower() in exts)


def embed_images(paths, model, processor, device):
    embeddings = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[i : i + BATCH_SIZE]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                pass
        if not imgs:
            continue
        inputs = processor(images=imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            pooled = model.vision_model(pixel_values=pixel_values).pooler_output
            feats = model.visual_projection(pooled)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().float().numpy())
    return np.concatenate(embeddings, axis=0) if embeddings else None


def main():
    device = get_device()
    print(f"Device: {device}")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    acct_dirs = sorted(
        d for d in IMAGES_DIR.iterdir()
        if d.is_dir() and d.name != "staging" and int(d.name) >= SYNTHETIC_ID_START
    )
    print(f"Found {len(acct_dirs)} synthetic account directories")

    all_embeddings = []
    all_user_ids   = []
    all_img_ids    = []

    for acct_dir in tqdm(acct_dirs, unit="account"):
        sid = int(acct_dir.name)
        img_paths = load_account_images(acct_dir)
        if not img_paths:
            continue
        per_image = embed_images(img_paths, model, processor, device)
        if per_image is None:
            continue
        for img_idx, vec in enumerate(per_image):
            all_embeddings.append(vec)
            all_user_ids.append(sid)
            all_img_ids.append(img_idx)

    embeddings = np.stack(all_embeddings)
    user_ids   = np.array(all_user_ids)
    img_ids    = np.array(all_img_ids)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, embeddings=embeddings, user_ids=user_ids, img_ids=img_ids)
    print(f"Saved {embeddings.shape} per-image embeddings → {OUT_NPZ}")


if __name__ == "__main__":
    main()
