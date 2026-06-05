# Converts per-image CLIP embeddings into per-user mean-pooled vectors.
# Input:  embeddings_60k.npz  (output of embed_clip_60k.py)
# Output: user_vectors_60k.npz  — needed for rq3_detection.py

from pathlib import Path

import numpy as np

DRIVE      = Path("/Volumes/Samsung PSSD T7 Media/FediData/embeddings")
IN_NPZ     = DRIVE / "embeddings_60k.npz"
OUT_NPZ    = DRIVE / "user_vectors_60k.npz"


def build_user_vectors(embeddings: np.ndarray, user_ids: np.ndarray):
    unique_users = np.unique(user_ids)
    vecs = []
    for uid in unique_users:
        mean_vec = embeddings[user_ids == uid].mean(axis=0)
        vecs.append(mean_vec)
    matrix = np.stack(vecs).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.maximum(norms, 1e-8)
    return matrix, unique_users


def main():
    print("Loading embeddings...")
    data = np.load(IN_NPZ)
    embeddings, user_ids = data["embeddings"], data["user_ids"]
    print(f"  {len(embeddings):,} image embeddings, {len(np.unique(user_ids)):,} unique users")

    print("Mean-pooling per user...")
    user_matrix, unique_users = build_user_vectors(embeddings, user_ids)
    print(f"  user_matrix shape: {user_matrix.shape}")

    np.savez(OUT_NPZ, user_vectors=user_matrix, user_ids=unique_users)
    print(f"Saved to {OUT_NPZ}")


if __name__ == "__main__":
    main()
