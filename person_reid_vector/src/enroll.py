from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .embedder import ReIDEmbedder


def main():
    parser = argparse.ArgumentParser(description="Create an averaged target-person Re-ID embedding.")
    parser.add_argument("--gallery", required=True, help="Directory containing target-person crops.")
    parser.add_argument("--output", default="outputs/target_embedding.npy")
    args = parser.parse_args()

    gallery = Path(args.gallery)
    paths = sorted(p for p in gallery.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not paths:
        raise SystemExit(f"No images found in {gallery}")

    embedder = ReIDEmbedder()
    embeddings = embedder.embed_images(paths)
    target = embeddings.mean(axis=0)
    target /= np.linalg.norm(target) + 1e-12

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, target.astype(np.float32))
    print(f"Enrolled {len(paths)} images -> {output}")
