from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .embedder import ReIDEmbedder
from .vector_db import VectorDB


def load_metadata(input_dir: Path, csv_path: Path | None):
    if csv_path:
        rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
        for row in rows:
            row["image_path"] = str((input_dir / row["image_path"]).resolve())
        return rows

    rows = []
    for path in sorted(input_dir.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        camera_id = path.parent.name
        rows.append({"image_path": str(path.resolve()), "camera_id": camera_id})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Embed camera person crops and index them in FAISS.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--db", default="outputs/vector_db")
    parser.add_argument("--metadata-csv", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input)
    rows = load_metadata(input_dir, Path(args.metadata_csv) if args.metadata_csv else None)
    if not rows:
        raise SystemExit("No person crops found.")

    paths = [Path(row["image_path"]) for row in rows]
    embedder = ReIDEmbedder()
    embeddings = embedder.embed_images(paths)
    db = VectorDB(args.db, dimension=embeddings.shape[1])
    db.add(embeddings, rows)
    print(f"Indexed {len(rows)} observations. Database size: {db.size}")
