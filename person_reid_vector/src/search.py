from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .embedder import ReIDEmbedder
from .vector_db import VectorDB


def main():
    parser = argparse.ArgumentParser(description="Search the vector DB for the target person.")
    parser.add_argument("--db", default="outputs/vector_db")
    parser.add_argument("--query", required=True, help="Target embedding .npy OR target image path")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=None, help="Optional cosine similarity threshold")
    parser.add_argument("--output", default=None, help="Optional JSON output")
    args = parser.parse_args()

    query_path = Path(args.query)
    if query_path.suffix.lower() == ".npy":
        query = np.load(query_path)
    else:
        query = ReIDEmbedder().embed_image(query_path)

    db = VectorDB(args.db)
    results = db.search(query, args.top_k, args.threshold)
    print(json.dumps(results, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
