from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class VectorDB:
    """Persistent FAISS cosine-similarity index with JSON metadata."""

    def __init__(self, directory: str | Path, dimension: int = 512):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "index.faiss"
        self.metadata_path = self.directory / "metadata.json"
        self.dimension = dimension
        self.metadata: list[dict[str, Any]] = []

        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        else:
            self.index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected embeddings shape (N, {self.dimension}), got {embeddings.shape}")
        if len(metadata) != len(embeddings):
            raise ValueError("metadata length must equal number of embeddings")
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        self.save()

    def search(self, query: np.ndarray, top_k: int = 10, min_similarity: float | None = None):
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise ValueError(f"Expected query dimension {self.dimension}, got {query.shape[1]}")
        faiss.normalize_L2(query)
        k = min(top_k, self.index.ntotal)
        if k == 0:
            return []
        scores, ids = self.index.search(query, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            if min_similarity is not None and float(score) < min_similarity:
                continue
            item = dict(self.metadata[int(idx)])
            item["similarity"] = float(score)
            item["rank"] = len(results) + 1
            results.append(item)
        return results

    def save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    @property
    def size(self) -> int:
        return self.index.ntotal
