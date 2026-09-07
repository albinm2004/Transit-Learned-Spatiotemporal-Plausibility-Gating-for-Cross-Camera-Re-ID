from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .embedder import ReIDEmbedder
from .vector_db import VectorDB


class CrossCameraReID:
    """Integration API intended to be called by the YOLO/tracking component."""

    def __init__(self, db_dir: str = "outputs/live_vector_db", device: str = "auto"):
        self.embedder = ReIDEmbedder(device=device)
        self.db = VectorDB(db_dir)

    def add_observation(self, image_path: str | Path, metadata: dict[str, Any]) -> np.ndarray:
        """Embed one YOLO person crop and add it to the vector database."""
        vector = self.embedder.embed_image(image_path)
        self.db.add(vector.reshape(1, -1), [metadata | {"image_path": str(image_path)}])
        return vector

    def identify(self, image_path: str | Path, top_k: int = 5, threshold: float = 0.55):
        """Find the closest known observations for a YOLO person crop."""
        vector = self.embedder.embed_image(image_path)
        return self.db.search(vector, top_k=top_k, min_similarity=threshold)

    def identify_vector(self, vector: np.ndarray, top_k: int = 5, threshold: float = 0.55):
        return self.db.search(vector, top_k=top_k, min_similarity=threshold)
