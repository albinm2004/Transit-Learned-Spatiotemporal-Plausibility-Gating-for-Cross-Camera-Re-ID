"""Frozen pretrained Re-ID appearance embedding extraction via torchreid OSNet.

No fine-tuning code lives here by design (see project scope): this wraps
torchreid's pretrained OSNet FeatureExtractor and runs it frozen, batch-embedding
person crops into L2-normalized appearance vectors for candidate generation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from transit.config import TransitConfig

logger = logging.getLogger(__name__)

_PLACEHOLDER_WEIGHT_KEYS = {"market1501", "msmt17"}


class ReidEmbedder:
    """Wraps a frozen, pretrained torchreid OSNet model for appearance embedding.

    Attributes:
        config: The TransitConfig this embedder was built from.
    """

    def __init__(self, config: TransitConfig) -> None:
        """Load a pretrained OSNet feature extractor according to the given config.

        Args:
            config: TransitConfig specifying `reid_model_name` (torchreid
                architecture, e.g. "osnet_x1_0"), `reid_weights_source` (a local
                checkpoint path or download URL -- NOT a bare dataset name), and
                `device`.

        Raises:
            ValueError: If `config.reid_weights_source` is still a placeholder
                dataset-name key ("market1501"/"msmt17") rather than an actual
                resolvable checkpoint path or URL.
        """
        from torchreid.utils import FeatureExtractor

        self.config = config
        weights_path = self._resolve_weights_path(config.reid_weights_source)

        logger.info(
            "Loading Re-ID model '%s' with weights '%s' on device '%s'",
            config.reid_model_name,
            weights_path,
            config.device,
        )
        self._extractor = FeatureExtractor(
            model_name=config.reid_model_name,
            model_path=weights_path,
            device=config.device,
        )

    @staticmethod
    def _resolve_weights_path(source: str) -> str:
        """Resolve `reid_weights_source` to a concrete checkpoint path/URL.

        TODO(tomorrow): configs/default.yaml ships `reid_weights_source:
        "market1501"` as a human-readable label of which pretrained weights to
        use, not an actual downloadable location -- torchreid does not expose a
        single canonical URL per (model, dataset) pair that's safe to hardcode
        here. Before running the Re-ID stage for real, download the appropriate
        OSNet checkpoint from the deep-person-reid model zoo
        (https://github.com/KaiyangZhou/deep-person-reid) and set
        `reid_weights_source` in the config to that local .pth path.

        Args:
            source: `config.reid_weights_source` value.

        Returns:
            The same string, if it is not a recognized placeholder key.

        Raises:
            ValueError: If `source` is a placeholder key rather than a real path/URL.
        """
        if source in _PLACEHOLDER_WEIGHT_KEYS:
            raise ValueError(
                f"reid_weights_source={source!r} is a placeholder dataset label, not a "
                "checkpoint path or URL. Download the pretrained OSNet checkpoint from the "
                "deep-person-reid model zoo and point reid_weights_source at that local "
                "file (see TODO(tomorrow) in ReidEmbedder._resolve_weights_path)."
            )
        return source

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """Compute L2-normalized appearance embeddings for a batch of person crops.

        Args:
            crops: A list of person-crop images as BGR numpy arrays (H, W, 3),
                e.g. produced by cropping Detection.bbox regions out of a frame.

        Returns:
            An (N, D) float32 array of L2-normalized embeddings, one row per
            input crop, in the same order as `crops`.
        """
        if not crops:
            return np.empty((0, 0), dtype=np.float32)

        features = self._extractor(crops)
        embeddings = features.detach().cpu().numpy().astype(np.float32)
        return _l2_normalize(embeddings)

    @staticmethod
    def aggregate(embeddings: np.ndarray) -> np.ndarray:
        """Aggregate multiple per-detection embeddings into one tracklet embedding.

        Uses mean pooling followed by re-normalization, a standard and robust
        choice for aggregating per-frame Re-ID features into a track-level
        signature.

        Args:
            embeddings: An (N, D) array of per-detection embeddings for a single
                tracklet.

        Returns:
            A single (D,) L2-normalized embedding representing the tracklet.

        Raises:
            ValueError: If `embeddings` is empty.
        """
        if embeddings.size == 0:
            raise ValueError("Cannot aggregate an empty embeddings array.")

        mean_embedding = embeddings.mean(axis=0, keepdims=True)
        return _l2_normalize(mean_embedding)[0]


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a 2D array, guarding against zero-norm rows."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms
