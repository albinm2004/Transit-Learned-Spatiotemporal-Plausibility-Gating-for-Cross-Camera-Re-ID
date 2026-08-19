"""Top-k cosine-similarity cross-camera candidate generation.

Turns per-tracklet appearance embeddings into a pool of candidate cross-camera
identity matches, which the baseline matcher (ablation A) or, eventually, the
learned plausibility gate (Part 3) then filters down to accepted matches.
"""

from __future__ import annotations

import logging

import numpy as np

from transit.data.schema import CandidateMatch

logger = logging.getLogger(__name__)

TrackletKey = tuple[str, int]  # (camera_id, track_id)


def generate_candidates(
    tracklet_embeddings: dict[TrackletKey, np.ndarray],
    top_k: int,
) -> list[CandidateMatch]:
    """Generate top-k cross-camera candidate matches by cosine similarity.

    For every tracklet, finds its `top_k` most similar tracklets among *other*
    cameras (same-camera tracklets are never candidates for each other, since a
    person cannot be their own cross-camera re-appearance). Embeddings are
    assumed to already be L2-normalized (see ReidEmbedder.aggregate), so cosine
    similarity reduces to a dot product.

    Args:
        tracklet_embeddings: Mapping from (camera_id, track_id) to that
            tracklet's aggregated (D,) appearance embedding.
        top_k: Number of nearest cross-camera candidates to keep per tracklet.

    Returns:
        A flat list of CandidateMatch objects (gate_score left as None), across
        all tracklets, each with appearance_similarity set.

    Raises:
        ValueError: If top_k is less than 1.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    keys = list(tracklet_embeddings.keys())
    if len(keys) < 2:
        return []

    embedding_matrix = np.stack([tracklet_embeddings[key] for key in keys])
    similarity_matrix = embedding_matrix @ embedding_matrix.T

    camera_ids = np.array([camera_id for camera_id, _ in keys])

    candidates: list[CandidateMatch] = []
    for i, (src_camera_id, src_track_id) in enumerate(keys):
        similarities = similarity_matrix[i].copy()
        same_camera_mask = camera_ids == src_camera_id
        similarities[same_camera_mask] = -np.inf  # exclude self + same-camera tracklets

        n_candidates = min(top_k, int(np.isfinite(similarities).sum()))
        if n_candidates == 0:
            continue

        top_indices = np.argpartition(-similarities, n_candidates - 1)[:n_candidates]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        for j in top_indices:
            dst_camera_id, dst_track_id = keys[j]
            candidates.append(
                CandidateMatch(
                    src_camera_id=src_camera_id,
                    src_track_id=src_track_id,
                    dst_camera_id=dst_camera_id,
                    dst_track_id=dst_track_id,
                    appearance_similarity=float(similarity_matrix[i, j]),
                )
            )

    logger.info("Generated %d candidate match(es) from %d tracklet(s)", len(candidates), len(keys))
    return candidates
