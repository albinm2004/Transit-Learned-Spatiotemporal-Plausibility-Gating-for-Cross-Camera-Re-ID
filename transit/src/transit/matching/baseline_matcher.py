"""Ablation-A: appearance-only nearest-neighbour cross-camera matching.

This baseline accepts the single highest-similarity candidate above a fixed
threshold for each source tracklet, with no plausibility gating whatsoever. It
exists to quantify how much the learned gate (Part 3) improves over pure
appearance similarity.
"""

from __future__ import annotations

import logging

from transit.data.schema import CandidateMatch

logger = logging.getLogger(__name__)


def match_baseline(candidates: list[CandidateMatch], threshold: float) -> list[CandidateMatch]:
    """Select the best appearance-only match per source tracklet above a threshold.

    For each distinct (src_camera_id, src_track_id), keeps only the candidate
    with the highest `appearance_similarity`, and only if that similarity is at
    or above `threshold`. Ties are broken by first-seen order in `candidates`.

    Args:
        candidates: Candidate matches, e.g. from
            matching/candidate_generator.py's `generate_candidates`.
        threshold: Minimum appearance_similarity required to accept a match.

    Returns:
        A list of accepted CandidateMatch objects, at most one per source tracklet.
    """
    best_by_source: dict[tuple[str, int], CandidateMatch] = {}

    for candidate in candidates:
        if candidate.appearance_similarity < threshold:
            continue

        source_key = (candidate.src_camera_id, candidate.src_track_id)
        current_best = best_by_source.get(source_key)
        if current_best is None or candidate.appearance_similarity > current_best.appearance_similarity:
            best_by_source[source_key] = candidate

    accepted = list(best_by_source.values())
    logger.info(
        "Baseline matcher accepted %d/%d candidate(s) above threshold %.3f",
        len(accepted),
        len(candidates),
        threshold,
    )
    return accepted
