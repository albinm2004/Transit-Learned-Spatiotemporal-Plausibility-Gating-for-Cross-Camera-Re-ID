"""Ranking and matching metrics for cross-camera re-identification.

Designed around a common ranked-candidates representation so the same metric
functions work for the appearance-only baseline matcher's output today, and the
learned gate's (appearance + plausibility) output later -- just by choosing which
CandidateMatch score field to rank by.
"""

from __future__ import annotations

import logging

from transit.data.schema import CandidateMatch

logger = logging.getLogger(__name__)

TrackletKey = tuple[str, int]  # (camera_id, track_id)


def build_ranked_candidates(
    candidates: list[CandidateMatch],
    score_attr: str = "appearance_similarity",
) -> dict[TrackletKey, list[TrackletKey]]:
    """Group candidates by source tracklet and rank destinations by a score field.

    Args:
        candidates: Candidate matches, e.g. from candidate_generator.py's
            `generate_candidates`.
        score_attr: Which CandidateMatch field to rank by -- typically
            "appearance_similarity" for the baseline, or "gate_score" once the
            gate has scored candidates.

    Returns:
        Mapping from source tracklet key (camera_id, track_id) to its candidate
        destination tracklet keys, sorted by descending score.

    Raises:
        ValueError: If any candidate is missing a value for `score_attr` (e.g.
            `gate_score` is None because the gate hasn't scored it yet).
    """
    grouped: dict[TrackletKey, list[CandidateMatch]] = {}
    for candidate in candidates:
        source_key = (candidate.src_camera_id, candidate.src_track_id)
        grouped.setdefault(source_key, []).append(candidate)

    ranked: dict[TrackletKey, list[TrackletKey]] = {}
    for source_key, group in grouped.items():
        scores = [getattr(c, score_attr) for c in group]
        if any(score is None for score in scores):
            raise ValueError(
                f"Cannot rank candidates for {source_key}: some candidates have "
                f"{score_attr}=None. Score every candidate before evaluating."
            )
        ordered = sorted(group, key=lambda c: getattr(c, score_attr), reverse=True)
        ranked[source_key] = [(c.dst_camera_id, c.dst_track_id) for c in ordered]

    return ranked


def rank_k_accuracy(
    ranked_candidates: dict[TrackletKey, list[TrackletKey]],
    ground_truth: dict[TrackletKey, set[TrackletKey]],
    k: int,
) -> float:
    """Fraction of queries whose true match appears in the top-k ranked candidates.

    Args:
        ranked_candidates: Per-query ranked destination candidates, as produced
            by `build_ranked_candidates`.
        ground_truth: Mapping from source tracklet key to the set of destination
            tracklet keys that are true cross-camera matches for it.
        k: Rank cutoff (e.g. 1 for Rank-1, 5 for Rank-5).

    Returns:
        Rank-k accuracy in [0, 1], averaged over queries that have at least one
        ground-truth match. Returns 0.0 if there are no such queries.

    Raises:
        ValueError: If k is less than 1.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    queries_with_gt = [key for key in ground_truth if ground_truth[key]]
    if not queries_with_gt:
        return 0.0

    hits = 0
    for source_key in queries_with_gt:
        top_k = set(ranked_candidates.get(source_key, [])[:k])
        if top_k & ground_truth[source_key]:
            hits += 1

    return hits / len(queries_with_gt)


def mean_average_precision(
    ranked_candidates: dict[TrackletKey, list[TrackletKey]],
    ground_truth: dict[TrackletKey, set[TrackletKey]],
) -> float:
    """Mean Average Precision over ranked candidate lists.

    For each query with at least one ground-truth match, computes average
    precision over its ranked candidate list (precision measured at each rank a
    true match is found), then averages across queries.

    Args:
        ranked_candidates: Per-query ranked destination candidates, as produced
            by `build_ranked_candidates`.
        ground_truth: Mapping from source tracklet key to the set of destination
            tracklet keys that are true cross-camera matches for it.

    Returns:
        mAP in [0, 1]. Returns 0.0 if there are no queries with ground truth.
    """
    average_precisions = []

    for source_key, true_matches in ground_truth.items():
        if not true_matches:
            continue

        ranked = ranked_candidates.get(source_key, [])
        num_hits = 0
        precision_sum = 0.0
        for rank, dst_key in enumerate(ranked, start=1):
            if dst_key in true_matches:
                num_hits += 1
                precision_sum += num_hits / rank

        average_precision = precision_sum / len(true_matches) if num_hits > 0 else 0.0
        average_precisions.append(average_precision)

    if not average_precisions:
        return 0.0
    return sum(average_precisions) / len(average_precisions)


def false_positive_match_rate(
    accepted_matches: list[CandidateMatch],
    ground_truth: dict[TrackletKey, set[TrackletKey]],
) -> float:
    """Fraction of accepted (thresholded) matches that are not true matches.

    Applies to the *decisions* a matcher actually commits to (e.g. baseline
    matcher's or gate's accepted single-best match per query), as opposed to the
    ranking metrics above which look at the full candidate list.

    Args:
        accepted_matches: The matches a matcher decided to accept, e.g. from
            matching/baseline_matcher.py's `match_baseline`.
        ground_truth: Mapping from source tracklet key to the set of destination
            tracklet keys that are true cross-camera matches for it.

    Returns:
        False positive rate in [0, 1]: (# accepted matches not in ground truth)
        / (# accepted matches). Returns 0.0 if there are no accepted matches.
    """
    if not accepted_matches:
        return 0.0

    false_positives = 0
    for match in accepted_matches:
        source_key = (match.src_camera_id, match.src_track_id)
        dst_key = (match.dst_camera_id, match.dst_track_id)
        if dst_key not in ground_truth.get(source_key, set()):
            false_positives += 1

    return false_positives / len(accepted_matches)
