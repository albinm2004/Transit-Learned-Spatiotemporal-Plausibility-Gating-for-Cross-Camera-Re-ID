"""Tests for eval/metrics.py.

Uses small hand-constructed synthetic CandidateMatch lists and ground truth --
no real dataset or trained model required.
"""

from __future__ import annotations

from transit.data.schema import CandidateMatch
from transit.eval.metrics import (
    build_ranked_candidates,
    false_positive_match_rate,
    mean_average_precision,
    rank_k_accuracy,
)


def _candidate(src_track: int, dst_camera: str, dst_track: int, similarity: float) -> CandidateMatch:
    return CandidateMatch(
        src_camera_id="cam1",
        src_track_id=src_track,
        dst_camera_id=dst_camera,
        dst_track_id=dst_track,
        appearance_similarity=similarity,
    )


def test_build_ranked_candidates_sorts_descending() -> None:
    """Candidates for one query should be ranked by descending score."""
    candidates = [
        _candidate(1, "cam2", 1, 0.3),
        _candidate(1, "cam2", 2, 0.9),
        _candidate(1, "cam2", 3, 0.6),
    ]

    ranked = build_ranked_candidates(candidates)

    assert ranked[("cam1", 1)] == [("cam2", 2), ("cam2", 3), ("cam2", 1)]


def test_build_ranked_candidates_raises_on_none_score() -> None:
    """A None gate_score should raise ValueError when ranking by gate_score."""
    candidates = [_candidate(1, "cam2", 1, 0.5)]  # gate_score defaults to None

    try:
        build_ranked_candidates(candidates, score_attr="gate_score")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_rank_k_accuracy_hit_within_top_k() -> None:
    """A true match appearing within the top-k ranked candidates should count as a hit."""
    ranked = {("cam1", 1): [("cam2", 2), ("cam2", 3), ("cam2", 1)]}
    ground_truth = {("cam1", 1): {("cam2", 1)}}

    assert rank_k_accuracy(ranked, ground_truth, k=1) == 0.0
    assert rank_k_accuracy(ranked, ground_truth, k=3) == 1.0


def test_rank_k_accuracy_ignores_queries_without_ground_truth() -> None:
    """Queries with an empty ground-truth set should not affect the denominator."""
    ranked = {("cam1", 1): [("cam2", 1)], ("cam1", 2): [("cam2", 2)]}
    ground_truth = {("cam1", 1): {("cam2", 1)}, ("cam1", 2): set()}

    assert rank_k_accuracy(ranked, ground_truth, k=1) == 1.0


def test_rank_k_accuracy_rejects_invalid_k() -> None:
    """k below 1 should raise ValueError."""
    try:
        rank_k_accuracy({}, {}, k=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_mean_average_precision_perfect_ranking() -> None:
    """A perfect ranking (true match first) should give AP = 1.0."""
    ranked = {("cam1", 1): [("cam2", 1), ("cam2", 2)]}
    ground_truth = {("cam1", 1): {("cam2", 1)}}

    assert mean_average_precision(ranked, ground_truth) == 1.0


def test_mean_average_precision_imperfect_ranking() -> None:
    """A true match at rank 2 of 2 should give AP = 0.5."""
    ranked = {("cam1", 1): [("cam2", 2), ("cam2", 1)]}
    ground_truth = {("cam1", 1): {("cam2", 1)}}

    assert mean_average_precision(ranked, ground_truth) == 0.5


def test_mean_average_precision_no_hit_is_zero() -> None:
    """A ranking with no true match present should give AP = 0.0."""
    ranked = {("cam1", 1): [("cam2", 2), ("cam2", 3)]}
    ground_truth = {("cam1", 1): {("cam2", 1)}}

    assert mean_average_precision(ranked, ground_truth) == 0.0


def test_mean_average_precision_empty_ground_truth_returns_zero() -> None:
    """No queries with ground truth should give mAP = 0.0."""
    assert mean_average_precision({}, {}) == 0.0


def test_false_positive_match_rate_all_correct() -> None:
    """All accepted matches being true matches should give FP rate 0.0."""
    accepted = [_candidate(1, "cam2", 1, 0.9)]
    ground_truth = {("cam1", 1): {("cam2", 1)}}

    assert false_positive_match_rate(accepted, ground_truth) == 0.0


def test_false_positive_match_rate_all_wrong() -> None:
    """No accepted matches being true matches should give FP rate 1.0."""
    accepted = [_candidate(1, "cam2", 2, 0.9)]
    ground_truth = {("cam1", 1): {("cam2", 1)}}

    assert false_positive_match_rate(accepted, ground_truth) == 1.0


def test_false_positive_match_rate_mixed() -> None:
    """A 50/50 mix of correct and incorrect accepted matches should give FP rate 0.5."""
    accepted = [_candidate(1, "cam2", 1, 0.9), _candidate(2, "cam2", 5, 0.8)]
    ground_truth = {("cam1", 1): {("cam2", 1)}, ("cam1", 2): {("cam2", 2)}}

    assert false_positive_match_rate(accepted, ground_truth) == 0.5


def test_false_positive_match_rate_empty_accepted_is_zero() -> None:
    """No accepted matches at all should give FP rate 0.0 (not undefined)."""
    assert false_positive_match_rate([], {("cam1", 1): {("cam2", 1)}}) == 0.0
