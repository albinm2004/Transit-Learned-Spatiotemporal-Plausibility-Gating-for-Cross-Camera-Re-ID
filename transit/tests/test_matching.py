"""Tests for matching/candidate_generator.py and matching/baseline_matcher.py.

Uses small hand-constructed synthetic embeddings -- no real Re-ID model needed.
"""

from __future__ import annotations

import numpy as np

from transit.matching.baseline_matcher import match_baseline
from transit.matching.candidate_generator import generate_candidates


def test_generate_candidates_excludes_same_camera() -> None:
    """Candidates should never pair a tracklet with another tracklet on the same camera."""
    embeddings = {
        ("cam1", 1): np.array([1.0, 0.0]),
        ("cam1", 2): np.array([1.0, 0.0]),  # identical, same camera as (cam1, 1)
        ("cam2", 1): np.array([0.0, 1.0]),
    }

    candidates = generate_candidates(embeddings, top_k=5)

    for candidate in candidates:
        assert candidate.src_camera_id != candidate.dst_camera_id


def test_generate_candidates_ranks_by_similarity() -> None:
    """The top-k candidates for a query should be its most cosine-similar cross-camera tracklets."""
    embeddings = {
        ("cam1", 1): np.array([1.0, 0.0]),
        ("cam2", 1): np.array([1.0, 0.0]),  # identical -> similarity 1.0
        ("cam2", 2): np.array([0.0, 1.0]),  # orthogonal -> similarity 0.0
    }

    candidates = generate_candidates(embeddings, top_k=5)
    query_candidates = [c for c in candidates if (c.src_camera_id, c.src_track_id) == ("cam1", 1)]

    assert query_candidates[0].dst_camera_id == "cam2"
    assert query_candidates[0].dst_track_id == 1
    assert query_candidates[0].appearance_similarity > query_candidates[1].appearance_similarity


def test_generate_candidates_respects_top_k() -> None:
    """No query should get more than top_k candidates."""
    embeddings = {("cam1", 1): np.array([1.0, 0.0])}
    embeddings.update({("cam2", i): np.array([1.0, 0.0]) for i in range(5)})

    candidates = generate_candidates(embeddings, top_k=2)
    query_candidates = [c for c in candidates if (c.src_camera_id, c.src_track_id) == ("cam1", 1)]

    assert len(query_candidates) == 2


def test_generate_candidates_rejects_invalid_top_k() -> None:
    """top_k below 1 should raise ValueError."""
    try:
        generate_candidates({("cam1", 1): np.array([1.0, 0.0])}, top_k=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_generate_candidates_handles_single_tracklet() -> None:
    """A single tracklet has no possible cross-camera candidates."""
    embeddings = {("cam1", 1): np.array([1.0, 0.0])}
    assert generate_candidates(embeddings, top_k=5) == []


def test_match_baseline_accepts_above_threshold() -> None:
    """match_baseline should accept the best candidate per source when above threshold."""
    embeddings = {
        ("cam1", 1): np.array([1.0, 0.0]),
        ("cam2", 1): np.array([1.0, 0.0]),
        ("cam2", 2): np.array([0.0, 1.0]),
    }
    candidates = generate_candidates(embeddings, top_k=5)

    accepted = match_baseline(candidates, threshold=0.5)
    accepted_by_source = {(c.src_camera_id, c.src_track_id): c for c in accepted}

    # (cam1, 1)'s best cross-camera match is (cam2, 1); (cam2, 2) never crosses
    # the threshold against (cam1, 1) so only one match should exist per source.
    match = accepted_by_source[("cam1", 1)]
    assert match.dst_camera_id == "cam2"
    assert match.dst_track_id == 1
    assert all(c.appearance_similarity >= 0.5 for c in accepted)


def test_match_baseline_rejects_below_threshold() -> None:
    """match_baseline should reject every candidate when none exceeds the threshold."""
    embeddings = {
        ("cam1", 1): np.array([1.0, 0.0]),
        ("cam2", 1): np.array([0.0, 1.0]),  # orthogonal -> similarity 0.0
    }
    candidates = generate_candidates(embeddings, top_k=5)

    accepted = match_baseline(candidates, threshold=0.9)
    assert accepted == []


def test_match_baseline_at_most_one_match_per_source() -> None:
    """match_baseline should keep only the single best match per source tracklet."""
    embeddings = {
        ("cam1", 1): np.array([1.0, 0.1]),
        ("cam2", 1): np.array([1.0, 0.0]),
        ("cam2", 2): np.array([1.0, 0.2]),
    }
    candidates = generate_candidates(embeddings, top_k=5)

    accepted = match_baseline(candidates, threshold=0.0)
    source_keys = [(c.src_camera_id, c.src_track_id) for c in accepted]
    assert len(source_keys) == len(set(source_keys))
