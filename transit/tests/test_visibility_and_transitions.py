"""Tests for preprocessing/visibility.py, transitions.py, and camera_pairs.py.

Uses small hand-constructed synthetic detections/calibration -- no real dataset,
video, or downloaded calibration files required.
"""

from __future__ import annotations

import numpy as np

from transit.data.schema import CameraCalibration, Detection
from transit.preprocessing.camera_pairs import (
    TIER_DISTANT,
    TIER_NEIGHBORING,
    TIER_OVERLAPPING,
    classify_camera_pairs,
)
from transit.preprocessing.transitions import compute_camera_pair_stats, derive_transition_events
from transit.preprocessing.visibility import compute_fov_footprint, compute_visibility_intervals


def _detection(camera_id: str, frame_idx: int, timestamp: float) -> Detection:
    return Detection(
        camera_id=camera_id,
        frame_idx=frame_idx,
        timestamp=timestamp,
        bbox=(0.0, 0.0, 10.0, 10.0),
        confidence=1.0,
    )


def test_compute_visibility_intervals_merges_close_detections() -> None:
    """Detections within max_gap_seconds on the same camera should merge into one interval."""
    detections = [
        _detection("cam1", 0, 0.0),
        _detection("cam1", 1, 0.1),
        _detection("cam1", 2, 0.2),
    ]

    intervals = compute_visibility_intervals({"obj1": detections}, max_gap_seconds=0.5)

    assert len(intervals) == 1
    assert intervals[0].camera_id == "cam1"
    assert intervals[0].start_time == 0.0
    assert intervals[0].end_time == 0.2


def test_compute_visibility_intervals_splits_on_large_gap() -> None:
    """A gap larger than max_gap_seconds should start a new interval."""
    detections = [
        _detection("cam1", 0, 0.0),
        _detection("cam1", 1, 0.1),
        _detection("cam1", 100, 20.0),
    ]

    intervals = compute_visibility_intervals({"obj1": detections}, max_gap_seconds=0.5)

    assert len(intervals) == 2
    assert intervals[0].end_time == 0.1
    assert intervals[1].start_time == 20.0


def test_compute_visibility_intervals_separates_by_camera() -> None:
    """Detections from two different cameras should produce separate intervals."""
    detections = [
        _detection("cam1", 0, 0.0),
        _detection("cam2", 0, 5.0),
    ]

    intervals = compute_visibility_intervals({"obj1": detections}, max_gap_seconds=0.5)

    camera_ids = {iv.camera_id for iv in intervals}
    assert camera_ids == {"cam1", "cam2"}


def test_derive_transition_events_between_cameras() -> None:
    """A cross-camera consecutive interval pair should yield one TransitionEvent."""
    detections = [
        _detection("cam1", 0, 0.0),
        _detection("cam1", 1, 1.0),
        _detection("cam2", 2, 3.0),
        _detection("cam2", 3, 4.0),
    ]

    intervals = compute_visibility_intervals({"obj1": detections}, max_gap_seconds=0.5)
    events = derive_transition_events(intervals)

    assert len(events) == 1
    event = events[0]
    assert event.object_id == "obj1"
    assert event.src_camera_id == "cam1"
    assert event.dst_camera_id == "cam2"
    assert event.src_exit_time == 1.0
    assert event.dst_entry_time == 3.0
    assert event.transit_time == 2.0


def test_derive_transition_events_ignores_same_camera_gaps() -> None:
    """Consecutive intervals on the same camera should not produce a transition event."""
    detections = [
        _detection("cam1", 0, 0.0),
        _detection("cam1", 100, 20.0),  # large gap, same camera
    ]

    intervals = compute_visibility_intervals({"obj1": detections}, max_gap_seconds=0.5)
    events = derive_transition_events(intervals)

    assert events == []


def test_compute_camera_pair_stats_aggregates_correctly() -> None:
    """compute_camera_pair_stats should compute correct mean/min/max/count per pair."""
    from transit.data.schema import TransitionEvent

    events = [
        TransitionEvent("obj1", "cam1", "cam2", 0.0, 2.0, 2.0),
        TransitionEvent("obj2", "cam1", "cam2", 0.0, 4.0, 4.0),
    ]

    stats = compute_camera_pair_stats(events)

    pair_stats = stats[("cam1", "cam2")]
    assert pair_stats.count == 2
    assert pair_stats.mean_transit_time == 3.0
    assert pair_stats.min_transit_time == 2.0
    assert pair_stats.max_transit_time == 4.0


def test_compute_fov_footprint_with_identity_homography() -> None:
    """An identity homography should project image corners unchanged."""
    calibration = CameraCalibration(
        camera_id="cam1",
        intrinsic_matrix=np.eye(3),
        extrinsic_matrix=np.eye(4),
        homography=np.eye(3),
    )

    footprint = compute_fov_footprint(calibration, image_width=100, image_height=50)

    expected = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]])
    np.testing.assert_allclose(footprint, expected)


def test_compute_fov_footprint_requires_homography() -> None:
    """Missing homography should raise ValueError."""
    calibration = CameraCalibration(
        camera_id="cam1",
        intrinsic_matrix=np.eye(3),
        extrinsic_matrix=np.eye(4),
        homography=None,
    )

    try:
        compute_fov_footprint(calibration, image_width=100, image_height=50)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_classify_camera_pairs_overlapping() -> None:
    """Two overlapping square footprints should be classified as 'overlapping'."""
    footprints = {
        "cam1": np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]),
        "cam2": np.array([[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]]),
    }

    tiers = classify_camera_pairs(footprints, neighbor_distance_threshold=1.0)
    assert tiers[("cam1", "cam2")] == TIER_OVERLAPPING


def test_classify_camera_pairs_neighboring() -> None:
    """Two non-overlapping but close footprints should be 'neighboring_non_overlapping'."""
    footprints = {
        "cam1": np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]),
        "cam2": np.array([[11.0, 0.0], [21.0, 0.0], [21.0, 10.0], [11.0, 10.0]]),
    }

    tiers = classify_camera_pairs(footprints, neighbor_distance_threshold=5.0)
    assert tiers[("cam1", "cam2")] == TIER_NEIGHBORING


def test_classify_camera_pairs_distant() -> None:
    """Two far-apart footprints should be 'distant_disconnected'."""
    footprints = {
        "cam1": np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]),
        "cam2": np.array([[1000.0, 0.0], [1010.0, 0.0], [1010.0, 10.0], [1000.0, 10.0]]),
    }

    tiers = classify_camera_pairs(footprints, neighbor_distance_threshold=5.0)
    assert tiers[("cam1", "cam2")] == TIER_DISTANT
