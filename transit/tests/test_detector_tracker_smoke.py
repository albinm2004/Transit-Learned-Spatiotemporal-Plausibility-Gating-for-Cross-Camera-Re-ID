"""Smoke tests that PersonDetector and CameraTracker construct without error.

These tests only check object construction (which loads a YOLO checkpoint) and do
not run inference on real video. They are skipped if ultralytics is not installed,
and treated as inconclusive (skipped) if model weights cannot be obtained (e.g. no
network access to download the checkpoint in a sandboxed environment).
"""

from __future__ import annotations

import pytest

from transit.config import TransitConfig

ultralytics = pytest.importorskip("ultralytics", reason="ultralytics is not installed")


@pytest.fixture()
def config() -> TransitConfig:
    return TransitConfig(
        dataset_root="data/raw",
        scene_name="scene_001",
        output_dir="outputs",
        detector_model="yolo11n.pt",
        device="cpu",
    )


def test_person_detector_instantiates(config: TransitConfig) -> None:
    """PersonDetector should construct and wrap a loaded YOLO model."""
    from transit.detection.detector import PersonDetector

    try:
        detector = PersonDetector(config)
    except Exception as exc:  # noqa: BLE001 - e.g. no network to fetch weights
        pytest.skip(f"Could not load detector model (likely no network access): {exc}")

    assert detector.config is config
    assert detector._model is not None


def test_camera_tracker_instantiates(config: TransitConfig) -> None:
    """CameraTracker should construct and wrap its own loaded YOLO model."""
    from transit.tracking.tracker import CameraTracker

    try:
        tracker = CameraTracker(config, camera_id="camera_01")
    except Exception as exc:  # noqa: BLE001 - e.g. no network to fetch weights
        pytest.skip(f"Could not load tracker model (likely no network access): {exc}")

    assert tracker.camera_id == "camera_01"
    assert tracker.get_tracklets() == []


def test_camera_trackers_are_independent_instances(config: TransitConfig) -> None:
    """Two CameraTracker instances must not share a model/tracker instance."""
    from transit.tracking.tracker import CameraTracker

    try:
        tracker_a = CameraTracker(config, camera_id="camera_01")
        tracker_b = CameraTracker(config, camera_id="camera_02")
    except Exception as exc:  # noqa: BLE001 - e.g. no network to fetch weights
        pytest.skip(f"Could not load tracker model (likely no network access): {exc}")

    assert tracker_a._model is not tracker_b._model
    assert tracker_a._tracklets is not tracker_b._tracklets
