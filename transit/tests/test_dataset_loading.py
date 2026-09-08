"""Tests for MTMCScene's path logic and the ground-truth/calibration JSON parsers,
against synthetic files matching the CONFIRMED nvidia/PhysicalAI-SmartSpaces
(MTMC_Tracking_2025) schema (see mtmc_dataset.py / calibration.py / dataset_export.py
docstrings). No real dataset download is needed -- these write small synthetic
fixture files to pytest's tmp_path.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from transit.data.calibration import load_camera_calibration
from transit.data.mtmc_dataset import MTMCScene
from transit.detection.dataset_export import load_ground_truth_annotations


def _make_scene_dir(tmp_path, camera_ids: list[str]):
    scene_dir = tmp_path / "Warehouse_000"
    videos_dir = scene_dir / "videos"
    videos_dir.mkdir(parents=True)
    for camera_id in camera_ids:
        (videos_dir / f"{camera_id}.mp4").write_bytes(b"")
    return scene_dir


def test_mtmc_scene_camera_ids_from_videos_dir(tmp_path) -> None:
    """camera_ids should list video stems under videos/, sorted, not per-camera subdirs."""
    scene_dir = _make_scene_dir(tmp_path, ["Camera_0001", "Camera_0000"])
    scene = MTMCScene(scene_dir)

    assert scene.camera_ids == ["Camera_0000", "Camera_0001"]


def test_mtmc_scene_camera_ids_empty_when_videos_dir_missing(tmp_path) -> None:
    """No videos/ directory should return an empty list, not raise."""
    scene = MTMCScene(tmp_path / "does_not_exist")

    assert scene.camera_ids == []


def test_mtmc_scene_video_path(tmp_path) -> None:
    """video_path should resolve to videos/<camera_id>.mp4."""
    scene_dir = _make_scene_dir(tmp_path, ["Camera_0000"])
    scene = MTMCScene(scene_dir)

    assert scene.video_path("Camera_0000") == scene_dir / "videos" / "Camera_0000.mp4"


def test_mtmc_scene_video_path_missing_raises(tmp_path) -> None:
    """A camera with no matching video file should raise FileNotFoundError."""
    scene_dir = _make_scene_dir(tmp_path, ["Camera_0000"])
    scene = MTMCScene(scene_dir)

    with pytest.raises(FileNotFoundError):
        scene.video_path("Camera_9999")


def test_mtmc_scene_calibration_and_ground_truth_paths_are_scene_level(tmp_path) -> None:
    """calibration_path/ground_truth_path point at ONE file per scene, not per camera."""
    scene_dir = _make_scene_dir(tmp_path, ["Camera_0000"])
    (scene_dir / "calibration.json").write_text("{}", encoding="utf-8")
    (scene_dir / "ground_truth.json").write_text("{}", encoding="utf-8")
    scene = MTMCScene(scene_dir)

    assert scene.calibration_path() == scene_dir / "calibration.json"
    assert scene.ground_truth_path() == scene_dir / "ground_truth.json"


def test_load_ground_truth_annotations_splits_per_camera(tmp_path) -> None:
    """A scene-level, frame-keyed ground_truth.json should split into per-camera,
    per-frame boxes, keeping only the requested object_type."""
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(
        json.dumps(
            {
                "0": [
                    {
                        "object_type": "Person",
                        "object_id": 7,
                        "2d_bounding_box_visible": {
                            "Camera_0000": [10.0, 20.0, 30.0, 40.0],
                            "Camera_0001": [1.0, 2.0, 3.0, 4.0],
                        },
                    },
                    {
                        # A non-person object (e.g. a robot/vehicle) must be dropped.
                        "object_type": "Forklift",
                        "object_id": 99,
                        "2d_bounding_box_visible": {"Camera_0000": [0.0, 0.0, 1.0, 1.0]},
                    },
                ],
                "5": [
                    {
                        "object_type": "Person",
                        "object_id": 7,
                        # Only visible on Camera_0000 at this frame.
                        "2d_bounding_box_visible": {"Camera_0000": [11.0, 21.0, 31.0, 41.0]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    annotations = load_ground_truth_annotations(gt_path)

    assert set(annotations.keys()) == {"Camera_0000", "Camera_0001"}
    assert set(annotations["Camera_0000"].keys()) == {0, 5}
    assert annotations["Camera_0000"][0][0].object_id == "7"
    assert annotations["Camera_0000"][0][0].bbox == (10.0, 20.0, 30.0, 40.0)
    assert set(annotations["Camera_0001"].keys()) == {0}
    # The Forklift object must not appear under any camera.
    assert all(box.object_id != "99" for boxes in annotations["Camera_0000"].values() for box in boxes)


def test_load_ground_truth_annotations_custom_object_types(tmp_path) -> None:
    """An explicit object_types filter should override the default person-only set."""
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(
        json.dumps(
            {
                "0": [
                    {
                        "object_type": "Forklift",
                        "object_id": 1,
                        "2d_bounding_box_visible": {"Camera_0000": [0.0, 0.0, 1.0, 1.0]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    annotations = load_ground_truth_annotations(gt_path, object_types={"Forklift"})

    assert annotations["Camera_0000"][0][0].object_id == "1"


def test_load_camera_calibration_finds_matching_sensor(tmp_path) -> None:
    """load_camera_calibration should pick the sensor whose id matches camera_id,
    among possibly multiple sensors and non-camera sensor types."""
    scene_dir = tmp_path / "Warehouse_000"
    scene_dir.mkdir()
    (scene_dir / "calibration.json").write_text(
        json.dumps(
            {
                "calibrationType": "cartesian",
                "sensors": [
                    {"type": "lidar", "id": "Lidar_0000"},
                    {
                        "type": "camera",
                        "id": "Camera_0000",
                        "coordinates": {"x": 1.0, "y": 2.0},
                        "intrinsicMatrix": [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
                        "extrinsicMatrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                        "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    },
                    {
                        "type": "camera",
                        "id": "Camera_0001",
                        "intrinsicMatrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "extrinsicMatrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    calibration = load_camera_calibration(scene_dir, "Camera_0000")

    assert calibration.camera_id == "Camera_0000"
    assert np.array_equal(calibration.intrinsic_matrix, np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]]))
    assert calibration.homography is not None


def test_load_camera_calibration_falls_back_to_camera_matrix(tmp_path) -> None:
    """A sensor with cameraMatrix but no extrinsicMatrix should still load."""
    scene_dir = tmp_path / "Warehouse_000"
    scene_dir.mkdir()
    (scene_dir / "calibration.json").write_text(
        json.dumps(
            {
                "sensors": [
                    {
                        "type": "camera",
                        "id": "Camera_0000",
                        "intrinsicMatrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "cameraMatrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    calibration = load_camera_calibration(scene_dir, "Camera_0000")

    assert calibration.extrinsic_matrix.shape == (3, 4)


def test_load_camera_calibration_missing_camera_raises_key_error(tmp_path) -> None:
    """A camera_id with no matching sensor should raise KeyError, not silently return None."""
    scene_dir = tmp_path / "Warehouse_000"
    scene_dir.mkdir()
    (scene_dir / "calibration.json").write_text(json.dumps({"sensors": []}), encoding="utf-8")

    with pytest.raises(KeyError):
        load_camera_calibration(scene_dir, "Camera_0000")


def test_load_camera_calibration_missing_file_raises_file_not_found(tmp_path) -> None:
    """No calibration.json for the scene at all should raise FileNotFoundError."""
    scene_dir = tmp_path / "Warehouse_000"
    scene_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        load_camera_calibration(scene_dir, "Camera_0000")
