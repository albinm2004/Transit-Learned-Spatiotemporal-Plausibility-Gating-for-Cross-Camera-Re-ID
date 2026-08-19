"""Camera field-of-view footprints and per-object per-camera visibility intervals.

These are ground-truth-identity-driven preprocessing steps: given detections
labeled with a shared global object_id (from MTMC ground truth), derive the time
intervals during which each object was visible on each camera. transitions.py then
turns consecutive intervals into TransitionEvents for gate training data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from transit.data.schema import CameraCalibration, Detection

logger = logging.getLogger(__name__)

_DEFAULT_MAX_GAP_SECONDS = 1.0


@dataclass
class VisibilityInterval:
    """A contiguous span of time during which one object was visible on one camera.

    Attributes:
        object_id: Global identity/object id.
        camera_id: Camera the object was visible on.
        start_time: Timestamp (seconds) the interval starts.
        end_time: Timestamp (seconds) the interval ends.
    """

    object_id: str
    camera_id: str
    start_time: float
    end_time: float


def compute_fov_footprint(calibration: CameraCalibration, image_width: int, image_height: int) -> np.ndarray:
    """Project a camera's image-plane extent onto the ground plane via its homography.

    Args:
        calibration: CameraCalibration for the camera, must have `homography` set.
        image_width: Camera image width in pixels.
        image_height: Camera image height in pixels.

    Returns:
        A (4, 2) array of [x, y] ground-plane points, the projection of the
        image's four corners, in the order top-left, top-right, bottom-right,
        bottom-left.

    Raises:
        ValueError: If `calibration.homography` is None.
    """
    if calibration.homography is None:
        raise ValueError(
            f"Camera '{calibration.camera_id}' has no homography; cannot compute an FOV footprint."
        )

    corners = np.array(
        [
            [0.0, 0.0],
            [image_width, 0.0],
            [image_width, image_height],
            [0.0, image_height],
        ],
        dtype=np.float64,
    )
    homogeneous_corners = np.concatenate([corners, np.ones((4, 1))], axis=1)
    projected = homogeneous_corners @ calibration.homography.T
    projected = projected[:, :2] / projected[:, [2]]
    return projected


def compute_visibility_intervals(
    object_detections: dict[str, list[Detection]],
    max_gap_seconds: float = _DEFAULT_MAX_GAP_SECONDS,
) -> list[VisibilityInterval]:
    """Derive per-object, per-camera visibility intervals from detections.

    Groups each object's detections by camera, sorts by timestamp, and merges
    consecutive detections into one interval as long as the gap between them
    does not exceed `max_gap_seconds` (bridging brief detector/tracker misses);
    a larger gap starts a new interval (e.g. the object left and later
    re-entered the same camera's view).

    Args:
        object_detections: Mapping from global object_id to all of that object's
            Detection instances, across all cameras.
        max_gap_seconds: Maximum time gap, in seconds, to bridge within a single
            visibility interval.

    Returns:
        A list of VisibilityInterval, unordered across objects/cameras.

    Raises:
        ValueError: If max_gap_seconds is negative.
    """
    if max_gap_seconds < 0:
        raise ValueError(f"max_gap_seconds must be >= 0, got {max_gap_seconds}")

    intervals: list[VisibilityInterval] = []

    for object_id, detections in object_detections.items():
        by_camera: dict[str, list[Detection]] = {}
        for detection in detections:
            by_camera.setdefault(detection.camera_id, []).append(detection)

        for camera_id, camera_detections in by_camera.items():
            ordered = sorted(camera_detections, key=lambda d: d.timestamp)
            interval_start = ordered[0].timestamp
            interval_end = ordered[0].timestamp

            for detection in ordered[1:]:
                if detection.timestamp - interval_end <= max_gap_seconds:
                    interval_end = detection.timestamp
                else:
                    intervals.append(VisibilityInterval(object_id, camera_id, interval_start, interval_end))
                    interval_start = detection.timestamp
                    interval_end = detection.timestamp

            intervals.append(VisibilityInterval(object_id, camera_id, interval_start, interval_end))

    logger.info("Computed %d visibility interval(s) for %d object(s)", len(intervals), len(object_detections))
    return intervals
