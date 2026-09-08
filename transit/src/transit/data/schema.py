"""Typed data structures shared across the Transit pipeline.

These dataclasses define the interchange format between pipeline stages:
detection -> tracking -> re-ID -> candidate generation -> (future) gating.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """A single person detection in one frame of one camera's video.

    Attributes:
        camera_id: Identifier of the camera this detection came from.
        frame_idx: Zero-based index of the frame within that camera's video.
        timestamp: Timestamp of the frame, in seconds since the start of the video
            (or since a scene-level epoch, once calibration/sync data is available).
        bbox: Bounding box as [xmin, ymin, xmax, ymax] in pixel coordinates.
        confidence: Detector confidence score for the "person" class, in [0, 1].
    """

    camera_id: str
    frame_idx: int
    timestamp: float
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass
class Tracklet:
    """A per-camera sequence of detections believed to be the same person.

    Attributes:
        track_id: Tracker-assigned identifier, unique within a single camera's
            tracking session (not unique across cameras).
        camera_id: Identifier of the camera this tracklet was produced on.
        detections: Ordered list of Detection objects belonging to this track.
        start_frame: Frame index of the first detection in this tracklet.
        end_frame: Frame index of the last detection in this tracklet.
    """

    track_id: int
    camera_id: str
    detections: list[Detection] = field(default_factory=list)
    start_frame: int = 0
    end_frame: int = 0


@dataclass
class CameraCalibration:
    """Calibration parameters for a single camera.

    Field names follow the NVIDIA PhysicalAI-SmartSpaces (MTMC_Tracking_2025)
    calibration JSON as best understood from public documentation/dataset examples.

    TODO: Verify these field names and shapes against real downloaded calibration
    files once the dataset is available locally -- the MTMC_Tracking_2025 release
    notes describe per-camera calibration JSON but exact key names should be
    confirmed rather than assumed here.

    NOTE: this calibration is used as an evaluation-time oracle baseline only
    (see gate/calibration_oracle.py) and must never be fed into the plausibility
    gate's training or inference path -- the gate is meant to work without it.

    Attributes:
        camera_id: Identifier of the camera.
        intrinsic_matrix: 3x3 camera intrinsic matrix (K).
        extrinsic_matrix: 4x4 (or 3x4) camera extrinsic matrix (world-to-camera).
        homography: 3x3 homography mapping image pixels to a ground-plane/map
            coordinate system, if provided by the dataset.
        translation_to_global: Translation vector (and/or transform) mapping this
            camera's local ground-plane coordinates into the scene's shared global
            coordinate system, if the dataset provides a separate per-camera-to-
            global alignment on top of the extrinsics.
    """

    camera_id: str
    intrinsic_matrix: np.ndarray
    extrinsic_matrix: np.ndarray
    homography: np.ndarray | None = None
    translation_to_global: np.ndarray | None = None


@dataclass
class TransitionEvent:
    """An observed (or candidate) transition of one object between two cameras.

    Produced by preprocessing/transitions.py from per-camera visibility intervals,
    and consumed by gate/features.py to build transition-time-likelihood features.

    Attributes:
        object_id: Global identity/object id the transition belongs to (ground
            truth during preprocessing; a candidate-match id at inference time).
        src_camera_id: Camera the object was last seen on before the transition.
        dst_camera_id: Camera the object was next seen on after the transition.
        src_exit_time: Timestamp (seconds) the object was last seen in src_camera_id.
        dst_entry_time: Timestamp (seconds) the object was first seen in dst_camera_id.
        transit_time: dst_entry_time - src_exit_time, in seconds. May be negative
            for briefly-overlapping cameras.
    """

    object_id: str
    src_camera_id: str
    dst_camera_id: str
    src_exit_time: float
    dst_entry_time: float
    transit_time: float


@dataclass
class CandidateMatch:
    """A candidate cross-camera identity match between two tracklets.

    Attributes:
        src_camera_id: Camera of the source tracklet.
        src_track_id: Track id of the source tracklet.
        dst_camera_id: Camera of the candidate destination tracklet.
        dst_track_id: Track id of the candidate destination tracklet.
        appearance_similarity: Cosine similarity between the two tracklets'
            (aggregated) Re-ID embeddings, in [-1, 1].
        gate_score: Plausibility score assigned by the learned gate, in [0, 1].
            None until the gate (Part 3) has scored this candidate.
    """

    src_camera_id: str
    src_track_id: int
    dst_camera_id: str
    dst_track_id: int
    appearance_similarity: float
    gate_score: float | None = None
