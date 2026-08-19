"""Per-camera person tracking built on Ultralytics' built-in ByteTrack tracking.

Ultralytics keeps tracker state (e.g. ByteTrack's track buffer) attached to the
YOLO model instance across calls when ``persist=True``. To keep camera tracks from
bleeding into each other, each CameraTracker owns its own model instance rather
than sharing one across cameras.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from transit.config import TransitConfig
from transit.data.schema import Detection, Tracklet

logger = logging.getLogger(__name__)

_COCO_PERSON_CLASS_ID = 0


class CameraTracker:
    """Tracks people across frames for a single camera.

    Attributes:
        config: The TransitConfig this tracker was built from.
        camera_id: Identifier of the camera this tracker is responsible for.
        weights_path: The checkpoint actually loaded (stock or fine-tuned).
    """

    def __init__(
        self,
        config: TransitConfig,
        camera_id: str,
        weights_path: str | Path | None = None,
    ) -> None:
        """Load a dedicated YOLO model for tracking on one camera.

        Args:
            config: TransitConfig specifying the default detector checkpoint,
                tracker config, confidence threshold, and device to run on.
            camera_id: Identifier of the camera this tracker will process.
            weights_path: Optional override for the checkpoint to load, e.g. a
                fine-tuned checkpoint from detection/train.py. Defaults to
                ``config.detector_model`` if not given.
        """
        from ultralytics import YOLO

        self.config = config
        self.camera_id = camera_id
        self.weights_path = str(weights_path) if weights_path is not None else config.detector_model
        logger.info(
            "Initializing tracker for camera '%s' with model '%s' and tracker '%s'",
            camera_id,
            self.weights_path,
            config.tracker_config,
        )
        self._model = YOLO(self.weights_path)
        self._tracklets: dict[int, Tracklet] = {}

    def update(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> list[Detection]:
        """Run tracking on a single new frame and update internal tracklet state.

        Args:
            frame: A single video frame as a BGR numpy array (H, W, 3).
            frame_idx: Index of this frame within the camera's video.
            timestamp: Timestamp (seconds) of this frame.

        Returns:
            The list of Detection objects produced for this frame (also folded
            into this tracker's accumulated Tracklet state).
        """
        results = self._model.track(
            frame,
            classes=[_COCO_PERSON_CLASS_ID],
            conf=self.config.detection_conf_threshold,
            tracker=self.config.tracker_config,
            device=self.config.device,
            persist=True,
            verbose=False,
        )

        frame_detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue
            for xyxy, conf, track_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.id.tolist()):
                detection = Detection(
                    camera_id=self.camera_id,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    confidence=conf,
                )
                frame_detections.append(detection)
                self._add_to_tracklet(int(track_id), detection)

        logger.debug(
            "Camera '%s' frame %d: %d tracked detection(s)", self.camera_id, frame_idx, len(frame_detections)
        )
        return frame_detections

    def get_tracklets(self) -> list[Tracklet]:
        """Return all tracklets accumulated so far for this camera.

        Returns:
            A list of Tracklet objects, one per distinct track_id observed.
        """
        return list(self._tracklets.values())

    def _add_to_tracklet(self, track_id: int, detection: Detection) -> None:
        tracklet = self._tracklets.get(track_id)
        if tracklet is None:
            tracklet = Tracklet(
                track_id=track_id,
                camera_id=self.camera_id,
                detections=[],
                start_frame=detection.frame_idx,
                end_frame=detection.frame_idx,
            )
            self._tracklets[track_id] = tracklet

        tracklet.detections.append(detection)
        tracklet.end_frame = detection.frame_idx
