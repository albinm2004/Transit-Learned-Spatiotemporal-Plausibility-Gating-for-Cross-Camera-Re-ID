"""Person detection using an Ultralytics YOLO model."""

from __future__ import annotations

import logging

import numpy as np

from transit.config import TransitConfig
from transit.data.schema import Detection

logger = logging.getLogger(__name__)

_COCO_PERSON_CLASS_ID = 0


class PersonDetector:
    """Wraps an Ultralytics YOLO model to detect people in individual frames.

    Attributes:
        config: The TransitConfig this detector was built from.
    """

    def __init__(self, config: TransitConfig) -> None:
        """Load a YOLO model according to the given config.

        Args:
            config: TransitConfig specifying the detector checkpoint, confidence
                threshold, and device to run on.
        """
        from ultralytics import YOLO

        self.config = config
        logger.info("Loading detector model '%s' on device '%s'", config.detector_model, config.device)
        self._model = YOLO(config.detector_model)

    def detect(
        self,
        frame: np.ndarray,
        camera_id: str = "",
        frame_idx: int = 0,
        timestamp: float = 0.0,
    ) -> list[Detection]:
        """Run person detection on a single frame.

        Args:
            frame: A single video frame as a BGR numpy array (H, W, 3).
            camera_id: Identifier of the camera the frame came from, stored on
                each resulting Detection.
            frame_idx: Index of this frame within the camera's video, stored on
                each resulting Detection.
            timestamp: Timestamp (seconds) of this frame, stored on each
                resulting Detection.

        Returns:
            A list of Detection objects for the "person" class with confidence
            above the configured threshold.
        """
        results = self._model.predict(
            frame,
            classes=[_COCO_PERSON_CLASS_ID],
            conf=self.config.detection_conf_threshold,
            device=self.config.device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for xyxy, conf in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
                detections.append(
                    Detection(
                        camera_id=camera_id,
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        confidence=conf,
                    )
                )

        logger.debug("Detected %d person(s) in frame %d of camera '%s'", len(detections), frame_idx, camera_id)
        return detections
