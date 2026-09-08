"""Loading of camera calibration data.

Reminder: calibration is an EVALUATION-time oracle baseline only. It must never be
fed into the plausibility gate's training or inference path -- see
gate/calibration_oracle.py for the only sanctioned use of this data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from transit.data.schema import CameraCalibration

logger = logging.getLogger(__name__)


def load_camera_calibration(scene_dir: str | Path, camera_id: str) -> CameraCalibration:
    """Load one camera's calibration from the scene's single calibration.json.

    Confirmed calibration.json structure (nvidia/PhysicalAI-SmartSpaces,
    MTMC_Tracking_2025 -- verified against the dataset's own Hugging Face README,
    Sept 2026) -- ONE file per scene, covering every camera::

        {
          "calibrationType": "cartesian",
          "sensors": [
            {
              "type": "camera",
              "id": "<camera_id>",
              "coordinates": {"x": float, "y": float},
              "intrinsicMatrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
              "extrinsicMatrix": [[...]],
              "cameraMatrix": [[...]],
              "homography": [[...]]
            },
            ...
          ]
        }

    This replaced an earlier provisional per-camera-file guess
    (``<scene_dir>/<camera_id>/calibration.json``) that did not match the real
    dataset -- calibration is one file per SCENE, with a "sensors" list covering
    every camera (and possibly non-camera sensor types, skipped here via
    ``"type": "camera"``). "coordinates" (a top-down x/y position) has no
    equivalent field on `CameraCalibration` and is intentionally not loaded here.

    Args:
        scene_dir: Path to the scene directory (see MTMCScene.calibration_path).
        camera_id: Identifier of the camera whose calibration should be loaded;
            must match a sensor's "id" (e.g. "Camera_0000").

    Returns:
        A populated CameraCalibration for the requested camera.

    Raises:
        FileNotFoundError: If no calibration.json exists for the scene.
        KeyError: If no camera sensor with the given id is found in it, or if a
            matching sensor has neither "extrinsicMatrix" nor "cameraMatrix".
    """
    scene_dir = Path(scene_dir)
    calib_path = scene_dir / "calibration.json"
    if not calib_path.exists():
        raise FileNotFoundError(f"No calibration file found at {calib_path}")

    with calib_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    sensors = raw.get("sensors", [])
    sensor = next(
        (s for s in sensors if s.get("type") == "camera" and str(s.get("id")) == str(camera_id)), None
    )
    if sensor is None:
        available = sorted(str(s.get("id")) for s in sensors if s.get("type") == "camera")
        raise KeyError(f"No camera sensor with id '{camera_id}' in {calib_path}; available: {available}")

    extrinsic = sensor.get("extrinsicMatrix") or sensor.get("cameraMatrix")
    if extrinsic is None:
        raise KeyError(
            f"Camera sensor '{camera_id}' in {calib_path} has neither 'extrinsicMatrix' nor 'cameraMatrix'"
        )

    logger.info("Loaded calibration for camera '%s' from %s", camera_id, calib_path)

    homography = sensor.get("homography")
    return CameraCalibration(
        camera_id=str(sensor.get("id", camera_id)),
        intrinsic_matrix=np.array(sensor["intrinsicMatrix"], dtype=np.float64),
        extrinsic_matrix=np.array(extrinsic, dtype=np.float64),
        homography=np.array(homography, dtype=np.float64) if homography is not None else None,
        translation_to_global=None,
    )
