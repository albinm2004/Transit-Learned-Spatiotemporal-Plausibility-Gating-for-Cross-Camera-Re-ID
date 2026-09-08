"""Scene-level access to a multi-target multi-camera (MTMC) dataset directory.

This module only handles locating cameras, videos, calibration, and ground-truth
files within a scene -- path-handling and iteration logic only. It is intentionally
importable and testable without any real scene directory existing on disk; methods
that need the directory to exist raise a clear FileNotFoundError at call time
instead of at construction time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_VIDEO_SUBDIR = "videos"
_VIDEO_EXTENSIONS = (".mp4",)


@dataclass
class MTMCScene:
    """Provides access to the cameras and files of a single MTMC scene.

    Confirmed layout (nvidia/PhysicalAI-SmartSpaces, MTMC_Tracking_2025 -- verified
    against the dataset's own Hugging Face README and file listing, Sept 2026)::

        <scene_dir>/videos/<camera_id>.mp4   # one file per camera, e.g. Camera_0000.mp4
        <scene_dir>/calibration.json         # ONE file for the WHOLE scene, all cameras
        <scene_dir>/ground_truth.json        # ONE file for the WHOLE scene, all cameras
        <scene_dir>/map.png                  # top-down visualization (unused here)
        <scene_dir>/depth_maps/              # per-camera depth (unused here)

    This replaced an earlier provisional per-camera-subdirectory guess
    (``<scene_dir>/<camera_id>/{video.mp4,calibration.json,ground_truth.json}``)
    that did not match the real dataset -- calibration and ground truth are each a
    single scene-level file, not one per camera, so ``calibration_path()`` and
    ``ground_truth_path()`` below take no ``camera_id`` argument.

    Note ``scene_dir`` itself sits one level below the dataset's train/val split,
    e.g. ``<dataset_root>/MTMC_Tracking_2025/train/Warehouse_000`` -- point
    ``TransitConfig.dataset_root`` at ``.../MTMC_Tracking_2025/train`` (or
    ``.../val``) accordingly, not at ``MTMC_Tracking_2025`` itself.

    Attributes:
        scene_dir: Path to the scene's root directory. Does not need to exist at
            construction time -- only when a method that reads the directory
            (e.g. camera_ids, video_path) is actually called.
    """

    scene_dir: Path

    def __post_init__(self) -> None:
        self.scene_dir = Path(self.scene_dir)

    @property
    def camera_ids(self) -> list[str]:
        """List camera identifiers available in this scene, sorted for determinism.

        Derived from the video filenames in ``<scene_dir>/videos/`` (e.g.
        "Camera_0000.mp4" -> "Camera_0000"). Returns an empty list if the videos/
        directory does not exist.
        """
        videos_dir = self.scene_dir / _VIDEO_SUBDIR
        if not videos_dir.exists():
            logger.warning("Videos directory does not exist: %s", videos_dir)
            return []

        cameras = sorted(
            entry.stem
            for entry in videos_dir.iterdir()
            if entry.is_file() and entry.suffix.lower() in _VIDEO_EXTENSIONS
        )
        logger.info("Found %d camera(s) in scene %s: %s", len(cameras), self.scene_dir, cameras)
        return cameras

    def video_path(self, camera_id: str) -> Path:
        """Return the path to a camera's video file.

        Args:
            camera_id: Camera identifier, e.g. "Camera_0000".

        Returns:
            Path to ``<scene_dir>/videos/<camera_id>.mp4``.

        Raises:
            FileNotFoundError: If the video file is missing.
        """
        video_path = self.scene_dir / _VIDEO_SUBDIR / f"{camera_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"No video file found for camera '{camera_id}' at {video_path}")
        return video_path

    def calibration_path(self) -> Path:
        """Return the path to the scene's single calibration.json (all cameras).

        Returns:
            Path to ``<scene_dir>/calibration.json``.

        Raises:
            FileNotFoundError: If the file is missing.
        """
        calib_path = self.scene_dir / "calibration.json"
        if not calib_path.exists():
            raise FileNotFoundError(f"No calibration file found at {calib_path}")
        return calib_path

    def ground_truth_path(self) -> Path:
        """Return the path to the scene's single ground_truth.json (all cameras).

        Returns:
            Path to ``<scene_dir>/ground_truth.json``.

        Raises:
            FileNotFoundError: If the file is missing.
        """
        gt_path = self.scene_dir / "ground_truth.json"
        if not gt_path.exists():
            raise FileNotFoundError(f"No ground-truth annotation file found at {gt_path}")
        return gt_path
