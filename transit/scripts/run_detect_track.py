#!/usr/bin/env python
"""CLI to run per-camera detection + tracking over a scene and save tracklets to disk.

Usage:
    python scripts/run_detect_track.py --config configs/default.yaml --scene scene_001
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from transit.config import load_config
from transit.data.mtmc_dataset import MTMCScene
from transit.tracking.tracker import CameraTracker

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the detect+track script."""
    parser = argparse.ArgumentParser(description="Run detection + tracking on every camera in a scene.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to a TransitConfig YAML file.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene name to process (overrides scene_name in the config file).",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "json"],
        default="parquet",
        help="Output file format for per-camera tracklets.",
    )
    return parser.parse_args()


def track_camera(tracker: CameraTracker, video_path: Path) -> None:
    """Run tracking over every frame of a single camera's video.

    Args:
        tracker: A CameraTracker for the camera that owns this video.
        video_path: Path to the camera's video file.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    with tqdm(total=total_frames or None, desc=f"Tracking {video_path.parent.name}", unit="frame") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_idx / fps
            tracker.update(frame, frame_idx=frame_idx, timestamp=timestamp)
            frame_idx += 1
            pbar.update(1)

    cap.release()


def save_tracklets(tracker: CameraTracker, output_path: Path, fmt: str) -> None:
    """Serialize a camera's tracklets to disk as one flat table of detections.

    Args:
        tracker: The CameraTracker whose accumulated tracklets should be saved.
        output_path: File path (without extension) to write to.
        fmt: Either "parquet" or "json".
    """
    rows = []
    for tracklet in tracker.get_tracklets():
        for detection in tracklet.detections:
            rows.append(
                {
                    "track_id": tracklet.track_id,
                    "camera_id": detection.camera_id,
                    "frame_idx": detection.frame_idx,
                    "timestamp": detection.timestamp,
                    "xmin": detection.bbox[0],
                    "ymin": detection.bbox[1],
                    "xmax": detection.bbox[2],
                    "ymax": detection.bbox[3],
                    "confidence": detection.confidence,
                }
            )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(output_path.with_suffix(".parquet"), index=False)
    else:
        df.to_json(output_path.with_suffix(".json"), orient="records")

    logger.info("Saved %d detections across %d tracklet(s) to %s", len(rows), len(tracker.get_tracklets()), output_path)


def main() -> None:
    """Entry point: run detection + tracking on every camera in the configured scene."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    args = parse_args()
    overrides = {"scene_name": args.scene} if args.scene else None
    config = load_config(args.config, overrides=overrides)

    scene = MTMCScene(config.scene_dir)
    camera_ids = scene.camera_ids
    if not camera_ids:
        logger.warning("No cameras found in scene directory %s", config.scene_dir)
        return

    for camera_id in camera_ids:
        logger.info("Processing camera '%s'", camera_id)
        tracker = CameraTracker(config, camera_id)
        video_path = scene.video_path(camera_id)
        track_camera(tracker, video_path)

        output_path = config.output_dir / config.scene_name / camera_id / "tracklets"
        save_tracklets(tracker, output_path, args.format)


if __name__ == "__main__":
    main()
