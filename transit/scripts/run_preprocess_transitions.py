#!/usr/bin/env python
"""CLI to derive visibility intervals, transition events, and camera-pair tiers.

Reads ground-truth annotations for every camera in the scene, builds per-object
per-camera visibility intervals, derives cross-camera transition events and their
per-camera-pair empirical statistics, and (where calibration is available)
classifies camera pairs into overlapping/neighboring/distant tiers.

This is ground-truth-identity-driven preprocessing meant to build the training
data gate/features.py will eventually consume -- it does not run the gate itself.

Usage:
    python scripts/run_preprocess_transitions.py --config configs/default.yaml --scene scene_001
"""

from __future__ import annotations

import argparse
import json
import logging

import cv2
import pandas as pd

from transit.config import load_config
from transit.data.calibration import load_camera_calibration
from transit.data.mtmc_dataset import MTMCScene
from transit.data.schema import Detection
from transit.detection.dataset_export import load_ground_truth_annotations
from transit.preprocessing.camera_pairs import classify_camera_pairs
from transit.preprocessing.transitions import compute_camera_pair_stats, derive_transition_events
from transit.preprocessing.visibility import compute_fov_footprint, compute_visibility_intervals

logger = logging.getLogger(__name__)

_DEFAULT_NEIGHBOR_DISTANCE_THRESHOLD = 10.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the transition-preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Derive visibility intervals, transition events, and camera-pair tiers for a scene."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to a TransitConfig YAML file.")
    parser.add_argument("--scene", type=str, default=None, help="Scene name to process (overrides config).")
    parser.add_argument(
        "--neighbor-distance-threshold",
        type=float,
        default=_DEFAULT_NEIGHBOR_DISTANCE_THRESHOLD,
        help="Ground-plane distance threshold for the neighboring-vs-distant camera-pair tier.",
    )
    return parser.parse_args()


def _load_object_detections(scene: MTMCScene, camera_id: str) -> dict[str, list[Detection]]:
    """Load a camera's ground truth as {object_id: [Detection, ...]}."""
    gt_path = scene.ground_truth_path(camera_id)
    annotations = load_ground_truth_annotations(gt_path)

    video_path = scene.video_path(camera_id)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    object_detections: dict[str, list[Detection]] = {}
    for frame_idx, boxes in annotations.items():
        for box in boxes:
            detection = Detection(
                camera_id=camera_id,
                frame_idx=frame_idx,
                timestamp=frame_idx / fps,
                bbox=box.bbox,
                confidence=1.0,
            )
            object_detections.setdefault(box.object_id, []).append(detection)

    return object_detections


def main() -> None:
    """Entry point: derive and save transition events + camera-pair tiers for the scene."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    args = parse_args()
    overrides = {"scene_name": args.scene} if args.scene else None
    config = load_config(args.config, overrides=overrides)

    scene = MTMCScene(config.scene_dir)
    camera_ids = scene.camera_ids
    if not camera_ids:
        logger.warning("No cameras found in scene directory %s", config.scene_dir)
        return

    merged_object_detections: dict[str, list[Detection]] = {}
    for camera_id in camera_ids:
        for object_id, detections in _load_object_detections(scene, camera_id).items():
            merged_object_detections.setdefault(object_id, []).extend(detections)

    intervals = compute_visibility_intervals(merged_object_detections)
    events = derive_transition_events(intervals)
    camera_pair_stats = compute_camera_pair_stats(events)

    output_dir = config.output_dir / config.scene_name
    output_dir.mkdir(parents=True, exist_ok=True)

    events_df = pd.DataFrame([vars(e) for e in events])
    events_df.to_parquet(output_dir / "transition_events.parquet", index=False)

    stats_summary = {
        f"{src}->{dst}": {
            "count": stats.count,
            "mean_transit_time": stats.mean_transit_time,
            "std_transit_time": stats.std_transit_time,
            "min_transit_time": stats.min_transit_time,
            "max_transit_time": stats.max_transit_time,
        }
        for (src, dst), stats in camera_pair_stats.items()
    }
    (output_dir / "camera_pair_stats.json").write_text(json.dumps(stats_summary, indent=2), encoding="utf-8")

    try:
        footprints = {}
        for camera_id in camera_ids:
            calibration = load_camera_calibration(config.scene_dir, camera_id)
            video_path = scene.video_path(camera_id)
            cap = cv2.VideoCapture(str(video_path))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            footprints[camera_id] = compute_fov_footprint(calibration, width, height)

        tiers = classify_camera_pairs(footprints, args.neighbor_distance_threshold)
        tiers_summary = {f"{a}->{b}": tier for (a, b), tier in tiers.items()}
        (output_dir / "camera_pair_tiers.json").write_text(json.dumps(tiers_summary, indent=2), encoding="utf-8")
    except FileNotFoundError as exc:
        logger.warning("Skipping camera-pair tiering: calibration not available (%s)", exc)

    logger.info(
        "Preprocessed %d visibility interval(s), %d transition event(s), %d camera pair(s) -> %s",
        len(intervals),
        len(events),
        len(camera_pair_stats),
        output_dir,
    )


if __name__ == "__main__":
    main()
