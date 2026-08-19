#!/usr/bin/env python
"""CLI to compute per-tracklet Re-ID appearance embeddings from saved tracklets.

Reads the per-camera tracklet tables written by run_detect_track.py, crops a
sample of each tracklet's detections out of the source video, embeds them with
ReidEmbedder, aggregates to one embedding per tracklet, and saves the result.

Usage:
    python scripts/run_embed.py --config configs/default.yaml --scene scene_001
"""

from __future__ import annotations

import argparse
import logging

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from transit.config import load_config
from transit.data.mtmc_dataset import MTMCScene
from transit.reid.embedder import ReidEmbedder

logger = logging.getLogger(__name__)

_MAX_CROPS_PER_TRACKLET = 8


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the embedding script."""
    parser = argparse.ArgumentParser(description="Compute Re-ID embeddings for every tracklet in a scene.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to a TransitConfig YAML file.")
    parser.add_argument("--scene", type=str, default=None, help="Scene name to process (overrides config).")
    parser.add_argument(
        "--max-crops-per-tracklet",
        type=int,
        default=_MAX_CROPS_PER_TRACKLET,
        help="Number of evenly-spaced detections to sample per tracklet for embedding.",
    )
    return parser.parse_args()


def _sample_rows(group: pd.DataFrame, max_crops: int) -> pd.DataFrame:
    """Evenly sample up to `max_crops` rows from a tracklet's detection rows."""
    if len(group) <= max_crops:
        return group
    indices = np.linspace(0, len(group) - 1, num=max_crops, dtype=int)
    return group.iloc[indices]


def _crop_detections(video_path, rows: pd.DataFrame) -> list[np.ndarray]:
    """Extract pixel crops for a set of detection rows from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    crops = []
    for _, row in rows.iterrows():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_idx"]))
        ok, frame = cap.read()
        if not ok:
            continue
        xmin, ymin, xmax, ymax = (int(row[c]) for c in ("xmin", "ymin", "xmax", "ymax"))
        crop = frame[max(0, ymin) : max(0, ymax), max(0, xmin) : max(0, xmax)]
        if crop.size > 0:
            crops.append(crop)

    cap.release()
    return crops


def main() -> None:
    """Entry point: embed every tracklet in the configured scene and save per camera."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    args = parse_args()
    overrides = {"scene_name": args.scene} if args.scene else None
    config = load_config(args.config, overrides=overrides)

    scene = MTMCScene(config.scene_dir)
    embedder = ReidEmbedder(config)

    for camera_id in scene.camera_ids:
        tracklets_path = config.output_dir / config.scene_name / camera_id / "tracklets.parquet"
        if not tracklets_path.exists():
            logger.warning("No tracklets found for camera '%s' at %s; skipping.", camera_id, tracklets_path)
            continue

        detections_df = pd.read_parquet(tracklets_path)
        video_path = scene.video_path(camera_id)

        track_ids: list[int] = []
        embeddings: list[np.ndarray] = []

        for track_id, group in tqdm(detections_df.groupby("track_id"), desc=f"Embedding {camera_id}"):
            sampled = _sample_rows(group, args.max_crops_per_tracklet)
            crops = _crop_detections(video_path, sampled)
            if not crops:
                continue

            per_crop_embeddings = embedder.embed(crops)
            track_ids.append(int(track_id))
            embeddings.append(embedder.aggregate(per_crop_embeddings))

        output_path = config.output_dir / config.scene_name / camera_id / "embeddings.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            track_ids=np.array(track_ids, dtype=np.int64),
            embeddings=np.stack(embeddings) if embeddings else np.empty((0, 0), dtype=np.float32),
        )
        logger.info("Saved %d tracklet embedding(s) for camera '%s' to %s", len(track_ids), camera_id, output_path)


if __name__ == "__main__":
    main()
