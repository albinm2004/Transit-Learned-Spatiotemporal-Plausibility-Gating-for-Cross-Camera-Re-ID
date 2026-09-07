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
from pathlib import Path

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


def _crop_all_tracklets(video_path, sampled_by_track: dict[int, pd.DataFrame]) -> dict[int, list[np.ndarray]]:
    """Extract pixel crops for every sampled row across all tracklets in one video pass.

    Seeking per-detection via `cv2.CAP_PROP_POS_FRAMES` (as an earlier version of
    this function did, once per tracklet) is unreliable on compressed/long-GOP
    video -- a seek can land on the nearest keyframe rather than the exact frame
    requested, silently producing the wrong crop -- and it reopens the video file
    once per tracklet, which is also slow. Instead this makes a single forward
    (`cap.read()`) pass over the video and crops every wanted detection, from any
    tracklet, as its frame is reached -- the same reliable pattern already used by
    `detection/dataset_export.py`'s `_export_camera_frames`.

    Args:
        video_path: Path to the camera's source video.
        sampled_by_track: track_id -> DataFrame of sampled detection rows to crop
            (as produced by `_sample_rows`).

    Returns:
        track_id -> list of pixel crops, in the same row order as the input
        DataFrame for that track. Tracks with no successfully-read crop map to
        an empty list.
    """
    # frame_idx -> list of (track_id, row_position, xmin, ymin, xmax, ymax) wanted from that frame.
    wanted: dict[int, list[tuple[int, int, int, int, int, int]]] = {}
    for track_id, rows in sampled_by_track.items():
        for row_position, (_, row) in enumerate(rows.iterrows()):
            frame_idx = int(row["frame_idx"])
            xmin, ymin, xmax, ymax = (int(row[c]) for c in ("xmin", "ymin", "xmax", "ymax"))
            wanted.setdefault(frame_idx, []).append((track_id, row_position, xmin, ymin, xmax, ymax))

    crops_by_track: dict[int, dict[int, np.ndarray]] = {track_id: {} for track_id in sampled_by_track}
    if not wanted:
        return {track_id: [] for track_id in sampled_by_track}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    remaining = dict(wanted)
    frame_idx = 0
    with tqdm(total=len(wanted), desc=f"Cropping {Path(video_path).name}", unit="frame") as pbar:
        while remaining:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in remaining:
                for track_id, row_position, xmin, ymin, xmax, ymax in remaining[frame_idx]:
                    crop = frame[max(0, ymin) : max(0, ymax), max(0, xmin) : max(0, xmax)]
                    if crop.size > 0:
                        crops_by_track[track_id][row_position] = crop
                del remaining[frame_idx]
                pbar.update(1)
            frame_idx += 1
    cap.release()

    if remaining:
        logger.warning(
            "%d wanted frame(s) were never reached in %s (video shorter than tracklets?): %s",
            len(remaining),
            video_path,
            sorted(remaining),
        )

    return {
        track_id: [crops_by_track[track_id][pos] for pos in sorted(crops_by_track[track_id])]
        for track_id in sampled_by_track
    }


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

        sampled_by_track: dict[int, pd.DataFrame] = {
            int(track_id): _sample_rows(group, args.max_crops_per_tracklet)
            for track_id, group in detections_df.groupby("track_id")
        }
        crops_by_track = _crop_all_tracklets(video_path, sampled_by_track)

        track_ids: list[int] = []
        embeddings: list[np.ndarray] = []
        for track_id, crops in tqdm(crops_by_track.items(), desc=f"Embedding {camera_id}"):
            if not crops:
                continue
            per_crop_embeddings = embedder.embed(crops)
            track_ids.append(track_id)
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
