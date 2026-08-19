#!/usr/bin/env python
"""CLI to run the appearance-only baseline matcher (ablation A) over a scene.

Reads per-camera tracklet embeddings (from run_embed.py), generates top-k
cross-camera candidates, applies the threshold-based baseline matcher, and saves
both the full candidate pool and the accepted matches.

Usage:
    python scripts/run_baseline.py --config configs/default.yaml --scene scene_001
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from transit.config import load_config
from transit.data.mtmc_dataset import MTMCScene
from transit.matching.baseline_matcher import match_baseline
from transit.matching.candidate_generator import generate_candidates

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the baseline matching script."""
    parser = argparse.ArgumentParser(description="Run the appearance-only baseline cross-camera matcher.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to a TransitConfig YAML file.")
    parser.add_argument("--scene", type=str, default=None, help="Scene name to process (overrides config).")
    return parser.parse_args()


def _load_tracklet_embeddings(config, scene: MTMCScene) -> dict[tuple[str, int], np.ndarray]:
    """Load every camera's saved tracklet embeddings into one combined mapping."""
    tracklet_embeddings: dict[tuple[str, int], np.ndarray] = {}

    for camera_id in scene.camera_ids:
        embeddings_path = config.output_dir / config.scene_name / camera_id / "embeddings.npz"
        if not embeddings_path.exists():
            logger.warning("No embeddings found for camera '%s' at %s; skipping.", camera_id, embeddings_path)
            continue

        data = np.load(embeddings_path)
        for track_id, embedding in zip(data["track_ids"], data["embeddings"]):
            tracklet_embeddings[(camera_id, int(track_id))] = embedding

    return tracklet_embeddings


def main() -> None:
    """Entry point: run candidate generation + baseline matching over the configured scene."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    args = parse_args()
    overrides = {"scene_name": args.scene} if args.scene else None
    config = load_config(args.config, overrides=overrides)

    scene = MTMCScene(config.scene_dir)
    tracklet_embeddings = _load_tracklet_embeddings(config, scene)
    if not tracklet_embeddings:
        logger.warning("No tracklet embeddings found; nothing to match.")
        return

    candidates = generate_candidates(tracklet_embeddings, top_k=config.candidate_top_k)
    accepted = match_baseline(candidates, threshold=config.baseline_similarity_threshold)

    output_dir = config.output_dir / config.scene_name
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([vars(c) for c in candidates]).to_parquet(output_dir / "candidates.parquet", index=False)
    pd.DataFrame([vars(c) for c in accepted]).to_parquet(output_dir / "baseline_matches.parquet", index=False)

    logger.info(
        "Generated %d candidate(s), baseline matcher accepted %d match(es) -> %s",
        len(candidates),
        len(accepted),
        output_dir,
    )


if __name__ == "__main__":
    main()
