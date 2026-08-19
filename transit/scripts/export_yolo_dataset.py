#!/usr/bin/env python
"""CLI to export a scene's ground-truth annotations into a YOLO training dataset.

Usage:
    python scripts/export_yolo_dataset.py --config configs/default.yaml --scene scene_001
"""

from __future__ import annotations

import argparse
import logging

from transit.config import load_config
from transit.detection.dataset_export import export_yolo_dataset

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the YOLO dataset export script."""
    parser = argparse.ArgumentParser(description="Export a scene's ground truth into a YOLO-format dataset.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to a TransitConfig YAML file.")
    parser.add_argument("--scene", type=str, default=None, help="Scene name to export (overrides config).")
    return parser.parse_args()


def main() -> None:
    """Entry point: export the configured scene into config.training.dataset_export_dir."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    args = parse_args()
    overrides = {"scene_name": args.scene} if args.scene else None
    config = load_config(args.config, overrides=overrides)

    dataset_yaml_path = export_yolo_dataset(config)
    logger.info("Dataset export complete: %s", dataset_yaml_path)


if __name__ == "__main__":
    main()
