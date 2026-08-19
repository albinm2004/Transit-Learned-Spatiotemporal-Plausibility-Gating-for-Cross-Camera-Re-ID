#!/usr/bin/env python
"""CLI to fine-tune the YOLO detector on a previously exported dataset.

Usage:
    python scripts/export_yolo_dataset.py --config configs/default.yaml --scene scene_001
    python scripts/train_yolo.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging

from transit.config import load_config
from transit.detection.train import train_yolo_detector

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the YOLO fine-tuning script."""
    parser = argparse.ArgumentParser(description="Fine-tune the YOLO detector on an exported scene dataset.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to a TransitConfig YAML file.")
    parser.add_argument(
        "--dataset-yaml",
        type=str,
        default=None,
        help="Path to dataset.yaml (defaults to <training.dataset_export_dir>/dataset.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: fine-tune the configured base checkpoint and log the resulting weights path."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    args = parse_args()
    config = load_config(args.config)

    dataset_yaml_path = args.dataset_yaml or (config.training.dataset_export_dir / "dataset.yaml")
    best_weights_path = train_yolo_detector(config, dataset_yaml_path)
    logger.info("Fine-tuned detector weights ready at: %s", best_weights_path)


if __name__ == "__main__":
    main()
