"""Fine-tune the YOLO detector on an exported scene-specific dataset.

Unlike the Re-ID and gate stages, detector fine-tuning is fully implemented and
meant to actually run once a scene has been downloaded and exported to YOLO format
via detection/dataset_export.py -- this is a well-scoped, mature-tooling
supervised fine-tune (Ultralytics' own `model.train()`), not a research unknown.
"""

from __future__ import annotations

import logging
from pathlib import Path

from transit.config import TransitConfig

logger = logging.getLogger(__name__)


def train_yolo_detector(config: TransitConfig, dataset_yaml_path: str | Path) -> Path:
    """Fine-tune the configured base YOLO checkpoint on an exported dataset.

    Wraps Ultralytics' ``model.train()``. Warehouse-specific imagery (unusual
    camera angles, occlusion from shelving/forklifts, uniforms) benefits from a
    real fine-tune of the stock COCO-pretrained detector rather than using it
    zero-shot.

    Args:
        config: TransitConfig; uses `config.resolved_base_checkpoint()` as the
            starting checkpoint and `config.training` for epochs/imgsz/batch/
            output_dir/device.
        dataset_yaml_path: Path to the dataset.yaml produced by
            detection/dataset_export.py's `export_yolo_dataset`.

    Returns:
        Path to the best fine-tuned checkpoint (``<run_dir>/weights/best.pt``).

    Raises:
        FileNotFoundError: If dataset_yaml_path does not exist.
    """
    from ultralytics import YOLO

    dataset_yaml_path = Path(dataset_yaml_path)
    if not dataset_yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset yaml not found at {dataset_yaml_path}; run detection/dataset_export.py first."
        )

    base_checkpoint = config.resolved_base_checkpoint()
    training = config.training
    training.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Fine-tuning '%s' on %s for %d epochs (imgsz=%d, batch=%d, device=%s)",
        base_checkpoint,
        dataset_yaml_path,
        training.epochs,
        training.imgsz,
        training.batch,
        config.device,
    )

    model = YOLO(base_checkpoint)
    model.train(
        data=str(dataset_yaml_path),
        epochs=training.epochs,
        imgsz=training.imgsz,
        batch=training.batch,
        device=config.device,
        project=str(training.output_dir),
        name="finetune",
        exist_ok=True,
    )

    best_weights_path = training.output_dir / "finetune" / "weights" / "best.pt"
    logger.info("Fine-tuning complete. Best weights: %s", best_weights_path)
    return best_weights_path
