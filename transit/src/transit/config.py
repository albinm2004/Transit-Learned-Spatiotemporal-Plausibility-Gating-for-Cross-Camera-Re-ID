"""Dataclass-based configuration for Transit, loaded from YAML.

Keeping configuration in a single typed object means every stage of the pipeline
(detection, tracking, and later re-ID/matching/gating) reads settings the same way,
and paths/hyperparameters never get hardcoded in individual modules.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_DETECTOR_MODEL = "yolo11x.pt"
_UPGRADED_DETECTOR_MODEL = "yolo26x.pt"


def _detect_default_model() -> str:
    """Pick the best available default detector checkpoint.

    Prefers ``yolo26x.pt`` over the baseline ``yolo11x.pt`` if the installed
    ultralytics version actually knows about that checkpoint (i.e. it appears in
    ultralytics' own table of downloadable model assets). This avoids silently
    assuming a newer checkpoint exists when the installed package predates it.
    """
    try:
        from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

        if _UPGRADED_DETECTOR_MODEL in GITHUB_ASSETS_NAMES:
            logger.info(
                "Detected '%s' as an available ultralytics checkpoint; using it as default.",
                _UPGRADED_DETECTOR_MODEL,
            )
            return _UPGRADED_DETECTOR_MODEL
    except Exception as exc:  # noqa: BLE001 - ultralytics may not be installed yet, or API may change
        logger.debug(
            "Could not query ultralytics for available checkpoints (%s); "
            "falling back to default '%s'.",
            exc,
            _DEFAULT_DETECTOR_MODEL,
        )

    return _DEFAULT_DETECTOR_MODEL


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu' based on actual torch/CUDA availability."""
    if device != "auto":
        return device

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception as exc:  # noqa: BLE001 - torch may not be installed yet
        logger.warning("Could not query torch for CUDA availability (%s); defaulting to 'cpu'.", exc)
        return "cpu"


@dataclass
class TransitConfig:
    """Top-level configuration for the Transit detection + tracking pipeline.

    Attributes:
        dataset_root: Root directory containing downloaded scene data.
        scene_name: Name of the scene (subdirectory of dataset_root) to process.
        output_dir: Directory where pipeline outputs are written.
        detector_model: Ultralytics detector checkpoint name.
        detection_conf_threshold: Minimum confidence to keep a detection.
        tracker_config: Ultralytics tracker config name (e.g. "bytetrack.yaml").
        device: "cuda", "cpu", or "auto" for runtime auto-detection.
    """

    dataset_root: Path
    scene_name: str
    output_dir: Path
    detector_model: str = _DEFAULT_DETECTOR_MODEL
    detection_conf_threshold: float = 0.4
    tracker_config: str = "bytetrack.yaml"
    device: str = "auto"

    def __post_init__(self) -> None:
        self.dataset_root = Path(self.dataset_root)
        self.output_dir = Path(self.output_dir)
        self.device = _resolve_device(self.device)

    @property
    def scene_dir(self) -> Path:
        """Directory containing this config's scene data."""
        return self.dataset_root / self.scene_name

    def to_dict(self) -> dict[str, Any]:
        """Serialize this config to a plain dict (paths as strings)."""
        data = asdict(self)
        data["dataset_root"] = str(self.dataset_root)
        data["output_dir"] = str(self.output_dir)
        return data


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> TransitConfig:
    """Load a TransitConfig from a YAML file, with optional field overrides.

    Args:
        path: Path to a YAML config file (see configs/default.yaml for the schema).
        overrides: Optional dict of field values to override after loading, e.g.
            values parsed from CLI flags.

    Returns:
        A populated TransitConfig instance.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})

    if raw.get("detector_model") is None:
        raw["detector_model"] = _detect_default_model()

    logger.info("Loaded config from %s", path)
    return TransitConfig(**raw)
