"""Dataclass-based configuration for Transit, loaded from YAML.

Keeping configuration in a single typed object means every stage of the pipeline
(detection, tracking, re-ID, matching, and eventually gating) reads settings the
same way, and paths/hyperparameters never get hardcoded in individual modules.
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
            "Could not query ultralytics for available checkpoints (%s); falling back to default '%s'.",
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
class TrainingConfig:
    """Configuration for YOLO detector fine-tuning (detection/train.py).

    Attributes:
        base_checkpoint: Checkpoint to fine-tune from. If None, resolved from the
            parent TransitConfig's detector_model (which itself may auto-detect
            yolo26x.pt vs yolo11x.pt).
        epochs: Number of fine-tuning epochs.
        imgsz: Training image size (square), passed to Ultralytics as `imgsz`.
        batch: Training batch size.
        frame_sample_stride: Sample every Nth annotated frame per camera when
            exporting ground-truth video frames to a YOLO training dataset, to
            avoid near-duplicate consecutive frames dominating the dataset.
        train_val_split: Fraction of exported images assigned to the train split;
            the remainder is used for validation.
        output_dir: Directory where fine-tuned weights and training run artifacts
            are written.
        dataset_export_dir: Directory where the exported YOLO-format dataset
            (images/, labels/, dataset.yaml) is written.
    """

    base_checkpoint: str | None = None
    epochs: int = 50
    imgsz: int = 960
    batch: int = 8
    frame_sample_stride: int = 10
    train_val_split: float = 0.85
    output_dir: Path = field(default_factory=lambda: Path("outputs/yolo_finetune"))
    dataset_export_dir: Path = field(default_factory=lambda: Path("outputs/yolo_dataset"))

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.dataset_export_dir = Path(self.dataset_export_dir)
        if not 0.0 < self.train_val_split < 1.0:
            raise ValueError(f"train_val_split must be in (0, 1), got {self.train_val_split}")


@dataclass
class TransitConfig:
    """Top-level configuration for the Transit pipeline.

    Attributes:
        dataset_root: Root directory containing downloaded scene data.
        scene_name: Name of the scene (subdirectory of dataset_root) to process.
        output_dir: Directory where pipeline outputs are written.
        detector_model: Ultralytics detector checkpoint name or path.
        detection_conf_threshold: Minimum confidence to keep a detection.
        tracker_config: Ultralytics tracker config name (e.g. "bytetrack.yaml").
        device: "cuda", "cpu", or "auto" for runtime auto-detection.
        reid_model_name: torchreid model architecture name (e.g. "osnet_x1_0").
        reid_weights_source: torchreid model-zoo key (e.g. "market1501", "msmt17")
            or a local path to a downloaded checkpoint.
        candidate_top_k: Number of nearest cross-camera candidates kept per
            tracklet during candidate generation.
        baseline_similarity_threshold: Minimum cosine similarity for the
            appearance-only baseline matcher to accept a match.
        gate_model_type: "logistic" or "mlp" — which gate architecture to use.
            Unused until gate/train_gate.py or gate/mlp_gate.py is implemented.
        training: Sub-config for YOLO detector fine-tuning.
    """

    dataset_root: Path
    scene_name: str
    output_dir: Path
    detector_model: str = _DEFAULT_DETECTOR_MODEL
    detection_conf_threshold: float = 0.4
    tracker_config: str = "bytetrack.yaml"
    device: str = "auto"
    reid_model_name: str = "osnet_x1_0"
    reid_weights_source: str = "market1501"
    candidate_top_k: int = 10
    baseline_similarity_threshold: float = 0.5
    gate_model_type: str = "logistic"
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        self.dataset_root = Path(self.dataset_root)
        self.output_dir = Path(self.output_dir)
        self.device = _resolve_device(self.device)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if self.gate_model_type not in {"logistic", "mlp"}:
            raise ValueError(f"gate_model_type must be 'logistic' or 'mlp', got {self.gate_model_type!r}")

    @property
    def scene_dir(self) -> Path:
        """Directory containing this config's scene data."""
        return self.dataset_root / self.scene_name

    def resolved_base_checkpoint(self) -> str:
        """The checkpoint to fine-tune from: training.base_checkpoint, or else detector_model."""
        return self.training.base_checkpoint or self.detector_model

    def to_dict(self) -> dict[str, Any]:
        """Serialize this config to a plain dict (paths as strings)."""
        data = asdict(self)
        data["dataset_root"] = str(self.dataset_root)
        data["output_dir"] = str(self.output_dir)
        data["training"]["output_dir"] = str(self.training.output_dir)
        data["training"]["dataset_export_dir"] = str(self.training.dataset_export_dir)
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
