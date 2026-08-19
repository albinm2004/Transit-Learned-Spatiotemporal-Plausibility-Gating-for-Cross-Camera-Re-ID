"""Smoke tests for TransitConfig loading."""

from __future__ import annotations

from pathlib import Path

from transit.config import TrainingConfig, TransitConfig, load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_load_config_from_yaml() -> None:
    """The default config file should load into a valid TransitConfig."""
    config = load_config(CONFIG_PATH)

    assert isinstance(config, TransitConfig)
    assert isinstance(config.dataset_root, Path)
    assert isinstance(config.output_dir, Path)
    assert config.scene_name
    assert config.detector_model
    assert 0.0 <= config.detection_conf_threshold <= 1.0
    assert config.tracker_config
    assert config.device in {"cuda", "cpu"}
    assert config.reid_model_name
    assert config.gate_model_type in {"logistic", "mlp"}
    assert isinstance(config.training, TrainingConfig)
    assert config.training.epochs > 0


def test_load_config_with_overrides() -> None:
    """Overrides passed to load_config should take precedence over YAML values."""
    config = load_config(CONFIG_PATH, overrides={"scene_name": "my_scene"})

    assert config.scene_name == "my_scene"


def test_scene_dir_property() -> None:
    """scene_dir should combine dataset_root and scene_name."""
    config = TransitConfig(
        dataset_root="data/raw",
        scene_name="scene_001",
        output_dir="outputs",
        device="cpu",
    )

    assert config.scene_dir == Path("data/raw") / "scene_001"


def test_resolved_base_checkpoint_falls_back_to_detector_model() -> None:
    """resolved_base_checkpoint() should use detector_model when training.base_checkpoint is unset."""
    config = TransitConfig(
        dataset_root="data/raw",
        scene_name="scene_001",
        output_dir="outputs",
        detector_model="yolo11x.pt",
        device="cpu",
    )

    assert config.resolved_base_checkpoint() == "yolo11x.pt"


def test_resolved_base_checkpoint_prefers_explicit_training_checkpoint() -> None:
    """resolved_base_checkpoint() should prefer an explicit training.base_checkpoint."""
    config = TransitConfig(
        dataset_root="data/raw",
        scene_name="scene_001",
        output_dir="outputs",
        detector_model="yolo11x.pt",
        device="cpu",
        training=TrainingConfig(base_checkpoint="yolo11n.pt"),
    )

    assert config.resolved_base_checkpoint() == "yolo11n.pt"


def test_invalid_gate_model_type_raises() -> None:
    """An unrecognized gate_model_type should raise ValueError."""
    try:
        TransitConfig(
            dataset_root="data/raw",
            scene_name="scene_001",
            output_dir="outputs",
            device="cpu",
            gate_model_type="not_a_real_model",
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass
