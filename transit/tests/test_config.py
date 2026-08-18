"""Smoke tests for TransitConfig loading."""

from __future__ import annotations

from pathlib import Path

from transit.config import TransitConfig, load_config

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
