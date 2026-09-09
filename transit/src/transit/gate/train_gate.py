"""Train and persist the logistic-regression plausibility gate."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from transit.config import TransitConfig
from transit.data.mtmc_dataset import MTMCScene
from transit.detection.dataset_export import load_ground_truth_annotations
from transit.eval.splits import split_by_identity
from transit.gate.features import _tracklet_identity_labels

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ["appearance_similarity", "transition_time_log_likelihood"]


def train_logistic_gate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    config: TransitConfig,
) -> tuple[Any, float]:
    """Fit a scaled logistic gate and return it with a 0.5 decision threshold."""
    if config.gate_model_type != "logistic":
        raise ValueError(f"train_logistic_gate requires gate_model_type='logistic', got {config.gate_model_type!r}")
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(train_features, train_labels)
    return model, 0.5


def _metrics(model: Pipeline, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probabilities = model.predict_proba(features)[:, 1]
    metrics = {"accuracy": float(accuracy_score(labels, probabilities >= 0.5))}
    metrics["auc"] = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan")
    return metrics


def train_scene_gate(
    feature_path: str | Path,
    scene_dir: str | Path,
    output_path: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, Any]:
    """Train on a feature table using the repository's identity split."""
    feature_path = Path(feature_path)
    scene = MTMCScene(scene_dir)
    table = pd.read_parquet(feature_path)
    required = set(FEATURE_COLUMNS + ["label", "src_camera_id", "src_track_id", "dst_camera_id", "dst_track_id"])
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Feature table is missing required columns: {sorted(missing)}")

    tracklets_by_camera = {
        camera_id: pd.read_parquet(feature_path.parent / camera_id / "tracklets.parquet")
        for camera_id in scene.camera_ids
    }
    identities = _tracklet_identity_labels(
        tracklets_by_camera,
        load_ground_truth_annotations(scene.ground_truth_path()),
        samples_per_tracklet=5,
    )
    train_ids, val_ids, eval_ids = split_by_identity(sorted(set(identities.values())), train_ratio, val_ratio)
    split_by_object = {object_id: "train" for object_id in train_ids}
    split_by_object.update({object_id: "val" for object_id in val_ids})
    split_by_object.update({object_id: "eval" for object_id in eval_ids})

    row_splits: list[str | None] = []
    for row in table.itertuples(index=False):
        source_identity = identities.get((row.src_camera_id, int(row.src_track_id)))
        destination_identity = identities.get((row.dst_camera_id, int(row.dst_track_id)))
        source_split = split_by_object.get(source_identity)
        destination_split = split_by_object.get(destination_identity)
        row_splits.append(source_split if source_split == destination_split else None)

    split_series = pd.Series(row_splits, index=table.index, dtype="object")
    train_table = table[split_series == "train"]
    val_table = table[split_series == "val"]
    if train_table.empty or val_table.empty:
        raise ValueError("Identity split produced an empty train or validation feature set")
    if train_table.label.nunique() < 2 or val_table.label.nunique() < 2:
        raise ValueError("Train and validation sets must each contain both label classes")

    config = TransitConfig(
        dataset_root=scene_dir,
        scene_name=scene.scene_dir.name,
        output_dir=feature_path.parent,
    )
    train_features = train_table[FEATURE_COLUMNS].to_numpy(np.float32)
    train_labels = train_table.label.to_numpy(np.int8)
    val_features = val_table[FEATURE_COLUMNS].to_numpy(np.float32)
    val_labels = val_table.label.to_numpy(np.int8)
    model, threshold = train_logistic_gate(train_features, train_labels, val_features, val_labels, config)
    train_metrics = _metrics(model, train_features, train_labels)
    val_metrics = _metrics(model, val_features, val_labels)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "feature_columns": FEATURE_COLUMNS}, output_path)

    coefficients = model.named_steps["classifier"].coef_[0].tolist()
    result = {
        "model_path": str(output_path),
        "identity_counts": {"train": len(train_ids), "val": len(val_ids), "eval": len(eval_ids)},
        "row_counts": {"train": len(train_table), "val": len(val_table)},
        "train": train_metrics,
        "validation": val_metrics,
        "standardized_coefficients": dict(zip(FEATURE_COLUMNS, coefficients)),
    }
    logger.info("Gate train metrics: %s", train_metrics)
    logger.info("Gate validation metrics: %s", val_metrics)
    logger.info("Gate standardized coefficients: %s", result["standardized_coefficients"])
    return result


__all__ = ["train_logistic_gate", "train_scene_gate"]
