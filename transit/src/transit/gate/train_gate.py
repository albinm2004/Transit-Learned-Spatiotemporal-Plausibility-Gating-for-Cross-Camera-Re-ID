"""Logistic-regression plausibility gate training."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from transit.config import TransitConfig

logger = logging.getLogger(__name__)

_THRESHOLD_SWEEP = np.linspace(0.01, 0.99, 99)


def _sweep_best_f1_threshold(val_probs: np.ndarray, val_labels: np.ndarray) -> tuple[float, float]:
    """Sweep decision thresholds on val predictions, returning the best-F1 one.

    Returns:
        (best_threshold, best_f1). If val_labels has no positives at all, F1 is
        undefined at every threshold; in that case falls back to threshold=0.5.
    """
    if val_labels.sum() == 0:
        logger.warning("Val split has no positive labels; cannot tune a threshold by F1. Defaulting to 0.5.")
        return 0.5, 0.0

    best_threshold, best_f1 = 0.5, -1.0
    for t in _THRESHOLD_SWEEP:
        preds = val_probs >= t
        tp = float(np.sum(preds & (val_labels == 1)))
        fp = float(np.sum(preds & (val_labels == 0)))
        fn = float(np.sum(~preds & (val_labels == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1, best_threshold = f1, float(t)

    return best_threshold, best_f1


def train_logistic_gate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    config: TransitConfig,
) -> tuple[Any, float]:
    """Fit a logistic-regression plausibility gate and tune its decision threshold.

    Fits `sklearn.linear_model.LogisticRegression` with `class_weight="balanced"`
    (true cross-camera matches are a small minority of the candidate pool, since
    each source tracklet gets `candidate_top_k` candidates but at most one is
    correct) on `(train_features, train_labels)`, then sweeps decision thresholds
    on `(val_features, val_labels)` to pick the threshold maximizing F1. This
    threshold is what turns a continuous `gate_score` into an accept/reject
    decision, analogous to matching/baseline_matcher.py's `threshold` parameter.

    Note: this threshold is tuned for a balanced precision/recall trade-off (F1).
    For the paper's reported operating point you may instead want to re-tune
    against eval/metrics.py's `false_positive_match_rate` at a fixed target
    recall -- do that as a second pass over `model.predict_proba(...)` on the
    eval split, using this function's `model` but a different threshold.

    Args:
        train_features: (N_train, F) feature matrix from gate/features.py's
            `build_gate_features`, built only from train-split identities.
        train_labels: (N_train,) binary labels from `build_hard_negative_labels`.
        val_features: (N_val, F) feature matrix, built only from val-split
            identities (see eval/splits.py), used solely for threshold tuning.
        val_labels: (N_val,) binary labels for val_features.
        config: TransitConfig (must have `config.gate_model_type == "logistic"`).

    Returns:
        (fitted_model, decision_threshold).

    Raises:
        ValueError: If `config.gate_model_type != "logistic"`, or if
            `train_labels` has no positive examples to fit against.
    """
    if config.gate_model_type != "logistic":
        raise ValueError(
            f"train_logistic_gate called with config.gate_model_type="
            f"{config.gate_model_type!r}; expected 'logistic'."
        )
    if train_labels.sum() == 0:
        raise ValueError(
            "train_labels has no positive examples -- cannot fit a gate with "
            "zero true cross-camera matches in the train split. This usually "
            "means the ground-truth tracklet linkage (see "
            "gate/ground_truth_linking.py) found no cross-camera transitions "
            "for any train-split identity; check the IoU-linking log output."
        )

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(train_features, train_labels)

    val_probs = model.predict_proba(val_features)[:, 1] if len(val_features) else np.array([])
    threshold, best_f1 = _sweep_best_f1_threshold(val_probs, val_labels)

    logger.info(
        "Trained logistic gate on %d example(s) (%.1f%% positive): "
        "coef=[appearance=%.4f, transition_log_likelihood=%.4f] intercept=%.4f; "
        "tuned threshold=%.3f (val F1=%.4f, n_val=%d)",
        len(train_labels),
        100.0 * train_labels.mean(),
        model.coef_[0][0],
        model.coef_[0][1],
        model.intercept_[0],
        threshold,
        best_f1,
        len(val_labels),
    )
    return model, threshold
