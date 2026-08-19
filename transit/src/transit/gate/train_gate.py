"""Logistic-regression plausibility gate training. STUB -- not implemented.

# TODO(tomorrow): implement once gate/features.py produces real feature matrices.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from transit.config import TransitConfig


def train_logistic_gate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    config: TransitConfig,
) -> tuple[Any, float]:
    """Fit a logistic-regression plausibility gate and tune its decision threshold.

    Intended implementation: fit `sklearn.linear_model.LogisticRegression` (or
    equivalent) on `(train_features, train_labels)`, then sweep decision
    thresholds on `(val_features, val_labels)` to pick the threshold that
    optimizes a chosen operating point (e.g. best F1, or false-positive-match-rate
    at a target recall) using eval/metrics.py's `false_positive_match_rate` /
    rank metrics as the objective. The resulting threshold is what turns a
    continuous `gate_score` into an accept/reject decision analogous to
    matching/baseline_matcher.py's `threshold` parameter.

    Args:
        train_features: (N_train, F) feature matrix from gate/features.py's
            `build_gate_features`, built only from train-split identities.
        train_labels: (N_train,) binary labels from `build_hard_negative_labels`.
        val_features: (N_val, F) feature matrix, built only from val-split
            identities (see eval/splits.py), used solely for threshold tuning.
        val_labels: (N_val,) binary labels for val_features.
        config: TransitConfig (uses `config.gate_model_type`, expected to be
            "logistic" when this function is called).

    Returns:
        (fitted_model, decision_threshold).

    Raises:
        NotImplementedError: Always -- this is a stub.
    """
    raise NotImplementedError(
        "gate/train_gate.py is a documented stub. TODO(tomorrow): implement "
        "logistic gate training once gate/features.py produces real feature "
        "matrices from a downloaded scene."
    )
