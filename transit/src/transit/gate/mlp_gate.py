"""Small-MLP plausibility gate training. STUB -- not implemented.

# TODO(tomorrow): implement once gate/features.py produces real feature matrices,
# and once the logistic baseline in gate/train_gate.py has an established result
# to compare against.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from transit.config import TransitConfig


def train_mlp_gate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    config: TransitConfig,
) -> tuple[Any, float]:
    """Fit a small-MLP plausibility gate and tune its decision threshold.

    Intended implementation: a small torch.nn.Module (1-2 hidden layers) trained
    with binary cross-entropy on `(train_features, train_labels)`, early-stopped
    against `(val_features, val_labels)`, with the same post-hoc threshold-tuning
    step described in gate/train_gate.py's `train_logistic_gate`. Only worth
    building out if the logistic gate underfits the transition-time-likelihood /
    appearance-similarity interaction -- decide after seeing logistic gate results.

    Args:
        train_features: (N_train, F) feature matrix from gate/features.py's
            `build_gate_features`, built only from train-split identities.
        train_labels: (N_train,) binary labels from `build_hard_negative_labels`.
        val_features: (N_val, F) feature matrix, built only from val-split
            identities (see eval/splits.py), used for early stopping and
            threshold tuning.
        val_labels: (N_val,) binary labels for val_features.
        config: TransitConfig (uses `config.gate_model_type`, expected to be
            "mlp" when this function is called).

    Returns:
        (fitted_model, decision_threshold).

    Raises:
        NotImplementedError: Always -- this is a stub.
    """
    raise NotImplementedError(
        "gate/mlp_gate.py is a documented stub. TODO(tomorrow): implement MLP "
        "gate training once gate/features.py produces real feature matrices, and "
        "only after the logistic baseline (gate/train_gate.py) has a result."
    )
