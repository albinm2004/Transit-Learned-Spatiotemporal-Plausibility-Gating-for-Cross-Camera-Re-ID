"""Object-identity-based train/val/eval splitting.

Splitting must be done by global object identity, not by frame or tracklet, so
that no identity's appearance leaks across splits -- otherwise Re-ID/matching/gate
evaluation numbers would be optimistically biased.
"""

from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)

_DEFAULT_SEED = 42


def split_by_identity(
    object_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int = _DEFAULT_SEED,
) -> tuple[list[str], list[str], list[str]]:
    """Split a list of global object ids into disjoint train/val/eval identity sets.

    Args:
        object_ids: All distinct global object ids in the scene(s) being split.
        train_ratio: Fraction of identities assigned to the train split.
        val_ratio: Fraction of identities assigned to the val split. The
            remainder (1 - train_ratio - val_ratio) is assigned to eval.
        seed: RNG seed, fixed by default for reproducible splits.

    Returns:
        (train_ids, val_ids, eval_ids), a partition of the (de-duplicated) input
        object_ids.

    Raises:
        ValueError: If train_ratio/val_ratio are not both in (0, 1) and don't sum
            to less than 1.
    """
    if not (0.0 < train_ratio < 1.0) or not (0.0 < val_ratio < 1.0):
        raise ValueError(f"train_ratio and val_ratio must each be in (0, 1), got {train_ratio}, {val_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1 (remainder is eval_ratio), "
            f"got {train_ratio} + {val_ratio} = {train_ratio + val_ratio}"
        )

    unique_ids = sorted(set(object_ids))
    shuffled = list(unique_ids)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    eval_ids = shuffled[n_train + n_val :]

    logger.info(
        "Split %d identity(ies) into train=%d, val=%d, eval=%d",
        n,
        len(train_ids),
        len(val_ids),
        len(eval_ids),
    )
    return train_ids, val_ids, eval_ids
