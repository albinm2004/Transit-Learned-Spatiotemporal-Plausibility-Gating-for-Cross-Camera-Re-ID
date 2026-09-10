"""Feature construction for the learned plausibility gate.

Builds the two-feature vector [appearance_similarity, transition_log_likelihood]
that gate/train_gate.py fits a logistic regression on, plus the binary labels used
to train it. See the module-level NOTE below for the density-estimation and
fallback strategy chosen for the transition-time likelihood feature.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import gaussian_kde

from transit.data.schema import CandidateMatch
from transit.preprocessing.transitions import CameraPairTransitionStats

logger = logging.getLogger(__name__)

TrackletKey = tuple[str, int]

# Below this many observed transit times for a camera pair, a Gaussian KDE is too
# unstable to trust; fall back to a parametric Gaussian (mean/std) instead, and
# below 2 samples even that isn't possible, so fall back further to a global prior
# pooled across every camera pair. This three-tier backoff is the "not yet decided"
# strategy the original stub flagged -- documenting the choice here rather than
# leaving it implicit.
_MIN_SAMPLES_FOR_KDE = 5
_MIN_SAMPLES_FOR_GAUSSIAN = 2
_MIN_STD = 1e-3  # floor to avoid a zero-variance Gaussian collapsing to a delta spike
_LOG_LIKELIHOOD_FLOOR = -50.0  # clip so one wildly implausible transit time can't blow up training


def _gaussian_log_pdf(x: float, mean: float, std: float) -> float:
    std = max(std, _MIN_STD)
    return float(-0.5 * np.log(2.0 * np.pi * std**2) - 0.5 * ((x - mean) / std) ** 2)


def _fit_density(transit_times: list[float]):
    """Fit a log-density estimator over observed transit times, or None if too few."""
    times = np.asarray(transit_times, dtype=np.float64)
    n = len(times)

    if n >= _MIN_SAMPLES_FOR_KDE and np.std(times) > _MIN_STD:
        try:
            kde = gaussian_kde(times)
            return lambda x: float(kde.logpdf([x])[0])
        except Exception as exc:  # noqa: BLE001 - degenerate inputs (e.g. near-singular) fall back below
            logger.debug("gaussian_kde fit failed (%s); falling back to parametric Gaussian.", exc)

    if n >= _MIN_SAMPLES_FOR_GAUSSIAN:
        mean, std = float(np.mean(times)), float(np.std(times))
        return lambda x: _gaussian_log_pdf(x, mean, std)

    return None


def _fit_global_density(camera_pair_stats: dict[tuple[str, str], CameraPairTransitionStats]):
    """Fit a fallback density pooled across every camera pair's transit times."""
    pooled: list[float] = []
    for stats in camera_pair_stats.values():
        pooled.extend(stats.transit_times)

    density = _fit_density(pooled)
    if density is not None:
        return density

    logger.warning(
        "Not enough total transit-time observations (%d) to fit even a global "
        "fallback density; transition_log_likelihood will be a constant floor "
        "(%.1f) for every candidate lacking pair-specific data.",
        len(pooled),
        _LOG_LIKELIHOOD_FLOOR,
    )
    return lambda x: _LOG_LIKELIHOOD_FLOOR


def build_gate_features(
    candidates: list[CandidateMatch],
    camera_pair_stats: dict[tuple[str, str], CameraPairTransitionStats],
    src_exit_times: dict[TrackletKey, float],
    dst_entry_times: dict[TrackletKey, float],
) -> np.ndarray:
    """Build the gate's input feature vector for each candidate match.

    Each row combines:
      1. `appearance_similarity`, already on CandidateMatch.
      2. `transition_log_likelihood`: the log-density of observing this
         candidate's transit time (dst_entry_time - src_exit_time) under the
         (src_camera_id, dst_camera_id) pair's empirical transit-time
         distribution (`camera_pair_stats`). Estimated via a Gaussian KDE when
         enough samples exist, a parametric Gaussian when there are a few, and a
         density pooled across all camera pairs when the specific pair has no
         (or too little) data -- see `_fit_density` / `_fit_global_density`.

    Explicitly excluded, by design: no camera calibration, intrinsics, extrinsics,
    homography, or hand-specified topology/tier labels are used as a feature here
    -- calibration is an evaluation-time oracle only (see gate/calibration_oracle.py).

    Args:
        candidates: Candidate matches to build feature vectors for (typically one
            split's output of matching/candidate_generator.py).
        camera_pair_stats: Empirical per-camera-pair transit-time distributions
            from preprocessing/transitions.py's `compute_camera_pair_stats`, fit
            on the *train* identity split only (see eval/splits.py) to avoid
            leaking val/eval identities into the likelihood estimate.
        src_exit_times: Mapping from (camera_id, track_id) to the timestamp that
            tracklet was last seen.
        dst_entry_times: Mapping from (camera_id, track_id) to the timestamp that
            tracklet was first seen.

    Returns:
        An (N, 2) feature matrix -- columns [appearance_similarity,
        transition_log_likelihood] -- one row per candidate, same order as
        `candidates`.
    """
    pair_densities = {pair: _fit_density(stats.transit_times) for pair, stats in camera_pair_stats.items()}
    global_density = _fit_global_density(camera_pair_stats)

    features = np.zeros((len(candidates), 2), dtype=np.float64)
    n_missing_timestamps = 0
    n_pair_fallback = 0

    for i, candidate in enumerate(candidates):
        src_key = (candidate.src_camera_id, candidate.src_track_id)
        dst_key = (candidate.dst_camera_id, candidate.dst_track_id)
        src_exit = src_exit_times.get(src_key)
        dst_entry = dst_entry_times.get(dst_key)

        if src_exit is None or dst_entry is None:
            n_missing_timestamps += 1
            transit_time = 0.0
            density_fn = global_density
        else:
            transit_time = dst_entry - src_exit
            pair_key = (candidate.src_camera_id, candidate.dst_camera_id)
            density_fn = pair_densities.get(pair_key)
            if density_fn is None:
                n_pair_fallback += 1
                density_fn = global_density

        log_likelihood = max(density_fn(transit_time), _LOG_LIKELIHOOD_FLOOR)
        features[i, 0] = candidate.appearance_similarity
        features[i, 1] = log_likelihood

    if n_missing_timestamps:
        logger.warning(
            "%d/%d candidate(s) had no exit/entry timestamp on record (src or dst "
            "tracklet key not found); their transit_time was treated as 0.0 and "
            "scored against the global fallback density.",
            n_missing_timestamps,
            len(candidates),
        )
    if n_pair_fallback:
        logger.info(
            "%d/%d candidate(s) used the global fallback density (their specific "
            "camera pair had no or too little transit-time data in the train split).",
            n_pair_fallback,
            len(candidates),
        )

    return features


def build_hard_negative_labels(
    candidates: list[CandidateMatch],
    ground_truth: dict[TrackletKey, set[TrackletKey]],
) -> np.ndarray:
    """Build binary labels for gate training.

    Positive candidates are true cross-camera matches (per `ground_truth`).
    Negative candidates are everything else -- notably including "hard
    negatives": high-appearance-similarity but implausible-transition-time
    candidates, since `matching/candidate_generator.py`'s top-k-by-appearance
    pool already concentrates candidates on exactly that failure mode without
    any extra weighting needed here. (A class-imbalance-aware loss is instead
    applied at training time -- see gate/train_gate.py's `class_weight="balanced"`
    -- rather than hand-weighting individual examples here.)

    Args:
        candidates: Candidate matches, in the same order as
            `build_gate_features`'s output rows.
        ground_truth: Mapping from source tracklet key to the set of destination
            tracklet keys that are true cross-camera matches for it.

    Returns:
        An (N,) binary label array, 1.0 for true matches, 0.0 otherwise.
    """
    labels = np.zeros(len(candidates), dtype=np.float64)
    for i, candidate in enumerate(candidates):
        src_key = (candidate.src_camera_id, candidate.src_track_id)
        dst_key = (candidate.dst_camera_id, candidate.dst_track_id)
        if dst_key in ground_truth.get(src_key, set()):
            labels[i] = 1.0
    return labels
