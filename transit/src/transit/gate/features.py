"""Feature construction for the learned plausibility gate. STUB -- not implemented.

# TODO(tomorrow): implement this once preprocessing has run on real data.
"""

from __future__ import annotations

import numpy as np

from transit.data.schema import CandidateMatch
from transit.preprocessing.transitions import CameraPairTransitionStats


def build_gate_features(
    candidates: list[CandidateMatch],
    camera_pair_stats: dict[tuple[str, str], CameraPairTransitionStats],
    src_exit_times: dict[tuple[str, int], float],
    dst_entry_times: dict[tuple[str, int], float],
) -> np.ndarray:
    """Build the gate's input feature vector for each candidate match.

    Each row will combine:
      1. `appearance_similarity` (already on CandidateMatch, from Re-ID cosine
         similarity).
      2. A transition-time log-likelihood: the log-probability of observing a
         transit time of (dst_entry_time - src_exit_time) for this
         (src_camera_id, dst_camera_id) pair, estimated from
         `camera_pair_stats[(src, dst)].transit_times` (e.g. via a KDE or a fitted
         parametric distribution over the empirical transit_times list produced by
         preprocessing/transitions.py's `compute_camera_pair_stats`). Camera pairs
         with too few observed transitions need a fallback/smoothing strategy
         (e.g. backing off to a global transit-time prior) -- not yet decided.
      3. Optional auxiliary features (candidates: elapsed real time since
         src_exit_time if the candidate is being scored online / streaming;
         relative track duration or confidence; not yet decided).

    Explicitly excluded, by design: no camera calibration, intrinsics, extrinsics,
    homography, or hand-specified topology/tier labels may be used as a feature
    here -- calibration is an evaluation-time oracle only
    (see gate/calibration_oracle.py) and using it as a gate input would defeat the
    project's core premise.

    Args:
        candidates: Candidate matches to build feature vectors for (typically the
            train/val-split output of matching/candidate_generator.py).
        camera_pair_stats: Empirical per-camera-pair transit-time distributions
            from preprocessing/transitions.py's `compute_camera_pair_stats`,
            fit on the *train* identity split only (see eval/splits.py) to avoid
            leaking val/eval identities into the likelihood estimate.
        src_exit_times: Mapping from (camera_id, track_id) to the timestamp that
            tracklet was last seen, for computing transit times.
        dst_entry_times: Mapping from (camera_id, track_id) to the timestamp that
            tracklet was first seen, for computing transit times.

    Returns:
        An (N, F) feature matrix, one row per candidate, in the same order as
        `candidates`.

    Raises:
        NotImplementedError: Always -- this is a stub.
    """
    raise NotImplementedError(
        "gate/features.py is a documented stub. TODO(tomorrow): implement feature "
        "construction once preprocessing/transitions.py has been run on a real "
        "downloaded scene and camera_pair_stats has real data to fit against."
    )


def build_hard_negative_labels(
    candidates: list[CandidateMatch],
    ground_truth: dict[tuple[str, int], set[tuple[str, int]]],
) -> np.ndarray:
    """Build binary labels for gate training, with hard-negative emphasis.

    Positive candidates are true cross-camera matches (per `ground_truth`).
    Negative candidates are all other generated candidates -- notably including
    "hard negatives": high-appearance-similarity but implausible-transition-time
    candidates, which are exactly the failure mode the gate is meant to catch. The
    exact hard-negative *weighting* scheme (vs. plain binary labels with a
    class-balanced loss) is not yet decided.

    Args:
        candidates: Candidate matches, in the same order as
            `build_gate_features`'s output rows.
        ground_truth: Mapping from source tracklet key to the set of destination
            tracklet keys that are true cross-camera matches for it.

    Returns:
        An (N,) binary label array, 1.0 for true matches, 0.0 otherwise.

    Raises:
        NotImplementedError: Always -- this is a stub.
    """
    raise NotImplementedError(
        "gate/features.py is a documented stub. TODO(tomorrow): implement label "
        "construction once ground-truth ID linkage is available from a real scene."
    )
