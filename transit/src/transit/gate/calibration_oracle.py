"""Calibration-based geometric-feasibility oracle. STUB -- not implemented.

EVALUATION-ONLY. This module computes a geometric plausibility signal directly
from ground-truth camera calibration (intrinsics/extrinsics/homography). It
exists purely to serve as an upper-bound / oracle comparison point in the paper's
results (i.e. "how well would a gate do if it were allowed to cheat and use
calibration directly") and must NEVER be wired into the learned gate's training
features (gate/features.py) or inference path (gate/train_gate.py,
gate/mlp_gate.py) -- doing so would defeat the entire premise of this project,
which is a gate that works *without* hand-specified/calibration-derived topology.

# TODO(tomorrow): implement once real calibration.json files are available and
# preprocessing/camera_pairs.py's footprint-based tiering has been validated
# against them.
"""

from __future__ import annotations

from transit.data.schema import CandidateMatch
from transit.preprocessing.camera_pairs import CameraPairTier


def compute_geometric_feasibility_oracle(
    candidates: list[CandidateMatch],
    camera_pair_tiers: dict[tuple[str, str], CameraPairTier],
) -> list[float]:
    """Score candidates by ground-truth-calibration-derived geometric feasibility.

    Intended implementation: for each candidate, use the src/dst cameras'
    calibration-derived FOV footprints and pair tier (from
    preprocessing/camera_pairs.py's `classify_camera_pairs`) plus their measured
    exit/entry timestamps to compute a purely geometric plausibility score --
    e.g. 1.0 for overlapping-camera candidates whose timing is consistent with
    simultaneous visibility, a distance/speed-plausibility score for
    neighbouring-tier candidates, and 0.0 for distant-disconnected-tier
    candidates whose implied travel speed is physically impossible. This is
    reported alongside the learned gate's results as an oracle upper bound, using
    the same eval/metrics.py functions (rank_k_accuracy, mean_average_precision,
    false_positive_match_rate) for a fair side-by-side comparison -- never used
    to train or run the actual gate.

    Args:
        candidates: Candidate matches to score, typically the eval-split output
            of matching/candidate_generator.py.
        camera_pair_tiers: Camera pair tiers from
            preprocessing/camera_pairs.py's `classify_camera_pairs`.

    Returns:
        One geometric feasibility score per candidate, in [0, 1], same order as
        `candidates`.

    Raises:
        NotImplementedError: Always -- this is a stub.
    """
    raise NotImplementedError(
        "gate/calibration_oracle.py is a documented stub. TODO(tomorrow): "
        "implement once real calibration.json files are available. Remember: "
        "evaluation-only -- never feed this into gate training/inference."
    )
