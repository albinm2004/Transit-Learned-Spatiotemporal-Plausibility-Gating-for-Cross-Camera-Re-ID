"""Calibration-based geometric-feasibility oracle.

EVALUATION-ONLY. This module computes a geometric plausibility signal directly
from ground-truth camera calibration (via preprocessing/camera_pairs.py's tiers,
themselves derived from calibration-projected FOV footprints). It exists purely
to serve as an upper-bound / oracle comparison point in the paper's results (i.e.
"how well would a gate do if it were allowed to cheat and use calibration
directly") and must NEVER be wired into the learned gate's training features
(gate/features.py) or inference path (gate/train_gate.py, gate/mlp_gate.py) --
doing so would defeat the entire premise of this project, which is a gate that
works *without* hand-specified/calibration-derived topology.
"""

from __future__ import annotations

import logging

from transit.data.schema import CandidateMatch
from transit.preprocessing.camera_pairs import TIER_DISTANT, TIER_NEIGHBORING, TIER_OVERLAPPING, CameraPairTier

logger = logging.getLogger(__name__)

# A candidate between overlapping cameras is geometrically unremarkable (the
# object can plausibly be seen on both at once); a candidate between distant,
# disconnected cameras implies a transition too large-scale for calibration
# alone to vouch for; neighboring cameras sit in between. These are deliberately
# coarse scores -- the oracle's whole point is to be the *simple*, calibration-
# only comparison point, not a second sophisticated model.
_TIER_SCORES: dict[CameraPairTier, float] = {
    TIER_OVERLAPPING: 1.0,
    TIER_NEIGHBORING: 0.5,
    TIER_DISTANT: 0.0,
}


def compute_geometric_feasibility_oracle(
    candidates: list[CandidateMatch],
    camera_pair_tiers: dict[tuple[str, str], CameraPairTier],
) -> list[float]:
    """Score candidates by ground-truth-calibration-derived geometric feasibility.

    Same-camera candidates (src_camera_id == dst_camera_id) score 1.0 trivially.
    Cross-camera candidates score by their pair's tier from
    preprocessing/camera_pairs.py's `classify_camera_pairs` (itself built from
    calibration-projected FOV footprints): 1.0 for overlapping, 0.5 for
    neighboring-non-overlapping, 0.0 for distant-disconnected. Reported alongside
    the learned gate's results using the same eval/metrics.py functions
    (rank_k_accuracy, mean_average_precision) for a fair side-by-side comparison.

    Args:
        candidates: Candidate matches to score, typically the eval-split output
            of matching/candidate_generator.py.
        camera_pair_tiers: Camera pair tiers from
            preprocessing/camera_pairs.py's `classify_camera_pairs`, keyed by
            (camera_id_a, camera_id_b) ordered lexicographically.

    Returns:
        One geometric feasibility score per candidate, in [0, 1], same order as
        `candidates`.
    """
    scores: list[float] = []
    n_unknown_pairs = 0

    for candidate in candidates:
        if candidate.src_camera_id == candidate.dst_camera_id:
            scores.append(1.0)
            continue

        pair_key = tuple(sorted((candidate.src_camera_id, candidate.dst_camera_id)))
        tier = camera_pair_tiers.get(pair_key)
        if tier is None:
            n_unknown_pairs += 1
            scores.append(0.0)
            continue

        scores.append(_TIER_SCORES.get(tier, 0.0))

    if n_unknown_pairs:
        logger.warning(
            "%d/%d candidate(s) referenced a camera pair with no tier on record "
            "(scored 0.0); this usually means camera-pair tiering "
            "(scripts/run_preprocess_transitions.py) was skipped or ran on a "
            "different camera set than candidate generation.",
            n_unknown_pairs,
            len(candidates),
        )

    return scores
