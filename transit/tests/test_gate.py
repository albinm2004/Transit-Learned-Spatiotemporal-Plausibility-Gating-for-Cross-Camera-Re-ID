"""Unit tests for the learned plausibility gate (gate/features.py, train_gate.py,
calibration_oracle.py, ground_truth_linking.py).

These use small synthetic data only -- no dataset download or GPU required, so
they can (and should) run before the real GPU pipeline run to sanity-check the
gate implementation in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transit.config import TransitConfig
from transit.data.schema import CandidateMatch
from transit.detection.dataset_export import GroundTruthBox
from transit.gate.calibration_oracle import compute_geometric_feasibility_oracle
from transit.gate.features import build_gate_features, build_hard_negative_labels
from transit.gate.ground_truth_linking import (
    index_tracklet_detections_by_frame,
    link_camera_tracklets_to_ground_truth,
)
from transit.gate.train_gate import train_logistic_gate
from transit.preprocessing.camera_pairs import TIER_DISTANT, TIER_NEIGHBORING, TIER_OVERLAPPING
from transit.preprocessing.transitions import CameraPairTransitionStats


def _make_config(**overrides) -> TransitConfig:
    defaults = dict(dataset_root="unused", scene_name="scene", output_dir="unused", device="cpu")
    defaults.update(overrides)
    return TransitConfig(**defaults)


class TestBuildGateFeatures:
    def test_uses_pair_specific_likelihood_when_available(self):
        stats = {
            ("A", "B"): CameraPairTransitionStats(
                src_camera_id="A",
                dst_camera_id="B",
                count=6,
                mean_transit_time=5.0,
                std_transit_time=1.0,
                min_transit_time=3.0,
                max_transit_time=7.0,
                transit_times=[3.0, 4.0, 5.0, 5.0, 6.0, 7.0],
            )
        }
        # A candidate whose transit time sits right at the pair's mean should
        # score a much higher log-likelihood than one far outside the observed range.
        plausible = CandidateMatch("A", 1, "B", 1, appearance_similarity=0.9)
        implausible = CandidateMatch("A", 2, "B", 2, appearance_similarity=0.9)
        exit_times = {("A", 1): 0.0, ("A", 2): 0.0}
        entry_times = {("B", 1): 5.0, ("B", 2): 500.0}

        features = build_gate_features([plausible, implausible], stats, exit_times, entry_times)

        assert features.shape == (2, 2)
        assert features[0, 0] == pytest.approx(0.9)
        assert features[0, 1] > features[1, 1]  # plausible transit time scores higher

    def test_falls_back_to_global_density_for_unseen_pair(self):
        stats = {
            ("A", "B"): CameraPairTransitionStats(
                src_camera_id="A",
                dst_camera_id="B",
                count=5,
                mean_transit_time=5.0,
                std_transit_time=1.0,
                min_transit_time=3.0,
                max_transit_time=7.0,
                transit_times=[3.0, 4.0, 5.0, 6.0, 7.0],
            )
        }
        # Candidate is for pair (C, D), which has no stats of its own.
        candidate = CandidateMatch("C", 1, "D", 1, appearance_similarity=0.5)
        features = build_gate_features(
            [candidate], stats, {("C", 1): 0.0}, {("D", 1): 5.0}
        )
        assert features.shape == (1, 2)
        assert np.isfinite(features[0, 1])

    def test_missing_timestamps_do_not_crash(self):
        candidate = CandidateMatch("A", 1, "B", 1, appearance_similarity=0.7)
        features = build_gate_features([candidate], {}, {}, {})
        assert features.shape == (1, 2)
        assert np.isfinite(features[0, 1])

    def test_empty_candidates_returns_empty_matrix(self):
        features = build_gate_features([], {}, {}, {})
        assert features.shape == (0, 2)


class TestBuildHardNegativeLabels:
    def test_labels_true_matches_positive(self):
        candidates = [
            CandidateMatch("A", 1, "B", 1, appearance_similarity=0.9),  # true match
            CandidateMatch("A", 1, "B", 2, appearance_similarity=0.8),  # hard negative
        ]
        ground_truth = {("A", 1): {("B", 1)}}
        labels = build_hard_negative_labels(candidates, ground_truth)
        np.testing.assert_array_equal(labels, [1.0, 0.0])

    def test_source_absent_from_ground_truth_is_all_negative(self):
        candidates = [CandidateMatch("A", 1, "B", 1, appearance_similarity=0.9)]
        labels = build_hard_negative_labels(candidates, {})
        np.testing.assert_array_equal(labels, [0.0])


class TestTrainLogisticGate:
    def test_learns_a_linearly_separable_toy_problem(self):
        rng = np.random.default_rng(0)
        n = 200
        # Positives: high appearance similarity AND high transition log-likelihood.
        pos = np.stack([rng.uniform(0.7, 1.0, n), rng.uniform(-2.0, 0.0, n)], axis=1)
        # Negatives: low on at least one axis.
        neg = np.stack([rng.uniform(0.0, 0.5, n), rng.uniform(-20.0, -10.0, n)], axis=1)
        features = np.concatenate([pos, neg])
        labels = np.concatenate([np.ones(n), np.zeros(n)])

        shuffle = rng.permutation(len(features))
        features, labels = features[shuffle], labels[shuffle]
        split = len(features) // 2
        config = _make_config(gate_model_type="logistic")

        model, threshold = train_logistic_gate(
            features[:split], labels[:split], features[split:], labels[split:], config
        )

        preds = model.predict_proba(features[split:])[:, 1] >= threshold
        accuracy = (preds == labels[split:]).mean()
        assert accuracy > 0.9
        # Both features should matter (positive coefficients on this toy problem).
        assert model.coef_[0][0] > 0
        assert model.coef_[0][1] > 0

    def test_rejects_wrong_model_type(self):
        config = _make_config(gate_model_type="mlp")
        with pytest.raises(ValueError):
            train_logistic_gate(np.zeros((2, 2)), np.array([0.0, 1.0]), np.zeros((2, 2)), np.array([0.0, 1.0]), config)

    def test_rejects_no_positive_train_labels(self):
        config = _make_config(gate_model_type="logistic")
        with pytest.raises(ValueError):
            train_logistic_gate(
                np.zeros((2, 2)), np.array([0.0, 0.0]), np.zeros((2, 2)), np.array([0.0, 1.0]), config
            )


class TestCalibrationOracle:
    def test_scores_by_tier(self):
        candidates = [
            CandidateMatch("A", 1, "A", 2, appearance_similarity=0.5),  # same camera
            CandidateMatch("A", 1, "B", 1, appearance_similarity=0.5),  # overlapping
            CandidateMatch("A", 1, "C", 1, appearance_similarity=0.5),  # neighboring
            CandidateMatch("A", 1, "D", 1, appearance_similarity=0.5),  # distant
            CandidateMatch("A", 1, "Z", 1, appearance_similarity=0.5),  # unknown pair
        ]
        tiers = {
            ("A", "B"): TIER_OVERLAPPING,
            ("A", "C"): TIER_NEIGHBORING,
            ("A", "D"): TIER_DISTANT,
        }
        scores = compute_geometric_feasibility_oracle(candidates, tiers)
        assert scores == [1.0, 1.0, 0.5, 0.0, 0.0]


class TestGroundTruthLinking:
    def test_index_tracklet_detections_by_frame(self):
        df = pd.DataFrame(
            {
                "track_id": [1, 1, 2],
                "frame_idx": [0, 1, 0],
                "xmin": [0.0, 1.0, 10.0],
                "ymin": [0.0, 1.0, 10.0],
                "xmax": [5.0, 6.0, 15.0],
                "ymax": [5.0, 6.0, 15.0],
            }
        )
        by_frame = index_tracklet_detections_by_frame(df)
        assert set(by_frame.keys()) == {0, 1}
        assert (1, (0.0, 0.0, 5.0, 5.0)) in by_frame[0]
        assert (2, (10.0, 10.0, 15.0, 15.0)) in by_frame[0]

    def test_links_track_to_majority_vote_identity(self):
        # track_id 1 overlaps ground-truth object "obj_A" on 3 frames, "obj_B" on 1.
        tracklet_detections = {
            0: [(1, (0.0, 0.0, 10.0, 10.0))],
            1: [(1, (0.0, 0.0, 10.0, 10.0))],
            2: [(1, (0.0, 0.0, 10.0, 10.0))],
            3: [(1, (0.0, 0.0, 10.0, 10.0))],
        }
        ground_truth_boxes = {
            0: [GroundTruthBox(object_id="obj_A", bbox=(0.0, 0.0, 10.0, 10.0))],
            1: [GroundTruthBox(object_id="obj_A", bbox=(0.0, 0.0, 10.0, 10.0))],
            2: [GroundTruthBox(object_id="obj_A", bbox=(0.0, 0.0, 10.0, 10.0))],
            3: [GroundTruthBox(object_id="obj_B", bbox=(0.0, 0.0, 10.0, 10.0))],
        }
        identity_map = link_camera_tracklets_to_ground_truth("cam", tracklet_detections, ground_truth_boxes)
        assert identity_map == {1: "obj_A"}

    def test_unmatched_track_is_omitted(self):
        tracklet_detections = {0: [(1, (0.0, 0.0, 10.0, 10.0))]}
        ground_truth_boxes = {0: [GroundTruthBox(object_id="obj_A", bbox=(100.0, 100.0, 110.0, 110.0))]}
        identity_map = link_camera_tracklets_to_ground_truth("cam", tracklet_detections, ground_truth_boxes)
        assert identity_map == {}
