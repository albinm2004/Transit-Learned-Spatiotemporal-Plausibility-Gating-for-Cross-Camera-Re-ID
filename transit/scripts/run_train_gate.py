#!/usr/bin/env python
"""CLI to train and evaluate the learned plausibility gate.

Runs the full Part-3 evaluation: links tracker tracklets to ground-truth
identities, splits by identity into train/val/eval, fits the transit-time
distributions on the train split only, trains the logistic-regression gate,
and reports Rank-1/Rank-5/mAP/false-positive-rate on the eval split for three
things side by side:
    - baseline_appearance_only: matching/baseline_matcher.py, no gating at all.
    - learned_gate:              this project's actual contribution.
    - calibration_oracle:        evaluation-only upper bound (never trained on).

Prerequisites (run these first, in order -- see NOTES_FOR_TOMORROW.md):
    python scripts/run_detect_track.py --config configs/default.yaml --scene <scene>
    python scripts/run_embed.py --config configs/default.yaml --scene <scene>
    python scripts/run_preprocess_transitions.py --config configs/default.yaml --scene <scene>
    python scripts/run_baseline.py --config configs/default.yaml --scene <scene>

Usage:
    python scripts/run_train_gate.py --config configs/default.yaml --scene scene_001
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from transit.config import TransitConfig, load_config
from transit.data.calibration import load_camera_calibration
from transit.data.mtmc_dataset import MTMCScene
from transit.data.schema import CandidateMatch, TransitionEvent
from transit.detection.dataset_export import load_ground_truth_annotations
from transit.eval.metrics import (
    build_ranked_candidates,
    false_positive_match_rate,
    mean_average_precision,
    rank_k_accuracy,
)
from transit.eval.splits import split_by_identity
from transit.gate.calibration_oracle import compute_geometric_feasibility_oracle
from transit.gate.features import build_gate_features, build_hard_negative_labels
from transit.gate.ground_truth_linking import (
    index_tracklet_detections_by_frame,
    link_camera_tracklets_to_ground_truth,
)
from transit.gate.train_gate import train_logistic_gate
from transit.matching.baseline_matcher import match_baseline
from transit.preprocessing.camera_pairs import classify_camera_pairs
from transit.preprocessing.transitions import compute_camera_pair_stats
from transit.preprocessing.visibility import compute_fov_footprint

logger = logging.getLogger(__name__)

TrackletKey = tuple[str, int]

_DEFAULT_TRAIN_RATIO = 0.6
_DEFAULT_VAL_RATIO = 0.2
_DEFAULT_IOU_THRESHOLD = 0.3
_DEFAULT_NEIGHBOR_DISTANCE_THRESHOLD = 10.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for gate training + evaluation."""
    parser = argparse.ArgumentParser(description="Train and evaluate the learned plausibility gate.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to a TransitConfig YAML file.")
    parser.add_argument("--scene", type=str, default=None, help="Scene name to train on (overrides config).")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["logistic", "mlp"],
        default=None,
        help="Gate architecture to train (overrides config.gate_model_type). Only 'logistic' is wired up so far.",
    )
    parser.add_argument("--train-ratio", type=float, default=_DEFAULT_TRAIN_RATIO, help="Fraction of identities for train.")
    parser.add_argument("--val-ratio", type=float, default=_DEFAULT_VAL_RATIO, help="Fraction of identities for val.")
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=_DEFAULT_IOU_THRESHOLD,
        help="Min IoU for a tracker detection to count as a vote for a ground-truth identity.",
    )
    parser.add_argument(
        "--neighbor-distance-threshold",
        type=float,
        default=_DEFAULT_NEIGHBOR_DISTANCE_THRESHOLD,
        help="Ground-plane distance threshold for the neighboring-vs-distant camera-pair tier (calibration oracle only).",
    )
    return parser.parse_args()


def _load_tracklets_df(config: TransitConfig, camera_id: str) -> pd.DataFrame:
    path = config.output_dir / config.scene_name / camera_id / "tracklets.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No tracklets found for camera '{camera_id}' at {path} -- run scripts/run_detect_track.py first."
        )
    return pd.read_parquet(path)


def build_identity_map(config: TransitConfig, scene: MTMCScene, iou_threshold: float) -> dict[TrackletKey, str]:
    """Link every camera's tracker tracklets to a ground-truth object_id (see
    gate/ground_truth_linking.py)."""
    all_ground_truth = load_ground_truth_annotations(scene.ground_truth_path())
    identity_map: dict[TrackletKey, str] = {}

    for camera_id in scene.camera_ids:
        tracklets_df = _load_tracklets_df(config, camera_id)
        by_frame = index_tracklet_detections_by_frame(tracklets_df)
        camera_map = link_camera_tracklets_to_ground_truth(
            camera_id, by_frame, all_ground_truth.get(camera_id, {}), iou_threshold=iou_threshold
        )
        for track_id, object_id in camera_map.items():
            identity_map[(camera_id, track_id)] = object_id

    logger.info(
        "Linked %d tracklet(s) to ground-truth identities across %d camera(s)",
        len(identity_map),
        len(scene.camera_ids),
    )
    return identity_map


def build_timing_maps(
    config: TransitConfig, scene: MTMCScene
) -> tuple[dict[TrackletKey, float], dict[TrackletKey, float]]:
    """Per-tracklet last-seen (exit) and first-seen (entry) timestamps, from tracklets.parquet."""
    exit_times: dict[TrackletKey, float] = {}
    entry_times: dict[TrackletKey, float] = {}

    for camera_id in scene.camera_ids:
        tracklets_df = _load_tracklets_df(config, camera_id)
        grouped = tracklets_df.groupby("track_id")["timestamp"].agg(["min", "max"])
        for track_id, row in grouped.iterrows():
            key = (camera_id, int(track_id))
            entry_times[key] = float(row["min"])
            exit_times[key] = float(row["max"])

    return exit_times, entry_times


def build_ground_truth_matches(identity_map: dict[TrackletKey, str]) -> dict[TrackletKey, set[TrackletKey]]:
    """For every linked tracklet, the set of OTHER-camera tracklets sharing its ground-truth identity."""
    by_object: dict[str, list[TrackletKey]] = {}
    for key, object_id in identity_map.items():
        by_object.setdefault(object_id, []).append(key)

    ground_truth: dict[TrackletKey, set[TrackletKey]] = {}
    for keys in by_object.values():
        for key in keys:
            ground_truth[key] = {other for other in keys if other != key and other[0] != key[0]}
    return ground_truth


def _identity_split(object_id: str | None, train_ids: set[str], val_ids: set[str]) -> str | None:
    if object_id is None:
        return None
    if object_id in train_ids:
        return "train"
    if object_id in val_ids:
        return "val"
    return "eval"


def split_candidates(
    candidates: list[CandidateMatch],
    identity_map: dict[TrackletKey, str],
    train_ids: set[str],
    val_ids: set[str],
) -> dict[str, list[CandidateMatch]]:
    """Route candidates into train/val/eval by their SOURCE tracklet's ground-truth identity.

    Candidates whose source tracklet has no known ground-truth identity (an
    unmatched/background tracker track) are dropped -- there is no reliable label
    to train or evaluate against for them.
    """
    buckets: dict[str, list[CandidateMatch]] = {"train": [], "val": [], "eval": []}
    n_dropped = 0

    for candidate in candidates:
        src_key = (candidate.src_camera_id, candidate.src_track_id)
        split = _identity_split(identity_map.get(src_key), train_ids, val_ids)
        if split is None:
            n_dropped += 1
            continue
        buckets[split].append(candidate)

    if n_dropped:
        logger.info("Dropped %d/%d candidate(s) with no ground-truth source identity", n_dropped, len(candidates))
    return buckets


def _ground_truth_for_split(
    ground_truth: dict[TrackletKey, set[TrackletKey]], split_candidates: list[CandidateMatch]
) -> dict[TrackletKey, set[TrackletKey]]:
    """Restrict `ground_truth` to just the source tracklet keys present in this split.

    Without this, mean_average_precision/rank_k_accuracy would iterate over EVERY
    identity's ground truth (train+val+eval combined), scoring train/val queries
    that were never given a chance to appear in this split's ranked candidates as
    complete misses -- silently deflating every metric. Scoping to the split's own
    source keys is what makes "eval-split mAP" mean what it says.
    """
    keys = {(c.src_camera_id, c.src_track_id) for c in split_candidates}
    return {key: ground_truth.get(key, set()) for key in keys}


def _evaluate(
    name: str,
    scored_candidates: list[CandidateMatch],
    ground_truth: dict[TrackletKey, set[TrackletKey]],
    score_attr: str,
    accepted_matches: list[CandidateMatch] | None = None,
) -> dict[str, float]:
    ranked = build_ranked_candidates(scored_candidates, score_attr=score_attr)
    result = {
        "rank1": rank_k_accuracy(ranked, ground_truth, k=1),
        "rank5": rank_k_accuracy(ranked, ground_truth, k=5),
        "mAP": mean_average_precision(ranked, ground_truth),
    }
    if accepted_matches is not None:
        result["false_positive_rate"] = false_positive_match_rate(accepted_matches, ground_truth)

    logger.info("[%s] %s", name, result)
    return result


def _dedupe_best_per_source(scored_candidates: list[CandidateMatch], score_attr: str) -> list[CandidateMatch]:
    """Keep only the single best-scoring candidate per source tracklet (mirrors
    matching/baseline_matcher.py's accept-one-per-query semantics)."""
    best_by_source: dict[TrackletKey, CandidateMatch] = {}
    for candidate in scored_candidates:
        key = (candidate.src_camera_id, candidate.src_track_id)
        score = getattr(candidate, score_attr)
        current_best = best_by_source.get(key)
        if current_best is None or score > getattr(current_best, score_attr):
            best_by_source[key] = candidate
    return list(best_by_source.values())


def main() -> None:
    """Entry point: link identities, split, train the gate, and evaluate against baseline + oracle."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    args = parse_args()
    overrides: dict[str, object] = {}
    if args.scene:
        overrides["scene_name"] = args.scene
    if args.model_type:
        overrides["gate_model_type"] = args.model_type
    config = load_config(args.config, overrides=overrides or None)

    if config.gate_model_type != "logistic":
        raise NotImplementedError(
            f"gate_model_type={config.gate_model_type!r} is not wired into this script yet -- "
            "only 'logistic' is implemented end-to-end so far (see gate/mlp_gate.py, still a stub)."
        )

    scene = MTMCScene(config.scene_dir)
    if not scene.camera_ids:
        logger.warning("No cameras found in scene directory %s", config.scene_dir)
        return

    output_dir = config.output_dir / config.scene_name
    candidates_path = output_dir / "candidates.parquet"
    if not candidates_path.exists():
        raise FileNotFoundError(f"No candidates found at {candidates_path} -- run scripts/run_baseline.py first.")

    candidates_df = pd.read_parquet(candidates_path)
    candidates = [
        CandidateMatch(
            src_camera_id=row.src_camera_id,
            src_track_id=int(row.src_track_id),
            dst_camera_id=row.dst_camera_id,
            dst_track_id=int(row.dst_track_id),
            appearance_similarity=float(row.appearance_similarity),
        )
        for row in candidates_df.itertuples()
    ]
    logger.info("Loaded %d candidate(s) from %s", len(candidates), candidates_path)

    # --- Ground-truth linkage (tracker track_id -> dataset object_id) ---
    identity_map = build_identity_map(config, scene, args.iou_threshold)
    exit_times, entry_times = build_timing_maps(config, scene)
    ground_truth = build_ground_truth_matches(identity_map)

    object_ids = sorted(set(identity_map.values()))
    if not object_ids:
        raise RuntimeError(
            "No tracklet was linked to any ground-truth identity -- check the IoU-linking log "
            "output above (camera-by-camera match counts) and consider lowering --iou-threshold."
        )
    train_ids, val_ids, _eval_ids = split_by_identity(object_ids, args.train_ratio, args.val_ratio)
    train_ids_set, val_ids_set = set(train_ids), set(val_ids)

    buckets = split_candidates(candidates, identity_map, train_ids_set, val_ids_set)
    logger.info(
        "Candidate split: train=%d val=%d eval=%d",
        len(buckets["train"]),
        len(buckets["val"]),
        len(buckets["eval"]),
    )

    # --- Camera-pair transit-time stats, fit on TRAIN identities only ---
    transition_events_path = output_dir / "transition_events.parquet"
    if not transition_events_path.exists():
        raise FileNotFoundError(
            f"No transition events found at {transition_events_path} -- run "
            "scripts/run_preprocess_transitions.py first."
        )
    events_df = pd.read_parquet(transition_events_path)
    train_events = [
        TransitionEvent(
            object_id=str(row.object_id),
            src_camera_id=row.src_camera_id,
            dst_camera_id=row.dst_camera_id,
            src_exit_time=float(row.src_exit_time),
            dst_entry_time=float(row.dst_entry_time),
            transit_time=float(row.transit_time),
        )
        for row in events_df.itertuples()
        if str(row.object_id) in train_ids_set
    ]
    camera_pair_stats = compute_camera_pair_stats(train_events)
    logger.info(
        "Fit transit-time stats for %d camera pair(s) from %d train-split transition event(s)",
        len(camera_pair_stats),
        len(train_events),
    )

    # --- Build gate features/labels per split ---
    train_features = build_gate_features(buckets["train"], camera_pair_stats, exit_times, entry_times)
    train_labels = build_hard_negative_labels(buckets["train"], ground_truth)
    val_features = build_gate_features(buckets["val"], camera_pair_stats, exit_times, entry_times)
    val_labels = build_hard_negative_labels(buckets["val"], ground_truth)
    eval_features = build_gate_features(buckets["eval"], camera_pair_stats, exit_times, entry_times)

    logger.info(
        "Feature matrices: train=%s (%.1f%% positive), val=%s (%.1f%% positive), eval=%s",
        train_features.shape,
        100.0 * train_labels.mean() if len(train_labels) else 0.0,
        val_features.shape,
        100.0 * val_labels.mean() if len(val_labels) else 0.0,
        eval_features.shape,
    )

    # --- Train the gate ---
    model, threshold = train_logistic_gate(train_features, train_labels, val_features, val_labels, config)

    model_path = output_dir / "gate_model.joblib"
    joblib.dump(
        {"model": model, "threshold": threshold, "feature_columns": ["appearance_similarity", "transition_log_likelihood"]},
        model_path,
    )
    logger.info("Saved trained gate -> %s", model_path)

    eval_candidates = buckets["eval"]
    eval_probs = model.predict_proba(eval_features)[:, 1] if len(eval_candidates) else np.array([])
    for candidate, prob in zip(eval_candidates, eval_probs):
        candidate.gate_score = float(prob)

    gate_accepted = _dedupe_best_per_source(
        [c for c in eval_candidates if c.gate_score >= threshold], "gate_score"
    )

    # --- Appearance-only baseline, on the same eval split ---
    baseline_accepted = match_baseline(eval_candidates, threshold=config.baseline_similarity_threshold)

    # --- Calibration oracle, evaluation-only ---
    have_oracle = False
    try:
        footprints = {}
        for camera_id in scene.camera_ids:
            calibration = load_camera_calibration(config.scene_dir, camera_id)
            cap = cv2.VideoCapture(str(scene.video_path(camera_id)))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            footprints[camera_id] = compute_fov_footprint(calibration, width, height)

        camera_pair_tiers = classify_camera_pairs(footprints, args.neighbor_distance_threshold)
        oracle_scores = compute_geometric_feasibility_oracle(eval_candidates, camera_pair_tiers)
        for candidate, score in zip(eval_candidates, oracle_scores):
            candidate.oracle_score = score
        have_oracle = True
    except FileNotFoundError as exc:
        logger.warning("Skipping calibration-oracle comparison: %s", exc)

    # --- Evaluate, scoped to this eval split's own queries only ---
    eval_ground_truth = _ground_truth_for_split(ground_truth, eval_candidates)

    results = {
        "baseline_appearance_only": _evaluate(
            "baseline", eval_candidates, eval_ground_truth, "appearance_similarity", baseline_accepted
        ),
        "learned_gate": _evaluate("gate", eval_candidates, eval_ground_truth, "gate_score", gate_accepted),
    }
    if have_oracle:
        results["calibration_oracle"] = _evaluate("oracle", eval_candidates, eval_ground_truth, "oracle_score")

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "gate_eval_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote evaluation results -> %s", results_path)


if __name__ == "__main__":
    main()
