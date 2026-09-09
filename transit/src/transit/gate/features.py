"""Build appearance and spatiotemporal training features for the gate."""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from transit.data.mtmc_dataset import MTMCScene
from transit.data.schema import CandidateMatch, TransitionEvent
from transit.detection.dataset_export import GroundTruthBox, load_ground_truth_annotations
from transit.preprocessing.transitions import CameraPairTransitionStats, compute_camera_pair_stats

logger = logging.getLogger(__name__)

TrackletKey = tuple[str, int]
FEATURE_COLUMNS = ["appearance_similarity", "transition_time_log_likelihood"]
_MIN_STD = 0.25
_MIN_IOU = 0.30
# Near-zero-variance camera pairs can create extreme Gaussian tails; -50 keeps them numerically stable.
_MIN_LOG_LIKELIHOOD = -50.0


def _log_normal_pdf(value: float, mean: float, std: float) -> float:
    std = max(float(std), _MIN_STD)
    z = (value - mean) / std
    log_likelihood = -0.5 * z * z - math.log(std) - 0.5 * math.log(2.0 * math.pi)
    return max(log_likelihood, _MIN_LOG_LIKELIHOOD)


def _global_stats(stats: dict[tuple[str, str], CameraPairTransitionStats]) -> tuple[float, float]:
    values = [value for pair_stats in stats.values() for value in pair_stats.transit_times]
    if not values:
        return 0.0, 1.0
    return float(np.mean(values)), max(float(np.std(values)), _MIN_STD)


def build_gate_features(
    candidates: list[CandidateMatch],
    camera_pair_stats: dict[tuple[str, str], CameraPairTransitionStats],
    src_exit_times: dict[TrackletKey, float],
    dst_entry_times: dict[TrackletKey, float],
) -> np.ndarray:
    """Build ``[appearance_similarity, transition_time_log_likelihood]`` rows."""
    global_mean, global_std = _global_stats(camera_pair_stats)
    features = np.empty((len(candidates), 2), dtype=np.float32)
    for index, candidate in enumerate(candidates):
        source_key = (candidate.src_camera_id, candidate.src_track_id)
        destination_key = (candidate.dst_camera_id, candidate.dst_track_id)
        transit_time = dst_entry_times[destination_key] - src_exit_times[source_key]
        stats = camera_pair_stats.get((candidate.src_camera_id, candidate.dst_camera_id))
        if stats is None or not stats.transit_times:
            mean, std = global_mean, global_std
        else:
            mean, std = stats.mean_transit_time, max(stats.std_transit_time, _MIN_STD)
        features[index] = [candidate.appearance_similarity, _log_normal_pdf(transit_time, mean, std)]
    return features


def build_hard_negative_labels(
    candidates: list[CandidateMatch],
    ground_truth: dict[TrackletKey, set[TrackletKey]],
) -> np.ndarray:
    """Return 1 for candidates whose destination is a true source match."""
    labels = np.zeros(len(candidates), dtype=np.float32)
    for index, candidate in enumerate(candidates):
        source_key = (candidate.src_camera_id, candidate.src_track_id)
        destination_key = (candidate.dst_camera_id, candidate.dst_track_id)
        labels[index] = float(destination_key in ground_truth.get(source_key, set()))
    return labels


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _tracklet_identity_labels(
    tracklets_by_camera: dict[str, pd.DataFrame],
    annotations: dict[str, dict[int, list[GroundTruthBox]]],
    samples_per_tracklet: int,
) -> dict[TrackletKey, str]:
    # Five evenly spaced detections per tracklet are IoU-matched at threshold 0.30;
    # tracks with no qualifying box are left unlabeled and excluded from gate labels.
    identities: dict[TrackletKey, str] = {}
    for camera_id, tracklets in tracklets_by_camera.items():
        for track_id, group in tracklets.groupby("track_id"):
            sample_count = min(samples_per_tracklet, len(group))
            sample = group.iloc[np.linspace(0, len(group) - 1, sample_count, dtype=int)]
            votes: list[str] = []
            for _, row in sample.iterrows():
                boxes = annotations.get(camera_id, {}).get(int(row.frame_idx), [])
                detection_box = (row.xmin, row.ymin, row.xmax, row.ymax)
                if boxes:
                    object_id, score = max(
                        ((box.object_id, _iou(detection_box, box.bbox)) for box in boxes),
                        key=lambda item: item[1],
                    )
                    if score >= _MIN_IOU:
                        votes.append(object_id)
            if votes:
                identities[(camera_id, int(track_id))] = Counter(votes).most_common(1)[0][0]
    return identities


def build_scene_gate_feature_table(
    scene_dir: str | Path,
    output_dir: str | Path,
    output_path: str | Path | None = None,
    samples_per_tracklet: int = 5,
) -> pd.DataFrame:
    """Build and save a train-ready feature table from completed scene outputs."""
    scene = MTMCScene(scene_dir)
    output_dir = Path(output_dir)
    output_path = Path(output_path or output_dir / "gate_features.parquet")

    candidates_df = pd.read_parquet(output_dir / "candidates.parquet")
    candidates = [CandidateMatch(**row) for row in candidates_df.to_dict("records")]
    events_df = pd.read_parquet(output_dir / "transition_events.parquet")
    events = [TransitionEvent(**row) for row in events_df.to_dict("records")]
    camera_pair_stats = compute_camera_pair_stats(events)

    tracklets_by_camera = {
        camera_id: pd.read_parquet(output_dir / camera_id / "tracklets.parquet")
        for camera_id in scene.camera_ids
    }
    annotations = load_ground_truth_annotations(scene.ground_truth_path())
    identities = _tracklet_identity_labels(tracklets_by_camera, annotations, samples_per_tracklet)

    src_exit_times: dict[TrackletKey, float] = {}
    dst_entry_times: dict[TrackletKey, float] = {}
    for camera_id, tracklets in tracklets_by_camera.items():
        for track_id, group in tracklets.groupby("track_id"):
            key = (camera_id, int(track_id))
            src_exit_times[key] = float(group.timestamp.max())
            dst_entry_times[key] = float(group.timestamp.min())

    valid_candidates = [
        candidate
        for candidate in candidates
        if (candidate.src_camera_id, candidate.src_track_id) in src_exit_times
        and (candidate.dst_camera_id, candidate.dst_track_id) in dst_entry_times
    ]
    ground_truth = {
        source_key: {
            destination_key
            for destination_key, destination_identity in identities.items()
            if destination_identity == source_identity
        }
        for source_key, source_identity in identities.items()
    }
    feature_matrix = build_gate_features(valid_candidates, camera_pair_stats, src_exit_times, dst_entry_times)
    table = pd.DataFrame(feature_matrix, columns=FEATURE_COLUMNS)
    table["label"] = build_hard_negative_labels(valid_candidates, ground_truth).astype(np.int8)
    for column in ("src_camera_id", "src_track_id", "dst_camera_id", "dst_track_id"):
        table[column] = [getattr(candidate, column) for candidate in valid_candidates]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)
    logger.info("Saved %d feature rows (%d positive) to %s", len(table), int(table.label.sum()), output_path)
    return table


__all__ = ["build_gate_features", "build_hard_negative_labels", "build_scene_gate_feature_table"]
