"""Link tracker-assigned tracklets to ground-truth global identities via IoU voting.

The tracker (ByteTrack, via CameraTracker) assigns track_ids that are only unique
within a single camera and have no knowledge of the dataset's ground-truth global
object_id. To train and evaluate the gate we need to know, for each
(camera_id, track_id) tracklet the tracker produced, which ground-truth object_id
(if any) it actually corresponds to -- this module derives that mapping by
per-frame IoU matching against ground_truth.json, then a majority vote per
tracklet.

This is standard MOT-evaluation-style association (akin to how MOTA/IDF1 assign
predicted tracks to ground-truth tracks). It is NOT part of the gate itself and is
never used at inference time in a deployed setting -- it exists purely to
construct training/eval labels for gate/features.py and gate/train_gate.py from a
scene's ground truth.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

import pandas as pd

from transit.detection.dataset_export import GroundTruthBox

logger = logging.getLogger(__name__)

_DEFAULT_IOU_THRESHOLD = 0.3

BBox = tuple[float, float, float, float]


def _iou(box_a: BBox, box_b: BBox) -> float:
    """Intersection-over-union of two [xmin, ymin, xmax, ymax] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0.0 else 0.0


def index_tracklet_detections_by_frame(
    tracklets_df: pd.DataFrame,
) -> dict[int, list[tuple[int, BBox]]]:
    """Turn one camera's tracklets.parquet (as saved by run_detect_track.py) into
    a per-frame index of (track_id, bbox) pairs, for IoU matching against ground
    truth.

    Args:
        tracklets_df: One camera's tracklets, as loaded from
            `<output_dir>/<scene>/<camera_id>/tracklets.parquet` -- one row per
            detection, with columns track_id, frame_idx, xmin, ymin, xmax, ymax
            (see scripts/run_detect_track.py's `save_tracklets`).

    Returns:
        Mapping from frame_idx to the list of (track_id, bbox) detections on that
        frame.
    """
    by_frame: dict[int, list[tuple[int, BBox]]] = defaultdict(list)
    for row in tracklets_df.itertuples():
        bbox = (float(row.xmin), float(row.ymin), float(row.xmax), float(row.ymax))
        by_frame[int(row.frame_idx)].append((int(row.track_id), bbox))
    return dict(by_frame)


def link_camera_tracklets_to_ground_truth(
    camera_id: str,
    tracklet_detections: dict[int, list[tuple[int, BBox]]],
    ground_truth_boxes: dict[int, list[GroundTruthBox]],
    iou_threshold: float = _DEFAULT_IOU_THRESHOLD,
) -> dict[int, str]:
    """Assign each tracker track_id on one camera to a ground-truth object_id.

    For every frame present in both `tracklet_detections` and `ground_truth_boxes`,
    each tracker detection is matched to its highest-IoU ground-truth box (if any
    box reaches `iou_threshold`), casting one "vote" for that box's object_id.
    Each track_id's final identity is its majority-vote object_id across all
    frames it appeared in.

    Args:
        camera_id: Camera these tracklets belong to (used only for logging).
        tracklet_detections: frame_idx -> [(track_id, bbox), ...], the tracker's
            own per-frame detections for this camera -- see
            `index_tracklet_detections_by_frame`.
        ground_truth_boxes: frame_idx -> [GroundTruthBox, ...] for this camera,
            e.g. one entry of detection/dataset_export.py's
            `load_ground_truth_annotations` output.
        iou_threshold: Minimum IoU for a tracker detection to be counted as a vote
            for a ground-truth box's object_id on a given frame.

    Returns:
        Mapping from track_id to its majority-vote ground-truth object_id. A
        track_id that never reaches `iou_threshold` against any ground-truth box
        on any frame is omitted (treated as an unmatched/background track -- e.g.
        a false-positive detection track).
    """
    votes: dict[int, Counter] = defaultdict(Counter)
    all_track_ids: set[int] = set()

    shared_frames = set(tracklet_detections) & set(ground_truth_boxes)
    for frame_idx in shared_frames:
        gt_boxes = ground_truth_boxes[frame_idx]
        for track_id, bbox in tracklet_detections[frame_idx]:
            all_track_ids.add(track_id)
            if not gt_boxes:
                continue

            best_iou, best_object_id = 0.0, None
            for gt_box in gt_boxes:
                iou = _iou(bbox, gt_box.bbox)
                if iou > best_iou:
                    best_iou, best_object_id = iou, gt_box.object_id

            if best_object_id is not None and best_iou >= iou_threshold:
                votes[track_id][best_object_id] += 1

    # Track ids that only ever appeared on frames with no ground truth at all
    # still need to be counted in the denominator below.
    for frame_idx, detections in tracklet_detections.items():
        for track_id, _ in detections:
            all_track_ids.add(track_id)

    identity_map = {track_id: counter.most_common(1)[0][0] for track_id, counter in votes.items()}

    logger.info(
        "Camera '%s': linked %d/%d tracklet(s) to a ground-truth identity (iou>=%.2f)",
        camera_id,
        len(identity_map),
        len(all_track_ids),
        iou_threshold,
    )
    return identity_map
