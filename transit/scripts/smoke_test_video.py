#!/usr/bin/env python
"""Ad hoc sanity check: run the detector (and, by default, the tracker) on ANY
local video file -- e.g. a downloaded YouTube CCTV clip -- with no dependency
on the MTMC dataset at all.

This is a smoke test, not an evaluation. It's useful for confirming the
detector/tracker actually work on real-world footage and for producing a quick
annotated demo video. It is NOT useful for validating Re-ID matching or the
plausibility gate: both require ground-truth identity correspondences across
*multiple* cameras viewing the same physical space, which a single arbitrary
clip cannot provide. See NOTES_FOR_TOMORROW.md for what a real evaluation run
needs.

Usage:
    python scripts/smoke_test_video.py --video path/to/clip.mp4
    python scripts/smoke_test_video.py --video path/to/clip.mp4 \
        --weights outputs/yolo_finetune/weights/best.pt
    python scripts/smoke_test_video.py --video path/to/clip.mp4 --detect-only --max-frames 300
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
from tqdm import tqdm

from transit.config import load_config
from transit.data.schema import Detection
from transit.detection.detector import PersonDetector
from transit.tracking.tracker import CameraTracker

logger = logging.getLogger(__name__)

_BOX_COLOR = (60, 200, 60)
_TEXT_COLOR = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the video smoke test."""
    parser = argparse.ArgumentParser(
        description="Run the detector (and optionally the tracker) on an arbitrary local video file."
    )
    parser.add_argument("--video", type=str, required=True, help="Path to a local video file to process.")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to a TransitConfig YAML file."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Checkpoint to load (e.g. a fine-tuned outputs/yolo_finetune/weights/best.pt). "
        "Defaults to the config's stock pretrained detector.",
    )
    parser.add_argument("--conf", type=float, default=None, help="Override the detection confidence threshold.")
    parser.add_argument("--device", type=str, default=None, help="Override the device ('cuda'/'cpu'/'auto').")
    parser.add_argument(
        "--tracker",
        type=str,
        default=None,
        help="Override the tracker config (ignored with --detect-only). Ultralytics ships two built-in "
        "options: 'bytetrack.yaml' (motion-only, the config default) and 'botsort.yaml' (adds camera-motion "
        "compensation + appearance re-ID; slower, often steadier through occlusion/crowding). Any other "
        "tracker (OC-SORT, StrongSORT, etc.) isn't wired in here -- see CameraTracker in "
        "src/transit/tracking/tracker.py for where to add one.",
    )
    parser.add_argument(
        "--camera-id", type=str, default="smoke_test", help="Label stored on each Detection/Tracklet."
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Run per-frame detection without tracking (no persistent IDs). Useful for isolating "
        "whether an issue is in the detector or the tracker.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many input frames (default: process the whole video).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Run inference on every Nth frame (default 1 = every frame); skipped frames are still "
        "written to the output video, unannotated, so playback speed/length is unchanged.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output annotated video path (default: <video_stem>_annotated.mp4 next to the input).",
    )
    return parser.parse_args()


def _current_frame_tracks(tracker: CameraTracker, frame_idx: int) -> list[tuple[int, Detection]]:
    """Return (track_id, detection) pairs for tracklets that were updated on this frame."""
    return [
        (t.track_id, t.detections[-1]) for t in tracker.get_tracklets() if t.detections and t.end_frame == frame_idx
    ]


def _draw_tracks(frame, track_pairs: list[tuple[int, Detection]]) -> None:
    """Draw boxes with persistent track IDs onto `frame`, in place."""
    for track_id, det in track_pairs:
        xmin, ymin, xmax, ymax = (int(v) for v in det.bbox)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), _BOX_COLOR, 2)
        label = f"id={track_id} {det.confidence:.2f}"
        cv2.putText(frame, label, (xmin, max(0, ymin - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT_COLOR, 1, cv2.LINE_AA)


def _draw_detections(frame, detections: list[Detection]) -> None:
    """Draw plain (untracked) boxes onto `frame`, in place."""
    for det in detections:
        xmin, ymin, xmax, ymax = (int(v) for v in det.bbox)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), _BOX_COLOR, 2)
        label = f"{det.confidence:.2f}"
        cv2.putText(frame, label, (xmin, max(0, ymin - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT_COLOR, 1, cv2.LINE_AA)


def main() -> None:
    """Entry point: annotate an arbitrary video with detections/tracks and save it."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    args = parse_args()
    if args.stride < 1:
        raise ValueError(f"--stride must be >= 1, got {args.stride}")

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    overrides: dict = {}
    if args.conf is not None:
        overrides["detection_conf_threshold"] = args.conf
    if args.device is not None:
        overrides["device"] = args.device
    if args.tracker is not None:
        overrides["tracker_config"] = args.tracker
    config = load_config(args.config, overrides=overrides or None)

    if args.tracker is not None and args.detect_only:
        logger.warning("--tracker '%s' has no effect with --detect-only (no tracker is used).", args.tracker)

    output_path = Path(args.output) if args.output else video_path.with_name(f"{video_path.stem}_annotated.mp4")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not read frame size from video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    if args.max_frames is not None:
        total_frames = min(total_frames, args.max_frames) if total_frames else args.max_frames

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {output_path}")

    tracker: CameraTracker | None = None
    detector: PersonDetector | None = None
    if args.detect_only:
        detector = PersonDetector(config, weights_path=args.weights)
        logger.info("Running in detect-only mode (no persistent track IDs).")
    else:
        tracker = CameraTracker(config, camera_id=args.camera_id, weights_path=args.weights)

    seen_track_ids: set[int] = set()
    frame_idx = 0

    with tqdm(total=total_frames, desc=f"Processing {video_path.name}", unit="frame") as pbar:
        while True:
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            timestamp = frame_idx / fps
            if frame_idx % args.stride == 0:
                if tracker is not None:
                    tracker.update(frame, frame_idx=frame_idx, timestamp=timestamp)
                    track_pairs = _current_frame_tracks(tracker, frame_idx)
                    _draw_tracks(frame, track_pairs)
                    seen_track_ids.update(track_id for track_id, _ in track_pairs)
                else:
                    detections = detector.detect(
                        frame, camera_id=args.camera_id, frame_idx=frame_idx, timestamp=timestamp
                    )
                    _draw_detections(frame, detections)

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()

    if tracker is not None:
        logger.info(
            "Done: %d frame(s) processed, %d distinct track ID(s) seen, output -> %s",
            frame_idx,
            len(seen_track_ids),
            output_path,
        )
    else:
        logger.info("Done: %d frame(s) processed, output -> %s", frame_idx, output_path)


if __name__ == "__main__":
    main()
