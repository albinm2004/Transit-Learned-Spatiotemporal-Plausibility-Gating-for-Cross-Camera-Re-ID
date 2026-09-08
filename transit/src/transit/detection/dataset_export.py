"""Export MTMC ground-truth person annotations into a YOLO-format training dataset.

Produces the standard Ultralytics detection dataset layout::

    <dataset_export_dir>/
        images/train/<camera_id>_<frame_idx>.jpg
        images/val/<camera_id>_<frame_idx>.jpg
        labels/train/<camera_id>_<frame_idx>.txt
        labels/val/<camera_id>_<frame_idx>.txt
        dataset.yaml

The ground-truth JSON schema assumed by ``load_ground_truth_annotations`` below
has been confirmed against the dataset's own Hugging Face README (Sept 2026),
replacing an earlier provisional guess -- see that function's docstring. One
thing that specific guess is still unverified: the exact ``object_type`` string
used for person annotations (``_PERSON_OBJECT_TYPES`` below assumes "Person");
``load_ground_truth_annotations`` logs any *other* object types it skips, so
check the log on first real run against a downloaded file.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from transit.config import TransitConfig
from transit.data.mtmc_dataset import MTMCScene

logger = logging.getLogger(__name__)

_PERSON_CLASS_ID = 0
_SPLIT_SEED = 42

# TODO: confirm this against a real downloaded ground_truth.json (see module
# docstring) -- the dataset may label other object types (vehicles, robots, etc.
# in some scenes) that must NOT be exported as YOLO "person" labels.
_PERSON_OBJECT_TYPES = {"Person"}


@dataclass
class GroundTruthBox:
    """A single ground-truth person bounding box on one frame.

    Attributes:
        object_id: Global/scene identity id of the annotated person.
        bbox: Bounding box as [xmin, ymin, xmax, ymax] in pixel coordinates.
    """

    object_id: str
    bbox: tuple[float, float, float, float]


def load_ground_truth_annotations(
    gt_path: str | Path, object_types: set[str] | None = None
) -> dict[str, dict[int, list[GroundTruthBox]]]:
    """Load a scene's ground truth and split it out per camera.

    MTMC_Tracking_2025 ships ONE ground_truth.json per SCENE (not per camera),
    keyed by frame id at the top level, with each object's 2D box given per
    camera it's visible in -- confirmed against the dataset's own Hugging Face
    README (Sept 2026), replacing an earlier provisional per-camera-file guess::

        {
          "<frame_id>": [
            {
              "object_type": "Person",
              "object_id": <int>,
              "2d_bounding_box_visible": {"<camera_id>": [xmin, ymin, xmax, ymax], ...}
              # (plus 3d_location / 3d_bounding_box_scale / 3d_bounding_box_rotation,
              # unused here)
            },
            ...
          ],
          ...
        }

    This is called ONCE per scene (the file can be very large, e.g. ~300MB for a
    25-camera warehouse scene) -- callers should NOT call this per camera.

    Args:
        gt_path: Path to the scene's ground_truth.json (see MTMCScene.ground_truth_path).
        object_types: Object type strings to keep; defaults to
            ``_PERSON_OBJECT_TYPES``. Anything else (vehicles, robots, etc., where
            present in a scene) is dropped -- see the module docstring re:
            confirming the exact person object_type string.

    Returns:
        camera_id -> {frame_idx -> [GroundTruthBox, ...]}.
    """
    object_types = object_types or _PERSON_OBJECT_TYPES
    gt_path = Path(gt_path)
    with gt_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    per_camera: dict[str, dict[int, list[GroundTruthBox]]] = {}
    skipped_types: set[str] = set()
    for frame_id_str, objects in raw.items():
        frame_idx = int(frame_id_str)
        for obj in objects:
            object_type = obj.get("object_type", "")
            if object_type not in object_types:
                skipped_types.add(object_type)
                continue
            object_id = str(obj["object_id"])
            boxes_by_camera = obj.get("2d_bounding_box_visible", {}) or {}
            for camera_id, bbox in boxes_by_camera.items():
                per_camera.setdefault(camera_id, {}).setdefault(frame_idx, []).append(
                    GroundTruthBox(object_id=object_id, bbox=tuple(float(v) for v in bbox))
                )

    if skipped_types:
        logger.info("Skipped non-%s object type(s) in %s: %s", sorted(object_types), gt_path, sorted(skipped_types))
    logger.info(
        "Loaded ground truth for %d camera(s) from %s (%d frame(s) total)",
        len(per_camera),
        gt_path,
        len(raw),
    )
    return per_camera


def bbox_to_yolo_label(
    bbox: tuple[float, float, float, float],
    img_width: int,
    img_height: int,
    class_id: int = _PERSON_CLASS_ID,
) -> str:
    """Convert a pixel-space [xmin, ymin, xmax, ymax] bbox to a YOLO label line.

    Args:
        bbox: Bounding box as [xmin, ymin, xmax, ymax] in pixel coordinates.
        img_width: Width of the image the bbox belongs to, in pixels.
        img_height: Height of the image the bbox belongs to, in pixels.
        class_id: YOLO class index to emit (always 0 / "person" for this project).

    Returns:
        A single YOLO label line: "<class_id> <cx> <cy> <w> <h>", with all four
        geometry values normalized to [0, 1] relative to image size.

    Raises:
        ValueError: If img_width or img_height is non-positive.
    """
    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"img_width and img_height must be positive, got {img_width}x{img_height}")

    xmin, ymin, xmax, ymax = bbox
    # Clip to image bounds so annotations that slightly overflow the frame (a
    # common artifact of ground-truth boxes near frame edges) don't produce
    # out-of-[0,1] YOLO coordinates.
    xmin = max(0.0, min(xmin, img_width))
    xmax = max(0.0, min(xmax, img_width))
    ymin = max(0.0, min(ymin, img_height))
    ymax = max(0.0, min(ymax, img_height))

    box_w = max(0.0, xmax - xmin)
    box_h = max(0.0, ymax - ymin)
    cx = xmin + box_w / 2.0
    cy = ymin + box_h / 2.0

    return (
        f"{class_id} "
        f"{cx / img_width:.6f} {cy / img_height:.6f} "
        f"{box_w / img_width:.6f} {box_h / img_height:.6f}"
    )


def sample_frame_indices(annotated_frame_indices: list[int], stride: int) -> list[int]:
    """Select every Nth annotated frame index, sorted ascending.

    Sampling from the set of *annotated* frames (rather than every raw video
    frame) avoids wasting the stride on frames that have no ground truth to export.

    Args:
        annotated_frame_indices: Frame indices that have ground-truth annotations.
        stride: Keep every `stride`-th frame (1 = keep all).

    Returns:
        The sorted, strided subset of frame indices.

    Raises:
        ValueError: If stride is less than 1.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    ordered = sorted(annotated_frame_indices)
    return ordered[::stride]


def train_val_split(items: list, train_ratio: float, seed: int = _SPLIT_SEED) -> tuple[list, list]:
    """Randomly split a list of items into train/val subsets.

    Args:
        items: Items to split (e.g. (camera_id, frame_idx) pairs).
        train_ratio: Fraction of items assigned to the train split.
        seed: RNG seed, fixed by default for reproducible exports.

    Returns:
        (train_items, val_items).

    Raises:
        ValueError: If train_ratio is not in (0, 1).
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    split_idx = round(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def chronological_split_by_camera(
    items: list[tuple[str, int]], train_ratio: float
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Split (camera_id, frame_idx) items into train/val, chronologically per camera.

    Unlike `train_val_split`, this does NOT shuffle. For each camera, its sampled
    frame indices are sorted ascending and the earliest `train_ratio` fraction
    becomes train, the remainder becomes val. This guarantees every train frame
    for a camera precedes every val frame for that camera in time.

    This matters because a random shuffle-and-split (as `train_val_split` does)
    can place near-duplicate, temporally-adjacent frames from the same camera on
    opposite sides of the train/val boundary. Consecutive frames of the same
    person barely differ, so the detector effectively "sees" val examples during
    training -- inflating validation accuracy without a real generalisation gain.
    A per-camera chronological split avoids that leakage; this is the split
    `export_yolo_dataset` actually uses.

    Args:
        items: (camera_id, frame_idx) pairs, e.g. the output of sampling frames
            per camera in `export_yolo_dataset`.
        train_ratio: Fraction of each camera's frames assigned to train.

    Returns:
        (train_items, val_items).

    Raises:
        ValueError: If train_ratio is not in (0, 1).
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    frames_by_camera: dict[str, list[int]] = {}
    for camera_id, frame_idx in items:
        frames_by_camera.setdefault(camera_id, []).append(frame_idx)

    train_items: list[tuple[str, int]] = []
    val_items: list[tuple[str, int]] = []
    for camera_id, frame_indices in frames_by_camera.items():
        ordered = sorted(frame_indices)
        split_idx = round(len(ordered) * train_ratio)
        train_items.extend((camera_id, frame_idx) for frame_idx in ordered[:split_idx])
        val_items.extend((camera_id, frame_idx) for frame_idx in ordered[split_idx:])

    return train_items, val_items


def export_yolo_dataset(config: TransitConfig) -> Path:
    """Export the configured scene's ground truth into a YOLO training dataset.

    For every camera in the scene: loads ground-truth annotations, samples frames
    per ``config.training.frame_sample_stride``, extracts those video frames as
    images, writes YOLO-format label files, splits into train/val per
    ``config.training.train_val_split``, and writes an Ultralytics ``dataset.yaml``.

    Args:
        config: TransitConfig with dataset_root/scene_name pointing at a real,
            downloaded scene, and a populated `training` sub-config.

    Returns:
        Path to the generated dataset.yaml.
    """
    scene = MTMCScene(config.scene_dir)
    export_dir = config.training.dataset_export_dir

    for split in ("train", "val"):
        (export_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (export_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_items: list[tuple[str, int]] = []  # (camera_id, frame_idx)
    # Loaded ONCE for the whole scene -- ground_truth.json is a single per-scene
    # file (can be very large), not one file per camera. See
    # load_ground_truth_annotations's docstring.
    all_annotations = load_ground_truth_annotations(scene.ground_truth_path())

    per_camera_annotations: dict[str, dict[int, list[GroundTruthBox]]] = {}

    for camera_id in scene.camera_ids:
        annotations = all_annotations.get(camera_id, {})
        if not annotations:
            logger.warning("No ground-truth annotations found for camera '%s'", camera_id)
        per_camera_annotations[camera_id] = annotations

        sampled_frames = sample_frame_indices(list(annotations.keys()), config.training.frame_sample_stride)
        all_items.extend((camera_id, frame_idx) for frame_idx in sampled_frames)

    train_items, val_items = chronological_split_by_camera(all_items, config.training.train_val_split)
    split_of: dict[tuple[str, int], str] = {}
    split_of.update({item: "train" for item in train_items})
    split_of.update({item: "val" for item in val_items})

    for camera_id in scene.camera_ids:
        video_path = scene.video_path(camera_id)
        _export_camera_frames(
            camera_id=camera_id,
            video_path=video_path,
            annotations=per_camera_annotations[camera_id],
            split_of=split_of,
            export_dir=export_dir,
        )

    dataset_yaml_path = write_dataset_yaml(export_dir)
    logger.info(
        "Exported YOLO dataset: %d train, %d val image(s) -> %s",
        len(train_items),
        len(val_items),
        dataset_yaml_path,
    )
    return dataset_yaml_path


def _export_camera_frames(
    camera_id: str,
    video_path: Path,
    annotations: dict[int, list[GroundTruthBox]],
    split_of: dict[tuple[str, int], str],
    export_dir: Path,
) -> None:
    """Extract and save the frames/labels for one camera that fall in split_of."""
    wanted_frames = {frame_idx for (cam, frame_idx) in split_of if cam == camera_id}
    if not wanted_frames:
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    frame_idx = 0
    with tqdm(total=len(wanted_frames), desc=f"Exporting {camera_id}", unit="frame") as pbar:
        while wanted_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in wanted_frames:
                split = split_of[(camera_id, frame_idx)]
                img_height, img_width = frame.shape[:2]

                stem = f"{camera_id}_{frame_idx}"
                image_path = export_dir / "images" / split / f"{stem}.jpg"
                label_path = export_dir / "labels" / split / f"{stem}.txt"

                cv2.imwrite(str(image_path), frame)
                lines = [
                    bbox_to_yolo_label(box.bbox, img_width, img_height) for box in annotations[frame_idx]
                ]
                label_path.write_text("\n".join(lines), encoding="utf-8")

                wanted_frames.discard(frame_idx)
                pbar.update(1)
            frame_idx += 1

    cap.release()
    if wanted_frames:
        logger.warning(
            "Camera '%s': %d annotated frame(s) were never reached in the video "
            "(video shorter than annotations?): %s",
            camera_id,
            len(wanted_frames),
            sorted(wanted_frames),
        )


def write_dataset_yaml(export_dir: str | Path) -> Path:
    """Write the Ultralytics dataset.yaml describing the exported dataset.

    Args:
        export_dir: Root of the exported YOLO dataset (contains images/, labels/).

    Returns:
        Path to the written dataset.yaml.
    """
    export_dir = Path(export_dir)
    dataset_yaml = {
        "path": str(export_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["person"],
    }
    dataset_yaml_path = export_dir / "dataset.yaml"
    with dataset_yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)
    return dataset_yaml_path
