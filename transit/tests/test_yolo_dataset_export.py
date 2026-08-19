"""Tests for detection/dataset_export.py's pure label-conversion and splitting math.

No real video files or ground-truth annotations are needed -- these test the math
on tiny synthetic examples.
"""

from __future__ import annotations

from transit.detection.dataset_export import (
    bbox_to_yolo_label,
    sample_frame_indices,
    train_val_split,
)


def test_bbox_to_yolo_label_centered_box() -> None:
    """A box exactly filling the image should normalize to a full-frame YOLO label."""
    label = bbox_to_yolo_label((0.0, 0.0, 100.0, 200.0), img_width=100, img_height=200)

    class_id, cx, cy, w, h = label.split()
    assert class_id == "0"
    assert float(cx) == 0.5
    assert float(cy) == 0.5
    assert float(w) == 1.0
    assert float(h) == 1.0


def test_bbox_to_yolo_label_quarter_box() -> None:
    """A box covering the top-left quadrant should have center at (0.25, 0.25)."""
    label = bbox_to_yolo_label((0.0, 0.0, 50.0, 50.0), img_width=100, img_height=100)

    class_id, cx, cy, w, h = label.split()
    assert class_id == "0"
    assert float(cx) == 0.25
    assert float(cy) == 0.25
    assert float(w) == 0.5
    assert float(h) == 0.5


def test_bbox_to_yolo_label_custom_class_id() -> None:
    """class_id should be emitted as passed."""
    label = bbox_to_yolo_label((10.0, 10.0, 20.0, 20.0), img_width=100, img_height=100, class_id=3)
    assert label.split()[0] == "3"


def test_bbox_to_yolo_label_clips_out_of_bounds_box() -> None:
    """A box that overflows the image bounds should be clipped, not raise."""
    label = bbox_to_yolo_label((-10.0, -10.0, 150.0, 150.0), img_width=100, img_height=100)

    class_id, cx, cy, w, h = label.split()
    assert float(cx) == 0.5
    assert float(cy) == 0.5
    assert float(w) == 1.0
    assert float(h) == 1.0


def test_bbox_to_yolo_label_rejects_non_positive_image_size() -> None:
    """Non-positive image dimensions should raise ValueError."""
    try:
        bbox_to_yolo_label((0.0, 0.0, 10.0, 10.0), img_width=0, img_height=100)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_sample_frame_indices_strides_sorted_frames() -> None:
    """Sampling should keep every Nth annotated frame, sorted ascending."""
    frames = [7, 3, 1, 9, 5, 11]
    sampled = sample_frame_indices(frames, stride=2)

    assert sampled == [1, 5, 9]


def test_sample_frame_indices_stride_one_keeps_all() -> None:
    """A stride of 1 should keep every annotated frame."""
    frames = [4, 2, 3]
    assert sample_frame_indices(frames, stride=1) == [2, 3, 4]


def test_sample_frame_indices_rejects_invalid_stride() -> None:
    """A stride below 1 should raise ValueError."""
    try:
        sample_frame_indices([1, 2, 3], stride=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_train_val_split_ratio_and_disjointness() -> None:
    """train_val_split should partition items disjointly, respecting the ratio."""
    items = [("cam1", i) for i in range(20)]
    train_items, val_items = train_val_split(items, train_ratio=0.75, seed=1)

    assert len(train_items) == 15
    assert len(val_items) == 5
    assert set(train_items).isdisjoint(set(val_items))
    assert set(train_items) | set(val_items) == set(items)


def test_train_val_split_is_deterministic_given_seed() -> None:
    """Repeated calls with the same seed should produce the same split."""
    items = [("cam1", i) for i in range(10)]
    split_a = train_val_split(items, train_ratio=0.6, seed=7)
    split_b = train_val_split(items, train_ratio=0.6, seed=7)

    assert split_a == split_b


def test_train_val_split_rejects_invalid_ratio() -> None:
    """A train_ratio outside (0, 1) should raise ValueError."""
    try:
        train_val_split([1, 2, 3], train_ratio=1.5)
        assert False, "Expected ValueError"
    except ValueError:
        pass
