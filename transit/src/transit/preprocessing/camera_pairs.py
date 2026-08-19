"""Classify camera pairs into overlapping / neighbouring / distant tiers.

Uses each camera's ground-plane FOV footprint (preprocessing/visibility.py's
`compute_fov_footprint`) purely as a descriptive/evaluation grouping -- e.g. to
report gate accuracy broken down by tier -- never as an input feature to the gate
itself (the gate must work without hand-specified topology).
"""

from __future__ import annotations

import logging
from itertools import combinations

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CameraPairTier = str  # one of "overlapping", "neighboring_non_overlapping", "distant_disconnected"

TIER_OVERLAPPING: CameraPairTier = "overlapping"
TIER_NEIGHBORING: CameraPairTier = "neighboring_non_overlapping"
TIER_DISTANT: CameraPairTier = "distant_disconnected"


def _polygon_intersection_area(polygon_a: np.ndarray, polygon_b: np.ndarray) -> float:
    """Intersection area of two convex polygons, via OpenCV's convex-convex intersection."""
    area, _ = cv2.intersectConvexConvex(
        polygon_a.astype(np.float32),
        polygon_b.astype(np.float32),
    )
    return float(area)


def _min_polygon_distance(polygon_a: np.ndarray, polygon_b: np.ndarray) -> float:
    """Approximate minimum distance between two polygons as the min pairwise vertex distance.

    This is an approximation (the true closest points may lie on an edge rather
    than at a vertex) but is adequate for coarse neighboring-vs-distant tiering.
    """
    diffs = polygon_a[:, None, :] - polygon_b[None, :, :]
    return float(np.min(np.linalg.norm(diffs, axis=-1)))


def classify_camera_pairs(
    footprints: dict[str, np.ndarray],
    neighbor_distance_threshold: float,
) -> dict[tuple[str, str], CameraPairTier]:
    """Classify every unordered camera pair into an overlap/adjacency tier.

    Args:
        footprints: Mapping from camera_id to that camera's (4, 2) ground-plane
            FOV footprint polygon, as produced by
            preprocessing/visibility.py's `compute_fov_footprint`.
        neighbor_distance_threshold: Maximum ground-plane distance (in the same
            units as the footprints) for two non-overlapping cameras to be
            classified as "neighboring_non_overlapping" rather than
            "distant_disconnected".

    Returns:
        A mapping from (camera_id_a, camera_id_b) -- ordered lexicographically --
        to its CameraPairTier.
    """
    tiers: dict[tuple[str, str], CameraPairTier] = {}

    for camera_a, camera_b in combinations(sorted(footprints), 2):
        polygon_a = footprints[camera_a]
        polygon_b = footprints[camera_b]

        intersection_area = _polygon_intersection_area(polygon_a, polygon_b)
        if intersection_area > 0.0:
            tier = TIER_OVERLAPPING
        else:
            distance = _min_polygon_distance(polygon_a, polygon_b)
            tier = TIER_NEIGHBORING if distance <= neighbor_distance_threshold else TIER_DISTANT

        tiers[(camera_a, camera_b)] = tier

    logger.info("Classified %d camera pair(s) into tiers", len(tiers))
    return tiers
