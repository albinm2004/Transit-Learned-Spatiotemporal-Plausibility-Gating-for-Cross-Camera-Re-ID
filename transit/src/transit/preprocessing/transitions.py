"""Derive cross-camera transition events and per-camera-pair empirical statistics.

Consumes VisibilityInterval objects from preprocessing/visibility.py (ground-truth
identity driven) to build the TransitionEvent training data that gate/features.py
will eventually turn into transition-time-likelihood features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from transit.data.schema import TransitionEvent
from transit.preprocessing.visibility import VisibilityInterval

logger = logging.getLogger(__name__)


@dataclass
class CameraPairTransitionStats:
    """Empirical distribution of observed transit times for one camera pair.

    Attributes:
        src_camera_id: Source camera.
        dst_camera_id: Destination camera.
        count: Number of observed transitions for this pair.
        mean_transit_time: Mean transit time, in seconds.
        std_transit_time: Standard deviation of transit time, in seconds.
        min_transit_time: Minimum observed transit time, in seconds.
        max_transit_time: Maximum observed transit time, in seconds.
        transit_times: The raw observed transit times, for downstream density
            estimation (e.g. a KDE or histogram used by gate/features.py).
    """

    src_camera_id: str
    dst_camera_id: str
    count: int
    mean_transit_time: float
    std_transit_time: float
    min_transit_time: float
    max_transit_time: float
    transit_times: list[float] = field(default_factory=list)


def derive_transition_events(intervals: list[VisibilityInterval]) -> list[TransitionEvent]:
    """Derive transition events from each object's chronologically ordered intervals.

    For every object, orders its VisibilityIntervals by start_time and emits one
    TransitionEvent for each consecutive pair of intervals that occur on
    different cameras (consecutive same-camera intervals, e.g. a brief tracking
    dropout and re-detection, are not transitions).

    Args:
        intervals: VisibilityInterval objects, as produced by
            preprocessing/visibility.py's `compute_visibility_intervals`.

    Returns:
        A list of TransitionEvent, one per observed cross-camera transition.
    """
    by_object: dict[str, list[VisibilityInterval]] = {}
    for interval in intervals:
        by_object.setdefault(interval.object_id, []).append(interval)

    events: list[TransitionEvent] = []
    for object_id, object_intervals in by_object.items():
        ordered = sorted(object_intervals, key=lambda iv: iv.start_time)
        for prev_interval, next_interval in zip(ordered, ordered[1:]):
            if prev_interval.camera_id == next_interval.camera_id:
                continue
            events.append(
                TransitionEvent(
                    object_id=object_id,
                    src_camera_id=prev_interval.camera_id,
                    dst_camera_id=next_interval.camera_id,
                    src_exit_time=prev_interval.end_time,
                    dst_entry_time=next_interval.start_time,
                    transit_time=next_interval.start_time - prev_interval.end_time,
                )
            )

    logger.info("Derived %d transition event(s) from %d object(s)", len(events), len(by_object))
    return events


def compute_camera_pair_stats(events: list[TransitionEvent]) -> dict[tuple[str, str], CameraPairTransitionStats]:
    """Compute per-camera-pair empirical transit-time distributions.

    Args:
        events: TransitionEvent objects, as produced by `derive_transition_events`.

    Returns:
        A mapping from (src_camera_id, dst_camera_id) to that pair's
        CameraPairTransitionStats.
    """
    grouped: dict[tuple[str, str], list[float]] = {}
    for event in events:
        key = (event.src_camera_id, event.dst_camera_id)
        grouped.setdefault(key, []).append(event.transit_time)

    stats: dict[tuple[str, str], CameraPairTransitionStats] = {}
    for (src_camera_id, dst_camera_id), transit_times in grouped.items():
        n = len(transit_times)
        mean = sum(transit_times) / n
        variance = sum((t - mean) ** 2 for t in transit_times) / n
        stats[(src_camera_id, dst_camera_id)] = CameraPairTransitionStats(
            src_camera_id=src_camera_id,
            dst_camera_id=dst_camera_id,
            count=n,
            mean_transit_time=mean,
            std_transit_time=variance**0.5,
            min_transit_time=min(transit_times),
            max_transit_time=max(transit_times),
            transit_times=transit_times,
        )

    logger.info("Computed transit-time stats for %d camera pair(s)", len(stats))
    return stats
