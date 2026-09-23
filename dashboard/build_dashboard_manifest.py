"""
build_dashboard_manifest.py
============================
Turns your Transit pipeline's outputs into the data the Investigator
dashboard (investigator.html) renders: a manifest of tracked identities,
their per-camera appearances, the plausibility gate's score at each hop,
and a short cropped video clip with the tracker box burned in for each
appearance.

Everything here is CPU-only (OpenCV + stdlib). It reads files your GPU
stages already produced (detection+tracking output, gate decisions) and
does not touch the detector, tracker, embedder, or gate itself -- so it
never needs the GPU workstation. Run it anywhere you can `pip install
opencv-python-headless`, including a laptop.

--------------------------------------------------------------------------
WHAT'S CONCRETE VS. WHAT YOU NEED TO ADAPT
--------------------------------------------------------------------------
The mechanical parts (cutting a clip, drawing a tracker box per frame,
assembling the manifest JSON, splicing it into the dashboard HTML) are
fully implemented and match investigator.html's documented schema exactly.

The two loader functions -- load_tracking_boxes() and load_gate_decisions()
-- read your actual output files, and I have not seen those files, so
they're written against my best inference of your pipeline (per-camera
Parquet tracking output, a gate-decisions JSON alongside gate_eval_results.
json) with every assumption marked "ADAPT:". Point them at your real
column/key names and everything downstream (clip cutting, manifest,
dashboard injection) works unchanged.

Run with --demo and no real data to sanity-check the mechanics end to end
using synthetic frames -- useful before wiring up real files.

--------------------------------------------------------------------------
A NOTE ON REAL gate_decisions.json SCALE
--------------------------------------------------------------------------
A real gate evaluation run re-scores each (person, from_camera, to_camera)
transition many times over -- e.g. once per short time window while both
cameras can see the person -- so raw gate_decisions.json can contain tens
of thousands of near-duplicate "accepted" records for what is really a
handful of actual camera-to-camera hops per person. collapse_and_limit()
below merges temporally-adjacent/overlapping records for the same
(person, from, to) triple into one appearance per real event, and then
keeps only the --max-appearances-per-person highest-confidence real
events per person, so the manifest/dashboard shows a clean, readable
investigation instead of thousands of timeline entries, and clip cutting
finishes in a reasonable time.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python3 build_dashboard_manifest.py \\
        --scene-dir outputs/Warehouse_015 \\
        --gate-decisions outputs/Warehouse_015/gate_decisions.json \\
        --raw-video-dir /path/to/real/Camera_NNNN.mp4/files \\
        --out-dir dashboard_build \\
        --dashboard-template investigator.html

    # sanity-check the mechanics with synthetic data, no real files needed:
    python3 build_dashboard_manifest.py --demo --out-dir dashboard_build

Writes:
    dashboard_build/manifest.json           -- the data, on its own
    dashboard_build/clips/<person>_<camera>.mp4
    dashboard_build/investigator_report.html -- the dashboard with your
                                                 real manifest spliced in,
                                                 ready to open directly
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit(
        "Missing dependencies. Install with:\n"
        "  pip install opencv-python-headless numpy --break-system-packages"
    )


# ==========================================================================
# Data model -- mirrors the schema documented at the top of investigator.html
# ==========================================================================

@dataclass
class Box:
    frame: int
    x: int
    y: int
    w: int
    h: int


@dataclass
class Appearance:
    camera: str
    start: float                       # seconds
    end: float                         # seconds
    accepted: bool
    is_entry: bool = False
    appearance_similarity: Optional[float] = None
    transit_plausibility: Optional[float] = None
    note: Optional[str] = None
    boxes: list[Box] = field(default_factory=list)
    clip_path: Optional[Path] = None   # filled in once the clip is cut

    def to_manifest_dict(self, clips_rel_dir: str) -> dict:
        d = {
            "camera": self.camera,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "accepted": self.accepted,
        }
        if self.is_entry:
            d["isEntry"] = True
        d["appearanceSimilarity"] = self.appearance_similarity
        d["transitPlausibility"] = self.transit_plausibility
        if self.note:
            d["note"] = self.note
        if self.clip_path is not None:
            d["clip"] = f"{clips_rel_dir}/{self.clip_path.name}"
        return d


@dataclass
class Person:
    id: str
    appearances: list[Appearance]


@dataclass
class Camera:
    id: str
    zone: str
    x: float  # 0-100, position on the dashboard's top-down map
    y: float


# ==========================================================================
# Loaders -- ADAPT these two to your actual pipeline output files
# ==========================================================================

def load_tracking_boxes(scene_dir: Path, camera: str, track_id: str) -> list[Box]:
    """
    Return the per-frame bounding box for one tracklet in one camera.

    ADAPT: this assumes `run_detect_track.py`'s Parquet output at
    outputs/<Scene>/<Camera>/tracks.parquet with columns
    [frame, track_id, x, y, w, h] (top-left x/y, pixel width/height).
    If your actual column names or file format differ, this is the only
    function that needs to change -- everything downstream just consumes
    the Box list this returns.
    """
    tracks_path = scene_dir / camera / "tracklets.parquet"
    if not tracks_path.exists():
        raise FileNotFoundError(
            f"{tracks_path} not found. ADAPT load_tracking_boxes() in this "
            f"script to point at wherever your pipeline actually writes "
            f"per-frame tracking boxes for {camera}."
        )
    import pandas as pd  # local import: only needed on the real-data path

    df = pd.read_parquet(tracks_path)
    df = df[df["track_id"] == int(track_id)].sort_values("frame_idx")
    return [
        Box(
            frame=int(r.frame_idx),
            x=int(r.xmin), y=int(r.ymin),
            w=int(r.xmax - r.xmin), h=int(r.ymax - r.ymin),
        )
        for r in df.itertuples()
    ]


def load_gate_decisions(gate_decisions_path: Path) -> list[dict]:
    """
    Return one record per candidate cross-camera match the gate scored,
    accepted or not.

    ADAPT: this assumes a JSON list of objects shaped like:
        {
          "person_id": "P-1047",
          "from_camera": "Camera_0000", "from_track_id": "...",
          "to_camera": "Camera_0002",   "to_track_id": "...",
          "start": 61.0, "end": 90.0,
          "appearance_similarity": 0.91,
          "transit_plausibility": 0.88,
          "accepted": true
        }
    `run_train_gate.py` / your evaluation script is the natural place to
    dump this alongside gate_eval_results.json -- right now that file only
    has aggregate Rank-1/mAP numbers, not per-candidate records, so you may
    need to add one write call there. If the real shape differs, adjust
    the field lookups in build_people() below rather than this loader.
    """
    if not gate_decisions_path.exists():
        raise FileNotFoundError(
            f"{gate_decisions_path} not found. ADAPT load_gate_decisions() "
            f"(or point --gate-decisions at wherever your gate's "
            f"per-candidate output actually lives)."
        )
    with open(gate_decisions_path) as f:
        return json.load(f)


# ==========================================================================
# Collapse + limit -- real gate_decisions.json is scored far more finely
# than "one record per real camera-to-camera hop"; see the module
# docstring's "A NOTE ON REAL gate_decisions.json SCALE" section.
# ==========================================================================

def _merge_cluster(cluster: list[dict]) -> dict:
    """Collapse a list of temporally-adjacent/overlapping records for the
    same (person, from_camera, to_camera) into one representative record
    spanning the whole cluster, keeping the highest-confidence record's
    scores and track id."""
    best = max(
        cluster,
        key=lambda r: (r.get("appearance_similarity") or 0) + (r.get("transit_plausibility") or 0),
    )
    return {
        "person_id": cluster[0]["person_id"],
        "from_camera": cluster[0]["from_camera"],
        "to_camera": cluster[0]["to_camera"],
        "to_track_id": best.get("to_track_id", ""),
        "start": min(r["start"] for r in cluster),
        "end": max(r["end"] for r in cluster),
        "appearance_similarity": best.get("appearance_similarity"),
        "transit_plausibility": best.get("transit_plausibility"),
        "accepted": True,
        "note": best.get("note"),
    }


def collapse_and_limit(gate_records: list[dict], max_per_person: int, merge_gap: float = 3.0) -> list[dict]:
    """Keep only accepted records (the confirmed path, not every rejected
    candidate the gate considered), merge repeated re-scorings of the same
    real event into one record each (a gap of more than `merge_gap`
    seconds between records for the same (person, from, to) triple counts
    as a new, separate real event rather than a continuation), then keep
    only the `max_per_person` highest-confidence real events per person,
    in chronological order."""
    accepted = [r for r in gate_records if r.get("accepted")]

    by_group: dict[tuple[str, str, str], list[dict]] = {}
    for r in accepted:
        key = (r["person_id"], r["from_camera"], r["to_camera"])
        by_group.setdefault(key, []).append(r)

    collapsed: list[dict] = []
    for key, records in by_group.items():
        records.sort(key=lambda r: r["start"])
        cluster: list[dict] = []
        cluster_end: float | None = None
        for r in records:
            if cluster and r["start"] > cluster_end + merge_gap:
                collapsed.append(_merge_cluster(cluster))
                cluster = []
                cluster_end = None
            cluster.append(r)
            cluster_end = r["end"] if cluster_end is None else max(cluster_end, r["end"])
        if cluster:
            collapsed.append(_merge_cluster(cluster))

    by_person: dict[str, list[dict]] = {}
    for r in collapsed:
        by_person.setdefault(r["person_id"], []).append(r)

    limited: list[dict] = []
    for pid, records in by_person.items():
        records.sort(key=lambda r: r["start"])
        kept = records[:max_per_person]
        limited.extend(kept)
        print(
            f"  {pid}: {len(records)} real events after merging -> keeping {len(kept)}",
            file=sys.stderr,
        )

    print(
        f"collapse_and_limit: {len(gate_records)} raw records -> "
        f"{len(accepted)} accepted -> {len(collapsed)} real events -> "
        f"{len(limited)} kept ({max_per_person}/person cap)",
        file=sys.stderr,
    )
    return limited


# ==========================================================================
# Clip cutting + overlay drawing -- concrete, no adaptation needed
# ==========================================================================

def _transcode_to_h264(path: Path) -> None:
    """Browsers can't play OpenCV's mp4v (MPEG-4 Part 2) output in a
    <video> tag -- only H.264/VP9/AV1. Re-encode in place via ffmpeg's
    libx264 so clips actually play in the dashboard."""
    tmp_path = path.with_suffix(".h264.mp4")
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(tmp_path)],
        capture_output=True,
    )
    if result.returncode != 0 or not tmp_path.exists():
        print(
            f"  [warn] ffmpeg transcode failed for {path}: "
            f"{result.stderr.decode(errors='replace')[:300]}",
            file=sys.stderr,
        )
        if tmp_path.exists():
            tmp_path.unlink()
        return
    tmp_path.replace(path)


def cut_clip_with_overlay(
    video_path: Path,
    boxes: list[Box],
    start_frame: int,
    end_frame: int,
    out_path: Path,
    fps: float,
    label: str,
) -> None:
    """Cut [start_frame, end_frame) from video_path and draw the tracked
    box on every frame, CCTV-overlay style (corner ticks + camera/time
    label burned in top-left) -- matching investigator.html's mock frames
    so real clips drop in with the same visual language."""
    boxes_by_frame = {b.frame: b for b in boxes}

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for frame_idx in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        box = boxes_by_frame.get(frame_idx)
        if box is not None:
            color = (75, 166, 227)  # BGR amber-ish, matches the dashboard accent
            cv2.rectangle(frame, (box.x, box.y), (box.x + box.w, box.y + box.h), color, 2)
            tick = 8
            for cx, cy, dx, dy in [
                (box.x, box.y, 1, 1),
                (box.x + box.w, box.y, -1, 1),
                (box.x, box.y + box.h, 1, -1),
                (box.x + box.w, box.y + box.h, -1, -1),
            ]:
                cv2.line(frame, (cx, cy), (cx + tick * dx, cy), color, 3)
                cv2.line(frame, (cx, cy), (cx, cy + tick * dy), color, 3)
        cv2.putText(frame, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    cap.release()
    _transcode_to_h264(out_path)


# ==========================================================================
# Manifest assembly
# ==========================================================================

@dataclass
class _PendingClip:
    """One clip still to be cut, grouped and processed per-camera in a
    single sequential video pass instead of a fresh open+seek per clip
    (seeking per clip was the original, much slower approach -- see
    cut_clips_for_camera below)."""
    appearance: Appearance
    start_frame: int
    end_frame: int
    out_path: Path
    label: str
    boxes_by_frame: dict[int, Box]


def cut_clips_for_camera(video_path: Path, pending: list[_PendingClip], fps: float) -> None:
    """Cut every pending clip for ONE camera's video in a single forward
    (sequential, no-seek) pass over that video, the same reliable pattern
    already used elsewhere in this pipeline (see run_embed.py's
    _crop_all_tracklets docstring) instead of opening the video and
    seeking fresh per clip, which is unreliable/slow on long videos."""
    if not pending:
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [skip clip] could not open {video_path}", file=sys.stderr)
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    not_yet_started = sorted(pending, key=lambda p: p.start_frame)
    active: list[tuple[_PendingClip, cv2.VideoWriter]] = []
    max_end_frame = max(p.end_frame for p in pending)

    frame_idx = 0
    while frame_idx <= max_end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        while not_yet_started and not_yet_started[0].start_frame <= frame_idx:
            clip = not_yet_started.pop(0)
            clip.out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(clip.out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            active.append((clip, writer))

        still_active = []
        for clip, writer in active:
            if frame_idx < clip.end_frame:
                out_frame = frame.copy() if len(active) > 1 else frame
                box = clip.boxes_by_frame.get(frame_idx)
                if box is not None:
                    color = (75, 166, 227)  # BGR amber-ish, matches the dashboard accent
                    cv2.rectangle(out_frame, (box.x, box.y), (box.x + box.w, box.y + box.h), color, 2)
                    tick = 8
                    for cx, cy, dx, dy in [
                        (box.x, box.y, 1, 1),
                        (box.x + box.w, box.y, -1, 1),
                        (box.x, box.y + box.h, 1, -1),
                        (box.x + box.w, box.y + box.h, -1, -1),
                    ]:
                        cv2.line(out_frame, (cx, cy), (cx + tick * dx, cy), color, 3)
                        cv2.line(out_frame, (cx, cy), (cx, cy + tick * dy), color, 3)
                cv2.putText(out_frame, clip.label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(out_frame)
                still_active.append((clip, writer))
            else:
                writer.release()
                _transcode_to_h264(clip.out_path)
                clip.appearance.clip_path = clip.out_path
        active = still_active
        frame_idx += 1

    for clip, writer in active:
        writer.release()
        _transcode_to_h264(clip.out_path)
        clip.appearance.clip_path = clip.out_path
    cap.release()


def build_people(
    gate_records: list[dict],
    scene_dir: Path,
    cut_clips: bool,
    out_dir: Path,
    raw_video_dir: Path | None = None,
    max_clip_seconds: float | None = None,
) -> list[Person]:
    """Group raw gate-decision records by person_id and turn each into a
    Person with an ordered list of Appearances. If cut_clips, also cuts a
    clip per appearance -- grouped and processed one camera-video at a
    time (see cut_clips_for_camera) rather than per appearance, so each
    source video is only opened and decoded once no matter how many
    people/appearances reference it. FPS is also cached per camera (see
    camera_fps below) -- opening this dataset's videos is expensive, so
    it must not be re-probed once per appearance. Call collapse_and_limit
    on gate_records before passing them here -- this function assumes it
    is already receiving a manageable, de-duplicated set of real events."""
    by_person: dict[str, list[dict]] = {}
    for r in gate_records:
        by_person.setdefault(r["person_id"], []).append(r)

    people: list[Person] = []
    pending_by_camera: dict[str, list[_PendingClip]] = {}
    fps_cache: dict[str, float] = {}

    def camera_fps(camera: str, video_path: Path) -> float:
        if camera not in fps_cache:
            cap = cv2.VideoCapture(str(video_path))
            fps_cache[camera] = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
        return fps_cache[camera]

    for pid, records in by_person.items():
        records.sort(key=lambda r: r["start"])
        appearances: list[Appearance] = []

        # the very first sighting for a person has nothing to link *from*
        first = records[0]
        appearances.append(Appearance(
            camera=first["from_camera"], start=first["start"] - 1e-6, end=first["start"],
            accepted=True, is_entry=True,
        ))

        for r in records:
            app = Appearance(
                camera=r["to_camera"], start=r["start"], end=r["end"],
                accepted=bool(r["accepted"]),
                appearance_similarity=r.get("appearance_similarity"),
                transit_plausibility=r.get("transit_plausibility"),
                note=r.get("note"),
            )
            if cut_clips and raw_video_dir is not None:
                # real dataset layout: flat "<camera_id>.mp4" files, not
                # "<camera>/source.mp4" -- point --raw-video-dir at the
                # folder that holds them, e.g. .../Warehouse_000/videos
                video_path = raw_video_dir / f"{app.camera}.mp4"
                if video_path.exists():
                    try:
                        boxes = load_tracking_boxes(scene_dir, app.camera, r.get("to_track_id", ""))
                        fps = camera_fps(app.camera, video_path)
                        if max_clip_seconds is not None and app.end - app.start > max_clip_seconds:
                            app.end = app.start + max_clip_seconds
                        start_frame, end_frame = int(app.start * fps), int(app.end * fps)
                        if end_frame <= start_frame:
                            end_frame = start_frame + 1
                        out_path = out_dir / "clips" / f"{pid}_{app.camera}_{start_frame}.mp4"
                        pending_by_camera.setdefault(app.camera, []).append(_PendingClip(
                            appearance=app,
                            start_frame=start_frame,
                            end_frame=end_frame,
                            out_path=out_path,
                            label=f"{app.camera} | {pid}",
                            boxes_by_frame={b.frame: b for b in boxes},
                        ))
                    except FileNotFoundError as e:
                        print(f"  [skip clip] {e}", file=sys.stderr)
                else:
                    print(f"  [skip clip] {video_path} not found", file=sys.stderr)
            appearances.append(app)

        people.append(Person(id=pid, appearances=appearances))

    if pending_by_camera:
        cameras_todo = list(pending_by_camera.items())
        for i, (camera, clips) in enumerate(cameras_todo, start=1):
            video_path = raw_video_dir / f"{camera}.mp4"
            fps = camera_fps(camera, video_path)
            print(
                f"  cutting {len(clips)} clip(s) from {camera} "
                f"({i}/{len(cameras_todo)} cameras)...",
                file=sys.stderr,
            )
            cut_clips_for_camera(video_path, clips, fps)

    people.sort(key=lambda p: p.id)
    return people


def build_manifest(scene: str, cameras: list[Camera], people: list[Person]) -> dict:
    return {
        "scene": scene,
        "cameras": [{"id": c.id, "zone": c.zone, "x": c.x, "y": c.y} for c in cameras],
        "people": [
            {
                "id": p.id,
                "appearances": [a.to_manifest_dict("clips") for a in p.appearances],
            }
            for p in people
        ],
    }


# ==========================================================================
# Dashboard injection -- splice the real manifest into investigator.html
# ==========================================================================

def inject_into_dashboard(manifest: dict, template_path: Path, out_path: Path) -> None:
    html = template_path.read_text(encoding="utf-8")
    start_marker = "/* DASHBOARD_DATA_START"
    end_marker = "/* DASHBOARD_DATA_END */"
    start = html.find(start_marker)
    end = html.find(end_marker)
    if start == -1 or end == -1:
        raise RuntimeError(
            "Could not find the DASHBOARD_DATA_START/END markers in "
            f"{template_path}. Did the template change? Falling back "
            "requires manually replacing the CAMERAS/MOCK_DATA block."
        )
    end += len(end_marker)

    cameras_js = json.dumps(manifest["cameras"], indent=2)
    data_js = json.dumps({"scene": manifest["scene"], "cameras": manifest["cameras"], "people": manifest["people"]}, indent=2)
    replacement = (
        f"{start_marker} -- injected by build_dashboard_manifest.py, "
        f"do not hand-edit below this line. */\n"
        f"const CAMERAS = {cameras_js};\n\n"
        f"const MOCK_DATA = {data_js};\n"
        f"{end_marker}"
    )
    new_html = html[:start] + replacement + html[end:]
    out_path.write_text(new_html, encoding="utf-8")


# ==========================================================================
# Demo mode -- exercises the full pipeline with synthetic data, no real
# files required. Useful to confirm the mechanics work before wiring up
# real tracking/gate output.
# ==========================================================================

def make_demo_video(path: Path, n_frames: int = 150, size=(320, 240), fps: float = 25.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), (30, 24, 18), dtype=np.uint8)
        cx = int(size[0] * 0.2 + (size[0] * 0.6) * (i / n_frames))
        cv2.circle(frame, (cx, size[1] // 2), 20, (90, 70, 55), -1)
        writer.write(frame)
    writer.release()


def run_demo(out_dir: Path, template_path: Path) -> None:
    print("Running in --demo mode: generating synthetic video + tracking data.")
    scene_dir = out_dir / "_demo_scene"
    cam = "Camera_0000"
    video_path = scene_dir / cam / "source.mp4"
    make_demo_video(video_path)

    boxes = [Box(frame=i, x=40 + i, y=90, w=40, h=90) for i in range(150)]

    clip_out = out_dir / "clips" / "P-DEMO_Camera_0000_0.mp4"
    cut_clip_with_overlay(video_path, boxes, 0, 150, clip_out, fps=25.0, label="Camera_0000 | P-DEMO")
    print(f"  wrote demo clip -> {clip_out}")

    cameras = [Camera("Camera_0000", "Demo Aisle", 20, 50), Camera("Camera_0001", "Demo Dock", 80, 50)]
    demo_appearance = Appearance(
        camera=cam, start=0, end=6, accepted=True, is_entry=True,
    )
    demo_appearance.clip_path = Path(clip_out.name)
    people = [Person(id="P-DEMO", appearances=[demo_appearance])]

    manifest = build_manifest("Demo_Scene", cameras, people)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  wrote {out_dir / 'manifest.json'}")

    report_path = out_dir / "investigator_report.html"
    inject_into_dashboard(manifest, template_path, report_path)
    print(f"  wrote {report_path}")
    print("Demo complete -- open investigator_report.html to see a real cut clip driving the dashboard shell.")


# ==========================================================================
# CLI
# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", type=Path, help="e.g. outputs/Warehouse_015")
    ap.add_argument("--gate-decisions", type=Path, help="per-candidate gate output JSON (see load_gate_decisions docstring)")
    ap.add_argument("--out-dir", type=Path, default=Path("dashboard_build"))
    ap.add_argument("--dashboard-template", type=Path, default=Path("investigator.html"))
    ap.add_argument("--no-clips", action="store_true", help="build manifest.json only, skip cutting video clips (fast iteration)")
    ap.add_argument(
        "--raw-video-dir",
        type=Path,
        default=None,
        help="Directory of the scene's real source videos, flat '<camera_id>.mp4' per file "
        "(e.g. data/raw/PhysicalAI-SmartSpaces/MTMC_Tracking_2025/train/Warehouse_000/videos). "
        "Required to cut real clips; if omitted, clip cutting is skipped even without --no-clips.",
    )
    ap.add_argument(
        "--max-appearances-per-person",
        type=int,
        default=15,
        help="Real gate_decisions.json re-scores each transition many times; this caps how many "
        "highest-confidence real (merged) transitions per person are kept, after collapsing "
        "duplicates. Default 15. See the module docstring's scale note.",
    )
    ap.add_argument(
        "--max-clip-seconds",
        type=float,
        default=None,
        help="Cap each cut clip's duration at this many seconds (trims from the start time), "
        "to bound total disk usage on constrained machines.",
    )
    ap.add_argument("--demo", action="store_true", help="run end-to-end on synthetic data, ignoring --scene-dir/--gate-decisions")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        run_demo(args.out_dir, args.dashboard_template)
        return

    if not args.scene_dir or not args.gate_decisions:
        ap.error("--scene-dir and --gate-decisions are required (or pass --demo)")

    if not args.no_clips and args.raw_video_dir is None:
        print(
            "warning: clip cutting requested but --raw-video-dir not set; "
            "clips will be skipped. Pass --raw-video-dir to cut real clips, "
            "or pass --no-clips to silence this warning.",
            file=sys.stderr,
        )

    # ADAPT: camera zone labels + map (x, y) positions are cosmetic --
    # set them to roughly match your scene's real top-down layout, or
    # pull them from the dataset's calibration/floor-plan metadata.
    # 25 cameras, 5 per row x 5 rows (Warehouse_000's real camera count).
    cameras = [
        Camera(f"Camera_{i:04d}", "Zone", x=10 + (i % 5) * 20, y=15 + (i // 5) * 20)
        for i in range(25)
    ]

    gate_records = load_gate_decisions(args.gate_decisions)
    gate_records = collapse_and_limit(gate_records, args.max_appearances_per_person)
    people = build_people(gate_records, args.scene_dir, cut_clips=not args.no_clips, out_dir=args.out_dir, raw_video_dir=args.raw_video_dir, max_clip_seconds=args.max_clip_seconds)
    manifest = build_manifest(args.scene_dir.name, cameras, people)

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.out_dir / 'manifest.json'} ({len(people)} people)")

    report_path = args.out_dir / "investigator_report.html"
    inject_into_dashboard(manifest, args.dashboard_template, report_path)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
