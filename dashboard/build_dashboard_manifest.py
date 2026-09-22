#!/usr/bin/env python3
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
USAGE
--------------------------------------------------------------------------
    python3 build_dashboard_manifest.py \\
        --scene-dir outputs/Warehouse_015 \\
        --gate-decisions outputs/Warehouse_015/gate_decisions.json \\
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
# Clip cutting + overlay drawing -- concrete, no adaptation needed
# ==========================================================================

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


# ==========================================================================
# Manifest assembly
# ==========================================================================

def build_people(gate_records: list[dict], scene_dir: Path, cut_clips: bool) -> list[Person]:
    """Group raw gate-decision records by person_id and turn each into a
    Person with an ordered list of Appearances, cutting a clip per
    appearance when tracking data + source video are available."""
    by_person: dict[str, list[dict]] = {}
    for r in gate_records:
        by_person.setdefault(r["person_id"], []).append(r)

    people: list[Person] = []
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
            if cut_clips:
                video_path = scene_dir / app.camera / "source.mp4"  # ADAPT if your raw video lives elsewhere
                if video_path.exists():
                    try:
                        boxes = load_tracking_boxes(scene_dir, app.camera, r.get("to_track_id", ""))
                        fps = cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FPS) or 25.0
                        start_frame, end_frame = int(app.start * fps), int(app.end * fps)
                        out_path = Path("clips") / f"{pid}_{app.camera}_{start_frame}.mp4"
                        cut_clip_with_overlay(
                            video_path, boxes, start_frame, end_frame, out_path, fps,
                            label=f"{app.camera} | {pid}",
                        )
                        app.clip_path = out_path
                    except FileNotFoundError as e:
                        print(f"  [skip clip] {e}", file=sys.stderr)
            appearances.append(app)

        people.append(Person(id=pid, appearances=appearances))

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
    html = template_path.read_text()
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
    out_path.write_text(new_html)


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
    demo_appearance.clip_path = clip_out.relative_to(out_dir).parent / clip_out.name if False else Path(clip_out.name)
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
    ap.add_argument("--demo", action="store_true", help="run end-to-end on synthetic data, ignoring --scene-dir/--gate-decisions")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        run_demo(args.out_dir, args.dashboard_template)
        return

    if not args.scene_dir or not args.gate_decisions:
        ap.error("--scene-dir and --gate-decisions are required (or pass --demo)")

    # ADAPT: camera zone labels + map (x, y) positions are cosmetic --
    # set them to roughly match your scene's real top-down layout, or
    # pull them from the dataset's calibration/floor-plan metadata.
    cameras = [
        Camera(f"Camera_{i:04d}", "Zone", x=10 + (i % 5) * 20, y=15 + (i // 5) * 30)
        for i in range(13)
    ]

    gate_records = load_gate_decisions(args.gate_decisions)
    people = build_people(gate_records, args.scene_dir, cut_clips=not args.no_clips)
    manifest = build_manifest(args.scene_dir.name, cameras, people)

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.out_dir / 'manifest.json'} ({len(people)} people)")

    report_path = args.out_dir / "investigator_report.html"
    inject_into_dashboard(manifest, args.dashboard_template, report_path)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
