# Transit

Transit is a research codebase for a final-year computer vision project on **learned
spatiotemporal plausibility gating for cross-camera person re-identification**. The
pipeline detects and tracks people per camera, extracts appearance embeddings,
generates candidate cross-camera identity matches by appearance similarity, and then
gates those candidates with a learned model of which camera-to-camera transitions are
physically plausible — replacing the usual hand-specified camera topology with a model
trained to recognize plausible vs. implausible transitions from data.

## Setup

```bash
# from the transit/ directory
pip install -e .
```

Point the pipeline at a downloaded scene by editing [configs/default.yaml](configs/default.yaml)
(or copying it to a scene-specific config):

```yaml
dataset_root: "data/raw"      # directory containing <scene_name>/<camera_id>/... 
scene_name: "scene_001"
output_dir: "outputs"
```

Expected scene layout (see [src/transit/data/mtmc_dataset.py](src/transit/data/mtmc_dataset.py)):

```
<dataset_root>/<scene_name>/<camera_id>/video.mp4
<dataset_root>/<scene_name>/<camera_id>/calibration.json
```

Run detection + tracking on every camera in a scene:

```bash
python scripts/run_detect_track.py --config configs/default.yaml --scene scene_001
```

Tracklets are written to `<output_dir>/<scene_name>/<camera_id>/tracklets.parquet`
(or `.json` with `--format json`).

## What's implemented so far

- Config loading (`TransitConfig`, YAML-driven, auto device detection).
- Typed data schema for detections, tracklets, and camera calibration.
- Scene/camera discovery and calibration loading (calibration field names are
  provisional — see TODOs in `data/schema.py` and `data/calibration.py`).
- Per-camera person detection (`PersonDetector`, Ultralytics YOLO).
- Per-camera tracking (`CameraTracker`, Ultralytics ByteTrack via `model.track`).
- CLI to run detection + tracking across a whole scene and save tracklets to disk.

## What's coming next

- **Re-ID** (`src/transit/reid/`): appearance embedding extraction for tracklets.
- **Matching** (`src/transit/matching/`): candidate cross-camera identity matches by
  appearance similarity.
- **Gating** (`src/transit/gate/`): the learned spatiotemporal plausibility model that
  filters candidate matches — the core contribution of this project.
- **Eval** (`src/transit/eval/`): tracking / re-ID / end-to-end MTMC evaluation metrics.
