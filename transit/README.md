# Transit

Transit is a research codebase for a final-year computer vision project on **learned
spatiotemporal plausibility gating for cross-camera person re-identification**. The
pipeline detects and tracks people per camera, extracts appearance embeddings,
generates candidate cross-camera identity matches by appearance similarity, and then
gates those candidates with a learned model of which camera-to-camera transitions are
physically plausible — replacing the usual hand-specified camera topology with a model
trained to recognize plausible vs. implausible transitions from data. Camera
calibration (shipped with the target dataset) is used only as an evaluation-time
oracle baseline, never as a gate input — the whole point is a gate that works without it.

## Setup

```bash
# from the transit/ directory
pip install -e .
```

`torchreid` is not consistently published to PyPI across versions; if `pip install`
fails on it, install from source:

```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

Point the pipeline at a downloaded scene by editing [configs/default.yaml](configs/default.yaml):

```yaml
dataset_root: "data/raw"      # directory containing <scene_name>/<camera_id>/...
scene_name: "scene_001"
output_dir: "outputs"
```

Expected scene layout (see [src/transit/data/mtmc_dataset.py](src/transit/data/mtmc_dataset.py)):

```
<dataset_root>/<scene_name>/<camera_id>/video.mp4
<dataset_root>/<scene_name>/<camera_id>/calibration.json
<dataset_root>/<scene_name>/<camera_id>/ground_truth.json   # filename TBD, see NOTES_FOR_TOMORROW.md
```

## Status

### Fully implemented and runnable

- Config (`TransitConfig`, YAML-driven, auto detector/device selection, `training` sub-config).
- Data schema, calibration loading, scene/camera discovery.
- Detection (`PersonDetector`), tracking (`CameraTracker`, ByteTrack).
- Re-ID embedding (`ReidEmbedder`, frozen pretrained OSNet via torchreid).
- Candidate generation (top-k cosine similarity) and the appearance-only baseline matcher.
- Preprocessing: visibility intervals, transition events + per-pair statistics, camera-pair tiering.
- Eval: identity-based splits, mAP / Rank-1 / Rank-5 / false-positive-match-rate.
- **YOLO detector fine-tuning — implemented and ready to run tomorrow**, not stubbed:
  `detection/dataset_export.py` (ground truth → YOLO dataset) and `detection/train.py`
  (`model.train()` wrapper), driven by `scripts/export_yolo_dataset.py` and `scripts/train_yolo.py`.

### Stubbed for tomorrow (documented signatures, `NotImplementedError`)

- `gate/features.py`, `gate/train_gate.py`, `gate/mlp_gate.py` — the learned
  plausibility gate itself (this project's actual research contribution).
- `gate/calibration_oracle.py` — the calibration-based evaluation-only oracle baseline.
- `scripts/run_train_gate.py` — CLI surface exists; prints a message and exits.

See [NOTES_FOR_TOMORROW.md](NOTES_FOR_TOMORROW.md) for the concrete next-steps checklist.
