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

### Downloading a scene

The target dataset is [nvidia/PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces)
(MTMC_Tracking_2025 edition). It's split `train/` (Warehouse_000-014) and `val/`
(Warehouse_015, Warehouse_016, Lab_000, Hospital_000) at the top level -- pull just
one scene (skip `depth_maps/`, which this project doesn't use):

```bash
pip install -U huggingface_hub
hf download nvidia/PhysicalAI-SmartSpaces --repo-type dataset --include "MTMC_Tracking_2025/train/Warehouse_000/**" --exclude "MTMC_Tracking_2025/train/Warehouse_000/depth_maps/**" --local-dir data/raw/PhysicalAI-SmartSpaces
```

(One line on purpose -- `\` line continuation doesn't work the same across bash/cmd/PowerShell,
so this is written to paste as-is anywhere. `huggingface-cli` is the old, now-deprecated command
name; use `hf` as shown. ~3.8 GB for Warehouse_000's videos + ground_truth.json +
calibration.json + map.png, vs. ~7+ GB if depth_maps/ is included.)

**If this fails with `[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate`**
(common on Windows behind a campus/institutional network or antivirus doing HTTPS
inspection -- the intercepting certificate is trusted by Windows but not by Python's
bundled `certifi`): first confirm `https://huggingface.co` loads fine in a normal
browser on the same machine, then run `pip install pip-system-certs` and retry -- this
makes Python trust whatever's in the Windows certificate store, which fixes it in one
shot. If the browser *also* shows a certificate warning, that's a different, real
network block rather than a trusted-but-unrecognized proxy, and needs different
troubleshooting (campus IT, a different network).

Then point the pipeline at it by editing [configs/default.yaml](configs/default.yaml):

```yaml
dataset_root: "data/raw/PhysicalAI-SmartSpaces/MTMC_Tracking_2025/train"   # note: train/, not MTMC_Tracking_2025/ itself
scene_name: "Warehouse_000"
output_dir: "outputs"
```

Confirmed scene layout (see [src/transit/data/mtmc_dataset.py](src/transit/data/mtmc_dataset.py) --
verified against the dataset's own Hugging Face README, Sept 2026; this replaced an
earlier provisional per-camera-subdirectory guess that did not match the real dataset):

```
<dataset_root>/<scene_name>/videos/<camera_id>.mp4   # one file per camera, e.g. Camera_0000.mp4
<dataset_root>/<scene_name>/calibration.json         # ONE file for the whole scene, all cameras
<dataset_root>/<scene_name>/ground_truth.json        # ONE file for the whole scene, all cameras
<dataset_root>/<scene_name>/map.png                  # top-down visualization (unused here)
```

One thing still unverified: the exact `object_type` string used for person
annotations in `ground_truth.json` (`load_ground_truth_annotations` assumes
`"Person"`; it logs any *other* object types it skips, so check the log on first
real run and adjust `_PERSON_OBJECT_TYPES` in
[src/transit/detection/dataset_export.py](src/transit/detection/dataset_export.py)
if needed).

## Quick sanity check (no dataset needed)

To confirm the detector/tracker work on real footage before touching the MTMC
dataset -- e.g. on a downloaded YouTube CCTV clip -- run:

```bash
python scripts/smoke_test_video.py --video path/to/clip.mp4
```

A ready-to-use sample clip is committed at
[samples/yolo11x_smoke_test_cctv_footage.mp4](samples/yolo11x_smoke_test_cctv_footage.mp4)
(~30 MB CCTV footage), so you can run this immediately with no setup:

```bash
python scripts/smoke_test_video.py --video samples/yolo11x_smoke_test_cctv_footage.mp4
```

This writes `<clip>_annotated.mp4` with boxes and persistent track IDs. It's a
smoke test only: a single video has no cross-camera ground truth, so it can't
validate Re-ID matching or the plausibility gate -- see `--help` for options
(`--detect-only`, `--weights` for a fine-tuned checkpoint, `--max-frames`, `--stride`).

A pre-rendered example of that output is committed at
[samples/yolo11x_smoke_test_cctv_footage_annotated.mp4](samples/yolo11x_smoke_test_cctv_footage_annotated.mp4)
(H.264, ~38 MB) so you can see the result without running anything. Note:
`smoke_test_video.py`'s own output uses `cv2.VideoWriter`'s `mp4v` codec and is
much larger (~110 MB for this clip) -- `*_annotated.mp4` is gitignored by
default for exactly that reason; this one file was re-encoded with
`ffmpeg -c:v libx264 -crf 23` before being committed as a deliberate exception.

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
