# Notes for tomorrow

1. **Download the MTMC_Tracking_2025 scene** and point `configs/default.yaml` (or a
   scene-specific config) at it: set `dataset_root` and `scene_name` so
   `<dataset_root>/<scene_name>/<camera_id>/` contains `video.mp4`,
   `calibration.json`, and a ground-truth annotation file.

2. **Verify field-name assumptions against the real downloaded files** before
   trusting any output built on them:
   - `src/transit/data/calibration.py` (`load_camera_calibration`) — assumed keys:
     `cameraId`, `intrinsicMatrix`, `extrinsicMatrix`, `homography`,
     `translationToGlobalCoordinates`.
   - `src/transit/detection/dataset_export.py` (`load_ground_truth_annotations`) —
     assumed structure: `{"frames": [{"frameId": ..., "objects": [{"objectId":
     ..., "bbox": [xmin, ymin, xmax, ymax]}]}]}`, and whether `bbox` is really
     xyxy vs. xywh.
   - `src/transit/data/mtmc_dataset.py` (`ground_truth_path`) — confirm the actual
     ground-truth filename (currently guesses `ground_truth.json` / `gt.json` /
     `labels.json`).

3. **Fine-tune the detector** (fully implemented, ready to run):
   ```bash
   python scripts/export_yolo_dataset.py --config configs/default.yaml --scene <scene>
   python scripts/train_yolo.py --config configs/default.yaml
   ```
   Point later stages at the fine-tuned checkpoint (`<training.output_dir>/finetune/weights/best.pt`)
   via each script's `--weights` flag or by setting `detector_model` in the config.

4. **Run the rest of the working pipeline in order** on real data, using the
   fine-tuned (or stock) detector:
   ```bash
   python scripts/run_detect_track.py --config configs/default.yaml --scene <scene>
   python scripts/run_embed.py --config configs/default.yaml --scene <scene>
   python scripts/run_preprocess_transitions.py --config configs/default.yaml --scene <scene>
   python scripts/run_baseline.py --config configs/default.yaml --scene <scene>
   ```
   Note: `run_embed.py` needs `reid_weights_source` in the config pointed at a real
   downloaded OSNet checkpoint path (not the placeholder `"market1501"` label) —
   see the TODO in `src/transit/reid/embedder.py`.

5. **Implement and run the gate, for real, in this order**:
   - `gate/features.py` — build `[appearance_similarity, transition-time
     log-likelihood, ...]` feature vectors from `preprocessing/transitions.py`
     output, fit only on the train identity split (`eval/splits.py`).
   - `gate/train_gate.py` — train the logistic-regression gate, tune threshold on val split.
   - `gate/calibration_oracle.py` — compute the calibration-based oracle baseline
     for comparison (evaluation-only, never fed into the gate).
   - Only then consider `gate/mlp_gate.py`, if the logistic gate underfits.
