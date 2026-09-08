# Notes for tomorrow

1. **DONE (Sept 2026):** the dataset layout and ground-truth/calibration JSON
   schemas were provisional guesses and have now been confirmed against the
   dataset's own Hugging Face README -- `src/transit/data/mtmc_dataset.py`,
   `src/transit/data/calibration.py`, and `src/transit/detection/dataset_export.py`
   (`load_ground_truth_annotations`) were rewritten to match. See README.md's
   "Downloading a scene" section for the real layout and a download command.

2. **Download a scene** (see README.md) and point `configs/default.yaml` at it:
   `dataset_root: ".../MTMC_Tracking_2025/train"`, `scene_name: "Warehouse_000"`.

3. **One thing still genuinely unverified** (small, isolated -- not a structural
   guess like #1 was): the exact `object_type` string used for person annotations
   in `ground_truth.json`. `_PERSON_OBJECT_TYPES = {"Person"}` in
   `src/transit/detection/dataset_export.py` is the assumption;
   `load_ground_truth_annotations` logs every *other* object type it skips, so
   check that log on the first real run against a downloaded file and adjust the
   constant if the log shows something unexpected (e.g. if "Person" turns out to
   be empty/wrong and everything gets skipped).

4. **Fine-tune the detector** (fully implemented, ready to run):
   ```bash
   python scripts/export_yolo_dataset.py --config configs/default.yaml --scene <scene>
   python scripts/train_yolo.py --config configs/default.yaml
   ```
   Point later stages at the fine-tuned checkpoint (`<training.output_dir>/finetune/weights/best.pt`)
   via each script's `--weights` flag or by setting `detector_model` in the config.

5. **Run the rest of the working pipeline in order** on real data, using the
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

6. **Implement and run the gate, for real, in this order**:
   - `gate/features.py` — build `[appearance_similarity, transition-time
     log-likelihood, ...]` feature vectors from `preprocessing/transitions.py`
     output, fit only on the train identity split (`eval/splits.py`).
   - `gate/train_gate.py` — train the logistic-regression gate, tune threshold on val split.
   - `gate/calibration_oracle.py` — compute the calibration-based oracle baseline
     for comparison (evaluation-only, never fed into the gate).
   - Only then consider `gate/mlp_gate.py`, if the logistic gate underfits.
