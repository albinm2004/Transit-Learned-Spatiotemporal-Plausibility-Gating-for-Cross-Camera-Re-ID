"""Learned spatiotemporal plausibility gating for cross-camera transitions.

This is Part 3 of the project -- the research contribution: gating candidate
cross-camera identity matches with a learned model of which camera-to-camera
transitions are physically plausible, without relying on hand-specified camera
topology or calibration at inference time.

- features.py: builds the [appearance_similarity, transition_log_likelihood]
  feature vectors and binary labels used to train the gate.
- train_gate.py: fits and threshold-tunes the primary logistic-regression gate.
- calibration_oracle.py: the calibration-based, evaluation-only oracle baseline
  (never fed into the gate's training or inference path).
- ground_truth_linking.py: IoU-based association from tracker track_ids to
  dataset ground-truth object_ids, used only to build training/eval labels.
- mlp_gate.py: upgrade path if the logistic gate underfits -- still a stub.

See scripts/run_train_gate.py for the end-to-end training + evaluation script.
"""
