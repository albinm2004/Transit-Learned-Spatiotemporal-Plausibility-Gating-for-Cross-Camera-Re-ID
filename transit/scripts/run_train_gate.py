#!/usr/bin/env python
"""CLI to train the learned plausibility gate. NOT YET IMPLEMENTED.

This script's argument surface is complete so the end-to-end pipeline is
visibly wired together, but gate/features.py, gate/train_gate.py, and
gate/mlp_gate.py are documented stubs -- running this prints a clear message and
exits rather than executing anything.

Usage (once implemented):
    python scripts/run_train_gate.py --config configs/default.yaml --scene scene_001
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the expected command-line arguments for gate training."""
    parser = argparse.ArgumentParser(description="Train the learned plausibility gate (not yet implemented).")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to a TransitConfig YAML file.")
    parser.add_argument("--scene", type=str, default=None, help="Scene name to train on (overrides config).")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["logistic", "mlp"],
        default=None,
        help="Gate architecture to train (overrides config.gate_model_type).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse args, then report that gate training is not yet implemented."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parse_args()  # validated eagerly so the CLI surface is exercised even though we exit next

    print(
        "Gate training is implemented separately (see gate/features.py, "
        "gate/train_gate.py, gate/mlp_gate.py) and is not executed by this pass. "
        "TODO(tomorrow): implement feature construction and gate training on a "
        "real downloaded scene, then wire this script to call them. See "
        "NOTES_FOR_TOMORROW.md for the full checklist."
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
