#!/usr/bin/env python3
"""Run the latent-space predictor demo."""

from __future__ import annotations

import argparse
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from latent_demo.pipeline import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the latent-space predictor demo.")
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "outputs"), help="Directory for artifacts.")
    args = parser.parse_args()

    metrics = run_demo(args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
