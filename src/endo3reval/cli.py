"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from endo3reval.pipeline import STAGES, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run official Endo3R inference on preprocessed SCARED and evaluate "
            "the saved depths with the official Video Depth Anything protocol."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scared.local.json"),
    )
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--limit-sequences", type=int, default=None)
    parser.add_argument(
        "--force-inference",
        action="store_true",
        help="Clear and rerun a sequence even when all prediction files exist",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.force_inference and args.stage not in ("infer", "all"):
        parser.error("--force-inference is only valid for infer/all")
    run_pipeline(
        args.config,
        stage=args.stage,
        force_inference=args.force_inference,
        limit_sequences=args.limit_sequences,
    )
