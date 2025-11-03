"""Standalone script to execute the voxel filtering dev pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Allow running as a stand-alone script by adding parent directory to sys.path.
    current_path = Path(__file__).resolve()
    package_root = current_path.parent
    project_root = package_root.parent
    sys.path.insert(0, str(project_root))
    from voxel_filtering_processor_dev.pipeline import run_dev_pipeline
else:
    from .pipeline import run_dev_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run voxel filtering development pipeline and generate tri-view diagnostics.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Path to configuration JSON file (default: ./config.json).",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="voxel_filtering_dev",
        help="Subdirectory under outputs/ for storing development results.",
    )
    parser.add_argument(
        "--baseline-time",
        type=int,
        default=1200,
        help="Baseline frame index to visualize (default: 1200).",
    )
    parser.add_argument(
        "--deltaf-time",
        type=int,
        default=1200,
        help="DeltaF/F0 frame index to visualize (default: 1200).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Chunk size for streaming dataset statistics (default: 5).",
    )
    parser.add_argument(
        "--start-stage",
        type=str,
        default="brain_region_filter",
        help="Pipeline stage to start from (default: brain_region_filter).",
    )
    parser.add_argument(
        "--end-stage",
        type=str,
        default="baseline_threshold_filter",
        help="Pipeline stage to end on (default: baseline_threshold_filter).",
    )
    parser.add_argument(
        "--baseline-threshold",
        type=float,
        default=5.0,
        help="Baseline threshold value; voxels below this baseline mean are removed.",
    )
    parser.add_argument(
        "--stage-inputs",
        type=Path,
        default=None,
        help="Optional JSON file describing per-stage input artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_dev_pipeline(
        config_path=args.config,
        output_subdir=args.output_subdir,
        baseline_time_index=args.baseline_time,
        deltaf_time_index=args.deltaf_time,
        chunk_size=args.chunk_size,
        baseline_threshold=args.baseline_threshold,
        start_stage=args.start_stage,
        end_stage=args.end_stage,
        stage_inputs_path=args.stage_inputs,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
