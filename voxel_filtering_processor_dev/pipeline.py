"""Orchestrator for the voxel filtering development pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .data_io import ensure_output_directory, load_config
from .stages import StageOutput, run_baseline_threshold_filter, run_brain_region_filter


def _load_stage_inputs(stage_inputs_path: Optional[Path]) -> tuple[Dict[str, Any], Optional[Path]]:
    """Load per-stage input overrides from JSON if available."""
    if stage_inputs_path is None:
        default_path = Path("voxel_filtering_stage_inputs.json")
        if default_path.exists():
            stage_inputs_path = default_path

    if stage_inputs_path is None:
        return {}, None

    stage_inputs_path = Path(stage_inputs_path)
    if not stage_inputs_path.exists():
        print(f"Stage inputs file not found at {stage_inputs_path}; using defaults from config.")
        return {}, None

    with stage_inputs_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Stage inputs file must contain a JSON object: {stage_inputs_path}")
    print(f"Loaded stage inputs from {stage_inputs_path}")
    return data, stage_inputs_path


def _stage_in_range(
    stage_name: str,
    start_stage: str,
    end_stage: str,
    stage_order: list[str],
) -> bool:
    """Determine whether a stage should be executed."""
    start_idx = stage_order.index(start_stage)
    end_idx = stage_order.index(end_stage)
    stage_idx = stage_order.index(stage_name)
    return start_idx <= stage_idx <= end_idx


def run_dev_pipeline(
    config_path: str | Path,
    output_subdir: str = "voxel_filtering_dev",
    baseline_time_index: int = 1200,
    deltaf_time_index: int = 1200,
    chunk_size: int = 5,
    baseline_threshold: float = 5.0,
    start_stage: str = "brain_region_filter",
    end_stage: str = "baseline_threshold_filter",
    stage_inputs_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run the staged voxel filtering pipeline."""
    stage_order = ["brain_region_filter", "baseline_threshold_filter"]
    if start_stage not in stage_order or end_stage not in stage_order:
        raise ValueError(f"start_stage and end_stage must be in {stage_order}.")
    if stage_order.index(start_stage) > stage_order.index(end_stage):
        raise ValueError(f"start_stage '{start_stage}' occurs after end_stage '{end_stage}'.")

    print("[0/4] Loading configuration...")
    config = load_config(config_path)
    data_paths = config.get("data_paths", {})
    base_directory = Path(data_paths.get("base_directory", "."))

    stage_inputs_map, stage_inputs_source = _load_stage_inputs(
        Path(stage_inputs_path) if stage_inputs_path is not None else None
    )

    output_dir = ensure_output_directory(base_directory, output_subdir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    stage_outputs: Dict[str, StageOutput] = {}
    final_summary: Optional[Dict[str, Any]] = None

    # Stage 1: Brain region filter
    if _stage_in_range("brain_region_filter", start_stage, end_stage, stage_order):
        overrides = stage_inputs_map.get("brain_region_filter", {})
        preprocessed_path = Path(overrides.get("deltaf_f0", data_paths["preprocessed_data"]))
        baseline_path = Path(overrides.get("baseline_timeseries", data_paths["preprocessed_data_baseline"]))
        brain_mask_path = Path(overrides.get("brain_mask", data_paths["brain_mask"]))
        brain_region_mask_path = Path(overrides.get("brain_region_mask", data_paths["brain_region_mask"]))
        exclude_regions = overrides.get("exclude_regions", [])
        output_suffix = overrides.get("output_suffix", "")

        stage1_output = run_brain_region_filter(
            base_directory=base_directory,
            output_dir=output_dir,
            image_dir=image_dir,
            preprocessed_path=preprocessed_path,
            baseline_path=baseline_path,
            brain_mask_path=brain_mask_path,
            brain_region_mask_path=brain_region_mask_path,
            baseline_time_index=baseline_time_index,
            deltaf_time_index=deltaf_time_index,
            chunk_size=chunk_size,
            stage_inputs_source=stage_inputs_source,
            exclude_regions=exclude_regions,
            output_suffix=output_suffix,
        )
        stage_outputs["brain_region_filter"] = stage1_output
        final_summary = stage1_output.summary

        stage_inputs_map.setdefault("baseline_threshold_filter", {})
        stage2_defaults = stage_inputs_map["baseline_threshold_filter"]
        stage2_defaults.setdefault("deltaf_f0", str(stage1_output.artifacts["filtered_h5"]))
        stage2_defaults.setdefault("baseline_timeseries", str(stage1_output.artifacts["filtered_baseline_timeseries"]))
        stage2_defaults.setdefault("mask", str(stage1_output.artifacts["filtered_mask"]))
        stage2_defaults.setdefault("output_suffix", stage1_output.summary.get("output_suffix", ""))

    # Stage 2: Baseline threshold filter
    if _stage_in_range("baseline_threshold_filter", start_stage, end_stage, stage_order):
        overrides = stage_inputs_map.get("baseline_threshold_filter", {})
        brain_mask_path = Path(overrides.get("brain_mask", data_paths["brain_mask"]))
        prev_stage = stage_outputs.get("brain_region_filter")

        stage2_output = run_baseline_threshold_filter(
            base_directory=base_directory,
            output_dir=output_dir,
            image_dir=image_dir,
            deltaf_time_index=deltaf_time_index,
            baseline_time_index=baseline_time_index,
            chunk_size=chunk_size,
            baseline_threshold=baseline_threshold,
            brain_mask_path=brain_mask_path,
            stage_inputs_source=stage_inputs_source,
            stage_inputs=overrides,
            previous_stage=prev_stage,
        )
        stage_outputs["baseline_threshold_filter"] = stage2_output
        final_summary = stage2_output.summary

    if final_summary is None:
        final_summary = {
            "stage": "no_stage_executed",
            "message": "No stages executed; check start_stage/end_stage configuration.",
        }

    print("\n" + "=" * 60)
    print("DEV PIPELINE COMPLETED")
    print("=" * 60)
    for key, value in final_summary.items():
        print(f"{key}: {value}")

    return final_summary
