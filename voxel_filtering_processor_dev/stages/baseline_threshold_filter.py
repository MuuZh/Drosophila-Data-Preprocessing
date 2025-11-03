"""Baseline-threshold filtering stage for the voxel filtering dev pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ..visualization import TriviewConfig, save_region_triview, save_triview
from .common import (
    DatasetMapping,
    StageOutput,
    build_stage_prefix,
    load_full_baseline,
    load_stage_mapping_and_mask,
    save_filtered_artifacts,
)


MULTI_SLICE_AXIS_INFO = {
    0: {"name": "Sagittal", "slice_label": "X", "axes_labels": ("Y", "Z")},
    1: {"name": "Coronal", "slice_label": "Y", "axes_labels": ("X", "Z")},
    2: {"name": "Axial", "slice_label": "Z", "axes_labels": ("X", "Y")},
}


def save_multislice_grid(
    volume: np.ndarray,
    mask: np.ndarray,
    axis: int,
    output_path: Path,
    title_prefix: str,
    colorbar_label: str,
    max_slices: int = 16,
) -> None:
    """Render a grid of slices across different depths for a given axis."""
    axis_info = MULTI_SLICE_AXIS_INFO[axis]
    num_available = volume.shape[axis]
    slice_indices = np.linspace(
        0, num_available - 1, num=min(max_slices, num_available), dtype=int
    )
    n_slices = slice_indices.size
    grid_cols = int(np.ceil(np.sqrt(n_slices)))
    grid_rows = int(np.ceil(n_slices / grid_cols))

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(
        3.5 * grid_cols, 3.5 * grid_rows))
    axes = np.atleast_2d(axes)

    for ax, idx in zip(axes.flat, slice_indices):
        plane = np.take(volume, idx, axis=axis)
        im = ax.imshow(plane.T, origin="lower", cmap="hot")
        mask_slice = np.take(mask, idx, axis=axis)
        if np.any(mask_slice):
            ax.contour(mask_slice.T.astype(float), levels=[
                       0.5], colors=["cyan"], linewidths=0.5)
        ax.set_title(f"{axis_info['slice_label']} = {idx}", fontsize=9)
        ax.set_xlabel(axis_info["axes_labels"][0])
        ax.set_ylabel(axis_info["axes_labels"][1])
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.035, label=colorbar_label)

    axes = np.atleast_2d(axes)
    axes_list = axes.flatten()
    for ax in axes_list[n_slices:]:
        ax.axis("off")

    fig.suptitle(f"{title_prefix} - {axis_info['name']} slices", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_baseline_threshold_filter(
    *,
    base_directory: Path,
    output_dir: Path,
    image_dir: Path,
    deltaf_time_index: int,
    baseline_time_index: int,
    chunk_size: int,
    baseline_threshold: float,
    brain_mask_path: Path,
    stage_inputs_source: Optional[Path],
    stage_inputs: Dict[str, str],
    previous_stage: Optional[StageOutput],
) -> StageOutput:
    """Execute stage 2: retain voxels whose baseline exceeds a threshold."""
    print("=" * 60)
    print("STAGE 2: Baseline Threshold Filter")
    print("=" * 60)

    deltaf_override = stage_inputs.get("deltaf_f0")
    baseline_override = stage_inputs.get("baseline_timeseries")
    mask_override = stage_inputs.get("mask")
    output_suffix = stage_inputs.get("output_suffix", "")

    if previous_stage is not None and (not deltaf_override or not baseline_override):
        deltaf_path = Path(previous_stage.artifacts["filtered_h5"])
        baseline_path = Path(
            previous_stage.artifacts["filtered_baseline_timeseries"])
        mask_volume_stage1 = previous_stage.mask_volume
        mapping_stage1 = previous_stage.mapping
        prev_prefix = previous_stage.prefix
        prev_kept = previous_stage.mapping.global_indices.size
    else:
        if not deltaf_override or not baseline_override:
            raise ValueError(
                "Baseline threshold stage requires 'deltaf_f0' and 'baseline_timeseries' paths "
                "when the brain_region_filter stage is skipped."
            )
        deltaf_path = Path(deltaf_override)
        baseline_path = Path(baseline_override)
        mapping_stage1, mask_volume_stage1 = load_stage_mapping_and_mask(
            filtered_h5_path=deltaf_path,
            mask_path=Path(mask_override) if mask_override else None,
        )
        prev_prefix = "brain_region_filter"
        prev_kept = mapping_stage1.global_indices.size

    print(f"Input DeltaF/F0 file: {deltaf_path}")
    print(f"Input baseline timeseries: {baseline_path}")
    print(f"Reference mask path: {brain_mask_path}")
    print(f"Previous stage prefix: {prev_prefix}")
    if output_suffix:
        print(f"Output suffix: {output_suffix}")

    baseline_memmap = np.load(baseline_path, mmap_mode="r")
    if baseline_memmap.ndim == 1:
        baseline_array = np.asarray(baseline_memmap, dtype=np.float32)
        baseline_time_idx_stage2 = 0
        baseline_time_vec = baseline_array
        baseline_time_label = "Baseline Values"
        baseline_max_vec = baseline_array
        baseline_mean_vec = baseline_array
    else:
        baseline_array = load_full_baseline(baseline_memmap)
        baseline_time_idx_stage2 = min(
            max(baseline_time_index, 0), baseline_array.shape[0] - 1)
        baseline_time_vec = baseline_array[baseline_time_idx_stage2, :].astype(
            np.float32, copy=False)
        baseline_time_label = f"Baseline Frame {baseline_time_idx_stage2}"
        baseline_max_vec = np.max(baseline_array, axis=0).astype(
            np.float32, copy=False)
        baseline_mean_vec = np.mean(
            baseline_array, axis=0).astype(np.float32, copy=False)
    baseline_keep_mask = baseline_mean_vec >= baseline_threshold
    kept_count = int(baseline_keep_mask.sum())
    removed_count = int(prev_kept - kept_count)

    print(f"Baseline threshold: {baseline_threshold}")
    print(f"    Voxels kept: {kept_count:,}")
    print(f"    Voxels removed: {removed_count:,}")

    initial_prefix = build_stage_prefix(
        "baseline_threshold_filter_initial",
        {"kept": prev_kept, "th": baseline_threshold},
    )
    stage_prefix = build_stage_prefix(
        "baseline_threshold_filter",
        {"kept": kept_count, "th": baseline_threshold},
    )

    # Add output_suffix to file names
    if output_suffix:
        initial_prefix = f"{initial_prefix}_{output_suffix}"
        stage_prefix = f"{stage_prefix}_{output_suffix}"

    with h5py.File(deltaf_path, "r") as handle:
        dataset = handle["deltaf_f0"]
        deltaf_array = np.asarray(dataset, dtype=np.float32)
    stage_timepoints = deltaf_array.shape[0]
    deltaf_time_idx_stage2 = min(
        max(deltaf_time_index, 0), stage_timepoints - 1)
    deltaf_time_vec = deltaf_array[deltaf_time_idx_stage2, :].astype(
        np.float32, copy=False)
    print("    Calculating DeltaF/F0 temporal mean for stage 2...")
    deltaf_mean_vec = np.mean(deltaf_array, axis=0).astype(
        np.float32, copy=False)

    print("    Mapping stage-1 vectors to brain volume...")
    stage1_volume_tasks = [
        ("Stage-1 baseline slice", baseline_time_vec, "baseline_time_volume"),
        ("Stage-1 baseline max", baseline_max_vec, "baseline_max_volume"),
        ("Stage-1 DeltaF/F0 slice", deltaf_time_vec, "deltaf_time_volume"),
        ("Stage-1 DeltaF/F0 mean", deltaf_mean_vec, "deltaf_mean_volume"),
    ]
    stage1_volumes: Dict[str, np.ndarray] = {}
    for description, vector, key in tqdm(stage1_volume_tasks, desc="    Volume mapping (stage-1)", leave=False):
        stage1_volumes[key] = mapping_stage1.values_to_volume(vector)

    baseline_time_volume = stage1_volumes["baseline_time_volume"]
    baseline_max_volume = stage1_volumes["baseline_max_volume"]
    deltaf_time_volume = stage1_volumes["deltaf_time_volume"]
    deltaf_mean_volume = stage1_volumes["deltaf_mean_volume"]
    overlay_mask_volume = mask_volume_stage1.astype(bool, copy=False)

    stage1_triview_tasks = [
        (baseline_time_volume, f"{initial_prefix}_baseline_slice_triview.png",
         f"{baseline_time_label} (Stage-1)", "Baseline (F0)", TriviewConfig(mode="slice")),
        (baseline_max_volume, f"{initial_prefix}_baseline_max_projection_triview.png",
         "Baseline Max Projection (Stage-1)", "Baseline (F0)", TriviewConfig(mode="max_projection")),
        (deltaf_time_volume, f"{initial_prefix}_deltaf_slice_triview.png",
         f"DeltaF/F0 Frame {deltaf_time_idx_stage2} (Stage-1)", "dF/F0", TriviewConfig(mode="slice")),
        (deltaf_mean_volume, f"{initial_prefix}_deltaf_mean_projection_triview.png",
         "DeltaF/F0 Mean Across Time (Stage-1)", "dF/F0", TriviewConfig(mode="mean_projection")),
    ]
    for volume, path_out, title_prefix, colorbar, config in tqdm(stage1_triview_tasks, desc="    Saving stage-1 tri-views", leave=False):
        save_triview(
            volume=volume,
            output_path=image_dir / path_out,
            title_prefix=title_prefix,
            config=config,
            overlay_mask=overlay_mask_volume,
            colorbar_label=colorbar,
        )

    mask_float = baseline_keep_mask.astype(np.float32, copy=False)
    filtered_mask_volume = mapping_stage1.mask_to_volume(baseline_keep_mask)
    filtered_mask_bool = filtered_mask_volume.astype(bool, copy=False)

    print("    Mapping baseline-threshold vectors back to brain volume...")
    stage2_volume_tasks = [
        ("Filtered baseline slice", baseline_time_vec *
         mask_float, "baseline_time_filtered"),
        ("Filtered baseline max", baseline_max_vec *
         mask_float, "baseline_max_filtered"),
        ("Filtered DeltaF/F0 slice", deltaf_time_vec *
         mask_float, "deltaf_time_filtered"),
        ("Filtered DeltaF/F0 mean", deltaf_mean_vec *
         mask_float, "deltaf_mean_filtered"),
    ]
    stage2_volumes: Dict[str, np.ndarray] = {}
    for description, vector, key in tqdm(stage2_volume_tasks, desc="    Volume mapping (filtered)", leave=False):
        stage2_volumes[key] = mapping_stage1.values_to_volume(vector)

    baseline_time_filtered = stage2_volumes["baseline_time_filtered"]
    baseline_max_filtered = stage2_volumes["baseline_max_filtered"]
    deltaf_time_filtered = stage2_volumes["deltaf_time_filtered"]
    deltaf_mean_filtered = stage2_volumes["deltaf_mean_filtered"]

    print("    Generating multi-slice grids for filtered baseline/DeltaF/F0...")
    multi_slice_tasks = [
        ("Baseline", baseline_time_filtered, filtered_mask_bool, "baseline"),
        ("DeltaF/F0", deltaf_time_filtered, filtered_mask_bool, "deltaf"),
    ]
    for label, volume, mask_bool, prefix in multi_slice_tasks:
        for axis, info in MULTI_SLICE_AXIS_INFO.items():
            output_path = image_dir / \
                f"{stage_prefix}_{prefix}_{info['name'].lower()}_multislice.png"
            print(
                f"      Saving {label} {info['name']} multi-slice grid to: {output_path}")
            save_multislice_grid(
                volume=volume,
                mask=mask_bool,
                axis=axis,
                output_path=output_path,
                title_prefix=f"{label} ({info['name']})",
                colorbar_label=label,
            )

    stage2_triview_tasks = [
        (baseline_time_filtered, f"{stage_prefix}_baseline_slice_triview_filtered.png",
         "Baseline Slice (Threshold Filtered)", "Baseline (F0)", TriviewConfig(mode="slice")),
        (baseline_max_filtered, f"{stage_prefix}_baseline_max_projection_triview_filtered.png",
         "Baseline Max Projection (Threshold Filtered)", "Baseline (F0)", TriviewConfig(mode="max_projection")),
        (deltaf_time_filtered, f"{stage_prefix}_deltaf_slice_triview_filtered.png",
         f"DeltaF/F0 Frame {deltaf_time_idx_stage2} (Threshold Filtered)", "dF/F0", TriviewConfig(mode="slice")),
        (deltaf_mean_filtered, f"{stage_prefix}_deltaf_mean_projection_triview_filtered.png",
         "DeltaF/F0 Mean Across Time (Threshold Filtered)", "dF/F0", TriviewConfig(mode="mean_projection")),
    ]
    for volume, path_out, title_prefix, colorbar, config in tqdm(stage2_triview_tasks, desc="    Saving post-filter tri-views", leave=False):
        save_triview(
            volume=volume,
            output_path=image_dir / path_out,
            title_prefix=title_prefix,
            config=config,
            overlay_mask=filtered_mask_bool,
            colorbar_label=colorbar,
        )

    save_region_triview(
        region_volume=filtered_mask_volume.astype(int, copy=False),
        output_path=image_dir /
        f"{stage_prefix}_mask_presence_slice_filtered.png",
        title_prefix="Mask Presence (Threshold Filtered)",
        config=TriviewConfig(mode="slice"),
        colorbar_label="Mask Presence",
        region_ticks=[0, 1],
    )

    voxel_indices_filtered = mapping_stage1.global_indices[baseline_keep_mask]
    artifact_paths = save_filtered_artifacts(
        output_dir=output_dir,
        stage_prefix=stage_prefix,
        stage_name="baseline_threshold_filter",
        preprocessed_path=deltaf_path,
        baseline_array=baseline_array,
        keep_mask=baseline_keep_mask,
        voxel_indices_filtered=voxel_indices_filtered,
        filtered_mask_volume=filtered_mask_volume,
        reference_mask_shape=mapping_stage1.brain_shape,
        reference_mask_path=brain_mask_path,
        stage_parameters={
            "baseline_threshold": baseline_threshold,
            "previous_stage_prefix": prev_prefix,
            "kept_voxels": kept_count,
            "removed_voxels": removed_count,
        },
        preloaded_deltaf=deltaf_array,
    )

    print("    Filtered HDF5 saved to:", artifact_paths["filtered_h5"])
    print("    Filtered mask saved to:", artifact_paths["filtered_mask"])
    print("    Filtered baseline timeseries saved to:",
          artifact_paths["filtered_baseline_timeseries"])

    summary: Dict[str, Any] = {
        "stage": "baseline_threshold_filter",
        "baseline_threshold": baseline_threshold,
        "input_deltaf_path": str(deltaf_path),
        "input_baseline_path": str(baseline_path),
        "previous_stage_prefix": prev_prefix,
        "previous_stage_kept_voxels": int(prev_kept),
        "kept_voxels": kept_count,
        "removed_voxels": removed_count,
        "output_suffix": output_suffix,
        "output_directory": str(output_dir),
        "filtered_data_path": str(artifact_paths["filtered_h5"]),
        "filtered_mask_path": str(artifact_paths["filtered_mask"]),
        "filtered_baseline_timeseries_path": str(artifact_paths["filtered_baseline_timeseries"]),
        "baseline_time_index": int(baseline_time_index),
        "deltaf_time_index": int(deltaf_time_idx_stage2),
        "stage_inputs_source": str(stage_inputs_source) if stage_inputs_source else None,
    }

    mapping_stage2 = DatasetMapping(
        global_indices=voxel_indices_filtered.astype(np.int64, copy=False),
        brain_shape=mapping_stage1.brain_shape,
    )

    output = StageOutput(
        name="baseline_threshold_filter",
        summary=summary,
        artifacts={key: Path(value) for key, value in artifact_paths.items()},
        mapping=mapping_stage2,
        mask_volume=filtered_mask_volume.astype(bool, copy=False),
        prefix=stage_prefix,
    )
    return output
