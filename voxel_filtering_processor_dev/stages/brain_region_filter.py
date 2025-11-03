"""Brain-region masking stage for the voxel filtering dev pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from tqdm.auto import tqdm

from ..data_io import (
    load_baseline_memmap,
    load_brain_mask,
    load_brain_region_mask,
    open_preprocessed_dataset,
)
from ..visualization import TriviewConfig, save_region_triview, save_triview
from .common import (
    DatasetMapping,
    StageOutput,
    build_stage_prefix,
    load_full_baseline,
    resolve_dataset_mapping,
    save_filtered_artifacts,
)


def run_brain_region_filter(
    *,
    base_directory: Path,
    output_dir: Path,
    image_dir: Path,
    preprocessed_path: Path,
    baseline_path: Path,
    brain_mask_path: Path,
    brain_region_mask_path: Path,
    baseline_time_index: int,
    deltaf_time_index: int,
    chunk_size: int,
    stage_inputs_source: Optional[Path],
    exclude_regions: list[int] | None = None,
    output_suffix: str = "",
) -> StageOutput:
    """Execute stage 1: retain voxels that fall within the supplied brain-region mask."""
    if exclude_regions is None:
        exclude_regions = []

    print("=" * 60)
    print("STAGE 1: Brain Region Filter")
    print("=" * 60)
    print(f"DeltaF/F0 data: {preprocessed_path}")
    print(f"Baseline data: {baseline_path}")
    print(f"Brain mask: {brain_mask_path}")
    print(f"Brain region mask: {brain_region_mask_path}")
    if exclude_regions:
        print(f"Excluding regions: {exclude_regions}")
    if output_suffix:
        print(f"Output suffix: {output_suffix}")

    print("[1/4] Loading brain and region masks...")
    brain_mask = load_brain_mask(brain_mask_path)
    brain_region_mask = load_brain_region_mask(brain_region_mask_path)
    if brain_mask.shape != brain_region_mask.shape:
        raise ValueError(
            f"Brain region mask shape {brain_region_mask.shape} does not match brain mask shape {brain_mask.shape}"
        )

    brain_mask_bool = brain_mask > 0
    brain_mask_shape = brain_mask_bool.shape

    # Create region mask: exclude background (0) and specified regions
    if exclude_regions:
        region_mask_bool = brain_region_mask > 0
        for region_id in exclude_regions:
            region_mask_bool = region_mask_bool & (brain_region_mask != region_id)
        excluded_voxel_count = np.sum(np.isin(brain_region_mask, exclude_regions))
        print(f"    Excluded {excluded_voxel_count:,} voxels from regions {exclude_regions}")
    else:
        region_mask_bool = brain_region_mask > 0

    print("[2/4] Loading baseline and DeltaF/F0 datasets...")
    baseline_memmap = load_baseline_memmap(baseline_path)
    print(f"    Baseline array shape: {baseline_memmap.shape}")

    with open_preprocessed_dataset(preprocessed_path) as handle:
        if "deltaf_f0" not in handle:
            raise KeyError("Dataset 'deltaf_f0' not found in preprocessed file.")
        dataset = handle["deltaf_f0"]
        if dataset.ndim != 2:
            raise ValueError(f"Expected 2D dataset, got shape {dataset.shape}")
        deltaf_array = np.asarray(dataset, dtype=np.float32)
        timepoints, num_voxels = deltaf_array.shape
        print(f"    DeltaF/F0 shape: {timepoints} frames x {num_voxels} voxels")
        voxel_indices = handle["voxel_indices"][:] if "voxel_indices" in handle else None

    initial_prefix = build_stage_prefix("brain_region_filter_initial", {"kept": num_voxels})
    if output_suffix:
        initial_prefix = f"{initial_prefix}_{output_suffix}"

    try:
        mapping_raw = resolve_dataset_mapping(
            brain_mask=brain_mask_bool,
            voxel_indices=None,
            num_voxels=num_voxels,
        )
        print("    Mapping voxels using provided brain mask (direct alignment).")
    except ValueError:
        if voxel_indices is None:
            raise
        print("    Direct alignment failed; using voxel_indices from source dataset.")
        mapping_raw = resolve_dataset_mapping(
            brain_mask=brain_mask_bool,
            voxel_indices=voxel_indices,
            num_voxels=num_voxels,
        )

    print("[3/4] Computing baseline and DeltaF/F0 summaries...")
    if baseline_memmap.ndim == 1:
        baseline_array = np.asarray(baseline_memmap, dtype=np.float32)
        baseline_time_label = "Baseline Values"
        baseline_reference_mode = "single_vector"
        baseline_reference_index = -1
    else:
        baseline_array = load_full_baseline(baseline_memmap)
        baseline_time_idx = min(max(baseline_time_index, 0), baseline_array.shape[0] - 1)
        baseline_time_vec = baseline_array[baseline_time_idx, :].astype(np.float32, copy=False)
        baseline_time_label = f"Baseline Frame {baseline_time_idx}"
        baseline_reference_mode = "timeseries"
        baseline_reference_index = int(baseline_time_idx)

    if baseline_memmap.ndim == 1:
        baseline_time_vec = baseline_array.astype(np.float32, copy=False)
        baseline_max_vec = baseline_array.astype(np.float32, copy=False)
        baseline_mean_vec = baseline_array.astype(np.float32, copy=False)
    else:
        baseline_max_vec = np.max(baseline_array, axis=0).astype(np.float32, copy=False)
        baseline_mean_vec = np.mean(baseline_array, axis=0).astype(np.float32, copy=False)

    deltaf_time_idx = min(max(deltaf_time_index, 0), timepoints - 1)
    deltaf_time_vec = deltaf_array[deltaf_time_idx, :].astype(np.float32, copy=False)
    print("      Calculating DeltaF/F0 temporal mean (full dataset)...")
    deltaf_mean_vec = np.mean(deltaf_array, axis=0).astype(np.float32, copy=False)

    print("      Mapping baseline and DeltaF/F0 vectors back to brain volume...")
    volume_tasks = [
        ("Baseline slice vector", baseline_time_vec, "baseline_time_volume"),
        ("Baseline max vector", baseline_max_vec, "baseline_max_volume"),
        ("DeltaF/F0 slice vector", deltaf_time_vec, "deltaf_time_volume"),
        ("DeltaF/F0 mean vector", deltaf_mean_vec, "deltaf_mean_volume"),
    ]
    mapped_volumes: Dict[str, np.ndarray] = {}
    for description, vector, key in tqdm(volume_tasks, desc="      Volume mapping", leave=False):
        mapped_volumes[key] = mapping_raw.values_to_volume(vector)

    baseline_time_volume = mapped_volumes["baseline_time_volume"]
    baseline_max_volume = mapped_volumes["baseline_max_volume"]
    deltaf_time_volume = mapped_volumes["deltaf_time_volume"]
    deltaf_mean_volume = mapped_volumes["deltaf_mean_volume"]
    region_volume = brain_region_mask.astype(np.int32, copy=False)
    overlay_mask_volume = region_mask_bool.astype(bool, copy=False)

    print("[4/4] Rendering diagnostics before filtering...")
    prefilter_triview_tasks = [
        (baseline_time_volume, f"{initial_prefix}_baseline_slice_triview.png", baseline_time_label, "Baseline (F0)", TriviewConfig(mode="slice")),
        (baseline_max_volume, f"{initial_prefix}_baseline_max_projection_triview.png", "Baseline Max Projection (all frames)", "Baseline (F0)", TriviewConfig(mode="max_projection")),
        (deltaf_time_volume, f"{initial_prefix}_deltaf_slice_triview.png", f"DeltaF/F0 Frame {deltaf_time_idx}", "dF/F0", TriviewConfig(mode="slice")),
        (deltaf_mean_volume, f"{initial_prefix}_deltaf_mean_projection_triview.png", "DeltaF/F0 Mean Across Time", "dF/F0", TriviewConfig(mode="mean_projection")),
    ]
    for volume, path_out, title_prefix, colorbar, config in tqdm(prefilter_triview_tasks, desc="      Saving pre-filter tri-views", leave=False):
        save_triview(
            volume=volume,
            output_path=image_dir / path_out,
            title_prefix=title_prefix,
            config=config,
            overlay_mask=overlay_mask_volume,
            colorbar_label=colorbar,
        )

    save_region_triview(
        region_volume=region_volume,
        output_path=image_dir / f"{initial_prefix}_brain_region_slice_triview.png",
        title_prefix="Brain Region Labels (Slice)",
        config=TriviewConfig(mode="slice"),
    )
    save_region_triview(
        region_volume=region_volume,
        output_path=image_dir / f"{initial_prefix}_brain_region_projection_triview.png",
        title_prefix="Brain Region Labels (Max Projection)",
        config=TriviewConfig(mode="max_projection"),
    )

    keep_mask = region_mask_bool.reshape(-1)[mapping_raw.global_indices] > 0
    kept_count = int(keep_mask.sum())
    removed_count = int(num_voxels - kept_count)
    stage_prefix = build_stage_prefix("brain_region_filter", {"kept": kept_count})
    if output_suffix:
        stage_prefix = f"{stage_prefix}_{output_suffix}"

    print("Applying brain region mask selection...")
    print(f"    Voxels kept: {kept_count:,}")
    print(f"    Voxels removed: {removed_count:,}")
    if exclude_regions:
        print(f"    Excluded regions: {exclude_regions}")

    filtered_mask_volume = mapping_raw.mask_to_volume(keep_mask)
    filtered_mask_bool = filtered_mask_volume.astype(bool, copy=False)

    print("      Mapping filtered vectors back to brain volume...")
    mask_float = keep_mask.astype(np.float32, copy=False)
    filtered_volume_tasks = [
        ("Filtered baseline slice", baseline_time_vec * mask_float, "baseline_time_filtered"),
        ("Filtered baseline max", baseline_max_vec * mask_float, "baseline_max_filtered"),
        ("Filtered DeltaF/F0 slice", deltaf_time_vec * mask_float, "deltaf_time_filtered"),
        ("Filtered DeltaF/F0 mean", deltaf_mean_vec * mask_float, "deltaf_mean_filtered"),
    ]
    filtered_volumes: Dict[str, np.ndarray] = {}
    for description, vector, key in tqdm(filtered_volume_tasks, desc="      Volume mapping (filtered)", leave=False):
        filtered_volumes[key] = mapping_raw.values_to_volume(vector)

    baseline_time_filtered = filtered_volumes["baseline_time_filtered"]
    baseline_max_filtered = filtered_volumes["baseline_max_filtered"]
    deltaf_time_filtered = filtered_volumes["deltaf_time_filtered"]
    deltaf_mean_filtered = filtered_volumes["deltaf_mean_filtered"]

    filtered_triview_tasks = [
        (baseline_time_filtered, f"{stage_prefix}_baseline_slice_triview_filtered.png", f"{baseline_time_label} (Filtered)", "Baseline (F0)", TriviewConfig(mode="slice")),
        (baseline_max_filtered, f"{stage_prefix}_baseline_max_projection_triview_filtered.png", "Baseline Max Projection (Filtered)", "Baseline (F0)", TriviewConfig(mode="max_projection")),
        (deltaf_time_filtered, f"{stage_prefix}_deltaf_slice_triview_filtered.png", f"DeltaF/F0 Frame {deltaf_time_idx} (Filtered)", "dF/F0", TriviewConfig(mode="slice")),
        (deltaf_mean_filtered, f"{stage_prefix}_deltaf_mean_projection_triview_filtered.png", "DeltaF/F0 Mean Across Time (Filtered)", "dF/F0", TriviewConfig(mode="mean_projection")),
    ]
    for volume, path_out, title_prefix, colorbar, config in tqdm(filtered_triview_tasks, desc="      Saving post-filter tri-views", leave=False):
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
        output_path=image_dir / f"{stage_prefix}_mask_presence_slice_filtered.png",
        title_prefix="Brain Region Presence (Filtered Slice)",
        config=TriviewConfig(mode="slice"),
        colorbar_label="Mask Presence",
        region_ticks=[0, 1],
    )

    voxel_indices_filtered = mapping_raw.global_indices[keep_mask]
    artifact_paths = save_filtered_artifacts(
        output_dir=output_dir,
        stage_prefix=stage_prefix,
        stage_name="brain_region_filter",
        preprocessed_path=preprocessed_path,
        baseline_array=baseline_array,
        keep_mask=keep_mask,
        voxel_indices_filtered=voxel_indices_filtered,
        filtered_mask_volume=filtered_mask_volume,
        reference_mask_shape=brain_mask_shape,
        reference_mask_path=brain_mask_path,
        stage_parameters={
            "baseline_reference_mode": baseline_reference_mode,
            "baseline_reference_index": baseline_reference_index,
            "deltaf_frame_index": int(deltaf_time_idx),
            "kept_voxels": int(kept_count),
            "removed_voxels": int(removed_count),
        },
        preloaded_deltaf=deltaf_array,
    )

    print("    Filtered HDF5 saved to:", artifact_paths["filtered_h5"])
    print("    Filtered mask saved to:", artifact_paths["filtered_mask"])
    print("    Filtered baseline timeseries saved to:", artifact_paths["filtered_baseline_timeseries"])

    summary: Dict[str, Any] = {
        "stage": "brain_region_filter",
        "total_voxels": int(num_voxels),
        "kept_voxels": kept_count,
        "removed_voxels": removed_count,
        "exclude_regions": exclude_regions,
        "output_suffix": output_suffix,
        "output_directory": str(output_dir),
        "filtered_data_path": str(artifact_paths["filtered_h5"]),
        "filtered_mask_path": str(artifact_paths["filtered_mask"]),
        "filtered_baseline_timeseries_path": str(artifact_paths["filtered_baseline_timeseries"]),
        "baseline_reference_mode": baseline_reference_mode,
        "baseline_reference_index": baseline_reference_index,
        "baseline_time_index": int(baseline_time_index),
        "deltaf_time_index": int(deltaf_time_idx),
        "stage_inputs_source": str(stage_inputs_source) if stage_inputs_source else None,
        "brain_mask_path": str(brain_mask_path),
        "brain_region_mask_path": str(brain_region_mask_path),
        "baseline_path": str(baseline_path),
        "deltaf_path": str(preprocessed_path),
    }

    output = StageOutput(
        name="brain_region_filter",
        summary=summary,
        artifacts={key: Path(value) for key, value in artifact_paths.items()},
        mapping=DatasetMapping(global_indices=voxel_indices_filtered.astype(np.int64, copy=False), brain_shape=brain_mask_shape),
        mask_volume=filtered_mask_volume.astype(bool, copy=False),
        prefix=stage_prefix,
    )
    return output
