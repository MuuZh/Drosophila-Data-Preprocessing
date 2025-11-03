"""Shared helpers for voxel filtering development stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import h5py
import numpy as np
from tqdm.auto import tqdm

from ..data_io import open_preprocessed_dataset


def _sanitize_value(value: int | float) -> str:
    """Create a filename-safe token for numeric values."""
    if isinstance(value, float):
        if not np.isfinite(value):
            return "nan"
        sign = "neg" if value < 0 else ""
        magnitude = abs(value)
        token = f"{magnitude:.3f}".rstrip("0").rstrip(".")
        token = token.replace(".", "p") if token else "0"
        return f"{sign}{token}"
    return str(value)


def build_stage_prefix(stage_name: str, parameters: Mapping[str, int | float | str]) -> str:
    """Construct a filename prefix that encodes stage parameters."""
    parts = [stage_name]
    for key, value in parameters.items():
        if isinstance(value, (int, float)):
            parts.append(f"{key}{_sanitize_value(value)}")
        else:
            parts.append(f"{key}{value}")
    return "_".join(parts)


@dataclass
class DatasetMapping:
    """Mapping utilities between flattened brain space and dataset columns."""

    global_indices: np.ndarray
    brain_shape: Tuple[int, int, int]

    def values_to_volume(self, values: np.ndarray) -> np.ndarray:
        """Map per-voxel values back to a 3D brain volume."""
        flat_count = int(np.prod(self.brain_shape))
        flat_volume = np.zeros(flat_count, dtype=np.float32)
        flat_volume[self.global_indices] = values.astype(np.float32, copy=False)
        return flat_volume.reshape(self.brain_shape)

    def mask_to_volume(self, mask: np.ndarray) -> np.ndarray:
        """Convert a boolean selection mask to a brain-volume mask."""
        flat_count = int(np.prod(self.brain_shape))
        flat_mask = np.zeros(flat_count, dtype=bool)
        flat_mask[self.global_indices] = mask.astype(bool, copy=False)
        return flat_mask.reshape(self.brain_shape)


@dataclass
class StageOutput:
    """Results from a single filtering stage."""

    name: str
    summary: Dict[str, Any]
    artifacts: Dict[str, Path]
    mapping: DatasetMapping
    mask_volume: np.ndarray
    prefix: str


def resolve_dataset_mapping(
    brain_mask: np.ndarray,
    voxel_indices: Optional[np.ndarray],
    num_voxels: int,
) -> DatasetMapping:
    """Determine mapping between dataset columns and flattened brain mask indices."""
    brain_mask_flat = brain_mask.reshape(-1)
    active_indices = np.flatnonzero(brain_mask_flat > 0)

    if voxel_indices is not None:
        voxel_indices = np.asarray(voxel_indices, dtype=np.int64)
        max_index = int(voxel_indices.max(initial=0))
        if max_index >= active_indices.size:
            raise ValueError(
                "voxel_indices exceed active mask size; verify that the brain mask matches the dataset."
            )
        mapping = active_indices[voxel_indices]
    else:
        if active_indices.size != num_voxels:
            raise ValueError(
                "Brain mask active voxel count does not match dataset columns; provide voxel_indices in the HDF5 file."
            )
        mapping = active_indices.astype(np.int64, copy=False)

    return DatasetMapping(global_indices=mapping, brain_shape=brain_mask.shape)


def compute_dataset_mean(dataset: h5py.Dataset) -> np.ndarray:
    """Compute mean across time axis by loading the full dataset."""
    print("      Loading full DeltaF/F0 dataset into memory for mean computation...")
    array = np.asarray(dataset, dtype=np.float32)
    return np.mean(array, axis=0).astype(np.float32)


def load_full_baseline(memmap: np.ndarray) -> np.ndarray:
    """Load the entire baseline array into memory once."""
    print("      Loading full baseline dataset into memory...")
    return np.asarray(memmap, dtype=np.float32)


def save_filtered_artifacts(
    output_dir: Path,
    stage_prefix: str,
    stage_name: str,
    preprocessed_path: Path,
    baseline_array: np.ndarray,
    keep_mask: np.ndarray,
    voxel_indices_filtered: np.ndarray,
    filtered_mask_volume: np.ndarray,
    reference_mask_shape: Tuple[int, int, int],
    reference_mask_path: Path,
    stage_parameters: Mapping[str, Any],
    preloaded_deltaf: Optional[np.ndarray] = None,
) -> Dict[str, Path]:
    """Persist filtered DeltaF/F0, baseline, and mask artifacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    keep_mask = np.asarray(keep_mask, dtype=bool)

    h5_path = output_dir / f"{stage_prefix}_filtered.h5"
    mask_path = output_dir / f"{stage_prefix}_mask.npy"
    baseline_ts_path = output_dir / f"{stage_prefix}_baseline_timeseries.npy"

    print(f"    [save] Writing filtered HDF5: {h5_path}")
    with h5py.File(h5_path, "w") as handle:
        if preloaded_deltaf is not None:
            filtered_data = preloaded_deltaf[:, keep_mask].astype(np.float32, copy=False)
        else:
            with open_preprocessed_dataset(preprocessed_path) as source_handle:
                dataset = source_handle["deltaf_f0"]
                filtered_data = np.asarray(dataset[:, keep_mask], dtype=np.float32)
        handle.create_dataset("deltaf_f0", data=filtered_data, compression="gzip")

        handle.create_dataset(
            "voxel_indices",
            data=voxel_indices_filtered.astype(np.int64, copy=False),
            compression="gzip",
        )
        handle.create_dataset(
            "keep_mask",
            data=keep_mask.astype(np.bool_),
            compression="gzip",
        )
        handle.attrs["reference_mask_shape"] = tuple(int(x) for x in reference_mask_shape)
        handle.attrs["reference_mask_path"] = str(reference_mask_path)
        handle.attrs["stage_name"] = stage_name
        handle.attrs["stage_prefix"] = stage_prefix
        for key, value in stage_parameters.items():
            handle.attrs[key] = value

    print(f"    [save] Writing mask: {mask_path}")
    np.save(mask_path, filtered_mask_volume.astype(np.uint8))

    print(f"    [save] Writing baseline timeseries: {baseline_ts_path}")
    if baseline_array.ndim == 1:
        baseline_timeseries = baseline_array[keep_mask].astype(np.float32, copy=False)[None, :]
    else:
        baseline_timeseries = baseline_array[:, keep_mask].astype(np.float32, copy=False)
    np.save(baseline_ts_path, baseline_timeseries)

    return {
        "filtered_h5": h5_path,
        "filtered_mask": mask_path,
        "filtered_baseline_timeseries": baseline_ts_path,
    }


def load_stage_mapping_and_mask(
    filtered_h5_path: Path,
    mask_path: Optional[Path] = None,
) -> tuple[DatasetMapping, np.ndarray]:
    """Load mapping and mask volume from a filtered stage artifact."""
    with h5py.File(filtered_h5_path, "r") as handle:
        if "voxel_indices" not in handle:
            raise KeyError(f"'voxel_indices' dataset missing from {filtered_h5_path}")
        voxel_indices = handle["voxel_indices"][:].astype(np.int64, copy=False)
        if "reference_mask_shape" not in handle.attrs:
            raise KeyError(f"'reference_mask_shape' attribute missing from {filtered_h5_path}")
        brain_shape = tuple(int(x) for x in handle.attrs["reference_mask_shape"])

    mapping = DatasetMapping(global_indices=voxel_indices, brain_shape=brain_shape)

    if mask_path is not None and Path(mask_path).exists():
        mask_volume = np.load(mask_path).astype(bool, copy=False)
    else:
        mask_volume = mapping.mask_to_volume(np.ones(voxel_indices.size, dtype=bool))

    return mapping, mask_volume
