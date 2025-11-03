"""Data loading utilities for voxel filtering development pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generator, Tuple

import h5py
import numpy as np


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load project configuration from JSON file."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def open_preprocessed_dataset(file_path: str | Path) -> h5py.File:
    """Open the preprocessed DeltaF/F0 dataset."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Preprocessed data file not found: {path}")
    return h5py.File(path, "r")


def load_baseline_memmap(file_path: str | Path) -> np.memmap:
    """Load baseline array using memory mapping to avoid loading into RAM."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline array file not found: {path}")
    return np.load(path, mmap_mode="r")


def load_brain_mask(file_path: str | Path) -> np.ndarray:
    """Load the binary brain mask."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Brain mask file not found: {path}")
    mask = np.load(path)
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D brain mask, got shape {mask.shape}")
    return mask


def load_brain_region_mask(file_path: str | Path) -> np.ndarray:
    """Load the labeled brain region mask."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Brain region mask file not found: {path}")
    mask = np.load(path)
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D brain region mask, got shape {mask.shape}")
    return mask


def ensure_output_directory(base_directory: Path, subdir: str) -> Path:
    """Create (if needed) and return the development output directory."""
    output_dir = base_directory / "outputs" / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_filtered_results(
    h5_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load previously saved filtered results."""
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"Filtered HDF5 file not found: {path}")

    with h5py.File(path, "r") as handle:
        if "deltaf_f0" not in handle:
            raise KeyError("Dataset 'deltaf_f0' missing from filtered results.")
        deltaf = handle["deltaf_f0"][:]

        if "voxel_indices" in handle:
            voxel_indices = handle["voxel_indices"][:]
        else:
            voxel_indices = np.empty(deltaf.shape[1], dtype=np.int64)

        if "keep_mask" in handle:
            keep_mask = handle["keep_mask"][:].astype(bool)
        else:
            keep_mask = np.ones(deltaf.shape[1], dtype=bool)

        attrs = dict(handle.attrs)

    return deltaf, voxel_indices, keep_mask, attrs


def iterate_hdf5_in_chunks(
    dataset: h5py.Dataset,
    chunk_size: int,
) -> Generator[np.ndarray, None, None]:
    """Yield dataset chunks along the time axis."""
    timepoints = dataset.shape[0]
    for start in range(0, timepoints, chunk_size):
        stop = min(start + chunk_size, timepoints)
        yield dataset[start:stop, :]
