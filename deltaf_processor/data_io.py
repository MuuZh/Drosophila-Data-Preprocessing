"""Utilities for loading calcium imaging data."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


def _normalize_path(path_value: str | Path, base_directory: str | Path | None) -> Path:
    """Resolve a data path string relative to the optional base directory."""
    path = Path(path_value)
    if path.is_absolute():
        return path

    if base_directory:
        base_dir = Path(base_directory)
        return (base_dir / path).expanduser().resolve()

    return path.expanduser().resolve()


@lru_cache(maxsize=None)
def _load_config(config_path: str | Path = "config.json") -> dict[str, Any]:
    """Load and cache the project configuration file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found at {config_file}")

    with config_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_data_path(
    key: str,
    config_path: str | Path = "config.json",
) -> Path:
    """
    Resolve a data path from config.json using `data_paths[key]`.

    Parameters
    ----------
    key : str
        Key in the config's ``data_paths`` section.
    config_path : str | Path, optional
        Path to the configuration file (default: ``config.json``).

    Returns
    -------
    Path
        Resolved absolute (or expanded) path for the requested data item.
    """
    config = _load_config(config_path)
    data_paths = config.get("data_paths", {})
    if key not in data_paths:
        raise KeyError(f"'{key}' not found in data_paths section of {config_path}")

    base_directory = data_paths.get("base_directory")
    resolved = _normalize_path(data_paths[key], base_directory)
    return resolved


def load_data(
    data_file: str | Path | None = None,
    n_voxels: int | str | None = None,
    config_path: str | Path = "config.json",
):
    """
    Load calcium imaging data with optional voxel sampling.

    Parameters
    ----------
    data_file : str | Path | None, optional
        Path to data file. If ``None``, uses ``raw_calcium_data`` from config.
    n_voxels : int | str | None, optional
        Number of voxels to randomly sample (if ``None`` or ``"all"``, load all).
    config_path : str | Path, optional
        Path to configuration file for resolving default data locations.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, Path]
        Loaded calcium imaging data, indices of selected voxels,
        and the resolved path to the data file.
    """
    if data_file is None:
        data_path = resolve_data_path("raw_calcium_data", config_path=config_path)
    else:
        data_path = _normalize_path(data_file, None)

    print(f"Loading data from: {data_path}")
    raw_data = np.load(data_path)
    print(f"Original data shape: {raw_data.shape}")

    # Handle string input for n_voxels
    if isinstance(n_voxels, str):
        if n_voxels.lower() == "all":
            n_voxels = None
        else:
            try:
                n_voxels = int(n_voxels)
            except ValueError:
                print(f"Warning: Invalid voxels value '{n_voxels}', loading all voxels")
                n_voxels = None

    if n_voxels is not None and n_voxels < raw_data.shape[1]:
        print(f"Loading {n_voxels:,} random voxels")
        np.random.seed(42)  # For reproducibility
        voxel_indices = np.random.choice(raw_data.shape[1], n_voxels, replace=False)
        voxel_indices = np.sort(voxel_indices)
        data = raw_data[:, voxel_indices]
    else:
        print("Loading all voxels")
        data = raw_data
        voxel_indices = np.arange(raw_data.shape[1])

    print(f"Loaded data shape: {data.shape}")
    return data, voxel_indices, data_path


def load_brain_mask(
    mask_file: str | Path | None = None,
    config_path: str | Path = "config.json",
):
    """
    Load the 3D brain mask array.

    Parameters
    ----------
    mask_file : str | Path | None, optional
        Path to mask file. If ``None``, uses ``brain_mask`` from config.
    config_path : str | Path, optional
        Path to configuration file for resolving default mask location.

    Returns
    -------
    tuple[np.ndarray, Path]
        Brain mask array and the resolved path to the mask file.
    """
    if mask_file is None:
        mask_path = resolve_data_path("brain_mask", config_path=config_path)
    else:
        mask_path = _normalize_path(mask_file, None)

    print(f"Loading brain mask from: {mask_path}")
    mask = np.load(mask_path)
    print(f"Brain mask shape: {mask.shape}")
    return mask, mask_path
