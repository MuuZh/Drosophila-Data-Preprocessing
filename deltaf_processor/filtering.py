"""Filtering helpers for ΔF/F0 processing."""

import numpy as np
from scipy import ndimage
from tqdm import tqdm


def conservative_gaussian_filtering(data, sigma=0.5):
    """
    Conservative Gaussian filtering for noise reduction.

    Parameters
    ----------
    data : np.ndarray
        Raw fluorescence data (timepoints, voxels).
    sigma : float
        Gaussian filter sigma parameter (default: 0.5 for conservative filtering).

    Returns
    -------
    np.ndarray
        Gaussian filtered data (or original data if sigma == 0).
    """
    if sigma == 0:
        print("Gaussian filtering skipped (sigma=0), using original data")
        return data

    print(f"Performing conservative Gaussian filtering (sigma={sigma})...")

    filtered_data = np.zeros_like(data)
    total_voxels = data.shape[1]

    progress_interval = max(1, total_voxels // 20)

    for voxel in tqdm(range(total_voxels), desc="Gaussian filtering"):
        filtered_data[:, voxel] = ndimage.gaussian_filter1d(data[:, voxel], sigma)

        if (voxel + 1) % progress_interval == 0:
            print(
                f"  Filtered {voxel + 1}/{total_voxels} voxels "
                f"({(voxel + 1) / total_voxels * 100:.1f}%)"
            )

    original_std = np.std(data)
    filtered_std = np.std(filtered_data)
    noise_reduction = (original_std - filtered_std) / original_std * 100

    print("Gaussian filtering completed:")
    print(f"  Original signal std: {original_std:.3f}")
    print(f"  Filtered signal std: {filtered_std:.3f}")
    print(f"  Noise reduction: {noise_reduction:.1f}%")

    return filtered_data
