"""High-level processing pipeline orchestration."""

import time
import numpy as np

from .deltaf import calculate_deltaf_cpu, calculate_deltaf_gpu
from .filtering import conservative_gaussian_filtering


def three_stage_processing_pipeline(
    raw_data,
    deltaf_method="gpu",
    gaussian_filtering=True,
    sigma=0.5,
    window_size=400,
    percentile=10.0,
):
    """
    Execute the three-stage ΔF/F0 processing pipeline.

    Parameters
    ----------
    raw_data : np.ndarray
        Raw motion-corrected calcium data (timepoints, voxels).
    deltaf_method : str
        Method for ΔF/F0 calculation ('gpu' or 'cpu').
    gaussian_filtering : bool
        Whether to perform Gaussian filtering (default: True).
    sigma : float
        Gaussian filter sigma parameter (default: 0.5).
    window_size : int
        Rolling window size for baseline calculation (default: 400).
    percentile : float
        Percentile for baseline calculation (default: 10.0).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        Final ΔF/F0 data, baseline array, and processing statistics.
    """

    print("=" * 60)
    print("THREE-STAGE ΔF/F0 PROCESSING PIPELINE")
    print("=" * 60)

    processing_stats = {
        "original_shape": raw_data.shape,
        "processing_time": {},
        "data_quality": {},
    }

    # Stage 1: Gaussian Filtering
    print("\nSTAGE 1: CONSERVATIVE GAUSSIAN FILTERING")
    print("-" * 40)

    stage1_start = time.time()

    if gaussian_filtering:
        filtered_data = conservative_gaussian_filtering(raw_data, sigma=sigma)
        processing_stats["gaussian_filtering"] = True
        processing_stats["gaussian_sigma"] = sigma
    else:
        print("Gaussian filtering disabled, using raw data")
        filtered_data = raw_data
        processing_stats["gaussian_filtering"] = False

    stage1_time = time.time() - stage1_start
    processing_stats["processing_time"]["stage1"] = stage1_time

    # Stage 2: ΔF/F0 Calculation with Baseline Estimation
    print("\nSTAGE 2: ΔF/F0 CALCULATION")
    print("-" * 40)
    print(f"Method: {deltaf_method.upper()} | Window: {window_size} | Percentile: {percentile}")

    stage2_start = time.time()

    if deltaf_method == "gpu":
        deltaf_result, baselines = calculate_deltaf_gpu(
            filtered_data,
            window_size=window_size,
            percentile=percentile,
        )
    else:
        deltaf_result, baselines = calculate_deltaf_cpu(
            filtered_data,
            window_size=window_size,
            percentile=percentile,
        )

    stage2_time = time.time() - stage2_start
    processing_stats["processing_time"]["stage2"] = stage2_time
    processing_stats["processing_time"]["total"] = stage1_time + stage2_time

    # Final Quality Assessment
    print("\nFINAL RESULTS")
    print("-" * 30)

    nan_count = np.sum(np.isnan(deltaf_result))
    inf_count = np.sum(np.isinf(deltaf_result))
    extreme_positive = np.sum(deltaf_result > 100)
    extreme_negative = np.sum(deltaf_result < -0.99)

    total_values = deltaf_result.size

    processing_stats["data_quality"] = {
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "extreme_positive": int(extreme_positive),
        "extreme_negative": int(extreme_negative),
        "total_values": int(total_values),
        "value_range": [float(np.min(deltaf_result)), float(np.max(deltaf_result))],
        "mean": float(np.mean(deltaf_result)),
        "std": float(np.std(deltaf_result)),
    }

    print("Data quality assessment:")
    print(f"  Total values: {total_values:,}")
    print(f"  NaN values: {nan_count} ({nan_count / total_values * 100:.4f}%)")
    print(f"  Inf values: {inf_count} ({inf_count / total_values * 100:.4f}%)")
    print(
        f"  Extreme positive (>100): {extreme_positive} ({extreme_positive / total_values * 100:.4f}%)"
    )
    print(
        f"  Extreme negative (<-0.99): {extreme_negative} ({extreme_negative / total_values * 100:.4f}%)"
    )
    print(
        "  Value range: "
        f"[{processing_stats['data_quality']['value_range'][0]:.3f}, "
        f"{processing_stats['data_quality']['value_range'][1]:.3f}]"
    )
    print(
        f"  Mean ± std: {processing_stats['data_quality']['mean']:.3f} ± "
        f"{processing_stats['data_quality']['std']:.3f}"
    )

    print("\nProcessing time breakdown:")
    print(f"  Stage 1 (Gaussian filtering): {stage1_time:.2f}s")
    print(f"  Stage 2 (ΔF/F0 + baseline estimation): {stage2_time:.2f}s")
    print(f"  Total: {stage1_time + stage2_time:.2f}s")

    return deltaf_result, baselines, processing_stats
