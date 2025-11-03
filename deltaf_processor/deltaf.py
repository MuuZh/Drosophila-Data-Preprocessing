"""ΔF/F0 computation utilities."""

import numpy as np


def calculate_deltaf_cpu(data, window_size=400, percentile=10):
    """
    CPU-based ΔF/F0 calculation with rolling baseline estimation.

    Parameters
    ----------
    data : np.ndarray
        Raw calcium imaging data (timepoints, voxels).
    window_size : int
        Rolling window size for baseline calculation.
    percentile : float
        Percentile for baseline calculation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ΔF/F0 data and the corresponding baseline array.
    """
    n_timepoints, n_voxels = data.shape
    baselines = np.zeros_like(data)

    print(f"Computing ΔF/F0 with CPU (window={window_size}, percentile={percentile})")

    # Step 0: Calculate global baseline for each voxel (fallback for edge frames)
    print("Computing global baseline for edge frame fallback...")
    global_baseline = np.percentile(data, percentile, axis=0)

    # Step 1: Calculate all baselines with edge padding using global fallback
    half_window = window_size // 2
    target_window_length = half_window * 2 + 1
    edge_padding_pairs = 0

    for t in range(n_timepoints):
        start = max(0, t - half_window)
        end = min(n_timepoints, t + half_window + 1)
        actual_window_size = end - start

        window_data = data[start:end, :]

        left_available = t - start
        right_available = end - t - 1

        missing_left = half_window - left_available
        if missing_left < 0:
            missing_left = 0
        missing_right = half_window - right_available
        if missing_right < 0:
            missing_right = 0

        missing_total = missing_left + missing_right

        if missing_total > 0:
            padded = np.empty(
                (actual_window_size + missing_total, n_voxels),
                dtype=data.dtype,
            )
            padded[:actual_window_size, :] = window_data
            padded[actual_window_size:, :] = global_baseline
            window_for_percentile = padded
            edge_padding_pairs += missing_total * n_voxels
        else:
            window_for_percentile = window_data

        baselines[t, :] = np.percentile(window_for_percentile, percentile, axis=0)

    if edge_padding_pairs > 0:
        print(
            f"Applied global baseline padding for {edge_padding_pairs:,} timepoint-voxel slots "
            f"to maintain window length {target_window_length}"
        )
    else:
        print(f"All frames used full rolling window length {target_window_length}")

    # Step 2: Compute ΔF/F0 without thresholding
    with np.errstate(divide="ignore", invalid="ignore"):
        deltaf_result = (data - baselines) / baselines

    deltaf_result = np.nan_to_num(deltaf_result, nan=0.0, posinf=0.0, neginf=0.0)

    return deltaf_result, baselines


def calculate_deltaf_gpu(data, window_size=400, percentile=10, gpu_memory_gb=8.0):
    """
    GPU-based ΔF/F0 calculation with rolling baseline estimation.

    Parameters
    ----------
    data : np.ndarray
        Raw calcium imaging data (timepoints, voxels).
    window_size : int
        Rolling window size for baseline calculation.
    percentile : float
        Percentile for baseline calculation.
    gpu_memory_gb : float
        GPU memory limit.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ΔF/F0 data and the corresponding baseline array.
    """
    try:
        import taichi as ti
    except ImportError as exc:  # pragma: no cover - import failure
        raise RuntimeError("Taichi is required for GPU ΔF/F0 calculation.") from exc

    try:
        ti.init(arch=ti.cuda, device_memory_GB=gpu_memory_gb)
        print(f"Taichi GPU initialized with {gpu_memory_gb:.1f} GB memory")

        n_timepoints, n_voxels = data.shape

        memory_per_voxel = n_timepoints * 4 * 3  # input, baseline, output arrays (float32)
        usable_memory = gpu_memory_gb * 0.8 * 1024**3  # Use 80% of GPU memory
        max_voxels_per_chunk = max(1, int(usable_memory / memory_per_voxel))

        print(f"Computing ΔF/F0 with GPU (window={window_size}, percentile={percentile})")
        print(
            f"GPU Memory: {gpu_memory_gb:.1f} GB, Max voxels per chunk: {max_voxels_per_chunk:,}"
        )

        if n_voxels <= max_voxels_per_chunk:
            print("  Processing in single chunk")
            result, baselines = process_deltaf_gpu_chunk(data, window_size, percentile)
            return result, baselines

        target_n_chunks = (n_voxels + max_voxels_per_chunk - 1) // max_voxels_per_chunk
        actual_chunk_size = (n_voxels + target_n_chunks - 1) // target_n_chunks
        n_chunks = target_n_chunks

        print(f"  Processing in {n_chunks} chunks with size ~{actual_chunk_size:,}")

        deltaf_result = np.zeros_like(data)
        baselines = np.zeros_like(data)

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * actual_chunk_size
            end_idx = min(start_idx + actual_chunk_size, n_voxels)
            current_chunk_size = end_idx - start_idx

            print(
                f"  Processing chunk {chunk_idx + 1}/{n_chunks}: voxels {start_idx}-{end_idx-1}, "
                f"size: {current_chunk_size:,}"
            )

            chunk_data = data[:, start_idx:end_idx]
            chunk_result, chunk_baseline = process_deltaf_gpu_chunk(
                chunk_data, window_size, percentile
            )
            deltaf_result[:, start_idx:end_idx] = chunk_result
            baselines[:, start_idx:end_idx] = chunk_baseline

            if chunk_idx < n_chunks - 1:
                ti.reset()
                ti.init(arch=ti.cuda, device_memory_GB=gpu_memory_gb)

        return deltaf_result, baselines

    except Exception as exc:  # pragma: no cover - failure path
        print(f"GPU processing failed: {exc}")
        raise

    finally:
        try:
            ti.reset()
        except Exception:
            pass


def process_deltaf_gpu_chunk(data, window_size, percentile):
    """Process a single chunk of data with GPU baseline estimation."""
    import taichi as ti

    n_timepoints, n_voxels = data.shape
    half_window = window_size // 2
    target_window_length = half_window * 2 + 1

    print(
        f"  Preparing global baselines and edge padding (target window length: {target_window_length})..."
    )
    global_baseline = np.percentile(data, percentile, axis=0)

    data_ti = ti.field(dtype=ti.f32, shape=(n_timepoints, n_voxels))
    f0_ti = ti.field(dtype=ti.f32, shape=(n_timepoints, n_voxels))
    result_ti = ti.field(dtype=ti.f32, shape=(n_timepoints, n_voxels))
    global_baseline_ti = ti.field(dtype=ti.f32, shape=n_voxels)

    data_ti.from_numpy(data.astype(np.float32))
    global_baseline_ti.from_numpy(global_baseline.astype(np.float32))

    edge_frame_count_ti = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def compute_baselines():
        for t, voxel_idx in ti.ndrange(n_timepoints, n_voxels):
            window_start = max(0, t - half_window)
            window_end = min(n_timepoints, t + half_window + 1)
            window_length = window_end - window_start

            window_min = data_ti[t, voxel_idx]
            window_max = window_min

            for w_idx in range(window_length):
                val = data_ti[window_start + w_idx, voxel_idx]
                window_min = min(window_min, val)
                window_max = max(window_max, val)

            left_available = t - window_start
            right_available = window_end - t - 1
            missing_left = half_window - left_available
            if missing_left < 0:
                missing_left = 0
            missing_right = half_window - right_available
            if missing_right < 0:
                missing_right = 0
            missing_total = missing_left + missing_right

            baseline_val = global_baseline_ti[voxel_idx]
            if missing_total > 0:
                window_min = min(window_min, baseline_val)
                window_max = max(window_max, baseline_val)
                ti.atomic_add(edge_frame_count_ti[None], missing_total)

            effective_length = window_length + missing_total
            f0_baseline = window_min

            if window_max != window_min:
                target_percentile = percentile / 100.0
                low = window_min
                high = window_max

                for _ in range(20):
                    mid = 0.5 * (low + high)
                    count_below = 0

                    for w_idx in range(window_length):
                        if data_ti[window_start + w_idx, voxel_idx] <= mid:
                            count_below += 1

                    if baseline_val <= mid:
                        count_below += missing_total

                    actual_percentile = float(count_below) / float(effective_length)

                    if actual_percentile < target_percentile:
                        low = mid
                    else:
                        high = mid

                f0_baseline = 0.5 * (low + high)

            f0_ti[t, voxel_idx] = f0_baseline

    @ti.kernel
    def compute_deltaf_f0():
        for t, voxel_idx in ti.ndrange(n_timepoints, n_voxels):
            f0_baseline = f0_ti[t, voxel_idx]
            raw_signal = data_ti[t, voxel_idx]
            if f0_baseline != 0.0:
                result_ti[t, voxel_idx] = (raw_signal - f0_baseline) / f0_baseline
            else:
                result_ti[t, voxel_idx] = 0.0

    edge_frame_count_ti[None] = 0
    compute_baselines()
    ti.sync()

    compute_deltaf_f0()
    ti.sync()

    edge_frame_count = edge_frame_count_ti[None]
    if edge_frame_count > 0:
        print(
            f"    GPU: Added global baseline padding for {edge_frame_count:,} timepoint-voxel slots "
            f"to maintain window length {target_window_length}"
        )
    else:
        print(
            f"    GPU: All baselines used full window length {target_window_length}"
        )

    result = result_ti.to_numpy()
    baselines = f0_ti.to_numpy()

    del data_ti, f0_ti, result_ti, global_baseline_ti, edge_frame_count_ti

    return result, baselines
