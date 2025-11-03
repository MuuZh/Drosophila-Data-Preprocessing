"""Visualization utilities for dF/F0 results."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # noqa: F401 imported for potential future styling
from pathlib import Path

from .data_io import resolve_data_path


def create_visualization(
    result,
    method,
    voxels_str,
    window_size=400,
    sigma=0.5,
    percentile=10.0,
):
    """Create and save standard dF/F0 visualizations."""
    print(f"Creating {method} visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    global_activity = np.mean(result, axis=1)
    axes[0, 0].plot(global_activity, alpha=0.8)
    axes[0, 0].set_title(
        f"Global Brain Activity ({method.upper()}, W={window_size}, P={percentile}%)"
    )
    axes[0, 0].set_xlabel("Time (frames)")
    axes[0, 0].set_ylabel("Mean dF/F0")
    axes[0, 0].grid(True, alpha=0.3)

    sample_data = result.flatten()
    if len(sample_data) > 100000:
        sample_data = np.random.choice(sample_data, 100000, replace=False)

    axes[0, 1].hist(sample_data, bins=50, alpha=0.7, edgecolor="black")
    axes[0, 1].set_title(
        f"dF/F0 Distribution ({method.upper()}, W={window_size}, P={percentile}%)"
    )
    axes[0, 1].set_xlabel("dF/F0")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True, alpha=0.3)

    n_samples = min(20, result.shape[1])
    sample_indices = np.linspace(0, result.shape[1] - 1, n_samples, dtype=int)

    for i, idx in enumerate(sample_indices):
        alpha = 0.7 if i < 5 else 0.3
        axes[1, 0].plot(result[:, idx], alpha=alpha, linewidth=1)

    axes[1, 0].set_title(f"Sample Voxel Time Series ({method.upper()}, W={window_size})")
    axes[1, 0].set_xlabel("Time (frames)")
    axes[1, 0].set_ylabel("dF/F0")
    axes[1, 0].grid(True, alpha=0.3)

    n_time_samples = min(200, result.shape[0])
    n_voxel_samples = min(100, result.shape[1])

    time_indices = np.linspace(0, result.shape[0] - 1, n_time_samples, dtype=int)
    voxel_indices = np.linspace(0, result.shape[1] - 1, n_voxel_samples, dtype=int)

    heatmap_data = result[np.ix_(time_indices, voxel_indices)]

    im = axes[1, 1].imshow(
        heatmap_data.T,
        aspect="auto",
        cmap="hot",
        vmin=np.percentile(heatmap_data, 1),
        vmax=np.percentile(heatmap_data, 99),
    )
    axes[1, 1].set_title(
        f"Activity Heatmap ({method.upper()}, W={window_size}, P={percentile}%)"
    )
    axes[1, 1].set_xlabel("Time (sampled frames)")
    axes[1, 1].set_ylabel("Voxels (sampled)")

    plt.colorbar(im, ax=axes[1, 1], label="dF/F0")
    plt.tight_layout()

    output_dir = Path("outputs/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_filename = (
        f"deltaf_threestage_{method}_results_{voxels_str}_sigma{sigma}_"
        f"window{window_size}_p{percentile}.png"
    )
    fig_path = output_dir / fig_filename

    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"{method.upper()} visualization saved to: {fig_path}")
    return fig_path


def create_brain_visualization(
    result,
    baselines,
    method,
    voxels_str,
    window_size=400,
    sigma=0.5,
    voxel_indices=None,
    percentile=10.0,
):
    """Create brain visualization with both signal and baseline projections."""
    print(f"Starting brain visualization for {method} method...")

    if baselines is None:
        raise ValueError("Baseline data is required for brain visualization.")

    try:
        mask_path = resolve_data_path("brain_mask")
        print(f"Checking for brain mask at: {mask_path}")

        if not mask_path.exists():
            print(f"Brain mask not found at {mask_path}, skipping brain visualization")
            return None

        print("Loading brain mask...")
        mask = np.load(mask_path)
        print(f"Brain mask loaded successfully, shape: {mask.shape}")

        unique_values = np.unique(mask)
        print(f"Mask unique values: {unique_values}")
        mask_flat = mask.reshape(-1)
        # Treat all non-zero mask entries as active voxels per CLAUDE guidelines
        active_mask_indices = np.flatnonzero(mask_flat > 0)
        n_active_voxels = active_mask_indices.size
        print(f"Number of active voxels in mask: {n_active_voxels:,}")

        print("Preparing data for visualization...")

        target_time_sec = 60
        sampling_rate_hz = 20
        target_frame = int(target_time_sec * sampling_rate_hz)

        if target_frame >= result.shape[0]:
            target_frame = result.shape[0] // 2
            actual_time = target_frame / sampling_rate_hz
            print(f"Warning: 60s exceeds data length, using t={actual_time:.1f}s (frame {target_frame})")
        else:
            print(f"Using t=60s (frame {target_frame}) for slice visualization")

        slice_activity = result[target_frame, :]
        print(f"Slice activity shape: {slice_activity.shape}")

        # Map processed voxels back to full brain volume using mask ordering
        if voxel_indices is None:
            selected_mask_indices = active_mask_indices
        else:
            print("Mapping sampled voxels back to full brain volume...")
            selected_mask_indices = active_mask_indices[np.asarray(voxel_indices)]

        full_brain_activity = np.zeros(mask_flat.size, dtype=slice_activity.dtype)
        full_brain_activity[selected_mask_indices] = slice_activity
        full_brain_volume = full_brain_activity.reshape(mask.shape)

        axial_slice = full_brain_volume[:, :, full_brain_volume.shape[2] // 2]
        sagittal_slice = full_brain_volume[full_brain_volume.shape[0] // 2, :, :]
        coronal_slice = full_brain_volume[:, full_brain_volume.shape[1] // 2, :]

        projection_brain_signal = np.mean(result, axis=0)
        full_projection = np.zeros(mask_flat.size, dtype=projection_brain_signal.dtype)
        full_projection[selected_mask_indices] = projection_brain_signal
        projection_brain_volume = full_projection.reshape(mask.shape)

        baseline_signal = np.mean(baselines, axis=0)
        full_baseline = np.zeros(mask_flat.size, dtype=baseline_signal.dtype)
        full_baseline[selected_mask_indices] = baseline_signal
        baseline_volume = full_baseline.reshape(mask.shape)

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, right=0.82, hspace=0.3, wspace=0.3)
        axes = np.array(
            [
                [
                    fig.add_subplot(gs[0, 0]),
                    fig.add_subplot(gs[0, 1]),
                    fig.add_subplot(gs[0, 2]),
                ],
                [
                    fig.add_subplot(gs[1, 0]),
                    fig.add_subplot(gs[1, 1]),
                    fig.add_subplot(gs[1, 2]),
                ],
            ]
        )

        non_zero_data = slice_activity[slice_activity != 0]
        if len(non_zero_data) > 0:
            slice_vmin, slice_vmax = np.percentile(non_zero_data, [1, 99])
            print(f"  Slice colorbar range: [{slice_vmin:.6f}, {slice_vmax:.6f}]")
        else:
            slice_vmin, slice_vmax = 0, 1
            print("  Slice colorbar range: [0, 1] (no data)")

        im1 = axes[0, 0].imshow(
            axial_slice.T, cmap="hot", vmin=slice_vmin, vmax=slice_vmax, origin="lower"
        )
        axes[0, 0].set_title(
            f"Axial Slice (Z={full_brain_volume.shape[2] // 2})\n"
            f"t=60s - {method.upper()}, W={window_size}, P={percentile}%"
        )
        axes[0, 0].set_xlabel("X")
        axes[0, 0].set_ylabel("Y")

        im2 = axes[0, 1].imshow(
            sagittal_slice.T, cmap="hot", vmin=slice_vmin, vmax=slice_vmax, origin="lower"
        )
        axes[0, 1].set_title(
            f"Sagittal Slice (X={full_brain_volume.shape[0] // 2})\n"
            f"t=60s - {method.upper()}, W={window_size}, P={percentile}%"
        )
        axes[0, 1].set_xlabel("Y")
        axes[0, 1].set_ylabel("Z")

        im3 = axes[0, 2].imshow(
            coronal_slice.T, cmap="hot", vmin=slice_vmin, vmax=slice_vmax, origin="lower"
        )
        axes[0, 2].set_title(
            f"Coronal Slice (Y={full_brain_volume.shape[1] // 2})\n"
            f"t=60s - {method.upper()}, W={window_size}, P={percentile}%"
        )
        axes[0, 2].set_xlabel("X")
        axes[0, 2].set_ylabel("Z")

        axial_proj = np.mean(projection_brain_volume, axis=2)
        sagittal_proj = np.mean(projection_brain_volume, axis=0)
        coronal_proj = np.mean(projection_brain_volume, axis=1)

        proj_data = np.concatenate(
            [
                axial_proj[axial_proj != 0],
                sagittal_proj[sagittal_proj != 0],
                coronal_proj[coronal_proj != 0],
            ]
        )
        if len(proj_data) > 0:
            proj_vmin, proj_vmax = np.percentile(proj_data, [1, 99])
            print(f"  Projection colorbar range: [{proj_vmin:.6f}, {proj_vmax:.6f}]")
        else:
            proj_vmin, proj_vmax = 0, 1
            print("  Projection colorbar range: [0, 1] (no data)")

        im4 = axes[1, 0].imshow(
            axial_proj.T, cmap="hot", vmin=proj_vmin, vmax=proj_vmax, origin="lower"
        )
        axes[1, 0].set_title(
            f"Axial Projection (Z-averaged)\nTime-averaged - {method.upper()}, W={window_size}, P={percentile}%"
        )
        axes[1, 0].set_xlabel("X")
        axes[1, 0].set_ylabel("Y")

        im5 = axes[1, 1].imshow(
            sagittal_proj.T, cmap="hot", vmin=proj_vmin, vmax=proj_vmax, origin="lower"
        )
        axes[1, 1].set_title(
            f"Sagittal Projection (X-averaged)\nTime-averaged - {method.upper()}, W={window_size}, P={percentile}%"
        )
        axes[1, 1].set_xlabel("Y")
        axes[1, 1].set_ylabel("Z")

        im6 = axes[1, 2].imshow(
            coronal_proj.T, cmap="hot", vmin=proj_vmin, vmax=proj_vmax, origin="lower"
        )
        axes[1, 2].set_title(
            f"Coronal Projection (Y-averaged)\nTime-averaged - {method.upper()}, W={window_size}, P={percentile}%"
        )
        axes[1, 2].set_xlabel("X")
        axes[1, 2].set_ylabel("Z")

        cbar_ax1 = fig.add_axes([0.84, 0.55, 0.02, 0.35])
        fig.colorbar(im1, cax=cbar_ax1, label="dF/F0 at t=60s (Slice)")

        cbar_ax2 = fig.add_axes([0.84, 0.10, 0.02, 0.35])
        fig.colorbar(im4, cax=cbar_ax2, label="Mean dF/F0 (Time-averaged)")

        print("Preparing to save brain visualization...")
        output_dir = Path("outputs/images")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory created/verified: {output_dir}")

        brain_filename = (
            f"brain_slices_and_projections_{method}_{voxels_str}_window{window_size}_"
            f"sigma{sigma}_p{percentile}.png"
        )
        brain_path = output_dir / brain_filename
        print(f"Saving brain visualization to: {brain_path}")

        plt.savefig(brain_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Brain visualization saved successfully to: {brain_path}")

        # Baseline maximum projection visualization
        baseline_slice = baseline_volume[:, :, baseline_volume.shape[2] // 2]
        baseline_nonzero = baseline_slice[baseline_slice != 0]
        if baseline_nonzero.size > 0:
            baseline_vmin, baseline_vmax = np.percentile(baseline_nonzero, [1, 99])
            print(
                f"Baseline slice colorbar range: [{baseline_vmin:.6f}, {baseline_vmax:.6f}]"
            )
        else:
            baseline_vmin, baseline_vmax = 0, 1
            print("Baseline slice colorbar range: [0, 1] (no data)")

        baseline_projections = [
            ("Axial Slice (Z mid, t=60s)", baseline_slice.T),
            ("Sagittal Slice (X mid, t=60s)", baseline_volume[baseline_volume.shape[0] // 2, :, :].T),
            ("Coronal Slice (Y mid, t=60s)", baseline_volume[:, baseline_volume.shape[1] // 2, :].T),
            ("Axial Max Projection (over time)", np.max(baseline_volume, axis=2).T),
            ("Sagittal Max Projection (over time)", np.max(baseline_volume, axis=0).T),
            ("Coronal Max Projection (over time)", np.max(baseline_volume, axis=1).T),
        ]

        fig_baseline = plt.figure(figsize=(20, 12))
        gs_baseline = fig_baseline.add_gridspec(2, 3, right=0.9, hspace=0.3, wspace=0.3)
        axes_baseline = np.array(
            [
                [
                    fig_baseline.add_subplot(gs_baseline[0, 0]),
                    fig_baseline.add_subplot(gs_baseline[0, 1]),
                    fig_baseline.add_subplot(gs_baseline[0, 2]),
                ],
                [
                    fig_baseline.add_subplot(gs_baseline[1, 0]),
                    fig_baseline.add_subplot(gs_baseline[1, 1]),
                    fig_baseline.add_subplot(gs_baseline[1, 2]),
                ],
            ]
        )
        for ax, (title, img) in zip(axes_baseline.flat, baseline_projections):
            im = ax.imshow(
                img,
                cmap="hot",
                vmin=baseline_vmin,
                vmax=baseline_vmax,
                origin="lower",
            )
            ax.set_title(f"{title}\n{method.upper()}, W={window_size}, P={percentile}%")
            ax.set_xlabel("Axis 1")
            ax.set_ylabel("Axis 2")

        cbar_ax = fig_baseline.add_axes([0.92, 0.1, 0.02, 0.8])
        cbar = fig_baseline.colorbar(
            im,
            cax=cbar_ax,
            label="Baseline (F0)",
        )
        cbar.ax.tick_params(labelsize=10)
        fig_baseline.canvas.draw()

        baseline_filename = (
            f"baseline_max_projections_{method}_{voxels_str}_window{window_size}_"
            f"sigma{sigma}_p{percentile}.png"
        )
        baseline_path = output_dir / baseline_filename
        plt.savefig(baseline_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Baseline projection visualization saved successfully to: {baseline_path}")
        return brain_path

    except Exception as exc:  # pragma: no cover - visualization failure fallback
        print(f"ERROR: Could not create brain visualization: {exc}")
        import traceback

        traceback.print_exc()
        return None
