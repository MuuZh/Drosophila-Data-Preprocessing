"""Command-line interface for the ΔF/F0 processing pipeline."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .data_io import load_brain_mask, load_data
from .pipeline import three_stage_processing_pipeline
from .visualization import create_brain_visualization, create_visualization


def build_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the ΔF/F0 processor CLI."""
    parser = argparse.ArgumentParser(description="Three-Stage ΔF/F0 Processor")
    parser.add_argument(
        "--method",
        choices=["cpu", "gpu"],
        default="gpu",
        help="ΔF/F0 calculation method (default: gpu)",
    )
    parser.add_argument(
        "--voxels",
        required=True,
        help='Number of voxels to process (integer or "all")',
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Input data file (default: use raw_calcium_data from config.json)",
    )
    parser.add_argument(
        "--output",
        help="Output file (auto-generated if not specified)",
    )

    # Gaussian filtering options
    parser.add_argument(
        "--no-filtering",
        action="store_true",
        help="Disable Gaussian filtering",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="Gaussian filter sigma parameter (default: 0.5)",
    )

    # ΔF/F0 calculation parameters
    parser.add_argument(
        "--window-size",
        type=int,
        default=400,
        help="Rolling window size for baseline calculation (default: 400)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=10.0,
        help="Percentile for baseline calculation (default: 10.0)",
    )

    # Baseline output options
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save baseline (F0) array to disk as a .npy file",
    )
    parser.add_argument(
        "--baseline-output",
        help="Path to the baseline output file (.npy). Implies --save-baseline if provided.",
    )

    # Output options
    parser.add_argument(
        "--no-save-files",
        dest="save_files",
        action="store_false",
        help="Skip saving HDF5 result files",
    )
    parser.add_argument(
        "--no-save-images",
        dest="save_images",
        action="store_false",
        help="Skip generating visualization images",
    )
    parser.add_argument(
        "--save-stats",
        action="store_true",
        help="Save processing statistics",
    )
    parser.add_argument(
        "--save-brain-viz",
        action="store_true",
        help="Generate brain visualization (requires brain mask)",
    )

    parser.set_defaults(save_files=True, save_images=True)
    return parser


def _convert_numpy_types(obj):
    """Convert numpy types to native Python types for serialization."""
    if isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run(args: argparse.Namespace) -> int:
    """Execute the CLI workflow based on parsed arguments."""
    baseline_requested = args.save_baseline or args.baseline_output is not None

    try:
        print("Configuration:")
        print(f"  Save files: {'Yes' if args.save_files else 'No'}")
        print(f"  Save images: {'Yes' if args.save_images else 'No'}")
        print(f"  Save stats: {'Yes' if args.save_stats else 'No'}")
        print(f"  Brain visualization: {'Yes' if args.save_brain_viz else 'No'}")
        print(f"  Save baseline: {'Yes' if baseline_requested else 'No'}")
        print()

        print("Loading calcium imaging data...")
        data, voxel_indices, data_path = load_data(args.data_file, args.voxels)
        print(f"Loaded data shape: {data.shape}")
        print(f"Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
        print(f"Data source: {data_path}")

        deltaf_result, baselines, stats = three_stage_processing_pipeline(
            data,
            deltaf_method=args.method,
            gaussian_filtering=not args.no_filtering,
            sigma=args.sigma,
            window_size=args.window_size,
            percentile=args.percentile,
        )

        subset_run = str(args.voxels).lower() != "all"
        subset_mask_path = _resolve_subset_mask_path(args) if subset_run else None

        if args.save_files:
            output_path = _resolve_output_path(args)
            baseline_path = _resolve_baseline_path(args) if baseline_requested else None
            _save_hdf5_results(
                output_path,
                deltaf_result,
                voxel_indices,
                args,
                stats,
                baseline_path,
                data_path,
                subset_mask_path,
            )
        else:
            output_path = None
            baseline_path = _resolve_baseline_path(args) if baseline_requested else None
            print("Skipping file save (--no-save-files specified)")

        if baseline_requested:
            baseline_path = baseline_path or _resolve_baseline_path(args)
            _save_baseline(baseline_path, baselines)

        if subset_run and subset_mask_path is not None:
            _save_subset_mask(subset_mask_path, voxel_indices)

        if args.save_stats:
            stats_path = _resolve_stats_path(output_path, args.voxels)
            _save_stats(stats_path, stats)

        if args.save_images:
            print("\nGenerating visualizations...")
            voxels_str = args.voxels if args.voxels == "all" else f"{args.voxels}voxels"
            create_visualization(
                deltaf_result,
                args.method,
                voxels_str,
                window_size=args.window_size,
                sigma=args.sigma,
                percentile=args.percentile,
            )

            if args.save_brain_viz:
                create_brain_visualization(
                    deltaf_result,
                    baselines,
                    args.method,
                    voxels_str,
                    window_size=args.window_size,
                    sigma=args.sigma,
                    voxel_indices=voxel_indices,
                    percentile=args.percentile,
                )
        else:
            print("Skipping image generation (--no-save-images specified)")

        print("\n" + "=" * 60)
        print("THREE-STAGE PROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        return 0

    except Exception as exc:  # pragma: no cover - top-level failure fallback
        print(f"\nProcessing failed: {exc}")
        import traceback

        traceback.print_exc()
        return 1


def _resolve_output_path(args: argparse.Namespace) -> Path:
    """Determine the output file path for the HDF5 results."""
    if args.output:
        return Path(args.output)

    voxel_str = args.voxels if args.voxels == "all" else f"{args.voxels}voxels"
    filter_str = "gauss" if not args.no_filtering else "nofilter"
    filename = (
        f"outputs/deltaf_threestage_{voxel_str}_{args.method}_{filter_str}_"
        f"sigma{args.sigma}_window{args.window_size}_p{args.percentile}.h5"
    )
    return Path(filename)


def _resolve_baseline_path(args: argparse.Namespace) -> Path:
    """Determine the baseline output path."""
    if args.baseline_output:
        return Path(args.baseline_output)

    voxel_str = args.voxels if args.voxels == "all" else f"{args.voxels}voxels"
    filter_str = "gauss" if not args.no_filtering else "nofilter"
    filename = (
        f"outputs/baseline/deltaf_baseline_{voxel_str}_{args.method}_{filter_str}_"
        f"sigma{args.sigma}_window{args.window_size}_p{args.percentile}.npy"
    )
    return Path(filename)


def _resolve_subset_mask_path(args: argparse.Namespace) -> Path:
    """Determine the output path for subset 3D masks."""
    voxel_str = args.voxels if str(args.voxels).lower() == "all" else f"{args.voxels}voxels"
    filter_str = "gauss" if not args.no_filtering else "nofilter"
    filename = (
        f"outputs/masks/subset_mask_{voxel_str}_{args.method}_{filter_str}_"
        f"sigma{args.sigma}_window{args.window_size}_p{args.percentile}.npy"
    )
    return Path(filename)


def _save_baseline(path: Path, baselines: np.ndarray) -> None:
    """Persist the baseline array to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, baselines.astype(np.float32))
    print(f"Baseline saved to: {path}")


def _save_hdf5_results(
    output_path: Path,
    deltaf_result: np.ndarray,
    voxel_indices,
    args: argparse.Namespace,
    stats: dict,
    baseline_path: Path | None,
    data_path: Path,
    subset_mask_path: Path | None,
) -> None:
    """Persist ΔF/F0 results and metadata to an HDF5 file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import h5py

    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("deltaf_f0", data=deltaf_result, compression="gzip")

        if voxel_indices is not None:
            handle.create_dataset("voxel_indices", data=voxel_indices, compression="gzip")
            handle.attrs["voxel_sampling"] = "random_subset"
            handle.attrs["total_voxels_available"] = len(voxel_indices)
            print(f"  Voxel indices saved: {len(voxel_indices):,} indices")
        else:
            handle.attrs["voxel_sampling"] = "all_voxels"
            handle.attrs["total_voxels_available"] = deltaf_result.shape[1]
            print(f"  Used all voxels: {deltaf_result.shape[1]:,}")

        handle.attrs["original_data_file"] = str(data_path)
        handle.attrs["original_data_shape"] = (
            f"2400x{handle.attrs['total_voxels_available']}"
        )
        handle.attrs["method"] = args.method
        handle.attrs["voxels"] = args.voxels
        handle.attrs["gaussian_filtering"] = not args.no_filtering
        handle.attrs["gaussian_sigma"] = args.sigma
        handle.attrs["processing_time"] = stats["processing_time"]["total"]

        if baseline_path is not None:
            handle.attrs["baseline_file"] = str(baseline_path)
        if subset_mask_path is not None:
            handle.attrs["subset_mask_file"] = str(subset_mask_path)

        for key, value in stats["data_quality"].items():
            handle.attrs[f"quality_{key}"] = value

    print(f"\nResults saved to: {output_path}")


def _resolve_stats_path(output_path: Path | None, voxels: str) -> Path:
    """Determine the statistics output path."""
    if output_path:
        return Path(str(output_path).replace(".h5", "_stats.json"))

    os.makedirs("outputs/analysis", exist_ok=True)
    filename = f"outputs/analysis/two_stage_stats_{voxels}_voxels_{int(time.time())}.json"
    return Path(filename)


def _save_stats(stats_path: Path, stats: dict) -> None:
    """Persist processing statistics to JSON."""
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w") as handle:
        json.dump(_convert_numpy_types(stats), handle, indent=2)
    print(f"Statistics saved to: {stats_path}")


def _save_subset_mask(mask_path: Path, voxel_indices) -> None:
    """Persist a 3D subset mask aligned with the brain mask."""
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    brain_mask, _ = load_brain_mask()
    mask_flat = brain_mask.reshape(-1)
    active_indices = np.flatnonzero(mask_flat == 1)

    subset_indices = active_indices[np.asarray(voxel_indices)]
    subset_mask_flat = np.zeros_like(mask_flat, dtype=np.uint8)
    subset_mask_flat[subset_indices] = 1

    subset_mask = subset_mask_flat.reshape(brain_mask.shape)
    np.save(mask_path, subset_mask)
    print(f"Subset 3D mask saved to: {mask_path}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)
