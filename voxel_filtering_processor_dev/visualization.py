"""Tri-view visualization helpers shared across development pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize


TriviewMode = Literal["slice", "max_projection", "mean_projection"]


@dataclass(frozen=True)
class TriviewConfig:
    """Configuration for tri-view extraction."""

    mode: TriviewMode = "slice"
    slice_indices: tuple[int | None, int | None, int | None] | None = None
    projection: Literal["max", "mean"] = "max"


def _extract_plane(
    volume: np.ndarray,
    axis: int,
    config: TriviewConfig,
) -> np.ndarray:
    """Extract 2D plane from 3D volume according to configuration."""
    if config.mode == "slice":
        if config.slice_indices is None:
            index = volume.shape[axis] // 2
        else:
            index = config.slice_indices[axis] or (volume.shape[axis] // 2)
        return np.take(volume, index, axis=axis)

    if config.mode == "max_projection":
        return np.max(volume, axis=axis)

    if config.mode == "mean_projection":
        return np.mean(volume, axis=axis)

    raise ValueError(f"Unsupported tri-view mode: {config.mode}")


def _extract_mask_plane(
    mask_volume: np.ndarray | None,
    axis: int,
    config: TriviewConfig,
) -> np.ndarray | None:
    """Prepare overlay mask plane matching the data extraction."""
    if mask_volume is None:
        return None

    if config.mode == "slice":
        if config.slice_indices is None:
            index = mask_volume.shape[axis] // 2
        else:
            index = config.slice_indices[axis] or (mask_volume.shape[axis] // 2)
        return np.take(mask_volume, index, axis=axis)

    # For projections use logical OR to capture all occupied voxels.
    return np.any(mask_volume, axis=axis)


def _style_imshow(ax: plt.Axes, data: np.ndarray, cmap: str, vmin: float | None, vmax: float | None) -> plt.AxesImage:
    """Render data with consistent orientation."""
    return ax.imshow(
        data.T,
        cmap=cmap,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )


def _add_mask_outline(ax: plt.Axes, mask_plane: np.ndarray, color: str) -> None:
    """Overlay mask outline using contour rendering."""
    if mask_plane.dtype != np.bool_:
        mask_plane = mask_plane > 0
    if not np.any(mask_plane):
        return
    ax.contour(
        mask_plane.T.astype(float),
        levels=[0.5],
        colors=[color],
        linewidths=0.8,
    )


def _panel_titles(mode: TriviewMode) -> tuple[str, str, str]:
    """Return axis labels for tri-view panels."""
    if mode == "slice":
        return ("Axial Slice (Z mid)", "Sagittal Slice (X mid)", "Coronal Slice (Y mid)")
    if mode == "max_projection":
        return ("Axial Max Projection", "Sagittal Max Projection", "Coronal Max Projection")
    return ("Axial Mean Projection", "Sagittal Mean Projection", "Coronal Mean Projection")


def save_triview(
    volume: np.ndarray,
    output_path: Path,
    title_prefix: str,
    config: TriviewConfig,
    overlay_mask: np.ndarray | None = None,
    overlay_color: str = "cyan",
    cmap: str = "hot",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
) -> None:
    """Save tri-view figure with optional mask overlay."""
    if volume.ndim != 3:
        raise ValueError(f"Tri-view expects 3D volume, received shape {volume.shape}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = _panel_titles(config.mode)

    for axis_idx, ax in enumerate(axes):
        plane = _extract_plane(volume, axis=axis_idx, config=config)
        img = _style_imshow(ax, plane, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{titles[axis_idx]}\n{title_prefix}")
        ax.set_xlabel(("X", "Y", "X")[axis_idx])
        ax.set_ylabel(("Y", "Z", "Z")[axis_idx])

        mask_plane = _extract_mask_plane(overlay_mask, axis=axis_idx, config=config)
        if mask_plane is not None:
            _add_mask_outline(ax, mask_plane, overlay_color)

    if colorbar_label:
        fig.tight_layout(rect=[0.0, 0.0, 0.9, 1.0])
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(img, cax=cbar_ax)
        cbar.set_label(colorbar_label)
    else:
        fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_region_triview(
    region_volume: np.ndarray,
    output_path: Path,
    title_prefix: str,
    config: TriviewConfig,
    cmap: ListedColormap | None = None,
    region_ticks: Sequence[int] | None = None,
    colorbar_label: str = "Brain Region ID",
) -> None:
    """Save tri-view showcasing labeled region mask."""
    if region_volume.ndim != 3:
        raise ValueError(f"Expected 3D region mask volume, got shape {region_volume.shape}")

    cmap = cmap or ListedColormap(plt.cm.get_cmap("tab20", 20).colors)
    norm = Normalize(vmin=region_volume.min(), vmax=region_volume.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = _panel_titles(config.mode)

    for axis_idx, ax in enumerate(axes):
        plane = _extract_plane(region_volume, axis=axis_idx, config=config)
        img = ax.imshow(
            plane.T,
            cmap=cmap,
            origin="lower",
            norm=norm,
            aspect="equal",
        )
        ax.set_title(f"{titles[axis_idx]}\n{title_prefix}")
        ax.set_xlabel(("X", "Y", "X")[axis_idx])
        ax.set_ylabel(("Y", "Z", "Z")[axis_idx])

    fig.tight_layout(rect=[0.0, 0.0, 0.9, 1.0])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.set_label(colorbar_label)
    if region_ticks:
        cbar.set_ticks(region_ticks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
