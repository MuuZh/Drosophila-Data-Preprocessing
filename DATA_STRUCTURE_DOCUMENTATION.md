# Drosophila Brain Calcium Imaging Dataset - Data Structure Documentation

## Overview

This document provides comprehensive documentation for the Drosophila brain calcium imaging dataset structure, loading methods, and usage patterns. The dataset contains:

- **Brain Activity Data**: Motion-corrected calcium imaging at 20Hz (2400 timepoints, 2,120,420 voxels)
- **Behavioral Data**: Movement velocities and auditory stimulus information at 50Hz
- **Brain Region Labels**: Anatomical region segmentation (47 regions, 0-46 labels)
- **Reference Templates**: JRC2018 fly brain template for spatial registration

---

## Table of Contents

1. [Core Dataset Files](#1-core-dataset-files)
   - [1.1 Raw Calcium Imaging Data](#11-raw-calcium-imaging-data)
   - [1.2 Brain Mask (Binary Mask)](#12-brain-mask-binary-mask)
   - [1.3 Behavioral Data](#13-behavioral-data)
   - [1.4 Preprocessed Data (ΔF/F0)](#14-preprocessed-data-δff0)
2. [Brain Region Label Files (YiSun Lab)](#2-brain-region-label-files-yisun-lab)
   - [2.1 Brain Region Mask](#21-brain-region-mask)
   - [2.2 Label Index (Region Names)](#22-label-index-region-names)
   - [2.3 Template Volume (NRRD)](#23-template-volume-nrrd)
   - [2.4 Label Volume (NRRD)](#24-label-volume-nrrd)
3. [Data Processing Pipelines](#3-data-processing-pipelines)
   - [3.1 Pipeline 1: ΔF/F0 Calculation](#31-pipeline-1-δff0-calculation-deltaf_processor)
   - [3.2 Pipeline 2: Quality Control Filtering](#32-pipeline-2-quality-control-filtering-voxel_filtering_processor_dev)
   - [3.3 Complete Workflow Example](#33-complete-workflow-example)
4. [File Relationships](#4-file-relationships)
5. [Troubleshooting](#5-troubleshooting)
6. [External Resources](#6-external-resources)
7. [Summary](#7-summary)

---

## 1. Core Dataset Files

### 1.1 Raw Calcium Imaging Data

**File**: `chat_motioncorrected_voxel.npy`

**Properties**:

- **Dimensions**: (2400, 2,120,420)
  - 2400 timepoints at 20Hz sampling rate = 120 seconds total duration
  - 2,120,420 active voxels
- **Size**: 18.96 GB
- **Data Type**: Motion-corrected fluorescence intensity
- **Format**: NumPy binary array

**Loading Example**:

```python
import numpy as np

# Memory-mapped loading (recommended for large files)
raw_data = np.load('chat_motioncorrected_voxel.npy', mmap_mode='r')
print(f"Shape: {raw_data.shape}")  # (2400, 2120420)

# Load specific timepoints
frame_100 = raw_data[100, :]  # Single frame
time_window = raw_data[100:200, :]  # 100 frames
```

---

### 1.2 Brain Mask (Binary Mask)

**File**: `brain_voxel_mask.npy`

**Properties**:

- **Dimensions**: (314, 147, 87)
- **Data Type**: Binary (0 or 1)
- **Active Voxels**: Sum of 1s = 2,120,420 (matches data column count)
- **Purpose**: Maps 2D data array (2,120,420 voxels) to 3D brain structure

**Voxel Index Mapping**

The voxel indices in the 2D data array correspond to the positions where `mask==1` in the 3D volume.

**Complete Mapping Example**:

```python
import numpy as np

# Load brain mask
mask = np.load('brain_voxel_mask.npy')  # Shape: (314, 147, 87)
print(f"Active voxels: {np.sum(mask == 1):,}")  # Should be 2,120,420

# Function to map 2D data back to 3D brain volume
def map_to_3d_brain(data_2d, mask):
    """
    Map 2D voxel data back to 3D brain volume

    Parameters:
    -----------
    data_2d : array (voxels,) or (timepoints, voxels)
        Data in 2D format with voxels corresponding to mask==1 positions
    mask : array (x, y, z)
        Brain mask with 1s for active voxels, 0s elsewhere

    Returns:
    --------
    volume : array (x, y, z) or (x, y, z, timepoints)
        3D brain volume with data mapped to spatial positions
    """
    if data_2d.ndim == 1:
        # Single timepoint
        volume = np.zeros(mask.shape, dtype=data_2d.dtype)
        volume[mask == 1] = data_2d
        return volume
    else:
        # Multiple timepoints
        n_timepoints = data_2d.shape[0]
        volume = np.zeros(mask.shape + (n_timepoints,), dtype=data_2d.dtype)
        volume[mask == 1, :] = data_2d.T
        return volume

# Example: Map single frame to 3D
raw_data = np.load('chat_motioncorrected_voxel.npy', mmap_mode='r')
frame_100 = raw_data[100, :]  # Shape: (2120420,)
volume_100 = map_to_3d_brain(frame_100, mask)  # Shape: (314, 147, 87)

# Create orthogonal slices for visualization
z_center = mask.shape[2] // 2  # Axial slice (Z=43)
x_center = mask.shape[0] // 2  # Sagittal slice (X=157)
y_center = mask.shape[1] // 2  # Coronal slice (Y=73)

axial_slice = volume_100[:, :, z_center]      # Top-down view
sagittal_slice = volume_100[x_center, :, :]   # Side view
coronal_slice = volume_100[:, y_center, :]    # Front view

# Visualize slices
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(axial_slice.T, cmap='hot', origin='lower')
axes[0].set_title(f'Axial (Z={z_center})')

axes[1].imshow(sagittal_slice.T, cmap='hot', origin='lower')
axes[1].set_title(f'Sagittal (X={x_center})')

axes[2].imshow(coronal_slice.T, cmap='hot', origin='lower')
axes[2].set_title(f'Coronal (Y={y_center})')

plt.tight_layout()
plt.savefig('brain_slices.png', dpi=200)
plt.close()
```

---

### 1.3 Behavioral Data

**File**: `Stim_Loc_Rec.pkl`

**Properties**:

- **Format**: Python pickle (dictionary)
- **Sampling Rate**: 50 Hz (behavioral), 20 Hz (brain)
- **Duration**: 120 seconds

**Dictionary Structure**:

| Key                        | Dimensions | Unit          | Sampling Rate | Description                     |
| -------------------------- | ---------- | ------------- | ------------- | ------------------------------- |
| `forward_vel_raw`          | (6000,)    | mm/s          | 50 Hz         | Forward movement velocity       |
| `side_vel_raw`             | (6000,)    | mm/s          | 50 Hz         | Lateral movement velocity       |
| `angular_vel_raw`          | (6000,)    | rad/s         | 50 Hz         | Angular movement velocity       |
| `auditory_stim_timepoints` | (10,)      | **seconds**   | N/A           | Stimulus onset times in seconds |
| `auditory_stim_type`       | (10,)      | type ID (1-5) | N/A           | Stimulus type for each stimulus |

**⚠️ IMPORTANT NOTES**:

- **Stimulus timepoints are stored in SECONDS**, not frames
- To convert to brain frames (20Hz): `frame = timepoint_seconds * 20`
- To convert to behavioral frames (50Hz): `frame = timepoint_seconds * 50`
- **Stimulus types**:
  - 1 = Sine wave (1s duration)
  - 2 = Pulse train (1s duration)
  - 3 = White noise (1s duration)
  - 4 = Sine + Pulse (2s duration, 1s each)
  - 5 = Pulse + Sine (2s duration, 1s each)

**Complete Loading Example**:

```python
import pickle
import numpy as np

# Load behavioral and stimulus data
with open('Stim_Loc_Rec.pkl', 'rb') as f:
    behavioral_data = pickle.load(f)

print("Available keys:", list(behavioral_data.keys()))

# Extract movement velocities (6000 samples, 50Hz, 120 seconds)
forward_vel_raw = behavioral_data['forward_vel_raw']  # Shape: (6000,), Unit: mm/s
side_vel_raw = behavioral_data['side_vel_raw']        # Shape: (6000,), Unit: mm/s
angular_vel_raw = behavioral_data['angular_vel_raw']  # Shape: (6000,), Unit: rad/s

# Extract stimulus information
auditory_stim_timepoints = behavioral_data['auditory_stim_timepoints']  # Shape: (10,), Unit: seconds
auditory_stim_type = behavioral_data['auditory_stim_type']              # Shape: (10,), Values: 1-5

print(f"Behavioral data length: {len(forward_vel_raw)} points (50Hz)")
print(f"Duration: {len(forward_vel_raw)/50:.1f} seconds")
print(f"Number of stimuli: {len(auditory_stim_timepoints)}")
print(f"Stimulus types: {np.unique(auditory_stim_type)}")

# Convert stimulus timepoints to frame indices
brain_sampling_rate = 20  # Hz
behavioral_sampling_rate = 50  # Hz

stim_frames_brain = (auditory_stim_timepoints * brain_sampling_rate).astype(int)
stim_frames_behavioral = (auditory_stim_timepoints * behavioral_sampling_rate).astype(int)

print(f"\nExample: Stimulus 1 at {auditory_stim_timepoints[0]:.1f}s")
print(f"  -> Brain frame (20Hz): {stim_frames_brain[0]}")
print(f"  -> Behavioral frame (50Hz): {stim_frames_behavioral[0]}")
```

---

### 1.4 Preprocessed Data (ΔF/F0)

**Properties**:

- **Format**: HDF5 with GZIP compression
- **Dataset Name**: `'deltaf_f0'`
- **Dimensions**: (2400, 2,120,420)
- **Preprocessing Method**: Rolling percentile baseline ΔF/F0 **only** (no additional detrending)
- **Formula**: ΔF/F0 = (F - F0) / F0, where F0 = 10th percentile of rolling window
- **Window Size**: 400 timepoints (~20 seconds at 20Hz)
- **Quality**: CPU vs GPU correlation >0.999

**Loading Example**:

```python
import h5py
import numpy as np

# Load complete preprocessed data (memory-mapped for efficiency)
with h5py.File('deltaf_results_allvoxels.h5', 'r') as f:
    print("Available datasets:", list(f.keys()))

    # Memory-mapped access (does not load entire array into RAM)
    deltaf_data = f['deltaf_f0']
    print(f"Shape: {deltaf_data.shape}")  # (2400, 2120420)

    # Load specific time window
    time_window = deltaf_data[100:200, :]  # 100 timepoints, all voxels

    # Load specific voxel subset
    voxel_subset = deltaf_data[:, :10000]  # All time, first 10k voxels

    # Load single frame for visualization
    frame_100 = deltaf_data[100, :]  # Shape: (2120420,)

# If you need to load the entire dataset into memory (requires ~38GB RAM)
with h5py.File('deltaf_results_allvoxels.h5', 'r') as f:
    deltaf_all = f['deltaf_f0'][:]  # Full array in RAM
```

---

---

## 2. Brain Region Label Files (YiSun Lab)

These files provide anatomical region segmentation aligned with the JRC2018 fly brain template.

**⚠️ IMPORTANT NOTE**: The optical lobe part of the fly whole brain (label IDs 1-4) was **not imaged** in this dataset. Only the central brain was kept.

---

### 2.1 Brain Region Mask

**File**: `brain_region_mask.npy`

**Properties**:

- **Dimensions**: (314, 147, 87) - Same as `brain_voxel_mask.npy`
- **Data Type**: `uint8`
- **Value Range**: 0-46
  - 0 = background (non-brain)
  - 1-46 = anatomical region identifiers
- **Alignment**: Matches YiSun Lab template coordinate system

**Usage Example**:

```python
import numpy as np

# Load region mask
region_mask = np.load('brain_region_mask.npy')

print(f"Region mask shape: {region_mask.shape}")
print(f"Value range: {region_mask.min()}-{region_mask.max()}")
print(f"Number of regions: {len(np.unique(region_mask)) - 1}")  # Exclude 0

# Create binary brain mask (matches functional voxel ordering)
binary_mask = (region_mask > 0).astype(np.uint8)

# Extract specific region (e.g., region 10)
region_10_voxels = (region_mask == 10)
print(f"Region 10 voxel count: {np.sum(region_10_voxels):,}")

# Count voxels per region
unique_labels, counts = np.unique(region_mask, return_counts=True)
print("\nVoxel counts per region:")
for label, count in zip(unique_labels, counts):
    if label > 0:  # Skip background
        print(f"  Region {label:2d}: {count:7,} voxels")
```

### 2.2 Label Index (Region Names)

**File**: `fly brain lable mask file (YiSun Lab)/label_index.pkl`

**Properties**:

- **Format**: pandas DataFrame (pickled)
- **Dimensions**: 47 rows × 2 columns
- **Columns**:
  - `'id'`: Region identifier (0-46, matches `brain_region_mask.npy` values)
  - `'name'`: Human-readable region name

**⚠️ CRITICAL: Pandas Pickle Compatibility Issue**

The pickle file was created with an older pandas version that uses `slice` objects for internal block placements. Modern pandas expects `BlockPlacement` objects. so we apply a compatibility patch before loading.

**Complete Loading Pattern with Compatibility Patch**:

```python
import os
import numpy as np
import pandas as pd
from pandas._libs.internals import BlockPlacement
from pandas.core.internals import blocks

def _patch_pandas_pickle_compatibility():

    original_new_block = blocks.new_block

    def compat_new_block(*args, **kwargs):
        # Handle kwargs
        if "placement" in kwargs and isinstance(kwargs["placement"], slice):
            placement_slice = kwargs["placement"]
            kwargs["placement"] = BlockPlacement(
                range(placement_slice.start or 0, placement_slice.stop)
            )
        else:
            # Handle args
            args = list(args)
            if len(args) > 1 and isinstance(args[1], slice):
                placement_slice = args[1]
                args[1] = BlockPlacement(
                    range(placement_slice.start or 0, placement_slice.stop)
                )
            args = tuple(args)
        return original_new_block(*args, **kwargs)

    blocks.new_block = compat_new_block

_patch_pandas_pickle_compatibility()

# Load label index

region_mask = np.load('brain_region_mask.npy')
label_df = pd.read_pickle('label_index.pkl').sort_values('id')

# Create lookup dictionary for easy access
label_lookup = dict(zip(label_df['id'], label_df['name']))

# Display information
print(f"Region mask shape: {region_mask.shape}")
print(f"Label range: {region_mask.min()}-{region_mask.max()}")
print(f"Number of regions: {label_df.shape[0]}")

print("\nLabel Index Table:")
print(label_df.head(10))

print("\nExample lookups:")
print(f"Region 10: {label_lookup.get(10, 'Unknown')}")
print(f"Region 20: {label_lookup.get(20, 'Unknown')}")

# Export to CSV for easier viewing
label_df.to_csv('label_index.csv', index=False)
print("\nLabel index exported to: label_index.csv")
```

---

### 2.3 Template Volume (NRRD)

**File**: `template.nrrd`

**Properties**:

- **Format**: NRRD (Nearly Raw Raster Data) with gzip compression
- **Dimensions**: (314, 147, 87)
- **Data Type**: `uint8`
- **Purpose**: JRC2018 fly brain template used for registration and spatial alignment
- **Visualization**: Can be viewed with ITK-SNAP, 3D-Slicer, or loaded with `pynrrd` package

---

### 2.4 Label Volume (NRRD)

**File**: `label.nrrd`

**Properties**:

- **Format**: NRRD (Nearly Raw Raster Data) with gzip compression
- **Dimensions**: (314, 147, 87)
- **Data Type**: `uint8`
- **Value Range**: 0-46 (same as `brain_region_mask.npy`)
- **Purpose**: Region label ID for each voxel in NRRD format (for compatibility with fMRI/neuroimaging software)
- **Visualization**: Can be viewed with ITK-SNAP, 3D-Slicer, or loaded with `pynrrd` package

**NRRD Loading Function**:

You can load NRRD files using the `pynrrd` package or implement your own parser:

```python
# Method 1: Using pynrrd package (recommended)
# Installation: pip install pynrrd
import nrrd

template, template_header = nrrd.read('template.nrrd')
label_volume, label_header = nrrd.read('label.nrrd')

print(f"Template shape: {template.shape}")
print(f"Label volume shape: {label_volume.shape}")
print(f"Header info: {template_header}")
```

```python
# Method 2: Custom NRRD parser (no external dependencies)
import gzip
import numpy as np
from typing import Tuple

def load_gzip_nrrd(path: str) -> Tuple[np.ndarray, dict]:
    """
    Parse a minimal NRRD header and load the attached gzip data block.

    Parameters:
    -----------
    path : str
        Path to NRRD file

    Returns:
    --------
    array : np.ndarray
        3D volume data
    header : dict
        NRRD header metadata (sizes, type, encoding, etc.)
    """
    with open(path, "rb") as file_obj:
            header_lines = []
            while True:
                line = file_obj.readline()
                if not line or line.strip() == b"":
                    break
                header_lines.append(line.decode("ascii").strip())
            header = {}
            for line in header_lines:
                if line.startswith("#") or line.startswith("NRRD"):
                    continue
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                header[key.strip()] = value.strip()
            sizes = tuple(int(val) for val in header["sizes"].split())
            dtype_map = {
                "uchar": np.uint8, "uint8": np.uint8,
                "short": np.int16, "ushort": np.uint16,
                "int": np.int32, "uint": np.uint32,
                "float": np.float32, "double": np.float64,
            }
            dtype = dtype_map[header["type"]]
            if header.get("encoding", "raw") != "gzip":
                raise ValueError(f"Unsupported encoding {header.get('encoding')}")
            gz = gzip.GzipFile(fileobj=file_obj)
            data = gz.read()
            array = np.frombuffer(data, dtype=dtype).reshape(sizes, order="F")
            return array, header

# Load template and label volumes
template, template_header = load_gzip_nrrd('template.nrrd')
label_volume, label_header = load_gzip_nrrd('label.nrrd')

print(f"Template shape: {template.shape}, dtype: {template.dtype}")
print(f"Label volume shape: {label_volume.shape}, dtype: {label_volume.dtype}")
print(f"Header sizes: {template_header['sizes']}")

```

---

## 3. Data Processing Pipelines

This section documents the processing pipelines used to transform raw calcium imaging data into analysis-ready ΔF/F0 signals with quality control filtering.

### 3.1 Pipeline 1: ΔF/F0 Calculation (`deltaf_processor`)

**Purpose**: Calculate ΔF/F0 (delta F over F0) from raw motion-corrected calcium imaging data using rolling percentile baseline estimation with optional Gaussian smoothing.

#### Configuration File

The pipeline reads data paths from `config.json`. Here's the required structure:

```json
{
  "data_paths": {
    "raw_calcium_data": "chat_motioncorrected_voxel.npy",
    "brain_mask": "brain_voxel_mask.npy"
  }
}
```

**Configuration Keys:**

| Key | Description | Required |
|-----|-------------|----------|
| `raw_calcium_data` | Path to raw motion-corrected calcium imaging data | Yes (unless using `--data-file`) |
| `brain_mask` | Path to 3D brain mask for visualization | No (only for `--save-brain-viz`) |

**Note**: You can override the data file path using the `--data-file` command-line argument.

#### Input Files

| File                  | Format | Description                                    | Configuration                                      |
| --------------------- | ------ | ---------------------------------------------- | -------------------------------------------------- |
| Raw calcium data      | `.npy` | Motion-corrected fluorescence (2400, N voxels) | `config.json`: `raw_calcium_data` or `--data-file` |
| Brain mask (optional) | `.npy` | 3D brain mask (314×147×87)                     | For visualization with `--save-brain-viz`          |

#### Processing Algorithm

**Stage 1: Gaussian Filtering (Optional)**

- **Purpose**: Temporal noise reduction while preserving signal dynamics
- **Method**: 1D Gaussian filter along time axis
- **Algorithm**: `scipy.ndimage.gaussian_filter1d`
- **Parameter**: `sigma` (default: 0.5, conservative value)
- **Skip**: Use `--no-filtering` flag

**Stage 2: Rolling Percentile Baseline**

```python
# For each timepoint t:
window_data = fluorescence[t - half_window : t + half_window + 1, :]
baseline[t, :] = percentile(window_data, percentile_value, axis=0)

# Edge handling (first/last ~200 frames):
# Missing samples padded with global baseline (percentile across entire recording)
```

**Stage 3: ΔF/F0 Calculation**

```python
deltaf_f0 = (F - F0) / F0

# Where:
# F  = fluorescence signal at time t
# F0 = rolling baseline at time t
# Division by zero handled → replaced with 0
```

**Key Parameters:**

- `window_size`: Rolling window size (default: 400 frames = 20 seconds at 20Hz)
- `percentile`: Baseline percentile (default: 10th percentile)
- `sigma`: Gaussian filter width (default: 0.5)
- `method`: 'cpu' or 'gpu' (default: 'gpu' for speed)

**GPU Processing (Taichi):**

- Automatic memory-aware chunking
- ~80% GPU memory utilization
- Bisection search for percentile (20 iterations)
- 10-100× faster than CPU for large datasets

#### Output Files

**1. ΔF/F0 HDF5 Package** (Primary output)

```
outputs/deltaf_threestage_{voxels}_{method}_{filter}_sigma{sigma}_window{window}_p{percentile}.h5
```

**HDF5 Structure:**

```python
import h5py

with h5py.File(output_h5, 'r') as f:
    # Datasets
    deltaf_f0 = f['deltaf_f0'][:]        # Shape: (2400, n_voxels), dtype: float32
    voxel_indices = f['voxel_indices'][:] # Shape: (n_voxels,), dtype: int64 (if subset)
    keep_mask = f['keep_mask'][:]         # Boolean mask (if filtered)

    # Attributes (metadata)
    print(f"Method: {f.attrs['method']}")
    print(f"Voxels: {f.attrs['voxels']}")
    print(f"Gaussian filtering: {f.attrs['gaussian_filtering']}")
    print(f"Processing time: {f.attrs['processing_time']:.2f}s")
    print(f"Original shape: {f.attrs['original_data_shape']}")
    print(f"Quality - NaN count: {f.attrs.get('quality_nan_count', 0)}")
    print(f"Quality - Mean: {f.attrs.get('quality_mean', 0):.4f}")
```

**2. Baseline Time Series** (Use `--save-baseline`)

```
outputs/baseline/deltaf_baseline_{voxels}_{method}_{filter}_sigma{sigma}_window{window}_p{percentile}.npy
```

**Critical**: This baseline file is **required** for the quality filtering pipeline (Stage 2: baseline threshold filter).

**3. Subset Mask** (Automatic for subset runs)

```
outputs/masks/subset_mask_{voxels}_{method}_{filter}_sigma{sigma}_window{window}_p{percentile}.npy
```

Binary 3D mask showing which voxels were processed in subset mode.

**4. Visualizations** (Use `--save-brain-viz`)

```
outputs/images/brain_overview_deltaf_{method}_{voxels}_*.png
outputs/images/brain_overview_baseline_{method}_{voxels}_*.png
```

2×3 grid showing axial/sagittal/coronal views for ΔF/F0 and baseline.

#### Usage Examples

**Basic Usage (GPU, all voxels)**

```bash
conda run -n base python -m deltaf_processor \
  --method gpu \
  --voxels all \
  --window-size 400 \
  --percentile 10 \
  --sigma 0.5 \
  --save-baseline \
  --save-brain-viz
```

**CPU Processing (subset for testing)**

```bash
conda run -n base python -m deltaf_processor \
  --method cpu \
  --voxels 100000 \
  --window-size 400 \
  --percentile 10 \
  --no-filtering
```

**Custom Data File**

```bash
conda run -n base python -m deltaf_processor \
  --method gpu \
  --voxels all \
  --data-file /path/to/custom_data.npy \
  --output /path/to/output.h5 \
  --baseline-output /path/to/baseline.npy
```

#### Command-Line Arguments

| Argument            | Type    | Default      | Description                                      |
| ------------------- | ------- | ------------ | ------------------------------------------------ |
| `--method`          | str     | `gpu`        | Processing method: 'cpu' or 'gpu'                |
| `--voxels`          | str/int | **Required** | Number of voxels or "all"                        |
| `--window-size`     | int     | 400          | Rolling window size (frames)                     |
| `--percentile`      | float   | 10.0         | Baseline percentile (0-100)                      |
| `--sigma`           | float   | 0.5          | Gaussian filter sigma                            |
| `--no-filtering`    | flag    | False        | Disable Gaussian filtering                       |
| `--save-baseline`   | flag    | False        | **Save baseline array (required for filtering)** |
| `--save-brain-viz`  | flag    | False        | Generate brain visualizations                    |
| `--data-file`       | str     | config.json  | Input data file path                             |
| `--output`          | str     | Auto         | Output HDF5 file path                            |
| `--baseline-output` | str     | Auto         | Baseline output path                             |

#### Memory Requirements

**CPU Method:**

- ~3× raw data size (input + baseline + output)
- For all voxels: ~3 × 2400 × 2,120,420 × 4 bytes ≈ **60 GB RAM**

**GPU Method:**

- Automatic chunking based on available GPU memory
- Can process full dataset with 8-16 GB GPU VRAM
- Host memory: ~2× data size for I/O

---

### 3.2 Pipeline 2: Quality Control Filtering (`voxel_filtering_processor_dev`)

**Purpose**: Post-processing pipeline to remove low-quality voxels from preprocessed ΔF/F0 data using anatomical and intensity-based filtering.

#### Two-Stage Architecture

1. **Stage 1**: Brain region filter (anatomical exclusion)
2. **Stage 2**: Baseline threshold filter (intensity-based quality control)

Stages run sequentially, with each stage's output feeding the next.

#### Input Files

**Required in `config.json`:**

| Config Key                   | File Type | Description                                |
| ---------------------------- | --------- | ------------------------------------------ |
| `preprocessed_data`          | HDF5      | ΔF/F0 data from deltaf_processor           |
| `preprocessed_data_baseline` | NPY       | Baseline time series from deltaf_processor |
| `brain_mask`                 | NPY       | Binary brain mask (314×147×87)             |
| `brain_region_mask`          | NPY       | Labeled anatomical regions (314×147×87)    |

**Optional: Stage Input Override**

Create `voxel_filtering_stage_inputs.json` to override inputs:

```json
{
  "brain_region_filter": {
    "deltaf_f0": "path/to/input.h5",
    "baseline_timeseries": "path/to/baseline.npy",
    "brain_mask": "path/to/mask.npy",
    "brain_region_mask": "path/to/region_mask.npy",
    "exclude_regions": [1, 2, 3, 4],
    "output_suffix": "custom"
  },
  "baseline_threshold_filter": {
    "deltaf_f0": "path/from/stage1.h5",
    "baseline_timeseries": "path/from/stage1.npy",
    "mask": "path/from/stage1_mask.npy",
    "baseline_threshold": 2.0,
    "output_suffix": "custom"
  }
}
```

#### Stage 1: Brain Region Filter

**Purpose**: Remove voxels from specific anatomical regions (e.g., exclude background or artifact regions).

**Algorithm**:

```python
# 1. Load brain region mask (labeled volume)
region_mask = load_brain_region_mask()  # Shape: (314, 147, 87), values: 0-46

# 2. Create inclusion mask
if exclude_regions:
    keep_mask = (region_mask > 0)  # Exclude background (0)
    for region_id in exclude_regions:
        keep_mask = keep_mask & (region_mask != region_id)
else:
    keep_mask = (region_mask > 0)  # Only exclude background

# 3. Map to voxel columns (critical step)
brain_mask_flat = brain_mask.reshape(-1)
active_indices = np.flatnonzero(brain_mask_flat > 0)
voxel_global_indices = active_indices[voxel_indices]  # if voxel_indices provided
region_keep_mask = keep_mask.reshape(-1)[voxel_global_indices]

# 4. Filter data
filtered_deltaf = deltaf_data[:, region_keep_mask]
filtered_baseline = baseline_data[:, region_keep_mask]
```

**Parameters**:

- `exclude_regions`: List of region IDs to exclude (default: [], only background excluded)
- `baseline_time_index`: Frame for visualization (default: 1200)
- `deltaf_time_index`: Frame for visualization (default: 1200)

**Outputs**:

```
outputs/voxel_filtering_dev/brain_region_filter_kept{N}_*.h5
outputs/voxel_filtering_dev/brain_region_filter_kept{N}_*_mask.npy
outputs/voxel_filtering_dev/brain_region_filter_kept{N}_*_baseline_timeseries.npy
outputs/voxel_filtering_dev/images/brain_region_filter_*.png
```

#### Stage 2: Baseline Threshold Filter

**Purpose**: Remove voxels with low baseline fluorescence (likely noise or poor signal-to-noise ratio).

**Algorithm**:

```python
# 1. Calculate baseline statistics
if baseline.ndim == 1:
    baseline_mean = baseline  # Already single vector
else:
    baseline_mean = np.mean(baseline, axis=0)  # Average over time

# 2. Apply threshold
keep_mask = baseline_mean >= baseline_threshold

# 3. Filter data
filtered_deltaf = deltaf_data[:, keep_mask]
filtered_baseline = baseline_data[:, keep_mask]
filtered_voxel_indices = voxel_indices[keep_mask]
```

**Parameters**:

- `baseline_threshold`: Minimum mean baseline value (default: 5.0)
- `baseline_time_index`: Frame for visualization (default: 1200)
- `deltaf_time_index`: Frame for visualization (default: 1200)

**Outputs**:

```
outputs/voxel_filtering_dev/baseline_threshold_filter_kept{N}_th{T}_*.h5
outputs/voxel_filtering_dev/baseline_threshold_filter_kept{N}_th{T}_*_mask.npy
outputs/voxel_filtering_dev/baseline_threshold_filter_kept{N}_th{T}_*_baseline_timeseries.npy
outputs/voxel_filtering_dev/images/baseline_threshold_filter_*.png
```

#### HDF5 Output Structure (Both Stages)

```python
import h5py

with h5py.File(filtered_h5, 'r') as f:
    # Datasets
    deltaf_f0 = f['deltaf_f0'][:]       # Filtered ΔF/F0
    voxel_indices = f['voxel_indices'][:] # Global indices of kept voxels
    keep_mask = f['keep_mask'][:]        # Boolean mask applied

    # Metadata attributes
    print(f"Stage: {f.attrs['stage_name']}")
    print(f"Kept voxels: {f.attrs['kept_voxels']}")
    print(f"Removed voxels: {f.attrs['removed_voxels']}")
    print(f"Reference mask shape: {f.attrs['reference_mask_shape']}")

    # Stage 2 specific
    if 'baseline_threshold' in f.attrs:
        print(f"Baseline threshold: {f.attrs['baseline_threshold']}")
        print(f"Previous stage kept: {f.attrs['previous_stage_kept_voxels']}")
```

#### Usage Examples

**Full Pipeline (Both Stages)**

```bash
conda run -n base python testing_part5_newpre/voxel_filtering_processor_dev/run_dev_pipeline.py \
  --config config.json \
  --baseline-threshold 5.0 \
  --baseline-time 1200 \
  --deltaf-time 1200 \
  --output-subdir voxel_filtering_dev
```

**Stage 1 Only (Region Filtering)**

```bash
conda run -n base python testing_part5_newpre/voxel_filtering_processor_dev/run_dev_pipeline.py \
  --config config.json \
  --end-stage brain_region_filter \
  --output-subdir region_filter_test
```

**Stage 2 Only (Threshold Filtering)**

```bash
# First create voxel_filtering_stage_inputs.json with Stage 1 outputs
conda run -n base python testing_part5_newpre/voxel_filtering_processor_dev/run_dev_pipeline.py \
  --config config.json \
  --start-stage baseline_threshold_filter \
  --baseline-threshold 10.0 \
  --stage-inputs voxel_filtering_stage_inputs.json
```

**Custom Region Exclusion**

```json
// voxel_filtering_stage_inputs.json
{
  "brain_region_filter": {
    "exclude_regions": [1, 5, 10, 15],
    "output_suffix": "exclude_regions_1_5_10_15"
  }
}
```

#### Command-Line Arguments

| Argument               | Type  | Default                     | Description                       |
| ---------------------- | ----- | --------------------------- | --------------------------------- |
| `--config`             | Path  | `config.json`               | Configuration file path           |
| `--output-subdir`      | str   | `voxel_filtering_dev`       | Output subdirectory               |
| `--baseline-threshold` | float | 5.0                         | Baseline mean threshold (Stage 2) |
| `--baseline-time`      | int   | 1200                        | Baseline frame for visualization  |
| `--deltaf-time`        | int   | 1200                        | ΔF/F0 frame for visualization     |
| `--start-stage`        | str   | `brain_region_filter`       | First stage to execute            |
| `--end-stage`          | str   | `baseline_threshold_filter` | Last stage to execute             |
| `--stage-inputs`       | Path  | None                        | Stage input override JSON         |

#### Memory Requirements

**Stage 1**: ~20 GB RAM (loads full ΔF/F0 dataset)
**Stage 2**: ~6 GB RAM (loads filtered dataset from Stage 1)

---

### 3.3 Complete Workflow Example

This section demonstrates the complete processing workflow from raw data to filtered ΔF/F0.

#### Step 1: Calculate ΔF/F0

```bash
# Process all voxels with GPU
conda run -n base python -m deltaf_processor \
  --method gpu \
  --voxels all \
  --window-size 400 \
  --percentile 10 \
  --sigma 0.5 \
  --save-baseline \
  --save-brain-viz \
  --output outputs/deltaf_allvoxels.h5 \
  --baseline-output outputs/baseline/baseline_allvoxels.npy
```

**Expected Output:**

```
outputs/deltaf_allvoxels.h5                    # ΔF/F0 data
outputs/baseline/baseline_allvoxels.npy        # Baseline time series
outputs/masks/subset_mask_*.npy                # Voxel mask (if subset)
outputs/images/brain_overview_deltaf_*.png     # Visualizations
```

#### Step 2: Update Configuration

```json
// config.json
{
  "data_paths": {
    "preprocessed_data": "outputs/deltaf_allvoxels.h5",
    "preprocessed_data_baseline": "outputs/baseline/baseline_allvoxels.npy",
    "brain_mask": "brain_voxel_mask.npy",
    "brain_region_mask": "fly brain lable mask file (YiSun Lab)/brain_region_mask.npy"
  }
}
```

#### Step 3: Quality Filtering

```bash
# Filter out low-quality voxels
conda run -n base python testing_part5_newpre/voxel_filtering_processor_dev/run_dev_pipeline.py \
  --config config.json \
  --baseline-threshold 5.0 \
  --output-subdir voxel_filtering_final
```

**Expected Output:**

```
outputs/voxel_filtering_final/
  brain_region_filter_kept{N}_filtered.h5
  brain_region_filter_kept{N}_mask.npy
  brain_region_filter_kept{N}_baseline_timeseries.npy
  baseline_threshold_filter_kept{M}_th5p0_filtered.h5
  baseline_threshold_filter_kept{M}_th5p0_mask.npy
  baseline_threshold_filter_kept{M}_th5p0_baseline_timeseries.npy
  images/*.png
```

#### Step 4: Load and Use Filtered Data

```python
import h5py
import numpy as np

# Load final filtered results
output_file = 'outputs/voxel_filtering_final/baseline_threshold_filter_kept1500000_th5p0_filtered.h5'

with h5py.File(output_file, 'r') as f:
    deltaf_filtered = f['deltaf_f0'][:]      # Shape: (2400, ~1500000)
    voxel_indices = f['voxel_indices'][:]     # Global indices

    print(f"Filtered data shape: {deltaf_filtered.shape}")
    print(f"Kept voxels: {len(voxel_indices):,}")
    print(f"Removed voxels: {f.attrs['removed_voxels']:,}")
    print(f"Baseline threshold: {f.attrs.get('baseline_threshold', 'N/A')}")

# Load 3D mask for spatial visualization
mask_3d = np.load('outputs/voxel_filtering_final/baseline_threshold_filter_kept1500000_th5p0_mask.npy')
print(f"3D Mask shape: {mask_3d.shape}")  # (314, 147, 87)

# Map filtered data to 3D volume
brain_mask = np.load('brain_voxel_mask.npy')
brain_flat = brain_mask.reshape(-1)
active_indices = np.flatnonzero(brain_flat > 0)
global_indices = active_indices[voxel_indices]

def values_to_3d_volume(values, global_indices, brain_shape=(314, 147, 87)):
    """Map 1D values back to 3D brain volume."""
    flat_volume = np.zeros(np.prod(brain_shape), dtype=np.float32)
    flat_volume[global_indices] = values
    return flat_volume.reshape(brain_shape)

# Visualize frame 1200
frame_1200 = deltaf_filtered[1200, :]
volume_1200 = values_to_3d_volume(frame_1200, global_indices)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(volume_1200[:, :, 43].T, cmap='hot', origin='lower')
axes[0].set_title('Axial (Z=43)')
axes[1].imshow(volume_1200[157, :, :].T, cmap='hot', origin='lower')
axes[1].set_title('Sagittal (X=157)')
axes[2].imshow(volume_1200[:, 73, :].T, cmap='hot', origin='lower')
axes[2].set_title('Coronal (Y=73)')
plt.tight_layout()
plt.savefig('filtered_brain_frame1200.png', dpi=200)
plt.close()
```

#### Workflow Summary

```
Raw Data (chat_motioncorrected_voxel.npy)
   ↓
[deltaf_processor] → ΔF/F0 + Baseline
   ↓
Update config.json
   ↓
[voxel_filtering_processor_dev]
   ├─ Stage 1: Brain region filter
   └─ Stage 2: Baseline threshold filter
   ↓
Filtered ΔF/F0 (high-quality voxels only)
   ↓
Your analysis scripts
```

---

## 4. File Relationships

```
Complete Data Flow Diagram:
===========================

Raw Imaging Data:
┌──────────────────────────────────┐
│ chat_motioncorrected_voxel.npy   │
│ (2400, 2,120,420)                │
│ Motion-corrected fluorescence    │
└────────────┬─────────────────────┘
             │
             │ [deltaf_processor]
             │ ↓ Gaussian filtering (optional)
             │ ↓ Rolling percentile baseline
             │ ↓ ΔF/F0 calculation
             │
             ├────────────────────────────┐
             ▼                            ▼
┌──────────────────────────────────┐  ┌─────────────────────────────────┐
│ ΔF/F0 HDF5                       │  │ Baseline Timeseries NPY         │
│ deltaf_threestage_*.h5           │  │ deltaf_baseline_*.npy           │
│ Dataset: 'deltaf_f0'             │  │ Shape: (2400, voxels)           │
│ (2400, voxels)                   │  │ ⚠️ Required for Stage 2 filter  │
└────────────┬─────────────────────┘  └────────────┬────────────────────┘
             │                                     │
             │ Update config.json paths            │
             │                                     │
             ├─────────────────────────────────────┤
             │                                     │
             │ [voxel_filtering_processor_dev]     │
             │ Stage 1: Brain region filter        │
             │ (exclude anatomical regions)        │
             │                                     │
             ▼                                     ▼
┌──────────────────────────────────┐  ┌─────────────────────────────────┐
│ Stage 1 Filtered HDF5            │  │ Stage 1 Baseline NPY            │
│ brain_region_filter_kept*.h5     │  │ brain_region_filter_*_baseline  │
│ (2400, ~540K voxels)             │  │ _timeseries.npy                 │
└────────────┬─────────────────────┘  └────────────┬────────────────────┘
             │                                     │
             │ Stage 2: Baseline threshold filter  │
             │ (remove low baseline intensity)     │
             │                                     │
             ▼                                     ▼
┌──────────────────────────────────┐  ┌─────────────────────────────────┐
│ Stage 2 Filtered HDF5 (FINAL)    │  │ Stage 2 Baseline NPY            │
│ baseline_threshold_filter_*.h5   │  │ baseline_threshold_filter_*     │
│ (2400, ~540K voxels)             │  │ _baseline_timeseries.npy        │
│ High-quality voxels only         │  │                                 │
└────────────┬─────────────────────┘  └─────────────────────────────────┘
             │
             │ Map via 3D mask
             ▼
┌──────────────────────────────────┐
│ 3D Mask NPY                      │
│ *_mask.npy                       │
│ (314, 147, 87)                   │
│ Binary: 0=filtered, 1=kept       │
└──────────────────────────────────┘

Spatial Reference Files:
========================
┌──────────────────────────────────┐
│ brain_voxel_mask.npy             │
│ (314, 147, 87)                   │
│ Binary: 0=outside, 1=brain       │
│ Sum(mask==1) = 2,120,420         │
└──────────────────────────────────┘

Brain Region Annotation:
========================
┌──────────────────────────────────┐
│ brain_region_mask.npy            │
│ (314, 147, 87)                   │
│ Values: 0-46 (region IDs)        │
└────────────┬─────────────────────┘
             │
             │ Lookup region names
             ▼
┌──────────────────────────────────┐
│ label_index.pkl                  │
│ pandas DataFrame (47 rows × 2)   │
│ Columns: 'id', 'name'
└──────────────────────────────────┘

Reference Anatomy (NRRD format):
=================================
┌──────────────────────────────────┐
│ template.nrrd                    │
│ (314, 147, 87) uint8             │
│ JRC2018 fly brain template       │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│ label.nrrd                       │
│ (314, 147, 87) uint8             │
│ Values: 0-46 (same as .npy)      │
│ For fMRI software compatibility  │
└──────────────────────────────────┘

Behavioral Data:
================
┌──────────────────────────────────┐
│ Stim_Loc_Rec.pkl                 │
│ Dictionary with keys:            │
│  - forward_vel_raw (6000,)       │
│  - side_vel_raw (6000,)          │
│  - angular_vel_raw (6000,)       │
│  - auditory_stim_timepoints (10,)│
│  - auditory_stim_type (10,)      │
│                                  │
│ Sampling: 50Hz                   │
│ Duration: 120 seconds            │
│ ⚠️ Timepoints in SECONDS!        │
└──────────────────────────────────┘

Coordinate System Alignment:
=============================
All spatial volumes aligned to same coordinate system:
  brain_voxel_mask.npy     (314, 147, 87)  ─┐
  brain_region_mask.npy    (314, 147, 87)  ─┼─ Same shape & alignment
  template.nrrd            (314, 147, 87)  ─┤
  label.nrrd               (314, 147, 87)  ─┘

Time Alignment:
===============
Brain data (20Hz):    2400 timepoints × 50ms = 120 seconds
Behavioral data (50Hz): 6000 timepoints × 20ms = 120 seconds
Stimulus timepoints:    10 events in seconds (multiply by sampling rate)
```

---

## 5. External Resources

### Brain Atlas Reference

**Virtual Fly Brain** - Interactive fly brain atlas:

- Website: https://v2.virtualflybrain.org/
- Quickly check anatomical regions and connectivity
- 3D visualization of brain structures
- Cross-reference with JRC2018 template

### Visualization Software

**Software for viewing NRRD files if needed**:

1. **ITK-SNAP** (http://www.itksnap.org/)

   - Free, open-source
   - Interactive 3D visualization
   - Region labeling and segmentation tools

2. **3D Slicer** (https://www.slicer.org/)
   - Free, open-source
   - Advanced medical imaging platform
   - Volume rendering and multi-planar views

### Python Packages

**For loading NRRD files**:

```bash
pip install pynrrd
```

Documentation: https://pynrrd.readthedocs.io/

---

## 6. Summary

This dataset provides comprehensive multi-modal data for studying Drosophila brain activity and behavior:

- **Spatial Resolution**: 314 × 147 × 87 voxels, ~2.1 million active voxels
- **Temporal Resolution**: 20Hz brain imaging, 50Hz behavioral tracking
- **Duration**: 120 seconds continuous recording
- **Stimuli**: 10 auditory stimuli (5 types)
- **Anatomical Annotations**: 47 brain regions (0-46 labels)
- **Preprocessing**: Rolling percentile ΔF/F0 normalization applied

**Key Data Files**:

1. `chat_motioncorrected_voxel.npy` - Raw motion-corrected calcium imaging
2. `brain_voxel_mask.npy` - Binary brain mask (critical for 2D↔3D mapping)
3. `Stim_Loc_Rec.pkl` - Behavioral velocities and stimulus timing
4. `brain_region_mask.npy` - Anatomical region labels (0-46)
5. `label_index.pkl` - Region names (requires pandas patch)
6. `template.nrrd` / `label.nrrd` - Reference anatomy in NRRD format

**Processing Pipelines**:

1. **deltaf_processor** (`deltaf_processor`)

   - Calculates ΔF/F0 from raw data
   - Rolling percentile baseline estimation
   - Optional Gaussian filtering
   - GPU/CPU processing modes
   - Outputs: ΔF/F0 HDF5 + Baseline timeseries NPY

2. **voxel_filtering_processor_dev** (`voxel_filtering_processor_dev`)
   - Two-stage quality control filtering
   - Stage 1: Brain region filter (anatomical exclusion)
   - Stage 2: Baseline threshold filter (intensity-based, threshold=5.0)
   - Outputs: Filtered ΔF/F0 HDF5 + 3D masks

**Typical Workflow**:

```
Raw Data → deltaf_processor → ΔF/F0 + Baseline
                                  ↓
                         Update config.json
                                  ↓
                    voxel_filtering_processor_dev
                    (Stage 1 + Stage 2 filtering)
                                  ↓
                      Filtered high-quality ΔF/F0
                                  ↓
                        Your analysis scripts
```

**Important Considerations**:

- **Voxel Mapping**: Voxel indices correspond to `mask==1` positions (critical for spatial analysis)
- **Stimulus Timepoints**: Stored in seconds (multiply by 20 for brain frames, 50 for behavioral frames)
- **Baseline Files**: Required for quality filtering pipeline
- **Memory Requirements**: 60 GB RAM for CPU processing, 8-16 GB GPU VRAM for GPU mode
- **Optical Lobe**: Not imaged (label IDs 1-4)

---
