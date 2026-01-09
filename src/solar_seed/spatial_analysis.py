"""
Spatial MI Analysis for Solar Seed
===================================

Computes MI maps over the solar disk to identify
where the highest (residual) MI occurs.

Question: Where on the Sun is the "extra information"?
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Optional
from dataclasses import dataclass

from solar_seed.mutual_info import mutual_information
from solar_seed.radial_profile import prepare_pair_for_residual_mi


@dataclass
class SpatialMIResult:
    """Result of the spatial MI analysis."""
    mi_map: NDArray[np.float64]  # MI per region
    grid_size: Tuple[int, int]  # (rows, cols)
    cell_size: Tuple[int, int]  # Pixel size per cell
    image_shape: Tuple[int, int]  # Original image size

    # Statistics
    mi_mean: float
    mi_std: float
    mi_max: float
    mi_min: float

    # Hotspot info
    hotspot_idx: Tuple[int, int]  # (row, col) of maximum
    hotspot_value: float


@dataclass
class SpatialComparisonResult:
    """Comparison of original MI and residual MI maps."""
    original: SpatialMIResult
    residual: SpatialMIResult

    # Difference analysis
    mi_reduction_map: NDArray[np.float64]  # original - residual
    mi_reduction_percent: NDArray[np.float64]  # (orig - res) / orig * 100

    # Where does the most MI remain after subtraction?
    residual_hotspot_idx: Tuple[int, int]
    residual_hotspot_value: float


def compute_spatial_mi_map(
    image_1: NDArray[np.float64],
    image_2: NDArray[np.float64],
    grid_size: Tuple[int, int] = (8, 8),
    bins: int = 32,
    min_valid_pixels: int = 100
) -> SpatialMIResult:
    """
    Computes a spatial MI map.

    Divides the images into a grid and computes MI per cell.

    Args:
        image_1: First image (e.g., 193 Å)
        image_2: Second image (e.g., 211 Å)
        grid_size: (rows, cols) number of grid cells
        bins: Bins for MI calculation (fewer for smaller regions)
        min_valid_pixels: Minimum number of non-zero pixels for MI calculation

    Returns:
        SpatialMIResult with MI map and statistics
    """
    rows, cols = grid_size
    h, w = image_1.shape

    cell_h = h // rows
    cell_w = w // cols

    mi_map = np.zeros((rows, cols), dtype=np.float64)

    for i in range(rows):
        for j in range(cols):
            # Extract region
            y_start = i * cell_h
            y_end = (i + 1) * cell_h if i < rows - 1 else h
            x_start = j * cell_w
            x_end = (j + 1) * cell_w if j < cols - 1 else w

            region_1 = image_1[y_start:y_end, x_start:x_end]
            region_2 = image_2[y_start:y_end, x_start:x_end]

            # Check if enough valid pixels present
            valid_mask = (region_1 > 0) & (region_2 > 0)
            n_valid = valid_mask.sum()

            if n_valid >= min_valid_pixels:
                mi_map[i, j] = mutual_information(
                    region_1[valid_mask],
                    region_2[valid_mask],
                    bins=bins
                )
            else:
                mi_map[i, j] = np.nan

    # Statistics (ignore NaN)
    valid_mi = mi_map[~np.isnan(mi_map)]
    if len(valid_mi) > 0:
        mi_mean = float(np.mean(valid_mi))
        mi_std = float(np.std(valid_mi))
        mi_max = float(np.max(valid_mi))
        mi_min = float(np.min(valid_mi))

        # Find hotspot
        max_idx = np.unravel_index(np.nanargmax(mi_map), mi_map.shape)
        hotspot_idx = (int(max_idx[0]), int(max_idx[1]))
        hotspot_value = float(mi_map[hotspot_idx])
    else:
        mi_mean = mi_std = mi_max = mi_min = 0.0
        hotspot_idx = (0, 0)
        hotspot_value = 0.0

    return SpatialMIResult(
        mi_map=mi_map,
        grid_size=grid_size,
        cell_size=(cell_h, cell_w),
        image_shape=image_1.shape,
        mi_mean=mi_mean,
        mi_std=mi_std,
        mi_max=mi_max,
        mi_min=mi_min,
        hotspot_idx=hotspot_idx,
        hotspot_value=hotspot_value
    )


def compute_spatial_residual_mi_map(
    image_1: NDArray[np.float64],
    image_2: NDArray[np.float64],
    grid_size: Tuple[int, int] = (8, 8),
    bins: int = 32,
    n_profile_bins: int = 100
) -> SpatialComparisonResult:
    """
    Computes and compares original and residual MI maps.

    Args:
        image_1: First image
        image_2: Second image
        grid_size: Grid size
        bins: Bins for MI
        n_profile_bins: Bins for radial profile

    Returns:
        SpatialComparisonResult with both maps and comparison
    """
    # Original MI map
    original = compute_spatial_mi_map(image_1, image_2, grid_size, bins)

    # Compute residuals
    residual_1, residual_2, _ = prepare_pair_for_residual_mi(
        image_1, image_2, n_bins=n_profile_bins
    )

    # Residual MI map
    residual = compute_spatial_mi_map(residual_1, residual_2, grid_size, bins)

    # Reduction analysis
    mi_reduction_map = original.mi_map - residual.mi_map

    # Percentage reduction (avoid division by 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        mi_reduction_percent = np.where(
            original.mi_map > 0,
            (original.mi_map - residual.mi_map) / original.mi_map * 100,
            0
        )

    return SpatialComparisonResult(
        original=original,
        residual=residual,
        mi_reduction_map=mi_reduction_map,
        mi_reduction_percent=mi_reduction_percent,
        residual_hotspot_idx=residual.hotspot_idx,
        residual_hotspot_value=residual.hotspot_value
    )


def find_top_hotspots(
    result: SpatialMIResult,
    n: int = 5
) -> List[Tuple[Tuple[int, int], float]]:
    """
    Finds the top N hotspots (highest MI values).

    Args:
        result: SpatialMIResult
        n: Number of hotspots

    Returns:
        List of ((row, col), mi_value) tuples
    """
    mi_map = result.mi_map.copy()
    mi_map = np.nan_to_num(mi_map, nan=-np.inf)

    # Flat indices sorted by value
    flat_indices = np.argsort(mi_map.ravel())[::-1]

    hotspots = []
    for idx in flat_indices[:n]:
        row, col = np.unravel_index(idx, mi_map.shape)
        value = result.mi_map[row, col]
        if not np.isnan(value):
            hotspots.append(((int(row), int(col)), float(value)))

    return hotspots


def get_region_coordinates(
    result: SpatialMIResult,
    grid_idx: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Returns the pixel coordinates of a grid region.

    Args:
        result: SpatialMIResult
        grid_idx: (row, col) in grid

    Returns:
        (y_start, y_end, x_start, x_end) in pixels
    """
    row, col = grid_idx
    cell_h, cell_w = result.cell_size
    h, w = result.image_shape
    rows, cols = result.grid_size

    y_start = row * cell_h
    y_end = (row + 1) * cell_h if row < rows - 1 else h
    x_start = col * cell_w
    x_end = (col + 1) * cell_w if col < cols - 1 else w

    return y_start, y_end, x_start, x_end


# ============================================================================
# ASCII VISUALIZATION
# ============================================================================

def mi_map_to_ascii(
    mi_map: NDArray[np.float64],
    width: int = 40,
    show_values: bool = False
) -> str:
    """
    Converts MI map to ASCII art for terminal output.

    Args:
        mi_map: 2D MI map
        width: Width of output in characters
        show_values: Show numerical values

    Returns:
        ASCII string of MI map
    """
    # Intensity characters (from low to high)
    chars = " ·:░▒▓█"

    rows, cols = mi_map.shape

    # Normalize to [0, 1]
    valid_mask = ~np.isnan(mi_map)
    if not valid_mask.any():
        return "No valid data"

    mi_min = np.nanmin(mi_map)
    mi_max = np.nanmax(mi_map)
    mi_range = mi_max - mi_min if mi_max > mi_min else 1.0

    normalized = (mi_map - mi_min) / mi_range
    normalized = np.nan_to_num(normalized, nan=0)

    # Calculate character width per cell
    char_per_cell = max(1, width // cols)

    lines = []

    # Top border
    lines.append("┌" + "─" * (cols * char_per_cell) + "┐")

    for i in range(rows):
        line = "│"
        for j in range(cols):
            if np.isnan(mi_map[i, j]):
                char = " "
            else:
                idx = int(normalized[i, j] * (len(chars) - 1))
                idx = min(idx, len(chars) - 1)
                char = chars[idx]
            line += char * char_per_cell
        line += "│"
        lines.append(line)

    # Bottom border
    lines.append("└" + "─" * (cols * char_per_cell) + "┘")

    # Legend
    lines.append(f"  MI: {mi_min:.3f} {'·' * 3} {mi_max:.3f}")

    return "\n".join(lines)


def print_spatial_comparison(
    result: SpatialComparisonResult,
    title: str = "Spatial MI Analysis"
) -> None:
    """
    Prints a formatted comparison of the spatial MI.

    Args:
        result: SpatialComparisonResult
        title: Title for the output
    """
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")

    print(f"\n  Grid: {result.original.grid_size[0]}x{result.original.grid_size[1]}")
    print(f"  Cell size: {result.original.cell_size[0]}x{result.original.cell_size[1]} pixels")

    # Original MI
    print(f"\n  ORIGINAL MI:")
    print(f"    Mean: {result.original.mi_mean:.4f} ± {result.original.mi_std:.4f}")
    print(f"    Range: [{result.original.mi_min:.4f}, {result.original.mi_max:.4f}]")
    print(f"    Hotspot: Cell {result.original.hotspot_idx} = {result.original.hotspot_value:.4f}")

    print(f"\n  ORIGINAL MI MAP:")
    print(mi_map_to_ascii(result.original.mi_map))

    # Residual MI
    print(f"\n  RESIDUAL MI (after geometry subtraction):")
    print(f"    Mean: {result.residual.mi_mean:.4f} ± {result.residual.mi_std:.4f}")
    print(f"    Range: [{result.residual.mi_min:.4f}, {result.residual.mi_max:.4f}]")
    print(f"    Hotspot: Cell {result.residual.hotspot_idx} = {result.residual.hotspot_value:.4f}")

    print(f"\n  RESIDUAL MI MAP:")
    print(mi_map_to_ascii(result.residual.mi_map))

    # Reduction analysis
    valid_reduction = result.mi_reduction_percent[~np.isnan(result.mi_reduction_percent)]
    if len(valid_reduction) > 0:
        mean_reduction = np.mean(valid_reduction)
        print(f"\n  REDUCTION FROM GEOMETRY SUBTRACTION:")
        print(f"    Mean reduction: {mean_reduction:.1f}%")

        # Where does the most MI remain?
        min_reduction_idx = np.unravel_index(
            np.nanargmin(result.mi_reduction_percent),
            result.mi_reduction_percent.shape
        )
        min_reduction = result.mi_reduction_percent[min_reduction_idx]
        print(f"    Smallest reduction: Cell {min_reduction_idx} ({min_reduction:.1f}%)")
        print(f"    → This region has the most 'extra information'")

    # Top hotspots in residual
    print(f"\n  TOP 5 RESIDUAL HOTSPOTS:")
    hotspots = find_top_hotspots(result.residual, n=5)
    for rank, (idx, value) in enumerate(hotspots, 1):
        coords = get_region_coordinates(result.residual, idx)
        print(f"    {rank}. Cell {idx}: MI={value:.4f} (Pixels {coords[0]}-{coords[1]}, {coords[2]}-{coords[3]})")


def create_disk_mask(
    shape: Tuple[int, int],
    center: Optional[Tuple[float, float]] = None,
    radius_fraction: float = 0.9
) -> NDArray[np.bool_]:
    """
    Creates a circular mask for the solar disk.

    Args:
        shape: (height, width) of the image
        center: (y, x) center, or None for image center
        radius_fraction: Radius as fraction of half image size

    Returns:
        Boolean mask (True = inside the disk)
    """
    h, w = shape
    if center is None:
        center = (h / 2, w / 2)

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    radius = min(h, w) / 2 * radius_fraction

    return r <= radius


def compute_disk_spatial_mi(
    image_1: NDArray[np.float64],
    image_2: NDArray[np.float64],
    grid_size: Tuple[int, int] = (8, 8),
    bins: int = 32
) -> SpatialComparisonResult:
    """
    Computes spatial MI only within the solar disk.

    Pixels outside the disk are set to 0.

    Args:
        image_1: First image
        image_2: Second image
        grid_size: Grid size
        bins: Bins for MI

    Returns:
        SpatialComparisonResult
    """
    # Create disk mask
    mask = create_disk_mask(image_1.shape)

    # Apply mask
    image_1_masked = image_1.copy()
    image_2_masked = image_2.copy()
    image_1_masked[~mask] = 0
    image_2_masked[~mask] = 0

    return compute_spatial_residual_mi_map(
        image_1_masked, image_2_masked, grid_size, bins
    )
