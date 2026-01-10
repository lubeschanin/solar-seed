"""
Radial Profile Analysis for Solar Seed
======================================

Subtracts radial geometry (limb darkening) to separate
"real" correlations from geometric ones.

Method:
1. Calculate mean intensity as a function of radius
2. Reconstruct geometry image from this profile
3. Residual = Original / Geometry Model
4. MI on residuals → shows non-geometric correlation
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class RadialProfile:
    """Result of radial profile calculation."""
    radii: NDArray[np.float64]  # Center points of radius bins
    intensities: NDArray[np.float64]  # Mean intensity per bin
    counts: NDArray[np.int64]  # Number of pixels per bin
    center: Tuple[float, float]  # Center used (y, x)
    n_bins: int


def find_disk_center(
    image: NDArray[np.float64],
    threshold_fraction: float = 0.1
) -> Tuple[float, float]:
    """
    Finds the center of the solar disk.

    Uses intensity centroid of pixels above threshold.

    Args:
        image: 2D image array
        threshold_fraction: Threshold as fraction of maximum

    Returns:
        (center_y, center_x) coordinates of the center
    """
    threshold = image.max() * threshold_fraction
    mask = image > threshold

    if not mask.any():
        # Fallback: Image center
        return image.shape[0] / 2, image.shape[1] / 2

    y_coords, x_coords = np.where(mask)
    weights = image[mask]

    center_y = np.average(y_coords, weights=weights)
    center_x = np.average(x_coords, weights=weights)

    return float(center_y), float(center_x)


def compute_radial_profile(
    image: NDArray[np.float64],
    n_bins: int = 100,
    center: Optional[Tuple[float, float]] = None,
    max_radius: Optional[float] = None
) -> RadialProfile:
    """
    Calculates the radial intensity profile of an image.

    Args:
        image: 2D image array
        n_bins: Number of radius bins
        center: (y, x) center, or None for automatic detection
        max_radius: Maximum radius, or None for image diagonal/2

    Returns:
        RadialProfile with radius bins and mean intensities
    """
    if center is None:
        center = find_disk_center(image)

    center_y, center_x = center

    # Coordinate grid
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    if max_radius is None:
        max_radius = min(image.shape) / 2

    # Bin edges
    bin_edges = np.linspace(0, max_radius, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate mean per bin
    intensities = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        counts[i] = mask.sum()
        if counts[i] > 0:
            intensities[i] = image[mask].mean()

    return RadialProfile(
        radii=bin_centers,
        intensities=intensities,
        counts=counts,
        center=center,
        n_bins=n_bins
    )


def reconstruct_from_profile(
    profile: RadialProfile,
    shape: Tuple[int, int]
) -> NDArray[np.float64]:
    """
    Reconstructs an image from the radial profile.

    Each pixel receives the mean intensity of its radius bin.

    Args:
        profile: RadialProfile with radius-intensity mapping
        shape: (height, width) of output image

    Returns:
        Reconstructed geometry image
    """
    center_y, center_x = profile.center

    # Coordinate grid
    y, x = np.ogrid[:shape[0], :shape[1]]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Bin edges
    max_radius = profile.radii[-1] + (profile.radii[1] - profile.radii[0]) / 2
    bin_width = max_radius / profile.n_bins

    # Assign each pixel to its bin
    bin_indices = np.clip((r / bin_width).astype(int), 0, profile.n_bins - 1)

    # Reconstruct image
    reconstructed = profile.intensities[bin_indices]

    return reconstructed


def compute_residual(
    image: NDArray[np.float64],
    model: NDArray[np.float64],
    method: str = "ratio",
    epsilon: float = 1.0
) -> NDArray[np.float64]:
    """
    Calculates the residual between image and model.

    Args:
        image: Original image
        model: Geometry model (from radial profile)
        method: "ratio" (image/model) or "difference" (image-model)
        epsilon: Small constant to avoid division by zero

    Returns:
        Residual image
    """
    if method == "ratio":
        # Ratio is better for multiplicative effects (limb darkening)
        return image / (model + epsilon)
    elif method == "difference":
        return image - model
    else:
        raise ValueError(f"Unknown method: {method}")


def subtract_radial_geometry(
    image: NDArray[np.float64],
    n_bins: int = 100,
    method: str = "ratio"
) -> Tuple[NDArray[np.float64], RadialProfile, NDArray[np.float64]]:
    """
    Complete workflow: Calculate profile, create model, compute residual.

    Args:
        image: Original image
        n_bins: Number of radius bins
        method: "ratio" or "difference"

    Returns:
        residual: Image without radial geometry
        profile: Calculated radial profile
        model: Reconstructed geometry image
    """
    profile = compute_radial_profile(image, n_bins=n_bins)
    model = reconstruct_from_profile(profile, image.shape)
    residual = compute_residual(image, model, method=method)

    return residual, profile, model


def prepare_pair_for_residual_mi(
    image_1: NDArray[np.float64],
    image_2: NDArray[np.float64],
    n_bins: int = 100,
    method: str = "ratio",
    shared_center: bool = True
) -> Tuple[NDArray[np.float64], NDArray[np.float64], dict]:
    """
    Prepares an image pair for residual MI analysis.

    Args:
        image_1: First image (e.g. 193 Å)
        image_2: Second image (e.g. 211 Å)
        n_bins: Number of radius bins
        method: Residual method
        shared_center: Use same center for both images

    Returns:
        residual_1: Residual of image 1
        residual_2: Residual of image 2
        info: Dictionary with profiles and models
    """
    if shared_center:
        # Find shared center (average of both centers)
        center_1 = find_disk_center(image_1)
        center_2 = find_disk_center(image_2)
        center = (
            (center_1[0] + center_2[0]) / 2,
            (center_1[1] + center_2[1]) / 2
        )
    else:
        center = None

    # Calculate profiles
    profile_1 = compute_radial_profile(image_1, n_bins=n_bins, center=center)
    profile_2 = compute_radial_profile(image_2, n_bins=n_bins, center=center)

    # Reconstruct models
    model_1 = reconstruct_from_profile(profile_1, image_1.shape)
    model_2 = reconstruct_from_profile(profile_2, image_2.shape)

    # Calculate residuals
    residual_1 = compute_residual(image_1, model_1, method=method)
    residual_2 = compute_residual(image_2, model_2, method=method)

    info = {
        "profile_1": profile_1,
        "profile_2": profile_2,
        "model_1": model_1,
        "model_2": model_2,
        "center": center if shared_center else (profile_1.center, profile_2.center),
        "method": method
    }

    return residual_1, residual_2, info
