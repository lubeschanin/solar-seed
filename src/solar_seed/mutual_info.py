"""
Mutual Information Calculation
==============================

Pure NumPy implementation of Mutual Information.

MI(X,Y) = H(X) + H(Y) - H(X,Y)

where H is the Shannon entropy.
"""

import numpy as np
from numpy.typing import NDArray

# Named constants
EPSILON = 1e-10      # Numerical stability in histogram normalization
MIN_SAMPLES = 100    # Minimum valid samples for MI calculation
DEFAULT_BINS = 64    # Default histogram bins


def _validate_inputs(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """
    Validate and filter inputs for MI calculation.

    Raises:
        ValueError: If arrays are empty, all NaN/Inf, or shape mismatch.

    Returns:
        Tuple of (x_valid, y_valid) flat arrays, or None if < MIN_SAMPLES valid.
    """
    x_flat = x.ravel().astype(np.float64)
    y_flat = y.ravel().astype(np.float64)

    if x_flat.size == 0 or y_flat.size == 0:
        raise ValueError("Input arrays must not be empty")

    if x_flat.size != y_flat.size:
        raise ValueError(
            f"Shape mismatch: x has {x_flat.size} elements, y has {y_flat.size}"
        )

    valid = np.isfinite(x_flat) & np.isfinite(y_flat)
    x_flat = x_flat[valid]
    y_flat = y_flat[valid]

    if x_flat.size == 0:
        raise ValueError("No valid (non-NaN/Inf) samples in input arrays")

    if x_flat.size < MIN_SAMPLES:
        return None

    return x_flat, y_flat


def compute_histogram_2d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    bins: int
) -> NDArray[np.float64]:
    """
    Computes 2D histogram for two arrays.

    Args:
        x: First array (will be flattened)
        y: Second array (will be flattened)
        bins: Number of bins per dimension

    Returns:
        2D histogram of size (bins, bins)
    """
    # Normalize to [0, bins-1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    x_norm = (x - x_min) / (x_max - x_min + EPSILON)
    y_norm = (y - y_min) / (y_max - y_min + EPSILON)
    
    x_bins = np.clip((x_norm * bins).astype(int), 0, bins - 1)
    y_bins = np.clip((y_norm * bins).astype(int), 0, bins - 1)
    
    # 2D histogram (faster variant with bincount)
    hist = np.zeros((bins, bins), dtype=np.float64)
    indices = x_bins.ravel() * bins + y_bins.ravel()
    counts = np.bincount(indices, minlength=bins * bins)
    hist = counts.reshape((bins, bins)).astype(np.float64)
    
    return hist


def entropy(p: NDArray[np.float64]) -> float:
    """
    Computes Shannon entropy.

    Args:
        p: Probability distribution (must sum to 1)

    Returns:
        Entropy in bits
    """
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return -float(np.sum(p * np.log2(p)))


def mutual_information(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    bins: int = DEFAULT_BINS
) -> float:
    """
    Computes Mutual Information between two arrays.

    MI(X,Y) = H(X) + H(Y) - H(X,Y)

    Args:
        x: First array (any shape, will be flattened)
        y: Second array (any shape, will be flattened)
        bins: Number of bins for histogram

    Returns:
        Mutual Information in bits

    Raises:
        ValueError: If arrays are empty, all NaN/Inf, or shape mismatch.

    Example:
        >>> x = np.random.randn(100, 100)
        >>> y = x + np.random.randn(100, 100) * 0.1  # Strongly correlated
        >>> mi = mutual_information(x, y)
        >>> mi > 1.0  # Should show high MI
        True
    """
    result = _validate_inputs(x, y)
    if result is None:
        return 0.0
    x_flat, y_flat = result

    # 2D Histogramm
    hist_2d = compute_histogram_2d(x_flat, y_flat, bins)
    
    # Marginal histograms
    hist_x = hist_2d.sum(axis=1)
    hist_y = hist_2d.sum(axis=0)
    
    # Normalize to probabilities
    n = hist_2d.sum()
    if n == 0:
        return 0.0
    
    p_xy = hist_2d / n
    p_x = hist_x / n
    p_y = hist_y / n
    
    # Entropies
    h_x = entropy(p_x)
    h_y = entropy(p_y)
    h_xy = entropy(p_xy.ravel())
    
    # MI = H(X) + H(Y) - H(X,Y)
    mi = h_x + h_y - h_xy
    
    return max(0.0, mi)


def normalized_mutual_information(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    bins: int = DEFAULT_BINS
) -> float:
    """
    Normalized Mutual Information.

    NMI = 2 * MI(X,Y) / (H(X) + H(Y))

    Args:
        x: First array
        y: Second array
        bins: Number of bins

    Returns:
        NMI in range [0, 1], where 1 = perfect correlation

    Raises:
        ValueError: If arrays are empty, all NaN/Inf, or shape mismatch.
    """
    result = _validate_inputs(x, y)
    if result is None:
        return 0.0
    x_flat, y_flat = result

    hist_2d = compute_histogram_2d(x_flat, y_flat, bins)
    hist_x = hist_2d.sum(axis=1)
    hist_y = hist_2d.sum(axis=0)
    
    n = hist_2d.sum()
    if n == 0:
        return 0.0
    
    p_xy = hist_2d / n
    p_x = hist_x / n
    p_y = hist_y / n
    
    h_x = entropy(p_x)
    h_y = entropy(p_y)
    h_xy = entropy(p_xy.ravel())
    
    mi = h_x + h_y - h_xy
    
    if h_x + h_y == 0:
        return 0.0
    
    nmi = 2 * mi / (h_x + h_y)
    return max(0.0, min(1.0, nmi))


def conditional_entropy(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    bins: int = DEFAULT_BINS
) -> float:
    """
    Conditional entropy H(X|Y).

    H(X|Y) = H(X,Y) - H(Y) = H(X) - MI(X,Y)

    Returns:
        H(X|Y) in bits

    Raises:
        ValueError: If arrays are empty, all NaN/Inf, or shape mismatch.
    """
    result = _validate_inputs(x, y)
    if result is None:
        return 0.0
    x_flat, y_flat = result

    hist_2d = compute_histogram_2d(x_flat, y_flat, bins)
    hist_y = hist_2d.sum(axis=0)
    
    n = hist_2d.sum()
    if n == 0:
        return 0.0
    
    p_xy = hist_2d / n
    p_y = hist_y / n
    
    h_xy = entropy(p_xy.ravel())
    h_y = entropy(p_y)
    
    return max(0.0, h_xy - h_y)
