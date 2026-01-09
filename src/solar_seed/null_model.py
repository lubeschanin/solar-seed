"""
Null Model for Hypothesis Test
==============================

Shuffle-based null model for statistical validation.

The idea: If two channels are independent, shuffling
one channel should not change the MI.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List

from solar_seed.mutual_info import mutual_information


def shuffle_array(arr: NDArray[np.float64], seed: int | None = None) -> NDArray[np.float64]:
    """
    Shuffles an array (flat).

    Args:
        arr: Input array
        seed: Random seed for reproducibility

    Returns:
        Shuffled array with same shape
    """
    rng = np.random.default_rng(seed)
    flat = arr.ravel().copy()
    rng.shuffle(flat)
    return flat.reshape(arr.shape)


def compute_null_distribution(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    n_shuffles: int = 100,
    bins: int = 64,
    seed: int | None = 42,
    verbose: bool = False
) -> Tuple[float, float, List[float]]:
    """
    Computes null distribution through repeated shuffling.

    Args:
        x: First array (remains unchanged)
        y: Second array (gets shuffled)
        n_shuffles: Number of shuffle iterations
        bins: Bins for MI calculation
        seed: Base seed for reproducibility
        verbose: Progress output

    Returns:
        mean: Mean MI under null hypothesis
        std: Standard deviation
        distribution: List of all MI values
    """
    mi_values: List[float] = []
    
    base_rng = np.random.default_rng(seed)
    
    for i in range(n_shuffles):
        # Generate new seed for this iteration
        shuffle_seed = base_rng.integers(0, 2**31)
        
        y_shuffled = shuffle_array(y, seed=shuffle_seed)
        mi = mutual_information(x, y_shuffled, bins)
        mi_values.append(mi)
        
        if verbose and (i + 1) % 20 == 0:
            print(f"     Shuffle {i+1}/{n_shuffles}...")
    
    return float(np.mean(mi_values)), float(np.std(mi_values)), mi_values


def compute_z_score(
    mi_real: float,
    mi_null_mean: float,
    mi_null_std: float
) -> float:
    """
    Computes Z-score.

    Z = (MI_real - MI_null_mean) / MI_null_std

    Args:
        mi_real: Observed MI
        mi_null_mean: Mean MI from null model
        mi_null_std: Standard deviation from null model

    Returns:
        Z-score (standard deviations above null model)
    """
    if mi_null_std < 1e-10:
        return float('inf') if mi_real > mi_null_mean else 0.0
    
    return (mi_real - mi_null_mean) / mi_null_std


def compute_p_value(
    mi_real: float,
    mi_null_distribution: List[float]
) -> float:
    """
    Computes empirical p-value.

    p = proportion of null MI values >= MI_real

    Args:
        mi_real: Observed MI
        mi_null_distribution: List of MI values from null model

    Returns:
        p-value (one-sided, right-tailed)
    """
    if not mi_null_distribution:
        return 1.0
    
    n_greater = sum(1 for mi in mi_null_distribution if mi >= mi_real)
    return n_greater / len(mi_null_distribution)


def interpret_result(
    z_score: float,
    p_value: float
) -> Tuple[str, str]:
    """
    Interprets statistical results.

    Returns:
        status: Short status string
        interpretation: Longer explanation
    """
    if z_score > 3 and p_value < 0.01:
        return (
            "✓ SIGNIFICANT",
            "MI is highly significantly higher than expected (p < 0.01). "
            "The channels share information that cannot be explained by chance."
        )
    elif z_score > 2 and p_value < 0.05:
        return (
            "~ TRENDING",
            "MI tends to be higher than expected (p < 0.05). "
            "Further investigation recommended."
        )
    elif z_score < -2:
        return (
            "? UNEXPECTEDLY LOW",
            "MI is lower than expected. "
            "This could indicate anti-correlation or data artifacts."
        )
    else:
        return (
            "✗ NOT SIGNIFICANT",
            "MI is within the expected range of the null model. "
            "No evidence for additional structure."
        )
