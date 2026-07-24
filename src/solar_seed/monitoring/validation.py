"""
Data Validation
===============

Validation functions to detect data errors before anomaly detection.
These gates run BEFORE statistical calculations to prevent invalid
data from contaminating baselines and thresholds.
"""

import numpy as np

from .constants import MIN_MI_THRESHOLD, MIN_ROI_STD


def validate_roi_variance(img1, img2, pair: str = None) -> dict:
    """
    Validate that ROI images have sufficient variance for meaningful MI.

    A constant (or near-constant) image will produce artificially low MI,
    which could be misinterpreted as a coupling break.

    Args:
        img1, img2: Residual images after geometry subtraction
        pair: Channel pair name for reporting

    Returns:
        dict with 'is_valid', 'error_type', 'error_reason', 'std1', 'std2'
    """
    # Compute standard deviation of each image
    std1 = np.nanstd(img1)
    std2 = np.nanstd(img2)

    if not np.isfinite(std1) or not np.isfinite(std2):
        return {
            'is_valid': False,
            'error_type': 'INVALID_IMAGE',
            'error_reason': f'Non-finite std dev: std1={std1}, std2={std2}',
            'std1': std1,
            'std2': std2,
        }

    if std1 < MIN_ROI_STD or std2 < MIN_ROI_STD:
        low_ch = []
        if std1 < MIN_ROI_STD:
            low_ch.append(f'ch1={std1:.2f}')
        if std2 < MIN_ROI_STD:
            low_ch.append(f'ch2={std2:.2f}')
        return {
            'is_valid': False,
            'error_type': 'CONSTANT_ROI',
            'error_reason': f'Near-constant image ({", ".join(low_ch)} DN std < {MIN_ROI_STD})',
            'std1': std1,
            'std2': std2,
        }

    return {
        'is_valid': True,
        'error_type': None,
        'error_reason': None,
        'std1': std1,
        'std2': std2,
    }


def validate_mi_measurement(delta_mi: float, pair: str = None,
                            baseline_mean: float = None) -> dict:
    """
    Validate MI measurement before it enters break detection.

    This gate runs BEFORE MAD/baseline calculation to prevent
    data errors from contaminating statistics.

    Args:
        delta_mi: Measured ΔMI value
        pair: Channel pair name (for reporting)
        baseline_mean: Baseline mean ΔMI for this pair, if known. When given,
            the low-MI gate becomes baseline-relative: max(0.02, 0.3*baseline).
            A fixed threshold of 0.05 would discard genuine breaks in
            weak-coupling pairs (e.g. 193-304 at 1k: baseline 0.07 ± 0.02).

    Returns:
        dict with 'is_valid', 'error_type', 'error_reason'
    """
    # Check for NaN/Inf
    if not np.isfinite(delta_mi):
        return {
            'is_valid': False,
            'error_type': 'INVALID_VALUE',
            'error_reason': f'Non-finite value: {delta_mi}',
        }

    # Check for suspiciously low MI (indicates data pipeline failure)
    if baseline_mean is not None and baseline_mean > 0:
        threshold = max(0.02, 0.3 * baseline_mean)
    else:
        threshold = MIN_MI_THRESHOLD
    if delta_mi < threshold:
        return {
            'is_valid': False,
            'error_type': 'BELOW_THRESHOLD',
            'error_reason': f'ΔMI={delta_mi:.4f} < {threshold:.4f} (likely data error)',
        }

    return {'is_valid': True, 'error_type': None, 'error_reason': None}
