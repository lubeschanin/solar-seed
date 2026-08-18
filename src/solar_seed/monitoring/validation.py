"""
Data Validation
===============

Validation functions to detect data errors before anomaly detection.
These gates run BEFORE statistical calculations to prevent invalid
data from contaminating baselines and thresholds.
"""

import numpy as np

from .constants import (
    EXTREME_LOW_BASELINE_FRACTION,
    MIN_MI_THRESHOLD,
    MIN_ROI_STD,
    ROBUSTNESS_MAX_CHANGE_PCT,
    SYNC_SPREAD_MAX_SEC,
)


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

    The gate is a pure noise floor (MIN_MI_THRESHOLD). It deliberately does NOT
    scale with the baseline: a deep coupling collapse is the signal this system
    exists to detect, and a baseline-relative gate discards exactly the largest
    events. See the constants module for the full rationale.

    Args:
        delta_mi: Measured ΔMI value
        pair: Channel pair name (for reporting)
        baseline_mean: Baseline mean ΔMI for this pair, if known. Used only to
            set the non-blocking 'is_extreme_low' flag - never to invalidate.

    Returns:
        dict with 'is_valid', 'error_type', 'error_reason', 'is_extreme_low'
    """
    # Check for NaN/Inf
    if not np.isfinite(delta_mi):
        return {
            'is_valid': False,
            'error_type': 'INVALID_VALUE',
            'error_reason': f'Non-finite value: {delta_mi}',
            'is_extreme_low': False,
        }

    # Noise floor: at or below this there is no coupling left to measure, which
    # is indistinguishable from a broken pipeline.
    if delta_mi < MIN_MI_THRESHOLD:
        return {
            'is_valid': False,
            'error_type': 'BELOW_THRESHOLD',
            'error_reason': (
                f'ΔMI={delta_mi:.4f} < {MIN_MI_THRESHOLD:.4f} '
                f'(at/below permutation noise floor)'
            ),
            'is_extreme_low': True,
        }

    # Valid, but worth flagging: a collapse deeper than any documented flare
    # signature. Kept in the statistics on purpose.
    is_extreme_low = bool(
        baseline_mean
        and baseline_mean > 0
        and delta_mi < EXTREME_LOW_BASELINE_FRACTION * baseline_mean
    )

    return {
        'is_valid': True,
        'error_type': None,
        'error_reason': None,
        'is_extreme_low': is_extreme_low,
    }


def assess_measurement_quality(
    *,
    time_spread_sec: float = None,
    robustness_check: dict = None,
    is_data_error: bool = False,
    break_vetoed=None,
    inherited_reasons: str = None,
) -> dict:
    """
    Decide quality_ok for a stored measurement AND say why.

    This is the single definition of "is this row clean", shared by the live
    monitor, the gap backfill and the 4k backfill. Every caller previously had
    its own copy of `spread < 60`, and only one of the four failure branches
    ever wrote a reason.

    Why the reason matters: a bare quality_ok=0 is unfilterable. 12513 of the
    12665 flagged rows in the database carry no reason at all, which is how an
    inverted anti-spike filter stayed invisible for three weeks. A row that is
    excluded must say what excluded it, or the exclusion cannot be reviewed.

    Args:
        time_spread_sec: Channel observation spread (artifact test A)
        robustness_check: Result of the 2x2 binning check (artifact test C)
        is_data_error: Measurement already classified as DATA_ERROR
        break_vetoed: Truthy veto recorded by break detection
        inherited_reasons: Reasons carried over from an earlier computation of
            the same row (used by backfill, which recomputes only some tests)

    Returns:
        dict with 'quality_ok' (bool) and 'veto_reason' (str or None).
        Multiple reasons are joined with '+' so the field stays greppable.
    """
    reasons = []

    if inherited_reasons:
        reasons.extend(
            r for r in (x.strip() for x in inherited_reasons.split('+')) if r
        )

    if is_data_error:
        reasons.append('data_error')

    if break_vetoed:
        # break_vetoed may be a bool flag or an explanatory string
        reasons.append(
            'break_vetoed' if break_vetoed is True else str(break_vetoed)
        )

    if time_spread_sec is not None and time_spread_sec > SYNC_SPREAD_MAX_SEC:
        reasons.append(f'time_sync({time_spread_sec:.0f}s)')

    if robustness_check is not None and robustness_check.get('is_robust') is False:
        change = robustness_check.get('change_pct')
        reasons.append(
            f'robustness({change:.0f}%)' if change is not None else 'robustness'
        )

    # De-duplicate by reason KIND, not by exact string. The same finding can
    # arrive twice in different detail - break detection reports a bare
    # 'robustness' while the robustness check reports 'robustness(36%)' - and
    # comparing full strings let both through as 'robustness+robustness(36%)'.
    # Keep the most detailed variant of each kind, in order of first appearance.
    best: dict[str, str] = {}
    order: list[str] = []
    for r in reasons:
        kind = r.split('(')[0].strip()
        if kind not in best:
            order.append(kind)
            best[kind] = r
        elif len(r) > len(best[kind]):
            best[kind] = r
    unique = [best[k] for k in order]

    return {
        'quality_ok': not unique,
        'veto_reason': '+'.join(unique) if unique else None,
    }
