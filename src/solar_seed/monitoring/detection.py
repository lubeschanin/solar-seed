"""
Anomaly Detection
=================

Coupling break detection, artifact detection, and anomaly classification.

Detection Pipeline:
1. validate_roi_variance() - Check image quality
2. validate_mi_measurement() - Check MI value validity
3. detect_artifact() - Check for single-frame anomalies
4. detect_coupling_break() - Formal break detection (median - k×MAD)
5. compute_robustness_check() - Verify stability under binning
6. compute_registration_shift() - Check spatial alignment
7. classify_anomaly_status() - Final actionable vs diagnostic classification
"""

from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING
import numpy as np

from .validation import validate_mi_measurement
from .constants import MIN_MI_THRESHOLD

if TYPE_CHECKING:
    from .coupling import CouplingMonitor


class AnomalyStatus:
    """Anomaly status constants."""
    DATA_ERROR = 'DATA_ERROR'            # Invalid data - do not process
    VALIDATED_BREAK = 'VALIDATED_BREAK'  # Actionable - all tests pass
    ANOMALY_VETOED = 'ANOMALY_VETOED'    # Diagnostic only - some test failed
    NORMAL = 'NORMAL'                     # No anomaly detected


class BreakType:
    """Break type constants for phase-gating."""
    PRECURSOR = 'PRECURSOR'      # Pre-flare: rising activity, stress buildup
    POSTCURSOR = 'POSTCURSOR'    # Post-flare: decay phase, relaxation
    AMBIGUOUS = 'AMBIGUOUS'      # Cannot determine: insufficient context


def detect_artifact(pair: str, current_mi: float, monitor: 'CouplingMonitor',
                    threshold_sigma: float = 3.0) -> dict | None:
    """
    Detect if current measurement might be an artifact (single-frame anomaly).

    An artifact is suspected if:
    - Current value deviates > threshold_sigma from recent mean
    - AND we have enough history to establish a baseline

    Returns dict with artifact info or None if no artifact suspected.
    """
    pair_history = [h for h in monitor.history if pair in h.get('coupling', {})]

    if len(pair_history) < 3:
        return None  # Not enough data to detect artifacts

    # Get recent values (last 6 readings, ~1 hour)
    recent = pair_history[-6:]
    recent_values = [h['coupling'][pair].get('delta_mi', 0)
                     for h in recent if 'delta_mi' in h['coupling'].get(pair, {})]

    if len(recent_values) < 3:
        return None

    mean_val = sum(recent_values) / len(recent_values)
    std_val = (sum((v - mean_val)**2 for v in recent_values) / len(recent_values)) ** 0.5

    if std_val < 0.001:  # Avoid division by zero
        return None

    deviation_sigma = abs(current_mi - mean_val) / std_val

    if deviation_sigma > threshold_sigma:
        return {
            'suspected': True,
            'deviation_sigma': deviation_sigma,
            'current': current_mi,
            'recent_mean': mean_val,
            'recent_std': std_val,
            'message': f"Jump of {deviation_sigma:.1f}σ from recent mean ({mean_val:.3f} ± {std_val:.3f})"
        }

    return None


def compute_registration_shift(img1, img2, max_shift: int = 10) -> dict:
    """
    Test B: Spatial registration sanity check using FFT cross-correlation.

    Computes the peak (dx, dy) shift between two images.
    Large shifts may indicate registration issues that cause MI artifacts.

    Args:
        img1, img2: Image arrays (same shape)
        max_shift: Maximum shift to search (pixels)

    Returns:
        dict with dx, dy, peak_value, is_centered
    """
    try:
        from scipy import fft

        # Use small central region for fast computation
        h, w = img1.shape
        cy, cx = h // 2, w // 2
        size = 256  # Small crop for speed

        crop1 = img1[cy-size:cy+size, cx-size:cx+size].astype(np.float64)
        crop2 = img2[cy-size:cy+size, cx-size:cx+size].astype(np.float64)

        # Normalize
        crop1 = (crop1 - np.mean(crop1)) / (np.std(crop1) + 1e-10)
        crop2 = (crop2 - np.mean(crop2)) / (np.std(crop2) + 1e-10)

        # FFT-based cross-correlation (much faster than correlate2d)
        f1 = fft.fft2(crop1)
        f2 = fft.fft2(crop2)
        corr = np.real(fft.ifft2(f1 * np.conj(f2)))

        # Shift zero-frequency to center
        corr = fft.fftshift(corr)

        # Find peak
        peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
        center = (corr.shape[0] // 2, corr.shape[1] // 2)

        dy = peak_idx[0] - center[0]
        dx = peak_idx[1] - center[1]
        peak_val = corr[peak_idx] / crop1.size  # Normalized

        # Is it well-centered? (peak within max_shift of center)
        shift_magnitude = np.sqrt(dx**2 + dy**2)
        is_centered = shift_magnitude <= max_shift

        return {
            'dx': int(dx),
            'dy': int(dy),
            'shift_pixels': float(shift_magnitude),
            'peak_correlation': float(peak_val),
            'is_centered': bool(is_centered),
            'max_allowed': max_shift,
        }

    except Exception as e:
        return {
            'dx': 0,
            'dy': 0,
            'shift_pixels': 0,
            'peak_correlation': 0,
            'is_centered': True,
            'error': str(e),
        }


def detect_coupling_break(pair: str, current_mi: float, monitor: 'CouplingMonitor',
                          window_minutes: int = 60, k: float = 2.0) -> dict:
    """
    Formal "Coupling Break" detection using rolling median and MAD.

    Definition (reviewer-proof):
        A Coupling Break occurs when:
        ΔMI(t) < median(last 60 min) - k × MAD(last 60 min)

    Data Quality Gate (runs FIRST):
        If current_mi < MIN_MI_THRESHOLD or is NaN/Inf, return DATA_ERROR.
        Invalid values are excluded from statistics to prevent contamination.

    Args:
        pair: Channel pair name
        current_mi: Current ΔMI value
        monitor: CouplingMonitor instance
        window_minutes: Rolling window size
        k: MAD multiplier (default 2.0 = ~95% interval)

    Returns:
        dict with break detection result and metadata
    """
    # === DATA QUALITY GATE (runs before any statistics) ===
    mi_validation = validate_mi_measurement(current_mi, pair)
    if not mi_validation['is_valid']:
        return {
            'is_break': False,
            'data_error': True,
            'error_type': mi_validation['error_type'],
            'error_reason': mi_validation['error_reason'],
            'current_mi': current_mi,
            'reason': f"DATA_ERROR: {mi_validation['error_reason']}",
        }

    pair_history = [h for h in monitor.history if pair in h.get('coupling', {})]

    # Filter to window AND exclude invalid values from statistics
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=window_minutes)

    window_values = []
    excluded_count = 0
    for h in pair_history:
        try:
            ts = datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00'))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= window_start:
                val = h['coupling'][pair].get('delta_mi')
                if val is not None:
                    # Validate historical value too (don't let bad data contaminate stats)
                    val_check = validate_mi_measurement(val, pair)
                    if val_check['is_valid']:
                        window_values.append(val)
                    else:
                        excluded_count += 1
        except:
            continue

    if len(window_values) < 3:
        return {
            'is_break': False,
            'reason': f'Insufficient data ({len(window_values)}/{3} points)',
            'n_points': len(window_values),
            'window_minutes': window_minutes,
        }

    # Compute robust statistics (median and MAD)
    window_values.sort()
    n = len(window_values)
    if n % 2 == 0:
        median = (window_values[n//2 - 1] + window_values[n//2]) / 2
    else:
        median = window_values[n//2]

    # MAD = median(|x_i - median|)
    deviations = sorted([abs(v - median) for v in window_values])
    if len(deviations) % 2 == 0:
        mad = (deviations[len(deviations)//2 - 1] + deviations[len(deviations)//2]) / 2
    else:
        mad = deviations[len(deviations)//2]

    # Scale MAD to approximate std (1.4826 for normal distribution)
    mad_scaled = mad * 1.4826

    # Threshold
    threshold = median - k * mad_scaled

    # Compute z_mad = (median - current) / MAD (positive = below median)
    # This is the number of MADs below median
    if mad_scaled > 0.001:
        z_mad = (median - current_mi) / mad_scaled
    else:
        z_mad = 0

    # Detect break: z_mad >= k means we're k MADs below median
    is_break = z_mad >= k

    return {
        'is_break': is_break,
        'current_mi': current_mi,
        'median': median,
        'mad': mad,
        'mad_scaled': mad_scaled,
        'threshold': threshold,
        'k': k,
        'z_mad': z_mad,  # MADs below median (positive = below)
        'n_points': n,
        'window_minutes': window_minutes,
        'criterion': f'ΔMI < median - {k}×MAD = {threshold:.4f}',
    }


def compute_robustness_check(img1, img2, original_mi: float, method: str = 'binning') -> dict:
    """
    Test C: Robustness check - verify MI is stable under preprocessing changes.

    Recomputes ΔMI with:
    - 2x2 binning (reduces resolution)

    If the drop is stable under binning, it's much harder to dismiss as artifact.

    Args:
        img1, img2: Original image arrays
        original_mi: MI computed at full resolution
        method: 'binning' (default)

    Returns:
        dict with robustness check results
    """
    try:
        from solar_seed.radial_profile import subtract_radial_geometry
        from solar_seed.control_tests import sector_ring_shuffle_test

        # 2x2 binning
        def bin2x2(img):
            h, w = img.shape
            h_new, w_new = h // 2, w // 2
            return img[:h_new*2, :w_new*2].reshape(h_new, 2, w_new, 2).mean(axis=(1, 3))

        binned1 = bin2x2(img1)
        binned2 = bin2x2(img2)

        # Recompute MI on binned data
        res1, _, _ = subtract_radial_geometry(binned1)
        res2, _, _ = subtract_radial_geometry(binned2)
        shuffle_result = sector_ring_shuffle_test(res1, res2, n_rings=10, n_sectors=12)
        binned_mi = shuffle_result.mi_original - shuffle_result.mi_sector_shuffled

        # Compare
        change_pct = (binned_mi - original_mi) / original_mi * 100 if original_mi else 0

        # Stable if change < 20%
        is_robust = abs(change_pct) < 20

        return {
            'original_mi': original_mi,
            'binned_mi': binned_mi,
            'change_pct': change_pct,
            'is_robust': is_robust,
            'method': '2x2 binning',
            'interpretation': 'STABLE' if is_robust else 'SENSITIVE to resolution',
        }

    except Exception as e:
        return {
            'original_mi': original_mi,
            'binned_mi': None,
            'change_pct': None,
            'is_robust': None,
            'error': str(e),
        }


def classify_break_type(trend_info: dict = None, goes_context: dict = None) -> dict:
    """
    Classify whether a validated break is a precursor or postcursor.

    Phase-Gating Rules:
    - PRECURSOR if: GOES rising OR (|trend| > 3%/h AND acc < 0)
    - POSTCURSOR if: GOES falling AND acc > 0 (relaxation)
    - AMBIGUOUS otherwise

    Args:
        trend_info: Trend data with slope_pct_per_hour and acceleration
        goes_context: GOES data with flux, derivative, and phase

    Returns:
        dict with break_type, reason, and is_alertable
    """
    # Extract values
    slope = abs(trend_info.get('slope_pct_per_hour', 0)) if trend_info else 0
    acc = trend_info.get('acceleration', 0) if trend_info else 0

    goes_rising = False
    goes_phase = 'unknown'
    if goes_context:
        goes_rising = goes_context.get('rising', False)
        goes_phase = goes_context.get('phase', 'unknown')

    # Rule 1: GOES rising → PRECURSOR
    if goes_rising:
        return {
            'break_type': BreakType.PRECURSOR,
            'reason': 'GOES activity rising',
            'is_alertable': True,
        }

    # Rule 2: Accelerating decoupling (steep negative trend, negative acc) → PRECURSOR
    if slope > 3.0 and acc < -1.0:
        return {
            'break_type': BreakType.PRECURSOR,
            'reason': f'Accelerating decoupling ({slope:.1f}%/h, acc={acc:.1f})',
            'is_alertable': True,
        }

    # Rule 3: GOES falling/decay AND positive acceleration (relaxation) → POSTCURSOR
    if goes_phase in ['decay', 'post-flare', 'falling'] or (not goes_rising and acc > 1.0):
        if acc > 0:
            return {
                'break_type': BreakType.POSTCURSOR,
                'reason': f'Decay phase relaxation (acc={acc:+.1f}%/h²)',
                'is_alertable': False,
            }

    # Rule 4: Positive acceleration without rising GOES → likely POSTCURSOR
    if acc > 2.0 and not goes_rising:
        return {
            'break_type': BreakType.POSTCURSOR,
            'reason': f'Relaxation signature (acc={acc:+.1f}%/h²)',
            'is_alertable': False,
        }

    # Cannot determine with confidence
    return {
        'break_type': BreakType.AMBIGUOUS,
        'reason': 'Insufficient phase context',
        'is_alertable': False,  # Conservative: don't alert if unsure
    }


def classify_anomaly_status(break_detection: dict, robustness_check: dict = None,
                            registration_check: dict = None, time_spread_sec: float = None,
                            trend_info: dict = None, goes_context: dict = None) -> dict:
    """
    Classify anomaly status into Actionable vs Diagnostic with Phase-Gating.

    Status levels:
        VALIDATED_BREAK (Actionable): z_mad >= k AND all tests PASS AND is PRECURSOR
        ANOMALY_VETOED (Diagnostic only): z_mad >= k BUT some test FAIL OR is POSTCURSOR
        NORMAL: z_mad < k

    Phase-Gating (critical for avoiding false alerts during decay):
        - PRECURSOR breaks → Actionable (issue alert)
        - POSTCURSOR breaks → Diagnostic only (no alert)
        - AMBIGUOUS breaks → Diagnostic only (conservative)

    Args:
        break_detection: Result from detect_coupling_break()
        robustness_check: Result from compute_robustness_check() (optional)
        registration_check: Result from compute_registration_shift() (optional)
        time_spread_sec: Time spread between channels in seconds (optional)
        trend_info: Trend data for phase-gating (slope, acceleration)
        goes_context: GOES context for phase-gating (rising, phase)

    Returns:
        dict with status, is_actionable, break_type, veto_reasons, and diagnostic info
    """
    # === DATA_ERROR: Invalid measurement (skip entirely) ===
    if break_detection.get('data_error'):
        return {
            'status': AnomalyStatus.DATA_ERROR,
            'is_actionable': False,
            'data_error': True,
            'error_type': break_detection.get('error_type'),
            'error_reason': break_detection.get('error_reason'),
            'z_mad': 0,
            'veto_reasons': [f"DATA_ERROR: {break_detection.get('error_reason', 'unknown')}"],
            'passed_tests': [],
            'failed_tests': ['data_quality'],
        }

    # No break detected = NORMAL
    if not break_detection.get('is_break') and not break_detection.get('vetoed'):
        z_mad = break_detection.get('z_mad', 0)
        if z_mad < break_detection.get('k', 2.0):
            return {
                'status': AnomalyStatus.NORMAL,
                'is_actionable': False,
                'z_mad': z_mad,
                'veto_reasons': [],
                'passed_tests': [],
                'failed_tests': [],
            }

    # Break candidate detected - check validation tests
    veto_reasons = []
    passed_tests = []
    failed_tests = []
    z_mad = break_detection.get('z_mad', 0)

    # Test A: Time alignment (<60s)
    if time_spread_sec is not None:
        if time_spread_sec <= 60:
            passed_tests.append(f'time_sync ({time_spread_sec:.0f}s)')
        else:
            failed_tests.append(f'time_sync ({time_spread_sec:.0f}s > 60s)')
            veto_reasons.append(f'time sync failed ({time_spread_sec:.0f}s)')

    # Test B: Registration shift (<10px)
    if registration_check:
        shift = registration_check.get('shift_pixels', 0)
        if registration_check.get('is_centered', True) and shift <= 10:
            passed_tests.append(f'registration ({shift:.1f}px)')
        else:
            failed_tests.append(f'registration ({shift:.1f}px > 10px)')
            veto_reasons.append(f'registration failed ({shift:.1f}px)')

    # Test C: Robustness (<20% binning change)
    if robustness_check:
        if robustness_check.get('is_robust') is True:
            change = robustness_check.get('change_pct', 0)
            passed_tests.append(f'robustness ({change:.1f}%)')
        elif robustness_check.get('is_robust') is False:
            change = robustness_check.get('change_pct', 0)
            failed_tests.append(f'robustness ({change:.1f}% > 20%)')
            veto_reasons.append(f'robustness failed ({change:.1f}%)')

    # Already vetoed by earlier logic
    if break_detection.get('vetoed'):
        if break_detection['vetoed'] not in [r.split()[0] for r in veto_reasons]:
            veto_reasons.append(break_detection['vetoed'])

    # Phase-Gating: classify break type (PRECURSOR vs POSTCURSOR)
    break_type_info = classify_break_type(trend_info, goes_context)
    break_type = break_type_info['break_type']
    phase_reason = break_type_info['reason']

    # Determine final status with phase-gating
    if veto_reasons:
        # Validation failed → diagnostic only
        status = AnomalyStatus.ANOMALY_VETOED
        is_actionable = False
    elif break_type == BreakType.PRECURSOR:
        # Validated + PRECURSOR → actionable alert
        status = AnomalyStatus.VALIDATED_BREAK
        is_actionable = True
    else:
        # Validated but POSTCURSOR/AMBIGUOUS → diagnostic only (phase-gated)
        status = AnomalyStatus.VALIDATED_BREAK  # Still validated, but...
        is_actionable = False  # ...not actionable due to phase
        veto_reasons.append(f'phase-gated: {break_type} ({phase_reason})')

    return {
        'status': status,
        'is_actionable': is_actionable,
        'break_type': break_type,
        'phase_reason': phase_reason,
        'z_mad': z_mad,
        'veto_reasons': veto_reasons,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
    }
