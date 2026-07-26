"""
Coupling Monitor
================

Track ΔMI coupling residuals over time for pre-flare detection.

Based on findings from Lubeschanin et al. (2026):
- Adjacent temperature layers (193-211 Å) show strongest coupling (0.59 ± 0.12 bits)
- Flare analysis showed -25% to -47% reduction during flares
- Chromospheric anchor (304 Å) shows highest temporal stability
"""

import copy
import json
import os
import sys
from pathlib import Path
from datetime import datetime

from .baselines import (
    PROVISIONAL_BASELINES,
    _mad_sigma,
    _median,
    load_baselines,
)
from .constants import (
    Z_SUDDEN_DROP_MODERATE,
    Z_SUDDEN_DROP_SEVERE,
    classify_status,
)

# Confidence levels ordered by rank. Plain string min() would compare
# lexicographically (min('high', 'medium') == 'high'!), so combining
# confidences must go through this ranking.
CONFIDENCE_RANK = {'none': 0, 'insufficient': 0, 'low': 1, 'medium': 2, 'high': 3}


def _min_confidence(a: str, b: str) -> str:
    """Return the weaker of two confidence labels (by rank, not alphabet)."""
    return min(a, b, key=lambda c: CONFIDENCE_RANK.get(c, 0))


class CouplingMonitor:
    """Track coupling residuals over time for pre-flare detection."""

    # Provisional fallback only - the live values come from
    # results/early_warning/baselines.json via load_baselines(). Kept as class
    # attributes because tests and external callers reference them.
    BASELINES_1K = PROVISIONAL_BASELINES['1k']
    BASELINES_4K = PROVISIONAL_BASELINES['4k']

    # Legacy alias for backwards compatibility
    BASELINES = BASELINES_1K

    # Flare analysis showed -25% to -47% reduction during flares
    ALERT_THRESHOLD = -0.25  # 25% below baseline triggers warning

    def __init__(self, history_file: Path = None, baseline_file: Path = None):
        self.history_file = history_file or Path("results/early_warning/coupling_history.json")
        self.history = self._load_history()
        self._baselines = load_baselines(baseline_file)

    def _load_history(self) -> list:
        """Load coupling history from file.

        A corrupt history is not silently swallowed: losing 24h of context
        puts every detector into "insufficient data" mode, which looks
        identical to a quiet Sun. Report it and move the bad file aside so the
        next run starts clean instead of failing again.
        """
        if not self.history_file.exists():
            return []
        try:
            with open(self.history_file) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            raise ValueError(f'expected a list, got {type(data).__name__}')
        except (json.JSONDecodeError, OSError, ValueError) as e:
            print(
                f"WARNING: coupling history unreadable ({e}); starting empty. "
                f"Detectors will report 'insufficient data' for ~1h.",
                file=sys.stderr,
            )
            try:
                self.history_file.replace(self.history_file.with_suffix('.corrupt'))
            except OSError:
                pass
            return []

    def _save_history(self):
        """Save coupling history atomically (temp file + replace).

        A plain open('w') truncates first: an interrupt mid-write leaves
        invalid JSON behind and _load_history then starts from scratch.
        """
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        # Keep last 24 hours (144 entries at 10min intervals)
        self.history = self.history[-144:]
        tmp = self.history_file.with_suffix(self.history_file.suffix + '.tmp')
        with open(tmp, 'w') as f:
            json.dump(self.history, f)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(self.history_file)

    def pair_series(self, pair: str, valid_only: bool = True) -> list[dict]:
        """
        History entries carrying a ΔMI reading for `pair`, oldest first.

        Args:
            valid_only: Drop readings that failed measurement validation.
                DATA_ERROR frames are stored with delta_mi = 0.0 (see
                _analyze_pair); letting those through wrecks the Theil-Sen
                slope and the y_mean normalisation in analyze_trend, and
                poisons the sudden-drop reference. detect_coupling_break
                already filtered them - this is the shared version.

        Returns:
            List of {'timestamp': str, 'delta_mi': float} dicts.
        """
        from .validation import validate_mi_measurement

        series = []
        for h in self.history:
            entry = h.get('coupling', {}).get(pair)
            if not isinstance(entry, dict):
                continue
            value = entry.get('delta_mi')
            if value is None:
                continue
            if entry.get('data_error') or entry.get('status') == 'DATA_ERROR':
                if valid_only:
                    continue
            if valid_only and not validate_mi_measurement(value, pair)['is_valid']:
                continue
            series.append({
                'timestamp': h.get('timestamp'),
                'delta_mi': value,
                'z_mad': entry.get('z_mad', 0),
            })
        return series

    def is_persistent_break(self, pair: str, current_is_break: bool, min_frames: int = 2) -> bool:
        """
        Check if a break persists for min_frames consecutive readings.

        Anti-spike filter: only confirm a break if the coupling was already
        depressed in the preceding frame(s). At 10-min cadence, min_frames=2
        means 20 minutes of persistence.

        The reference level is the median of the frames *before* the candidate
        break window, not the rolling 60-min median used by
        detect_coupling_break. That rolling median follows a sustained collapse
        downwards, so z_mad falls back below threshold after a frame or two -
        which used to let transient two-frame spikes through while filtering
        out exactly the sustained plateaus this check is meant to confirm.

        Args:
            pair: Channel pair (e.g. '193-211')
            current_is_break: Whether current frame shows a break
            min_frames: Minimum consecutive depressed frames required

        Returns:
            True if break is persistent, False if likely spike/artifact
        """
        if not current_is_break:
            return False

        n_previous = min_frames - 1
        if n_previous <= 0:
            return True

        series = self.pair_series(pair)
        candidate = series[-n_previous:] if n_previous else []
        if len(candidate) < n_previous:
            # Not enough history, can't confirm persistence
            return False

        # Pre-break reference window: the frames before the candidate frames.
        reference_window = [e['delta_mi'] for e in series[:-n_previous]][-12:]
        if len(reference_window) < 3:
            return False

        reference = _median(reference_window)
        sigma = _mad_sigma(reference_window, reference)
        # With a degenerate spread fall back to a 10% relative drop.
        threshold = reference - 2.0 * sigma if sigma > 1e-6 else reference * 0.90

        return all(e['delta_mi'] < threshold for e in candidate)

    def detect_sudden_drop(self, pair: str, delta_mi: float, lookback: int = 3,
                           baseline_std: float = None) -> dict:
        """
        Detect sudden relative drop in coupling.

        This catches pre-flare drops that are still above baseline but represent
        a significant decrease from recent readings.

        The reference is the MEDIAN of the lookback window, not its maximum.
        With max(), the reference sits systematically above the typical level -
        for 193-211 the measured spread is sigma/mu = 0.24, so the maximum of
        three draws lands ~20% high on average and a perfectly ordinary next
        reading already registers as a >15% "drop". That single line produced
        69% of all predictions in the database (10196 of 14708).

        Severity is graded in sigma, not in percent, for the same reason the
        status thresholds are: a 15% drop is 0.9 sigma for 193-211 but only
        0.3 sigma for 193-304 at 4k, so one percentage produced wildly
        different false-alarm rates per pair.

        Args:
            pair: Channel pair (e.g. '193-211')
            delta_mi: Current ΔMI value
            lookback: Number of previous readings to compare (default: 3 = ~30 min)
            baseline_std: Baseline sigma for this pair. Without it the drop
                cannot be expressed in sigma and no severity is assigned -
                drop_pct is still reported for context.

        Returns:
            Dict with drop detection results
        """
        series = self.pair_series(pair)

        if len(series) < lookback:
            return {
                'sudden_drop': False,
                'drop_pct': 0,
                'drop_sigma': 0,
                'reference_value': None,
                'severity': None,
                'reason': f'Not enough history ({len(series)}/{lookback})'
            }

        # Get recent values (excluding current)
        recent_values = [e['delta_mi'] for e in series[-lookback:]]

        # Median of recent values as reference (robust "normal" level)
        reference = _median(recent_values)

        # Percent drop is kept for display and for the stored record
        drop_pct = (delta_mi - reference) / reference if reference > 0 else 0

        # ...but the decision is made in sigma below the recent level.
        if baseline_std and baseline_std > 0:
            drop_sigma = (reference - delta_mi) / baseline_std
        else:
            drop_sigma = 0.0

        if not baseline_std or baseline_std <= 0:
            sudden_drop, severity = False, None
        elif drop_sigma >= Z_SUDDEN_DROP_SEVERE:
            sudden_drop, severity = True, 'SEVERE'
        elif drop_sigma >= Z_SUDDEN_DROP_MODERATE:
            sudden_drop, severity = True, 'MODERATE'
        else:
            sudden_drop, severity = False, None

        return {
            'sudden_drop': sudden_drop,
            'drop_pct': drop_pct,
            'drop_sigma': drop_sigma,
            'reference_value': reference,
            'current_value': delta_mi,
            'severity': severity,
            'lookback_minutes': lookback * 10,  # Assuming 10min intervals
        }

    def get_baselines(self, resolution: str = '1k') -> dict:
        """Get baselines for the specified resolution.

        Reads the measured table loaded at construction time
        (results/early_warning/baselines.json), falling back to the
        provisional hardcoded values when no measured file exists.
        """
        key = '4k' if resolution == '4k' else '1k'
        return self._baselines.get(key, PROVISIONAL_BASELINES[key])

    @property
    def baseline_source(self) -> str:
        """Where the active baselines came from ('provisional' or a path)."""
        return self._baselines.get('_meta', {}).get('source', 'unknown')

    def compute_residual(self, pair: str, delta_mi: float, resolution: str = '1k') -> dict:
        """Compute residual r(t) = (ΔMI - baseline) / std with sudden drop detection.

        Args:
            pair: Channel pair (e.g. '193-211')
            delta_mi: Current ΔMI value
            resolution: Data resolution ('1k' or '4k') - affects baseline selection
        """
        baselines = self.get_baselines(resolution)
        if pair not in baselines:
            return {'residual': 0, 'deviation_pct': 0, 'status': 'unknown', 'sudden_drop': None}

        baseline = baselines[pair]
        residual = (delta_mi - baseline['mean']) / baseline['std']
        # Kept for display and for the stored record; no longer decides status.
        deviation_pct = (delta_mi - baseline['mean']) / baseline['mean']

        # Check for sudden drop (relative to recent readings), graded in sigma
        drop_info = self.detect_sudden_drop(
            pair, delta_mi, baseline_std=baseline['std'])

        # Status from the z-score alone, so the criterion means the same thing
        # for every pair regardless of its relative spread.
        status = classify_status(residual)

        # A sharp drop from the recent level still raises ELEVATED even when
        # the absolute level is nominal - that is the pre-flare case the
        # absolute threshold cannot see.
        if status == 'NORMAL' and drop_info['sudden_drop']:
            status = 'ELEVATED'

        return {
            'residual': residual,
            'deviation_pct': deviation_pct,
            'status': status,
            'sudden_drop': drop_info
        }

    def _theil_sen_slope(self, values: list) -> float:
        """Compute robust Theil-Sen median slope estimator."""
        n = len(values)
        if n < 2:
            return 0.0

        slopes = []
        for i in range(n):
            for j in range(i + 1, n):
                if j != i:
                    slopes.append((values[j] - values[i]) / (j - i))

        if not slopes:
            return 0.0

        slopes.sort()
        mid = len(slopes) // 2
        if len(slopes) % 2 == 0:
            return (slopes[mid - 1] + slopes[mid]) / 2
        return slopes[mid]

    def analyze_trend(self, pair: str) -> dict:
        """Analyze recent trend in coupling using robust Theil-Sen estimator."""
        # valid_only: a DATA_ERROR frame is stored as delta_mi = 0.0 and would
        # dominate both the Theil-Sen slope and the y_mean normalisation below.
        pair_history = self.pair_series(pair)
        n_available = len(pair_history)

        # Base result with metadata
        base_result = {
            'method': 'Theil-Sen',
            'interval_min': 10,  # Assumed interval between readings
            'window_max': 12,    # Max window size (2 hours)
        }

        # Minimum 3 points for any trend
        MIN_POINTS = 3
        if n_available < MIN_POINTS:
            if n_available == 0:
                return {
                    **base_result,
                    'trend': 'NO_DATA',
                    'slope_pct_per_hour': 0,
                    'n_points': 0,
                    'window_min': 0,
                    'confidence': 'none',
                    'reason': 'No readings available'
                }
            else:
                return {
                    **base_result,
                    'trend': 'COLLECTING',
                    'slope_pct_per_hour': 0,
                    'n_points': n_available,
                    'window_min': n_available * 10,
                    'confidence': 'insufficient',
                    'reason': f'Need {MIN_POINTS} points, have {n_available}'
                }

        # Rolling window: last 12 points (2 hours) or all available
        window_size = min(12, n_available)
        recent = pair_history[-window_size:]
        values = [e['delta_mi'] for e in recent]
        n = len(values)

        # Calculate actual time span from timestamps
        try:
            t_first = datetime.fromisoformat(recent[0]['timestamp'].replace('Z', '+00:00'))
            t_last = datetime.fromisoformat(recent[-1]['timestamp'].replace('Z', '+00:00'))
            window_min = (t_last - t_first).total_seconds() / 60
        except (KeyError, ValueError, AttributeError, TypeError):
            window_min = n * 10  # Fallback: assume 10min intervals

        # Robust Theil-Sen slope
        slope = self._theil_sen_slope(values)

        # Mean value for normalization
        y_mean = sum(values) / n if n > 0 else 1

        # Normalize slope to % per hour using the ACTUAL cadence from
        # timestamps (slope is per reading index). Falls back to the old
        # 10-min assumption (6 readings/h) if the window has zero span.
        if window_min > 0 and n > 1:
            readings_per_hour = (n - 1) / (window_min / 60.0)
        else:
            readings_per_hour = 6  # Fallback: assume 10min intervals
        slope_per_hour = slope * readings_per_hour / y_mean * 100 if y_mean else 0

        # Acceleration: compare first half vs second half slopes
        acceleration = 0
        if n >= 6:
            first_half = values[:n//2]
            second_half = values[n//2:]
            slope1 = self._theil_sen_slope(first_half)
            slope2 = self._theil_sen_slope(second_half)
            acceleration = (slope2 - slope1) / y_mean * 100 if y_mean else 0

        # Confidence based on sample size
        if n >= 9:
            confidence = 'high'
        elif n >= 6:
            confidence = 'medium'
        else:
            confidence = 'low'

        # Thresholds for trend classification
        EPSILON = 3.0      # %/hour for stable vs trending
        EPSILON_ACC = 2.0  # acceleration threshold

        # Determine trend label
        if abs(slope_per_hour) < EPSILON:
            trend = 'STABLE'
        elif slope_per_hour < -EPSILON:
            if acceleration < -EPSILON_ACC:
                trend = 'ACCELERATING_DOWN'  # Getting worse faster
            else:
                trend = 'DECLINING'
        else:  # slope_per_hour > EPSILON
            if acceleration > EPSILON_ACC:
                trend = 'ACCELERATING_UP'
            else:
                trend = 'RISING'

        return {
            **base_result,
            'trend': trend,
            'slope_pct_per_hour': slope_per_hour,
            'acceleration': acceleration,
            'n_points': n,
            'window_min': window_min,
            'confidence': confidence
        }

    def add_reading(self, timestamp: str, coupling_data: dict):
        """Add a new coupling reading to history.

        Stores a deep copy: the caller (run_coupling_analysis) keeps writing
        _quality / _validation / _transfer_state into the same dict after this
        returns, and a stored reference would pull all of that into the
        history file on the next save.
        """
        self.history.append({
            'timestamp': timestamp,
            'coupling': copy.deepcopy(coupling_data),
        })
        self._save_history()

    def detect_transfer_state(self, robustness_checks: dict = None,
                              time_spread_sec: float = None) -> dict | None:
        """
        Detect potential energy transfer between layers.

        TRANSFER_STATE: When chromospheric anchor (193-304) strengthens
        while coronal coupling (193-211) weakens - may indicate
        energy reorganization before flare.

        If channels involved have failed robustness checks or time_sync fails,
        state is marked as 'degraded' (diagnostic only, not actionable).

        Args:
            robustness_checks: Dict of robustness check results by pair
            time_spread_sec: Time spread between channel observations (>60s = ASYNC)

        Returns dict with state info or None if not detected.
        """
        # Need trends for both pairs
        trend_304 = self.analyze_trend('193-304')
        trend_211 = self.analyze_trend('193-211')

        # Require at least medium confidence
        if trend_304.get('confidence') in ['none', 'low']:
            return None
        if trend_211.get('confidence') in ['none', 'low']:
            return None

        slope_304 = trend_304.get('slope_pct_per_hour', 0)
        slope_211 = trend_211.get('slope_pct_per_hour', 0)

        # Thresholds for transfer detection
        RISING_THRESHOLD = 3.0   # %/hour
        FALLING_THRESHOLD = -3.0  # %/hour

        # Check degradation conditions
        degraded = False
        degraded_reasons = []

        # 1. Time sync failure (ASYNC)
        if time_spread_sec is not None and time_spread_sec > 60:
            degraded = True
            degraded_reasons.append(f'ASYNC (channels {time_spread_sec:.0f}s apart)')

        # 2. Robustness failures
        if robustness_checks:
            for pair in ['193-211', '193-304']:
                rob = robustness_checks.get(pair, {})
                if rob.get('is_robust') is False:
                    degraded = True
                    change = rob.get('change_pct', 0)
                    degraded_reasons.append(f'{pair} robustness failed (Δbin={change:.1f}%)')

        # Transfer state: 304 rising while 211 falling
        if slope_304 > RISING_THRESHOLD and slope_211 < FALLING_THRESHOLD:
            result = {
                'state': 'TRANSFER_STATE',
                'description': 'Chromospheric anchor strengthening, coronal coupling weakening',
                'slope_193_304': slope_304,
                'slope_193_211': slope_211,
                'confidence': _min_confidence(trend_304['confidence'], trend_211['confidence']),
                'interpretation': 'Possible energy reorganization / magnetic stress buildup',
                'degraded': degraded,
                'degraded_reasons': degraded_reasons,
            }
            if degraded:
                result['interpretation'] = 'DIAGNOSTIC ONLY — ' + result['interpretation']
            return result

        # Inverse: recovery after flare?
        if slope_304 < FALLING_THRESHOLD and slope_211 > RISING_THRESHOLD:
            result = {
                'state': 'RECOVERY_STATE',
                'description': 'Coronal coupling recovering, chromospheric anchor releasing',
                'slope_193_304': slope_304,
                'slope_193_211': slope_211,
                'confidence': _min_confidence(trend_304['confidence'], trend_211['confidence']),
                'interpretation': 'Possible post-flare recovery / relaxation',
                'degraded': degraded,
                'degraded_reasons': degraded_reasons,
            }
            if degraded:
                result['interpretation'] = 'DIAGNOSTIC ONLY — ' + result['interpretation']
            return result

        return None
