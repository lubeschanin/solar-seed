"""
Coupling Monitor
================

Track ΔMI coupling residuals over time for pre-flare detection.

Based on findings from Lubeschanin et al. (2026):
- Adjacent temperature layers (193-211 Å) show strongest coupling (0.59 ± 0.12 bits)
- Flare analysis showed -25% to -47% reduction during flares
- Chromospheric anchor (304 Å) shows highest temporal stability
"""

import json
from pathlib import Path
from datetime import datetime


class CouplingMonitor:
    """Track coupling residuals over time for pre-flare detection."""

    # Baseline values by resolution
    # 1k: From synoptic data (spatial aliasing affects 304Å pairs)
    # 4k: From JSOC full-resolution backfill (accurate MI values)
    BASELINES_1K = {
        '193-211': {'mean': 0.59, 'std': 0.12},   # Scale-invariant
        '193-304': {'mean': 0.07, 'std': 0.02},   # Underestimated at 1k
        '171-193': {'mean': 0.17, 'std': 0.04},
        '211-335': {'mean': 0.28, 'std': 0.06},
    }

    BASELINES_4K = {
        '193-211': {'mean': 1.03, 'std': 0.31},   # ~1.7x higher than 1k
        '193-304': {'mean': 0.32, 'std': 0.12},   # ~4.5x higher than 1k
        '171-193': {'mean': 0.29, 'std': 0.07},   # Estimated 1.7x scaling
        '211-335': {'mean': 0.48, 'std': 0.10},   # Estimated 1.7x scaling
    }

    # Legacy alias for backwards compatibility
    BASELINES = BASELINES_1K

    # Flare analysis showed -25% to -47% reduction during flares
    ALERT_THRESHOLD = -0.25  # 25% below baseline triggers warning

    def __init__(self, history_file: Path = None):
        self.history_file = history_file or Path("results/early_warning/coupling_history.json")
        self.history = self._load_history()

    def _load_history(self) -> list:
        """Load coupling history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def _save_history(self):
        """Save coupling history to file."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        # Keep last 24 hours (144 entries at 10min intervals)
        self.history = self.history[-144:]
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f)

    def is_persistent_break(self, pair: str, current_is_break: bool, min_frames: int = 2) -> bool:
        """
        Check if a break persists for min_frames consecutive readings.

        Anti-spike filter: Only confirm break if it persists across multiple
        frames. At 10-min cadence, min_frames=2 means 20 minutes persistence.

        Args:
            pair: Channel pair (e.g. '193-211')
            current_is_break: Whether current frame shows a break
            min_frames: Minimum consecutive frames required (default: 2)

        Returns:
            True if break is persistent, False if likely spike/artifact
        """
        if not current_is_break:
            return False

        # Check history for previous break detections
        # Note: break_detected is stored in coupling_measurements when we add it
        pair_history = [
            h for h in self.history[-min_frames:]
            if pair in h.get('coupling', {})
        ]

        if len(pair_history) < min_frames - 1:
            # Not enough history, can't confirm persistence
            return False

        # Check if previous frames also had breaks
        # Use z_mad > 2.0 (break DETECTION threshold), not is_break (which may be vetoed)
        # This avoids catch-22 where vetoed breaks prevent future confirmations
        BREAK_THRESHOLD = 2.0
        previous_breaks = sum(
            1 for h in pair_history[-(min_frames-1):]
            if abs(h.get('coupling', {}).get(pair, {}).get('z_mad', 0)) > BREAK_THRESHOLD
        )

        return previous_breaks >= min_frames - 1

    def detect_sudden_drop(self, pair: str, delta_mi: float, lookback: int = 3) -> dict:
        """
        Detect sudden relative drop in coupling.

        This catches pre-flare drops that are still above baseline but represent
        a significant decrease from recent readings.

        Args:
            pair: Channel pair (e.g. '193-211')
            delta_mi: Current ΔMI value
            lookback: Number of previous readings to compare (default: 3 = ~30 min)

        Returns:
            Dict with drop detection results
        """
        pair_history = [
            h for h in self.history
            if pair in h.get('coupling', {})
        ]

        if len(pair_history) < lookback:
            return {
                'sudden_drop': False,
                'drop_pct': 0,
                'reference_value': None,
                'reason': f'Not enough history ({len(pair_history)}/{lookback})'
            }

        # Get recent values (excluding current)
        recent_values = [
            h['coupling'][pair]['delta_mi']
            for h in pair_history[-lookback:]
        ]

        # Use max of recent values as reference (captures the "normal" level)
        reference = max(recent_values)

        # Calculate drop percentage
        if reference > 0:
            drop_pct = (delta_mi - reference) / reference
        else:
            drop_pct = 0

        # Thresholds for sudden drop detection
        SUDDEN_DROP_THRESHOLD = -0.15  # 15% drop from recent max
        SEVERE_DROP_THRESHOLD = -0.25  # 25% drop = more urgent

        if drop_pct < SEVERE_DROP_THRESHOLD:
            sudden_drop = True
            severity = 'SEVERE'
        elif drop_pct < SUDDEN_DROP_THRESHOLD:
            sudden_drop = True
            severity = 'MODERATE'
        else:
            sudden_drop = False
            severity = None

        return {
            'sudden_drop': sudden_drop,
            'drop_pct': drop_pct,
            'reference_value': reference,
            'current_value': delta_mi,
            'severity': severity,
            'lookback_minutes': lookback * 10,  # Assuming 10min intervals
        }

    def get_baselines(self, resolution: str = '1k') -> dict:
        """Get baselines for the specified resolution."""
        if resolution == '4k':
            return self.BASELINES_4K
        return self.BASELINES_1K

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
        deviation_pct = (delta_mi - baseline['mean']) / baseline['mean']

        # Check for sudden drop (relative to recent readings)
        drop_info = self.detect_sudden_drop(pair, delta_mi)

        # Determine status - combine absolute threshold AND sudden drop
        if deviation_pct < self.ALERT_THRESHOLD:
            status = 'ALERT'
        elif deviation_pct < -0.15:
            status = 'WARNING'
        elif deviation_pct < -0.10:
            status = 'ELEVATED'
        elif drop_info['sudden_drop'] and drop_info['severity'] == 'SEVERE':
            # Sudden severe drop even if above baseline
            status = 'ELEVATED'
        elif drop_info['sudden_drop'] and drop_info['severity'] == 'MODERATE':
            # Sudden moderate drop
            status = 'ELEVATED'
        else:
            status = 'NORMAL'

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
        pair_history = [h for h in self.history if pair in h.get('coupling', {})]
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
        values = [h['coupling'][pair]['delta_mi'] for h in recent]
        n = len(values)

        # Calculate actual time span from timestamps
        try:
            t_first = datetime.fromisoformat(recent[0]['timestamp'].replace('Z', '+00:00'))
            t_last = datetime.fromisoformat(recent[-1]['timestamp'].replace('Z', '+00:00'))
            window_min = (t_last - t_first).total_seconds() / 60
        except (KeyError, ValueError, AttributeError):
            window_min = n * 10  # Fallback: assume 10min intervals

        # Robust Theil-Sen slope
        slope = self._theil_sen_slope(values)

        # Mean value for normalization
        y_mean = sum(values) / n if n > 0 else 1

        # Normalize slope to % per hour (assuming 10min intervals)
        slope_per_hour = slope * 6 / y_mean * 100 if y_mean else 0

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
        """Add a new coupling reading to history."""
        self.history.append({
            'timestamp': timestamp,
            'coupling': coupling_data
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
                'confidence': min(trend_304['confidence'], trend_211['confidence']),
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
                'confidence': min(trend_304['confidence'], trend_211['confidence']),
                'interpretation': 'Possible post-flare recovery / relaxation',
                'degraded': degraded,
                'degraded_reasons': degraded_reasons,
            }
            if degraded:
                result['interpretation'] = 'DIAGNOSTIC ONLY — ' + result['interpretation']
            return result

        return None
