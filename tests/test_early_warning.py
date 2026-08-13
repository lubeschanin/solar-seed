"""
Tests for Solar Early Warning System
"""

import pytest
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from early_warning import (
    classify_flare,
    assess_geomagnetic_risk,
    CouplingMonitor,
    fetch_json,
    FLARE_THRESHOLDS,
    compute_registration_shift,
    detect_coupling_break,
    store_coupling_reading,
)


class TestFlareClassification:
    """Test X-ray flux to flare class conversion."""

    def test_x_class_flare(self):
        """X-class flares: >= 10^-4 W/m²"""
        flare_class, severity = classify_flare(1e-4)
        assert flare_class == "X1.0"
        assert severity == 4

        flare_class, severity = classify_flare(5e-4)
        assert flare_class == "X5.0"
        assert severity == 4

    def test_m_class_flare(self):
        """M-class flares: >= 10^-5 W/m²"""
        flare_class, severity = classify_flare(1e-5)
        assert flare_class == "M1.0"
        assert severity == 3

        flare_class, severity = classify_flare(5e-5)
        assert flare_class == "M5.0"
        assert severity == 3

    def test_c_class_flare(self):
        """C-class flares: >= 10^-6 W/m²"""
        flare_class, severity = classify_flare(1e-6)
        assert flare_class == "C1.0"
        assert severity == 2

        flare_class, severity = classify_flare(3.5e-6)
        assert flare_class == "C3.5"
        assert severity == 2

    def test_b_class_flare(self):
        """B-class flares: >= 10^-7 W/m²"""
        flare_class, severity = classify_flare(1e-7)
        assert flare_class == "B1.0"
        assert severity == 1

        flare_class, severity = classify_flare(5.4e-7)
        assert flare_class == "B5.4"
        assert severity == 1

    def test_a_class_quiet(self):
        """A-class (quiet): < 10^-7 W/m²"""
        flare_class, severity = classify_flare(1e-8)
        assert flare_class == "A"
        assert severity == 0

    def test_threshold_boundaries(self):
        """Test exact threshold boundaries."""
        # Just below X-class
        flare_class, _ = classify_flare(9.99e-5)
        assert flare_class.startswith("M")

        # Just below M-class
        flare_class, _ = classify_flare(9.99e-6)
        assert flare_class.startswith("C")


class TestGeomagneticRisk:
    """Test solar wind risk assessment."""

    def test_quiet_conditions(self):
        """Quiet: Bz positive, low speed."""
        solar_wind = {
            'mag': {'bz': 5.0, 'bt': 6.0},
            'plasma': {'speed': 350, 'density': 5}
        }
        risk, level = assess_geomagnetic_risk(solar_wind)
        assert level == 0
        assert "QUIET" in risk

    def test_low_risk_moderate_bz(self):
        """Low risk: moderate southward Bz."""
        solar_wind = {
            'mag': {'bz': -3.0, 'bt': 5.0},
            'plasma': {'speed': 400, 'density': 8}
        }
        risk, level = assess_geomagnetic_risk(solar_wind)
        assert level >= 1
        assert "LOW" in risk or "southward" in risk.lower()

    def test_moderate_risk_strong_bz(self):
        """Moderate risk: strong southward Bz."""
        solar_wind = {
            'mag': {'bz': -7.0, 'bt': 10.0},
            'plasma': {'speed': 550, 'density': 10}  # Elevated speed for moderate risk
        }
        risk, level = assess_geomagnetic_risk(solar_wind)
        assert level >= 2

    def test_high_risk_extreme_conditions(self):
        """High risk: very strong southward Bz + high speed."""
        solar_wind = {
            'mag': {'bz': -15.0, 'bt': 20.0},
            'plasma': {'speed': 800, 'density': 25}
        }
        risk, level = assess_geomagnetic_risk(solar_wind)
        assert level == 3
        assert "HIGH" in risk

    def test_high_speed_contribution(self):
        """High solar wind speed increases risk."""
        base = {'mag': {'bz': -5.0}, 'plasma': {'speed': 400, 'density': 5}}
        fast = {'mag': {'bz': -5.0}, 'plasma': {'speed': 750, 'density': 5}}

        _, base_level = assess_geomagnetic_risk(base)
        _, fast_level = assess_geomagnetic_risk(fast)

        assert fast_level > base_level

    def test_missing_data(self):
        """Handle missing solar wind data gracefully."""
        risk, level = assess_geomagnetic_risk(None)
        assert level == 0
        assert "Unknown" in risk

        risk, level = assess_geomagnetic_risk({})
        assert level == 0


#: Baselines used across the tests in this module. Roughly the values measured
#: over six months of monitoring. Tests must never read the production
#: results/early_warning/baselines.json - results would then depend on how
#: recently someone ran `baselines --recompute`.
TEST_BASELINES = {
    '1k': {
        '193-211': {'mean': 0.80, 'std': 0.14},
        '193-304': {'mean': 0.18, 'std': 0.06},
    },
    '4k': {
        '193-211': {'mean': 0.80, 'std': 0.17},
        '193-304': {'mean': 0.11, 'std': 0.05},
    },
    '_meta': {'source': 'test fixture'},
}


def make_monitor(tmp_path, baselines=None):
    """CouplingMonitor with isolated history AND baseline files."""
    from solar_seed.monitoring.baselines import save_baselines

    baseline_path = tmp_path / 'baselines.json'
    save_baselines(baselines or TEST_BASELINES, baseline_path)
    return CouplingMonitor(
        history_file=tmp_path / 'history.json',
        baseline_file=baseline_path,
    )


class TestCouplingMonitor:
    """Test coupling residual tracking."""

    @pytest.fixture
    def monitor(self, tmp_path):
        """Create a fresh monitor with temp history and temp baselines."""
        return make_monitor(tmp_path)

    def test_baselines_defined(self, monitor):
        """Baselines resolve for the key pairs."""
        active = monitor.get_baselines('1k')
        assert '193-211' in active
        assert '193-304' in active
        assert active['193-211']['mean'] == 0.80
        assert active['193-211']['std'] == 0.14

    def test_baselines_come_from_file_not_class_constants(self, monitor):
        """The active baselines are the loaded ones, not the provisional table.

        The class constants are a cold-start fallback only. They had drifted
        far from the measured distributions - 193-304 at 4k was coded 0.32
        against a measured 0.107, which made 65% of all 4k readings fire ALERT.
        """
        assert monitor.baseline_source == 'test fixture'
        assert monitor.get_baselines('1k')['193-211']['mean'] != \
            monitor.BASELINES_1K['193-211']['mean']

    def test_provisional_fallback_when_no_file(self, tmp_path):
        """With no baseline file the provisional table is used and flagged."""
        m = CouplingMonitor(
            history_file=tmp_path / 'history.json',
            baseline_file=tmp_path / 'absent.json',
        )
        assert m.baseline_source == 'provisional'
        assert m.get_baselines('1k') == m.BASELINES_1K

    def test_get_baselines_method(self, monitor):
        """get_baselines returns the table for the requested resolution."""
        for pair, expected in TEST_BASELINES['1k'].items():
            assert monitor.get_baselines('1k')[pair] == expected
        for pair, expected in TEST_BASELINES['4k'].items():
            assert monitor.get_baselines('4k')[pair] == expected
        # Default should be 1k
        assert monitor.get_baselines()['193-211'] == TEST_BASELINES['1k']['193-211']

    def test_unmeasured_pairs_fall_back_to_provisional(self, monitor):
        """Pairs absent from the measured file still resolve.

        The monitor only samples 193-211 and 193-304, so 171-193 and 211-335
        never reach the sample threshold and keep their provisional values.
        """
        assert monitor.get_baselines('1k')['211-335'] == \
            monitor.BASELINES_1K['211-335']

    def test_residual_with_resolution_parameter(self, monitor):
        """compute_residual uses the baseline matching the resolution."""
        delta_mi = 0.18  # nominal at 1k, well above the 4k baseline

        result_1k = monitor.compute_residual('193-304', delta_mi, resolution='1k')
        assert result_1k['deviation_pct'] == pytest.approx(0.0, abs=0.01)

        result_4k = monitor.compute_residual('193-304', delta_mi, resolution='4k')
        assert result_4k['deviation_pct'] > 0.5  # far above the 4k baseline

    def test_residual_4k_status_classification(self, monitor):
        """Status classification with 4k baselines (193-304: 0.11 ± 0.05)."""
        at_z = lambda z: 0.11 + z * 0.05
        for z, expected in ((-0.5, 'NORMAL'), (-1.7, 'ELEVATED'),
                            (-2.5, 'WARNING'), (-3.5, 'ALERT')):
            assert monitor.compute_residual(
                '193-304', at_z(z), resolution='4k')['status'] == expected

    def test_deep_alert_can_be_out_of_reach_for_weak_pairs(self, monitor):
        """ΔMI is floored, so a weak pair may not span the ALERT threshold.

        193-304 at 4k has a measured baseline near 0.107 ± 0.051 and ΔMI
        cannot go below MIN_MI_THRESHOLD, which puts its floor at about
        -1.9 sigma - above the -3 sigma ALERT line. That is a fact about the
        pair's dynamic range, not a threshold to be lowered until every pair
        can alarm; the sudden-drop path still covers it.
        """
        from solar_seed.monitoring.constants import MIN_MI_THRESHOLD, Z_ALERT

        base = monitor.get_baselines('4k')['193-304']
        z_floor = (MIN_MI_THRESHOLD - base['mean']) / base['std']
        assert z_floor > Z_ALERT  # ALERT unreachable for this pair
        assert monitor.compute_residual(
            '193-304', MIN_MI_THRESHOLD, resolution='4k')['status'] != 'ALERT'

    # Status is decided in sigma. With the fixture baseline 0.80 ± 0.14:
    #   z = -1.5 -> 0.590   (ELEVATED)
    #   z = -2.0 -> 0.520   (WARNING)
    #   z = -3.0 -> 0.380   (ALERT)

    def test_residual_normal(self, monitor):
        """Normal coupling: at the baseline."""
        result = monitor.compute_residual('193-211', 0.80)
        assert result['residual'] == pytest.approx(0.0, abs=0.1)
        assert result['status'] == 'NORMAL'

    def test_residual_elevated(self, monitor):
        """Elevated: below -1.5 sigma."""
        result = monitor.compute_residual('193-211', 0.80 - 1.7 * 0.14)
        assert result['residual'] < -1.5
        assert result['status'] == 'ELEVATED'

    def test_residual_warning(self, monitor):
        """Warning: below -2 sigma."""
        result = monitor.compute_residual('193-211', 0.80 - 2.5 * 0.14)
        assert result['residual'] < -2.0
        assert result['status'] == 'WARNING'

    def test_residual_alert(self, monitor):
        """Alert: below -3 sigma (flare precursor)."""
        result = monitor.compute_residual('193-211', 0.80 - 3.5 * 0.14)
        assert result['residual'] < -3.0
        assert result['status'] == 'ALERT'

    def test_status_is_pair_independent_in_sigma(self, monitor):
        """The same z gives the same status for pairs with different spread.

        This is the whole point of the change: a fixed -25% meant -1.44 sigma
        for 193-211 but only -0.52 sigma for 193-304 at 4k, so one nominal
        threshold produced very different false-alarm rates per pair.
        """
        for pair in ('193-211', '193-304'):
            base = monitor.get_baselines('1k')[pair]
            at_z = lambda z: base['mean'] + z * base['std']
            assert monitor.compute_residual(pair, at_z(-0.5))['status'] == 'NORMAL'
            assert monitor.compute_residual(pair, at_z(-1.7))['status'] == 'ELEVATED'
            assert monitor.compute_residual(pair, at_z(-2.5))['status'] == 'WARNING'
            assert monitor.compute_residual(pair, at_z(-3.5))['status'] == 'ALERT'

    def test_residual_unknown_pair(self, monitor):
        """Unknown pair returns safe defaults."""
        result = monitor.compute_residual('999-888', 0.5)
        assert result['status'] == 'unknown'
        assert result['residual'] == 0

    def test_history_persistence(self, monitor):
        """History is saved and loaded correctly."""
        timestamp = datetime.now(timezone.utc).isoformat()
        coupling_data = {
            '193-211': {'delta_mi': 0.55, 'status': 'NORMAL'}
        }

        monitor.add_reading(timestamp, coupling_data)
        assert len(monitor.history) == 1

        # Create new monitor with same file
        monitor2 = CouplingMonitor(history_file=monitor.history_file)
        assert len(monitor2.history) == 1
        assert monitor2.history[0]['coupling']['193-211']['delta_mi'] == 0.55

    def test_history_limit(self, monitor):
        """History is limited to 144 entries (24 hours)."""
        for i in range(200):
            monitor.add_reading(f"2026-01-01T{i:02d}:00:00", {'test': i})

        assert len(monitor.history) == 144

    def test_trend_no_data(self, monitor):
        """No data returns NO_DATA status."""
        result = monitor.analyze_trend('193-211')
        assert result['trend'] == 'NO_DATA'
        assert result['n_points'] == 0

    def test_trend_collecting(self, monitor):
        """1-2 data points returns COLLECTING with reason."""
        monitor.add_reading("2026-01-01T10:00:00", {'193-211': {'delta_mi': 0.59}})
        result = monitor.analyze_trend('193-211')
        assert result['trend'] == 'COLLECTING'
        assert result['n_points'] == 1
        assert result['confidence'] == 'insufficient'
        assert 'Need 3 points' in result['reason']
        assert result['method'] == 'Theil-Sen'

    def test_trend_stable(self, monitor):
        """Stable trend: minimal change."""
        # Add 6 readings with stable values
        for i in range(6):
            monitor.add_reading(
                f"2026-01-01T{10+i}:00:00",
                {'193-211': {'delta_mi': 0.59 + (i % 2) * 0.005}}
            )

        result = monitor.analyze_trend('193-211')
        assert result['trend'] == 'STABLE'
        assert result['confidence'] == 'medium'
        assert result['n_points'] == 6
        # Check metadata
        assert result['method'] == 'Theil-Sen'
        assert result['window_min'] > 0  # Time span calculated
        assert 'window_max' in result

    def test_trend_declining(self, monitor):
        """Declining trend: significant decrease."""
        # Add readings with decreasing values
        values = [0.60, 0.55, 0.50, 0.45, 0.40, 0.35]
        for i, val in enumerate(values):
            monitor.add_reading(
                f"2026-01-01T{10+i}:00:00",
                {'193-211': {'delta_mi': val}}
            )

        result = monitor.analyze_trend('193-211')
        assert result['trend'] in ['DECLINING', 'ACCELERATING_DOWN']
        assert result['slope_pct_per_hour'] < 0
        assert 'acceleration' in result

    def test_trend_high_confidence(self, monitor):
        """High confidence with 9+ data points."""
        for i in range(10):
            monitor.add_reading(
                f"2026-01-01T{10+i}:00:00",
                {'193-211': {'delta_mi': 0.59}}
            )

        result = monitor.analyze_trend('193-211')
        assert result['confidence'] == 'high'
        assert result['n_points'] == 10

    def test_theil_sen_robust(self, monitor):
        """Theil-Sen slope is robust to outliers."""
        # Add readings with one outlier
        values = [0.50, 0.51, 0.52, 0.90, 0.54, 0.55]  # 0.90 is outlier
        for i, val in enumerate(values):
            monitor.add_reading(
                f"2026-01-01T{10+i}:00:00",
                {'193-211': {'delta_mi': val}}
            )

        result = monitor.analyze_trend('193-211')
        # Should still detect rising trend despite outlier
        assert result['slope_pct_per_hour'] > 0

    def test_transfer_state_detection(self, monitor):
        """Detect TRANSFER_STATE when 304 rises and 211 falls."""
        # Add readings with diverging trends at realistic 10-min cadence
        # (slope normalization uses the ACTUAL timestamp spacing)
        # 193-304: rising (0.07 -> 0.10)
        # 193-211: falling (0.59 -> 0.50)
        for i in range(8):
            minutes = i * 10
            monitor.add_reading(
                f"2026-01-01T{10 + minutes // 60}:{minutes % 60:02d}:00",
                {
                    '193-304': {'delta_mi': 0.07 + i * 0.005},  # Rising
                    '193-211': {'delta_mi': 0.59 - i * 0.015}   # Falling
                }
            )

        transfer = monitor.detect_transfer_state()
        assert transfer is not None
        assert transfer['state'] == 'TRANSFER_STATE'
        assert transfer['slope_193_304'] > 0
        assert transfer['slope_193_211'] < 0

    def test_no_transfer_state_when_both_stable(self, monitor):
        """No transfer state when both pairs are stable."""
        for i in range(8):
            monitor.add_reading(
                f"2026-01-01T{10+i}:00:00",
                {
                    '193-304': {'delta_mi': 0.07 + (i % 2) * 0.001},
                    '193-211': {'delta_mi': 0.59 + (i % 2) * 0.001}
                }
            )

        transfer = monitor.detect_transfer_state()
        assert transfer is None

    # TEST_BASELINES 193-211 @1k: mean 0.80, std 0.14
    # → depressed means below 0.80 - 1.5*0.14 = 0.59
    QUIET = 0.90
    LOW = 0.48

    def test_persistence_no_history(self, monitor):
        """No history = not persistent (min_frames=2 needs one previous frame)."""
        result = monitor.is_persistent_break(
            '193-211', current_is_break=True, current_delta_mi=self.LOW, min_frames=2)
        assert result is False

    @staticmethod
    def _fill(monitor, values, start_hour=0):
        """Append `values` as consecutive 10-minute readings for 193-211."""
        for i, v in enumerate(values):
            monitor.add_reading(
                f"2026-01-01T{start_hour + i // 6:02d}:{(i % 6) * 10:02d}:00",
                {'193-211': {'delta_mi': v}},
            )

    def test_persistence_previous_and_current_depressed(self, monitor):
        """Previous frame depressed and current depressed = persistent."""
        self._fill(monitor, [0.90, 0.91, 0.89, 0.90, 0.50])
        assert monitor.is_persistent_break(
            '193-211', current_is_break=True, current_delta_mi=self.LOW, min_frames=2) is True

    def test_persistence_rejects_single_frame_spike(self, monitor):
        """Previous frame still quiet = single-frame spike, not persistent.

        Regression guard against the inverted filter: with a rolling reference
        window, an isolated excursion *passed* precisely because the window
        preceding it was still undisturbed.
        """
        self._fill(monitor, [0.90, 0.91, 0.89, 0.90, 0.90])
        assert monitor.is_persistent_break(
            '193-211', current_is_break=True, current_delta_mi=self.LOW, min_frames=2) is False

    def test_persistence_survives_sustained_collapse(self, monitor):
        """A collapse longer than any plausible rolling window stays persistent.

        Regression guard for the inverted filter. Both earlier variants judged
        the frame against a window drawn from the recent past (60-min median,
        then a 12-frame pre-break window). Once such a window has filled with
        depressed values its median sits on the plateau and its spread shrinks,
        so no plateau value can be 2 sigma below it and every frame is vetoed
        as a "spike" - the exact inverse of the intent.

        The collapse here is deliberately longer than 12 frames: the previous
        test suite used four, which is why it passed while production, running
        on 144 frames of history, vetoed 265 of 297 frames of a real 44-hour
        depression (193-211, 11-13 Aug 2026).
        """
        self._fill(monitor, [0.90, 0.91, 0.89, 0.90] + [0.47] * 20)
        assert monitor.is_persistent_break(
            '193-211', current_is_break=True, current_delta_mi=self.LOW, min_frames=2) is True

    def test_persistence_real_plateau_confirmed(self, monitor):
        """Real 193-211 plateau of 12-13 Aug 2026 must confirm, not veto.

        These are measured values from monitoring.db. Against the quiet-Sun
        baseline they sit ~2.7 sigma low; against their own rolling median they
        are indistinguishable from noise (median 0.316, sigma 0.009).
        """
        self._fill(monitor, [0.318, 0.320, 0.322, 0.330, 0.400, 0.416, 0.380, 0.382])
        assert monitor.is_persistent_break(
            '193-211', current_is_break=True, current_delta_mi=0.407, min_frames=2) is True

    def test_persistence_real_spike_vetoed(self, monitor):
        """Real single-frame excursion of 11 Aug 2026 17:27 must be vetoed.

        ΔMI 0.345 sitting between 0.701 and 0.640 - a textbook spike that the
        inverted filter confirmed as "PERSISTENT (2+ frames)", because it
        judged the spike against the still-undisturbed window before it.

        Framed as production hit it: the spike is the previous frame and the
        current frame has already recovered to 0.640.
        """
        self._fill(monitor, [0.616, 0.578, 0.684, 0.701, 0.345])
        assert monitor.is_persistent_break(
            '193-211', current_is_break=True, current_delta_mi=0.640, min_frames=2) is False

    def test_persistence_three_frames(self, monitor):
        """Three frames required: both preceding frames must be depressed."""
        self._fill(monitor, [0.90, 0.91, 0.89, 0.90, 0.50, 0.48])
        assert monitor.is_persistent_break(
            '193-211', current_is_break=True, current_delta_mi=self.LOW, min_frames=3) is True

        # Only the most recent previous frame depressed → not persistent over 3
        monitor.history = []
        self._fill(monitor, [0.90, 0.91, 0.89, 0.90, 0.90, 0.50])
        assert monitor.is_persistent_break(
            '193-211', current_is_break=True, current_delta_mi=self.LOW, min_frames=3) is False

    def test_persistence_insufficient_history_for_min_frames(self, monitor):
        """Fewer previous frames than min_frames requires = not persistent."""
        self._fill(monitor, [0.50])
        assert monitor.is_persistent_break(
            '193-211', current_is_break=True, current_delta_mi=self.LOW, min_frames=3) is False

    def test_persistence_current_frame_is_checked(self, monitor):
        """A quiet current frame cannot be persistent, however depressed the past.

        The previous implementation never looked at the current value at all -
        it took only the boolean and inspected history.
        """
        self._fill(monitor, [0.90, 0.91, 0.89, 0.90, 0.47, 0.47])
        assert monitor.is_persistent_break(
            '193-211', current_is_break=True, current_delta_mi=self.QUIET, min_frames=2) is False

    def test_persistence_uses_resolution_baseline(self, monitor):
        """The 4k baseline table is selected when resolution='4k'."""
        # 193-304: 1k mean 0.18/std 0.06 → thr 0.09; 4k mean 0.11/std 0.05 → thr 0.035
        self._fill(monitor, [0.90] * 4)
        for i, v in enumerate([0.05, 0.05]):
            monitor.add_reading(
                f"2026-01-01T05:{i * 10:02d}:00", {'193-304': {'delta_mi': v}})
        assert monitor.is_persistent_break(
            '193-304', current_is_break=True, current_delta_mi=0.05,
            min_frames=2, resolution='1k') is True
        assert monitor.is_persistent_break(
            '193-304', current_is_break=True, current_delta_mi=0.05,
            min_frames=2, resolution='4k') is False

    def test_persistence_unknown_pair(self, monitor):
        """A pair with no baseline cannot be judged = not persistent."""
        assert monitor.is_persistent_break(
            '94-131', current_is_break=True, current_delta_mi=0.01, min_frames=2) is False

    def test_persistence_not_current_break(self, monitor):
        """If current is not a break, return False immediately."""
        self._fill(monitor, [0.90, 0.91, 0.89, 0.90, 0.50])
        assert monitor.is_persistent_break(
            '193-211', current_is_break=False, current_delta_mi=self.LOW, min_frames=2) is False


class TestSuddenDropDetector:
    """Test sudden drop detection for pre-flare warnings.

    The reference level is the MEDIAN of the lookback window. It used to be the
    maximum, which sits systematically above the typical level (for 193-211 the
    measured sigma/mu is 0.24, so max-of-three lands ~20% high) and made an
    ordinary reading look like a 15% drop. That produced 69% of all stored
    predictions.

    Severity is graded in sigma below the reference, not in percent. With the
    fixture baseline 0.70 ± 0.14 and a window median of 0.90:
        MODERATE (>= 1.25 sigma) -> reading <= 0.725
        SEVERE   (>= 2.50 sigma) -> reading <= 0.550
    """

    #: Baseline low enough that the drops below stay ABOVE it - the whole
    #: point of the sudden-drop detector is to catch a fall that the absolute
    #: threshold would miss.
    BASELINES = {
        '1k': {'193-211': {'mean': 0.70, 'std': 0.14}},
        '4k': {'193-211': {'mean': 0.70, 'std': 0.14}},
        '_meta': {'source': 'test fixture'},
    }

    @pytest.fixture
    def monitor(self, tmp_path):
        """Create monitor with typical history (median of window = 0.90)."""
        m = make_monitor(tmp_path, self.BASELINES)
        m.history = [
            {'timestamp': '2026-01-01T10:00:00', 'coupling': {'193-211': {'delta_mi': 0.90}}},
            {'timestamp': '2026-01-01T10:10:00', 'coupling': {'193-211': {'delta_mi': 0.95}}},
            {'timestamp': '2026-01-01T10:20:00', 'coupling': {'193-211': {'delta_mi': 0.88}}},
        ]
        return m

    def test_reference_is_median_not_max(self, monitor):
        """The reference is the window median, not its maximum."""
        result = monitor.compute_residual('193-211', 0.90)
        assert result['sudden_drop']['reference_value'] == pytest.approx(0.90)

    def test_no_drop_normal_status(self, monitor):
        """No significant drop = NORMAL status."""
        result = monitor.compute_residual('193-211', 0.90)
        assert result['sudden_drop']['sudden_drop'] is False
        # Status based on absolute threshold (0.90 > baseline 0.59)
        assert result['status'] == 'NORMAL'

    def test_typical_reading_is_not_a_drop(self, monitor):
        """A reading at the window's own level must not register as a drop.

        Regression guard for the max-reference bias: with max(0.95) as
        reference, 0.88 - a value already present in the window - came out as
        a -7% "drop", and anything mildly below it crossed -15%.
        """
        result = monitor.compute_residual('193-211', 0.88)
        assert result['sudden_drop']['sudden_drop'] is False

    def test_moderate_drop_detected(self, monitor):
        """1.25-2.5 sigma below the median = MODERATE severity."""
        # 0.90 - 1.7*0.14 = 0.662
        result = monitor.compute_residual('193-211', 0.662)
        assert result['sudden_drop']['sudden_drop'] is True
        assert result['sudden_drop']['severity'] == 'MODERATE'
        assert result['sudden_drop']['drop_sigma'] == pytest.approx(1.7, abs=0.05)

    def test_severe_drop_detected(self, monitor):
        """2.5+ sigma below the median = SEVERE severity."""
        # 0.90 - 3.0*0.14 = 0.48
        result = monitor.compute_residual('193-211', 0.48)
        assert result['sudden_drop']['sudden_drop'] is True
        assert result['sudden_drop']['severity'] == 'SEVERE'

    def test_sudden_drop_triggers_elevated_status(self, monitor):
        """A sharp drop raises ELEVATED even while the level is nominal."""
        # 0.68 is only -0.14 sigma from the 0.70 baseline (NORMAL on level),
        # but 1.57 sigma below the recent median of 0.90.
        result = monitor.compute_residual('193-211', 0.68)
        assert result['residual'] > -1.5          # nominal by absolute level
        assert result['sudden_drop']['sudden_drop'] is True
        assert result['status'] == 'ELEVATED'

    def test_no_severity_without_baseline_sigma(self, monitor):
        """Without a sigma the drop cannot be graded, so nothing fires."""
        info = monitor.detect_sudden_drop('193-211', 0.40, baseline_std=None)
        assert info['sudden_drop'] is False
        assert info['severity'] is None
        assert info['drop_pct'] < 0  # still reported for context

    def test_m3_preflare_scenario(self, monitor):
        """Simulate the M3 pre-flare drop.

        Timeline 0.917 → 0.953 → 0.875 → 0.714. Against the median (0.917) the
        drop is 0.203 bits = 1.45 sigma, so MODERATE. This is the case that
        pinned the MODERATE threshold at 1.25 rather than 1.5 sigma: the
        documented precursor must stay detectable. Both severities map to
        status ELEVATED, so the operational outcome is unchanged.
        """
        monitor.history = [
            {'timestamp': '2026-01-11T21:38:00', 'coupling': {'193-211': {'delta_mi': 0.917}}},
            {'timestamp': '2026-01-11T21:48:00', 'coupling': {'193-211': {'delta_mi': 0.953}}},
            {'timestamp': '2026-01-11T21:59:00', 'coupling': {'193-211': {'delta_mi': 0.875}}},
        ]
        result = monitor.compute_residual('193-211', 0.714)

        assert result['sudden_drop']['sudden_drop'] is True
        assert result['sudden_drop']['severity'] == 'MODERATE'
        # 0.203 bits below the median = 1.45 sigma at the fixture's 0.14
        assert result['sudden_drop']['drop_sigma'] == pytest.approx(1.45, abs=0.05)

        # Still triggers ELEVATED (pre-flare warning!)
        assert result['status'] == 'ELEVATED'

    def test_empty_history_no_crash(self, tmp_path):
        """Empty history should not crash."""
        m = make_monitor(tmp_path, self.BASELINES)
        m.history = []
        result = m.compute_residual('193-211', 0.5)
        assert result['sudden_drop']['sudden_drop'] is False
        assert 'reason' in result['sudden_drop']


class TestAlertThresholds:
    """Test that alert thresholds match paper findings."""

    def test_deep_flare_collapse_reaches_alert(self, tmp_path):
        """The deep end of the documented -25%..-47% flare collapse alarms.

        For 193-211 the measured sigma/mu is ~0.17, so -47% is about -2.7
        sigma (WARNING) and a -55% collapse clears the -3 sigma ALERT line.
        """
        monitor = make_monitor(tmp_path)
        baseline = monitor.get_baselines('1k')['193-211']['mean']

        assert monitor.compute_residual(
            '193-211', baseline * 0.45)['status'] == 'ALERT'
        assert monitor.compute_residual(
            '193-211', baseline * 0.53)['status'] == 'WARNING'

    def test_shallow_collapse_needs_the_sudden_drop_path(self, tmp_path):
        """A -25% dip is only -1.4 sigma and the level alone does not flag it.

        Worth stating plainly: switching to sigma means the shallow end of the
        documented flare range no longer alarms on absolute level. It is still
        caught, but by the sudden-drop detector - a 25% fall from a steady run
        is 1.4 sigma below the recent median, past the 1.25 MODERATE line. The
        two paths are complementary, and dropping either would open a gap.
        """
        monitor = make_monitor(tmp_path)
        base = monitor.get_baselines('1k')['193-211']
        collapsed = base['mean'] * 0.75

        # Absolute level alone: not flagged
        assert monitor.compute_residual('193-211', collapsed)['status'] == 'NORMAL'

        # Same value after a steady run at baseline: caught as a sudden drop
        monitor.history = [
            {'timestamp': f'2026-03-01T10:{i * 10:02d}:00',
             'coupling': {'193-211': {'delta_mi': base['mean']}}}
            for i in range(3)
        ]
        result = monitor.compute_residual('193-211', collapsed)
        assert result['sudden_drop']['severity'] == 'MODERATE'
        assert result['status'] == 'ELEVATED'

    def test_deep_collapse_still_classified_not_discarded(self, tmp_path):
        """A -70% collapse must reach ALERT, not be dropped as a data error."""
        monitor = make_monitor(tmp_path)
        baseline = monitor.get_baselines('1k')['193-211']['mean']

        result = monitor.compute_residual('193-211', baseline * 0.30)
        assert result['status'] == 'ALERT'

    def test_pre_flare_detection_window(self, tmp_path):
        """A destabilisation below -1.5 sigma must be flagged before the peak."""
        monitor = make_monitor(tmp_path)
        base = monitor.get_baselines('1k')['193-211']

        pre_flare = base['mean'] - 1.7 * base['std']
        result = monitor.compute_residual('193-211', pre_flare)
        assert result['status'] in ['WARNING', 'ELEVATED']


class TestDataFetching:
    """Test data fetching utilities."""

    def test_fetch_json_invalid_url(self):
        """Invalid URL returns None."""
        result = fetch_json("https://invalid.example.com/nonexistent", timeout=5)
        assert result is None

    @patch('early_warning.urlopen')
    def test_fetch_json_success(self, mock_urlopen):
        """Successful fetch returns parsed JSON."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "ok"}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = fetch_json("https://example.com/api")
        assert result == {"status": "ok"}

    @patch('early_warning.urlopen')
    def test_fetch_json_malformed(self, mock_urlopen):
        """Malformed JSON returns None."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'not valid json'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = fetch_json("https://example.com/api")
        assert result is None


class TestIntegration:
    """Integration tests (require network)."""

    @pytest.mark.integration
    def test_goes_xray_live(self):
        """Test live GOES X-ray data fetch."""
        from early_warning import get_goes_xray

        result = get_goes_xray()
        # Should return data or None (network issues)
        if result:
            assert 'flux' in result
            assert 'flare_class' in result
            assert result['flux'] >= 0

    @pytest.mark.integration
    def test_dscovr_live(self):
        """Test live DSCOVR data fetch."""
        from early_warning import get_dscovr_solar_wind

        result = get_dscovr_solar_wind()
        if result:
            assert 'plasma' in result or 'mag' in result


class TestValidationChecks:
    """Test reviewer-proof validation functions."""

    @pytest.fixture
    def temp_monitor(self):
        """Create a monitor with temp file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        monitor = CouplingMonitor(history_file=temp_path)
        yield monitor
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    def test_registration_shift_aligned(self):
        """Test registration check with aligned images."""
        import numpy as np

        # Create identical images with random structure
        np.random.seed(42)
        img = np.random.randn(512, 512).astype(np.float32)

        result = compute_registration_shift(img, img)
        assert result['is_centered'] == True
        assert result['shift_pixels'] < 2  # Should be near zero
        assert result['dx'] == 0 or abs(result['dx']) <= 1
        assert result['dy'] == 0 or abs(result['dy']) <= 1

    def test_registration_shift_misaligned(self):
        """Test registration check with shifted images."""
        import numpy as np
        from scipy.ndimage import shift as ndshift

        # Create image with structure
        np.random.seed(42)
        img1 = np.random.randn(512, 512).astype(np.float32)

        # Shift by 20 pixels
        img2 = ndshift(img1, (20, 15), mode='constant', cval=0)

        result = compute_registration_shift(img1, img2, max_shift=10)
        # Should detect the large shift
        assert result['shift_pixels'] > 10
        assert result['is_centered'] == False

    def test_coupling_break_detection_normal(self, temp_monitor):
        """Test break detection with normal values."""
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)

        # Add history with some variation (not perfectly stable)
        values = [0.55, 0.58, 0.62, 0.57, 0.60, 0.59, 0.61, 0.56, 0.58, 0.60]
        for i, val in enumerate(values):
            ts = (now - timedelta(minutes=5*i)).isoformat()
            temp_monitor.add_reading(ts, {'193-211': {'delta_mi': val}})

        # Test with value within normal range (near median ~0.585)
        result = detect_coupling_break('193-211', 0.57, temp_monitor)
        assert result['is_break'] == False
        assert result['n_points'] >= 3

    def test_coupling_break_detection_break(self, temp_monitor):
        """Test break detection with anomalous value."""
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)

        # Add history with realistic variation
        values = [0.55, 0.58, 0.62, 0.57, 0.60, 0.59, 0.61, 0.56, 0.58, 0.60]
        for i, val in enumerate(values):
            ts = (now - timedelta(minutes=5*i)).isoformat()
            temp_monitor.add_reading(ts, {'193-211': {'delta_mi': val}})

        # Test with very low value (break) - well below any history value
        result = detect_coupling_break('193-211', 0.30, temp_monitor)
        assert result['is_break'] == True
        assert result['z_mad'] > 2  # Significant positive z_mad = MADs below median
        assert 'median' in result
        assert 'threshold' in result

    def test_coupling_break_criterion_format(self, temp_monitor):
        """Test break detection returns proper criterion string."""
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)

        for i in range(5):
            ts = (now - timedelta(minutes=5*i)).isoformat()
            temp_monitor.add_reading(ts, {'193-211': {'delta_mi': 0.59}})

        result = detect_coupling_break('193-211', 0.55, temp_monitor)
        assert 'criterion' in result
        assert 'MAD' in result['criterion']
        assert 'median' in result['criterion']

    def test_coupling_break_insufficient_data(self, temp_monitor):
        """Test break detection with insufficient data."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        # Only add 1 reading (recent)
        temp_monitor.add_reading(now.isoformat(), {'193-211': {'delta_mi': 0.59}})

        result = detect_coupling_break('193-211', 0.50, temp_monitor)
        assert result['is_break'] == False
        assert 'Insufficient' in result.get('reason', '')


class TestStoreCouplingReading:
    """Test store_coupling_reading quality_ok and trigger_kind logic."""

    @pytest.fixture
    def mock_db(self):
        """Create a temp MonitoringDB and patch get_monitoring_db."""
        from solar_seed.monitoring.db import MonitoringDB
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        db = MonitoringDB(db_path=db_path)
        with patch('early_warning.get_monitoring_db', return_value=db):
            yield db
        db.close()
        db_path.unlink()

    def test_trigger_kind_elevated_threshold(self, mock_db):
        """A z between -1.5 and -2 gets trigger_kind=THRESHOLD(-1.5)."""
        coupling = {
            '193-211': {
                'delta_mi': 0.53,
                'status': 'ELEVATED',
                'deviation_pct': -0.12,
                'residual': -1.7,
                'trend': 'STABLE',
            },
            '_quality': {'resolution': '1024x1024', 'resolution_class': '1k'},
        }

        store_coupling_reading("2026-01-15T12:00:00", coupling)

        cursor = mock_db.conn.cursor()
        cursor.execute("SELECT trigger_kind, trigger_value, trigger_threshold FROM predictions")
        row = cursor.fetchone()
        assert row['trigger_kind'] == 'THRESHOLD'
        assert row['trigger_value'] == pytest.approx(-1.7)
        assert row['trigger_threshold'] == pytest.approx(-1.5)

    def test_trigger_kind_all_levels(self, mock_db):
        """Each sigma band maps to the matching trigger_threshold."""
        cases = [
            (-3.5, -3.0, 'ALERT'),
            (-2.5, -2.0, 'WARNING'),
            (-1.7, -1.5, 'ELEVATED'),
        ]
        for i, (residual, expected_threshold, status) in enumerate(cases):
            # Hours apart so each opens its own episode rather than folding
            ts = f"2026-01-15T{12 + i * 2}:00:00"
            coupling = {
                '193-211': {
                    'delta_mi': 0.50,
                    'status': status,
                    'deviation_pct': -0.3,
                    'residual': residual,
                    'trend': 'STABLE',
                },
                '_quality': {'resolution': '1024x1024', 'resolution_class': '1k'},
            }
            store_coupling_reading(ts, coupling)

        cursor = mock_db.conn.cursor()
        cursor.execute("SELECT trigger_threshold FROM predictions ORDER BY prediction_time")
        rows = cursor.fetchall()
        assert len(rows) == 3
        for row, (_, expected, _) in zip(rows, cases):
            assert row['trigger_threshold'] == pytest.approx(expected)

    def test_quality_ok_stored(self, mock_db):
        """quality_ok, robustness_score, sync_delta_s passed through to DB."""
        coupling = {
            '193-211': {
                'delta_mi': 0.59,
                'status': 'NORMAL',
                'trend': 'STABLE',
            },
            '_quality': {'resolution': '1024x1024', 'time_spread_sec': 8.5},
            '_validation': {
                'robustness_checks': {
                    '193-211': {'is_robust': True, 'change_pct': 2.1},
                },
            },
        }

        store_coupling_reading("2026-01-15T12:00:00", coupling)

        cursor = mock_db.conn.cursor()
        cursor.execute("SELECT quality_ok, robustness_score, sync_delta_s FROM coupling_measurements")
        row = cursor.fetchone()
        assert row['quality_ok'] == 1
        assert row['robustness_score'] == pytest.approx(2.1)
        assert row['sync_delta_s'] == pytest.approx(8.5)

    def test_quality_ok_false_on_data_error(self, mock_db):
        """DATA_ERROR status sets quality_ok=False."""
        coupling = {
            '193-211': {
                'delta_mi': 0.0,
                'status': 'DATA_ERROR',
            },
            '_quality': {'resolution': '1024x1024'},
        }

        store_coupling_reading("2026-01-15T12:00:00", coupling)

        cursor = mock_db.conn.cursor()
        cursor.execute("SELECT quality_ok FROM coupling_measurements")
        row = cursor.fetchone()
        assert row['quality_ok'] == 0

    def test_quality_ok_false_on_failed_robustness(self, mock_db):
        """Failed robustness check sets quality_ok=False."""
        coupling = {
            '193-211': {
                'delta_mi': 0.59,
                'status': 'NORMAL',
            },
            '_quality': {'resolution': '1024x1024', 'time_spread_sec': 10.0},
            '_validation': {
                'robustness_checks': {
                    '193-211': {'is_robust': False, 'change_pct': 35.0},
                },
            },
        }

        store_coupling_reading("2026-01-15T12:00:00", coupling)

        cursor = mock_db.conn.cursor()
        cursor.execute("SELECT quality_ok, robustness_score FROM coupling_measurements")
        row = cursor.fetchone()
        assert row['quality_ok'] == 0
        assert row['robustness_score'] == pytest.approx(35.0)
