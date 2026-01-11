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


class TestCouplingMonitor:
    """Test coupling residual tracking."""

    @pytest.fixture
    def monitor(self):
        """Create a fresh monitor with temp file."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        monitor = CouplingMonitor(history_file=temp_path)
        yield monitor
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    def test_baselines_defined(self, monitor):
        """Verify baseline values exist for key pairs."""
        assert '193-211' in monitor.BASELINES
        assert '193-304' in monitor.BASELINES
        assert monitor.BASELINES['193-211']['mean'] == 0.59
        assert monitor.BASELINES['193-211']['std'] == 0.12

    def test_residual_normal(self, monitor):
        """Normal coupling: within 1σ of baseline."""
        # 193-211 baseline: 0.59 ± 0.12
        result = monitor.compute_residual('193-211', 0.59)
        assert result['residual'] == pytest.approx(0.0, abs=0.1)
        assert result['status'] == 'NORMAL'

    def test_residual_elevated(self, monitor):
        """Elevated: 10-15% below baseline."""
        # 193-211: 0.59 - 12% = 0.52
        result = monitor.compute_residual('193-211', 0.52)
        assert result['deviation_pct'] < -0.10
        assert result['status'] in ['ELEVATED', 'WARNING']

    def test_residual_warning(self, monitor):
        """Warning: 15-25% below baseline."""
        # 193-211: 0.59 - 20% = 0.47
        result = monitor.compute_residual('193-211', 0.47)
        assert result['deviation_pct'] < -0.15
        assert result['status'] == 'WARNING'

    def test_residual_alert(self, monitor):
        """Alert: >25% below baseline (flare precursor)."""
        # 193-211: 0.59 - 30% = 0.41
        result = monitor.compute_residual('193-211', 0.41)
        assert result['deviation_pct'] < -0.25
        assert result['status'] == 'ALERT'

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
        # Add readings with diverging trends
        # 193-304: rising (0.07 -> 0.10)
        # 193-211: falling (0.59 -> 0.50)
        for i in range(8):
            monitor.add_reading(
                f"2026-01-01T{10+i}:00:00",
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

    def test_persistence_no_history(self, monitor):
        """No history = not persistent."""
        result = monitor.is_persistent_break('193-211', current_is_break=True, min_frames=2)
        assert result is False

    def test_persistence_single_frame(self, monitor):
        """Single previous frame with break = not persistent (need 2)."""
        # Add one reading with z_mad > 2.0 (break detected)
        monitor.add_reading(
            "2026-01-01T10:00:00",
            {'193-211': {'delta_mi': 0.5, 'z_mad': 5.0}}  # Break
        )
        result = monitor.is_persistent_break('193-211', current_is_break=True, min_frames=2)
        # Need 1 previous break (min_frames-1=1), have 1 → persistent
        assert result is True

    def test_persistence_previous_no_break(self, monitor):
        """Previous frame had no break = not persistent."""
        # Add reading WITHOUT break (z_mad < 2.0)
        monitor.add_reading(
            "2026-01-01T10:00:00",
            {'193-211': {'delta_mi': 0.5, 'z_mad': 1.0}}  # No break
        )
        result = monitor.is_persistent_break('193-211', current_is_break=True, min_frames=2)
        assert result is False

    def test_persistence_vetoed_still_counts(self, monitor):
        """Vetoed break (is_break=False but z_mad>2) should still count for persistence."""
        # Add reading where break was DETECTED but VETOED
        # This tests the fix: we check z_mad, not is_break
        monitor.add_reading(
            "2026-01-01T10:00:00",
            {'193-211': {'delta_mi': 0.5, 'z_mad': 10.0, 'is_break': False, 'break_vetoed': 'spike'}}
        )
        result = monitor.is_persistent_break('193-211', current_is_break=True, min_frames=2)
        # z_mad > 2.0 → counts as break for persistence
        assert result is True

    def test_persistence_three_frames(self, monitor):
        """Three frames required: need 2 previous breaks."""
        # Add two readings with breaks
        monitor.add_reading("2026-01-01T10:00:00", {'193-211': {'z_mad': 5.0}})
        monitor.add_reading("2026-01-01T10:10:00", {'193-211': {'z_mad': 6.0}})

        result = monitor.is_persistent_break('193-211', current_is_break=True, min_frames=3)
        # Need 2 previous breaks (min_frames-1=2), have 2 → persistent
        assert result is True

    def test_persistence_not_current_break(self, monitor):
        """If current is not a break, return False immediately."""
        monitor.add_reading("2026-01-01T10:00:00", {'193-211': {'z_mad': 5.0}})
        result = monitor.is_persistent_break('193-211', current_is_break=False, min_frames=2)
        assert result is False


class TestAlertThresholds:
    """Test that alert thresholds match paper findings."""

    def test_flare_coupling_reduction(self):
        """Paper shows 25-47% coupling reduction during flares."""
        monitor = CouplingMonitor()

        # Simulate flare-level reduction (30% below baseline)
        baseline = monitor.BASELINES['193-211']['mean']
        flare_value = baseline * 0.70  # 30% reduction

        result = monitor.compute_residual('193-211', flare_value)
        assert result['status'] == 'ALERT'

    def test_pre_flare_detection_window(self):
        """Coupling anomaly should trigger before flare peak."""
        monitor = CouplingMonitor()

        # 15% reduction should trigger WARNING
        baseline = monitor.BASELINES['193-211']['mean']
        pre_flare = baseline * 0.85

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
            assert result['flux'] > 0

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
