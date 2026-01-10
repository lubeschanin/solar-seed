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

    def test_trend_insufficient_data(self, monitor):
        """Trend requires at least 3 data points."""
        result = monitor.analyze_trend('193-211')
        assert result['trend'] == 'insufficient_data'

    def test_trend_stable(self, monitor):
        """Stable trend: minimal change."""
        # Add 6 readings with stable values
        for i in range(6):
            monitor.add_reading(
                f"2026-01-01T{10+i}:00:00",
                {'193-211': {'delta_mi': 0.59 + (i % 2) * 0.01}}
            )

        result = monitor.analyze_trend('193-211')
        assert result['trend'] == 'STABLE'

    def test_trend_dropping(self, monitor):
        """Dropping trend: significant decrease."""
        # Add readings with decreasing values
        values = [0.60, 0.55, 0.50, 0.45, 0.40, 0.35]
        for i, val in enumerate(values):
            monitor.add_reading(
                f"2026-01-01T{10+i}:00:00",
                {'193-211': {'delta_mi': val}}
            )

        result = monitor.analyze_trend('193-211')
        assert result['trend'] in ['DROPPING', 'DECLINING']
        assert result['slope_pct_per_hour'] < 0


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
