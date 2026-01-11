"""
Tests for Phase Classification Logic
====================================

Tests the core phase classification functions that determine
BASELINE, ACTIVE, PRE-FLARE, POST-EVENT, RECOVERY, ELEVATED-QUIET states.
"""

import pytest
from solar_seed.monitoring.constants import (
    Phase,
    DivergenceType,
    classify_phase_goes_only,
    classify_phase_experimental,
    classify_phase_parallel,
    classify_divergence_type,
)


# =============================================================================
# GOES-ONLY CLASSIFICATION
# =============================================================================

class TestClassifyPhaseGoesOnly:
    """Test GOES-only phase classification (traditional approach)."""

    def test_no_goes_data_returns_baseline(self):
        """No GOES data should return BASELINE."""
        phase, reason = classify_phase_goes_only(goes_flux=None)
        assert phase == Phase.BASELINE
        assert "No GOES" in reason

    def test_m_class_flare_returns_active(self):
        """M-class flux (>=1e-5) should return ACTIVE."""
        phase, reason = classify_phase_goes_only(goes_flux=1e-5, goes_class="M1.0")
        assert phase == Phase.ACTIVE
        assert "M/X-class" in reason

    def test_x_class_flare_returns_active(self):
        """X-class flux (>=1e-4) should return ACTIVE."""
        phase, reason = classify_phase_goes_only(goes_flux=1e-4, goes_class="X1.0")
        assert phase == Phase.ACTIVE
        assert "M/X-class" in reason

    def test_c_class_rising_returns_active(self):
        """C-class flux rising should return ACTIVE."""
        phase, reason = classify_phase_goes_only(
            goes_flux=5e-6, goes_rising=True, goes_class="C5.0"
        )
        assert phase == Phase.ACTIVE
        assert "C-class flare" in reason

    def test_c_class_falling_returns_recovery(self):
        """C-class flux falling should return RECOVERY."""
        phase, reason = classify_phase_goes_only(
            goes_flux=2e-6, goes_rising=False, goes_class="C2.0"
        )
        assert phase == Phase.RECOVERY
        assert "Post-flare decay" in reason

    def test_b_class_returns_baseline(self):
        """B-class flux (<1e-6) should return BASELINE."""
        phase, reason = classify_phase_goes_only(goes_flux=5e-7, goes_class="B5.0")
        assert phase == Phase.BASELINE
        assert "Quiet" in reason

    def test_a_class_returns_baseline(self):
        """A-class flux (<1e-7) should return BASELINE."""
        phase, reason = classify_phase_goes_only(goes_flux=5e-8, goes_class="A5.0")
        assert phase == Phase.BASELINE

    def test_threshold_boundary_m_class(self):
        """Test exact M-class boundary (1e-5)."""
        # Exactly at threshold = ACTIVE
        phase, _ = classify_phase_goes_only(goes_flux=1e-5)
        assert phase == Phase.ACTIVE

        # Just below = depends on rising/falling
        phase, _ = classify_phase_goes_only(goes_flux=9.9e-6, goes_rising=True)
        assert phase == Phase.ACTIVE  # C-class rising

        phase, _ = classify_phase_goes_only(goes_flux=9.9e-6, goes_rising=False)
        assert phase == Phase.RECOVERY  # C-class falling

    def test_threshold_boundary_c_class(self):
        """Test C-class boundary (1e-6)."""
        # At C-class threshold, not rising
        phase, _ = classify_phase_goes_only(goes_flux=1e-6, goes_rising=False)
        assert phase == Phase.RECOVERY

        # Just below C-class = BASELINE
        phase, _ = classify_phase_goes_only(goes_flux=9e-7)
        assert phase == Phase.BASELINE


# =============================================================================
# EXPERIMENTAL (ΔMI-INTEGRATED) CLASSIFICATION
# =============================================================================

class TestClassifyPhaseExperimental:
    """Test ΔMI-integrated phase classification."""

    def _make_pairs(self, z_211=0, z_304=0, trend_211=0, trend_304=0):
        """Helper to create pairs_data dict."""
        return {
            '193-211': {'residual': z_211, 'slope_pct_per_hour': trend_211},
            '193-304': {'residual': z_304, 'slope_pct_per_hour': trend_304},
        }

    # --- ACTIVE (high GOES) ---

    def test_active_m_class(self):
        """M-class should return ACTIVE regardless of ΔMI."""
        pairs = self._make_pairs(z_211=10, z_304=10)  # High anomaly
        phase, reason = classify_phase_experimental(pairs, goes_flux=1e-5, goes_class="M1.0")
        assert phase == Phase.ACTIVE

    def test_active_c_class_rising(self):
        """C-class rising should return ACTIVE."""
        pairs = self._make_pairs()
        phase, reason = classify_phase_experimental(
            pairs, goes_flux=5e-6, goes_rising=True, goes_class="C5.0"
        )
        assert phase == Phase.ACTIVE

    # --- PRE-FLARE (destabilization) ---

    def test_pre_flare_negative_anomaly_goes_rising(self):
        """Negative anomaly + GOES rising = PRE-FLARE."""
        pairs = self._make_pairs(z_211=-3)  # Negative anomaly
        phase, reason = classify_phase_experimental(
            pairs, goes_flux=5e-7, goes_rising=True
        )
        assert phase == Phase.PRE_FLARE
        assert "destabilizing" in reason

    def test_pre_flare_coronal_decoupling(self):
        """Coronal decoupling (z_211 < -2, trend < -3) = PRE-FLARE."""
        pairs = self._make_pairs(z_211=-2.5, trend_211=-5)
        phase, reason = classify_phase_experimental(pairs, goes_flux=5e-7)
        assert phase == Phase.PRE_FLARE
        assert "decoupling" in reason

    def test_not_pre_flare_if_trend_not_negative(self):
        """z_211 < -2 but trend >= -3 should NOT be PRE-FLARE."""
        pairs = self._make_pairs(z_211=-2.5, trend_211=-2)  # Trend not negative enough
        phase, reason = classify_phase_experimental(pairs, goes_flux=5e-7)
        assert phase != Phase.PRE_FLARE

    # --- POST-EVENT (reorganizing) ---

    def test_post_event_high_anomaly_goes_quiet(self):
        """High anomaly (>5σ) + GOES quiet = POST-EVENT."""
        pairs = self._make_pairs(z_304=6)  # High chromospheric anomaly
        phase, reason = classify_phase_experimental(pairs, goes_flux=5e-7)
        assert phase == Phase.POST_EVENT
        assert "193-304" in reason  # Trigger pair identified

    def test_post_event_relaxing_vs_reorganizing(self):
        """POST-EVENT should distinguish relaxing (falling) vs reorganizing."""
        # Relaxing (trend < -3)
        pairs = self._make_pairs(z_304=6, trend_304=-5)
        phase, reason = classify_phase_experimental(pairs, goes_flux=5e-7)
        assert phase == Phase.POST_EVENT
        assert "Relaxing" in reason

        # Reorganizing (trend >= -3)
        pairs = self._make_pairs(z_304=6, trend_304=0)
        phase, reason = classify_phase_experimental(pairs, goes_flux=5e-7)
        assert phase == Phase.POST_EVENT
        assert "Reorganizing" in reason

    def test_post_event_chromosphere_restructuring(self):
        """z_304 > 4 with rising trend = chromosphere restructuring."""
        pairs = self._make_pairs(z_304=4.5, trend_304=2)
        phase, reason = classify_phase_experimental(pairs, goes_flux=5e-7)
        assert phase == Phase.POST_EVENT
        assert "Chromosphere" in reason

    def test_post_event_trigger_pair_211(self):
        """Trigger pair should be 193-211 when it has higher |z|."""
        pairs = self._make_pairs(z_211=7, z_304=5)  # 211 dominates
        phase, reason = classify_phase_experimental(pairs, goes_flux=5e-7)
        assert phase == Phase.POST_EVENT
        assert "193-211" in reason

    # --- ELEVATED-QUIET (structurally active but stable) ---

    def test_elevated_quiet_stable(self):
        """max_z > 3 with stable trends = ELEVATED-QUIET."""
        pairs = self._make_pairs(z_211=3.5, trend_211=1, trend_304=1)
        phase, reason = classify_phase_experimental(pairs, goes_flux=5e-7)
        assert phase == Phase.ELEVATED_QUIET
        assert "stable" in reason

    def test_elevated_quiet_with_trend(self):
        """max_z > 3 with significant trend still = ELEVATED-QUIET."""
        pairs = self._make_pairs(z_211=3.5, trend_211=5)  # Rising trend
        phase, reason = classify_phase_experimental(pairs, goes_flux=5e-7)
        assert phase == Phase.ELEVATED_QUIET
        assert "↑" in reason or "trend" in reason

    # --- RECOVERY (GOES decaying) ---

    def test_recovery_c_class_falling(self):
        """C-class falling with low ΔMI = RECOVERY."""
        pairs = self._make_pairs(z_211=1, z_304=1)  # Low anomaly
        phase, reason = classify_phase_experimental(
            pairs, goes_flux=2e-6, goes_rising=False, goes_class="C2.0"
        )
        assert phase == Phase.RECOVERY

    # --- BASELINE (quiet) ---

    def test_baseline_all_quiet(self):
        """Low GOES + low ΔMI = BASELINE."""
        pairs = self._make_pairs(z_211=1, z_304=1)
        phase, reason = classify_phase_experimental(pairs, goes_flux=5e-7)
        assert phase == Phase.BASELINE
        assert "Quiet" in reason

    def test_baseline_no_goes(self):
        """No GOES data + low ΔMI = BASELINE."""
        pairs = self._make_pairs(z_211=1, z_304=1)
        phase, reason = classify_phase_experimental(pairs, goes_flux=None)
        assert phase == Phase.BASELINE

    def test_empty_pairs_returns_baseline(self):
        """Empty pairs dict should return BASELINE."""
        phase, reason = classify_phase_experimental({}, goes_flux=5e-7)
        assert phase == Phase.BASELINE


# =============================================================================
# PARALLEL CLASSIFICATION (DIVERGENCE DETECTION)
# =============================================================================

class TestClassifyPhaseParallel:
    """Test parallel phase classification and divergence detection."""

    def _make_pairs(self, z_211=0, z_304=0, trend_211=0, trend_304=0):
        """Helper to create pairs_data dict."""
        return {
            '193-211': {'residual': z_211, 'slope_pct_per_hour': trend_211},
            '193-304': {'residual': z_304, 'slope_pct_per_hour': trend_304},
        }

    def test_consistent_when_both_agree(self):
        """Both classifiers agree = not divergent."""
        pairs = self._make_pairs(z_211=1, z_304=1)
        result = classify_phase_parallel(pairs, goes_flux=5e-7)

        assert result['current'][0] == Phase.BASELINE
        assert result['experimental'][0] == Phase.BASELINE
        assert result['is_divergent'] is False

    def test_divergent_when_disagree(self):
        """Classifiers disagree = divergent."""
        # GOES says BASELINE, ΔMI sees POST-EVENT
        pairs = self._make_pairs(z_304=6)  # High anomaly
        result = classify_phase_parallel(pairs, goes_flux=5e-7)

        assert result['current'][0] == Phase.BASELINE
        assert result['experimental'][0] == Phase.POST_EVENT
        assert result['is_divergent'] is True

    def test_divergence_note_content(self):
        """Divergence note should describe the disagreement."""
        pairs = self._make_pairs(z_304=6)
        result = classify_phase_parallel(pairs, goes_flux=5e-7)

        assert "BASELINE" in result['divergence_note']
        assert "POST-EVENT" in result['divergence_note']

    def test_both_agree_on_active(self):
        """Both should agree on ACTIVE for high GOES."""
        pairs = self._make_pairs(z_304=6)
        result = classify_phase_parallel(pairs, goes_flux=1e-5, goes_class="M1.0")

        assert result['current'][0] == Phase.ACTIVE
        assert result['experimental'][0] == Phase.ACTIVE
        assert result['is_divergent'] is False


# =============================================================================
# DIVERGENCE TYPE CLASSIFICATION
# =============================================================================

class TestClassifyDivergenceType:
    """Test divergence type categorization."""

    def test_no_divergence_returns_none(self):
        """Same phase = no divergence type."""
        result = classify_divergence_type(
            phase_goes=Phase.BASELINE,
            phase_experimental=Phase.BASELINE
        )
        assert result is None

    def test_precursor_goes_quiet_rising(self):
        """GOES quiet but rising = potential PRECURSOR."""
        result = classify_divergence_type(
            phase_goes=Phase.BASELINE,
            phase_experimental=Phase.ELEVATED_QUIET,
            goes_trend_rising=True
        )
        assert result == DivergenceType.PRECURSOR

    def test_post_event_recent_flare(self):
        """GOES quiet with recent flare = POST_EVENT."""
        result = classify_divergence_type(
            phase_goes=Phase.BASELINE,
            phase_experimental=Phase.POST_EVENT,
            goes_trend_rising=False,
            recent_flare_hours=12
        )
        assert result == DivergenceType.POST_EVENT

    def test_post_event_goes_active(self):
        """GOES active, ΔMI quiet = POST_EVENT (GOES leading)."""
        result = classify_divergence_type(
            phase_goes=Phase.ACTIVE,
            phase_experimental=Phase.BASELINE
        )
        assert result == DivergenceType.POST_EVENT

    def test_post_event_goes_recovery(self):
        """GOES recovery, ΔMI different = POST_EVENT."""
        result = classify_divergence_type(
            phase_goes=Phase.RECOVERY,
            phase_experimental=Phase.BASELINE
        )
        assert result == DivergenceType.POST_EVENT

    def test_unconfirmed_no_context(self):
        """GOES quiet, no trend info, no flare = UNCONFIRMED."""
        result = classify_divergence_type(
            phase_goes=Phase.BASELINE,
            phase_experimental=Phase.ELEVATED_QUIET,
            goes_trend_rising=False,
            recent_flare_hours=None
        )
        assert result == DivergenceType.UNCONFIRMED

    def test_unconfirmed_old_flare(self):
        """GOES quiet with old flare (>24h) = UNCONFIRMED."""
        result = classify_divergence_type(
            phase_goes=Phase.BASELINE,
            phase_experimental=Phase.POST_EVENT,
            goes_trend_rising=False,
            recent_flare_hours=30  # >24h ago
        )
        assert result == DivergenceType.UNCONFIRMED


# =============================================================================
# EDGE CASES AND REGRESSION TESTS
# =============================================================================

class TestPhaseClassificationEdgeCases:
    """Edge cases and regression tests."""

    def test_goes_flux_zero(self):
        """Zero flux should return BASELINE."""
        phase, _ = classify_phase_goes_only(goes_flux=0)
        assert phase == Phase.BASELINE

    def test_negative_residual_not_pre_flare_without_trend(self):
        """Negative residual alone (without trend) should not be PRE-FLARE."""
        pairs = {'193-211': {'residual': -3, 'slope_pct_per_hour': 0}}
        phase, _ = classify_phase_experimental(pairs, goes_flux=5e-7, goes_rising=False)
        assert phase != Phase.PRE_FLARE

    def test_max_z_calculation_ignores_underscore_keys(self):
        """max_z calculation should ignore keys starting with _."""
        pairs = {
            '193-211': {'residual': 2},
            '_metadata': {'residual': 100},  # Should be ignored
        }
        phase, _ = classify_phase_experimental(pairs, goes_flux=5e-7)
        # If _metadata wasn't ignored, this would be POST-EVENT (z=100)
        assert phase == Phase.BASELINE

    def test_phase_constants_exist(self):
        """Verify all phase constants are defined."""
        assert Phase.BASELINE == 'BASELINE'
        assert Phase.ELEVATED_QUIET == 'ELEVATED-QUIET'
        assert Phase.POST_EVENT == 'POST-EVENT'
        assert Phase.RECOVERY == 'RECOVERY'
        assert Phase.PRE_FLARE == 'PRE-FLARE'
        assert Phase.ACTIVE == 'ACTIVE'

    def test_divergence_type_constants_exist(self):
        """Verify all divergence type constants are defined."""
        assert DivergenceType.PRECURSOR == 'PRECURSOR'
        assert DivergenceType.POST_EVENT == 'POST_EVENT'
        assert DivergenceType.UNCONFIRMED == 'UNCONFIRMED'
        assert DivergenceType.TRUE_POSITIVE == 'TRUE_POSITIVE'
        assert DivergenceType.FALSE_POSITIVE == 'FALSE_POSITIVE'
