"""
Tests für Mutual Information Berechnung
=======================================
"""

import numpy as np
import pytest

from solar_seed.mutual_info import (
    mutual_information,
    normalized_mutual_information,
    conditional_entropy,
    compute_histogram_2d,
    entropy,
    EPSILON,
    MIN_SAMPLES,
    DEFAULT_BINS,
)
from solar_seed.null_model import (
    compute_null_distribution,
    compute_z_score,
    compute_p_value
)


class TestEntropy:
    """Tests für Shannon-Entropie."""
    
    def test_uniform_distribution(self):
        """Gleichverteilung sollte maximale Entropie haben."""
        n_bins = 64
        p = np.ones(n_bins) / n_bins
        h = entropy(p)
        expected = np.log2(n_bins)  # log2(64) = 6
        assert abs(h - expected) < 0.001
    
    def test_single_peak(self):
        """Einpunktverteilung sollte Entropie 0 haben."""
        p = np.zeros(64)
        p[0] = 1.0
        h = entropy(p)
        assert h == 0.0
    
    def test_empty_array(self):
        """Leeres Array sollte 0 zurückgeben."""
        p = np.array([])
        h = entropy(p)
        assert h == 0.0


class TestMutualInformation:
    """Tests für MI-Berechnung."""
    
    def test_identical_arrays(self):
        """Identische Arrays sollten hohe MI haben."""
        rng = np.random.default_rng(42)
        x = rng.random((100, 100))
        mi = mutual_information(x, x)
        assert mi > 5.0  # Sollte nahe maximaler Entropie sein
    
    def test_independent_arrays(self):
        """Unabhängige Arrays sollten niedrige MI haben."""
        rng = np.random.default_rng(42)
        x = rng.random((100, 100))
        y = rng.random((100, 100))
        mi = mutual_information(x, y)
        # Bei echtem Rauschen sollte MI < 0.5 sein
        assert mi < 1.0
    
    def test_correlated_arrays(self):
        """Korrelierte Arrays sollten messbare MI haben."""
        rng = np.random.default_rng(42)
        x = rng.random((100, 100))
        noise = rng.random((100, 100)) * 0.1
        y = x + noise  # Stark korreliert
        mi = mutual_information(x, y)
        assert mi > 2.0
    
    def test_mi_non_negative(self):
        """MI sollte immer >= 0 sein."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            x = rng.random((50, 50))
            y = rng.random((50, 50))
            mi = mutual_information(x, y)
            assert mi >= 0.0
    
    def test_small_array(self):
        """Zu kleine Arrays sollten 0 zurückgeben."""
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        mi = mutual_information(x, y)
        assert mi == 0.0


class TestNormalizedMI:
    """Tests für normalisierte MI."""
    
    def test_nmi_range(self):
        """NMI sollte in [0, 1] liegen."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            x = rng.random((100, 100))
            y = rng.random((100, 100))
            nmi = normalized_mutual_information(x, y)
            assert 0.0 <= nmi <= 1.0
    
    def test_identical_arrays_nmi(self):
        """Identische Arrays sollten NMI ≈ 1 haben."""
        rng = np.random.default_rng(42)
        x = rng.random((100, 100))
        nmi = normalized_mutual_information(x, x)
        assert nmi > 0.95


class TestNullModel:
    """Tests für das Nullmodell."""
    
    def test_null_distribution_independent(self):
        """Bei unabhängigen Daten sollte Z ≈ 0 sein."""
        rng = np.random.default_rng(42)
        x = rng.exponential(100, (100, 100))
        y = rng.exponential(100, (100, 100))
        
        mi_real = mutual_information(x, y)
        mi_null_mean, mi_null_std, _ = compute_null_distribution(
            x, y, n_shuffles=50, seed=42
        )
        
        z = compute_z_score(mi_real, mi_null_mean, mi_null_std)
        # Z sollte nahe 0 sein (innerhalb von ±3)
        assert abs(z) < 3.0
    
    def test_null_distribution_correlated(self):
        """Bei korrelierten Daten sollte Z >> 0 sein."""
        rng = np.random.default_rng(42)
        x = rng.random((100, 100))
        y = x + rng.random((100, 100)) * 0.1
        
        mi_real = mutual_information(x, y)
        mi_null_mean, mi_null_std, _ = compute_null_distribution(
            x, y, n_shuffles=50, seed=42
        )
        
        z = compute_z_score(mi_real, mi_null_mean, mi_null_std)
        # Z sollte deutlich positiv sein
        assert z > 3.0
    
    def test_z_score_calculation(self):
        """Z-Score Berechnung sollte korrekt sein."""
        z = compute_z_score(10.0, 5.0, 2.5)
        assert z == 2.0
    
    def test_z_score_zero_std(self):
        """Z-Score bei std=0 sollte inf zurückgeben."""
        z = compute_z_score(10.0, 5.0, 0.0)
        assert z == float('inf')


class TestHistogram2D:
    """Tests für 2D-Histogramm."""
    
    def test_histogram_shape(self):
        """Histogramm sollte korrekte Shape haben."""
        x = np.random.random(1000)
        y = np.random.random(1000)
        hist = compute_histogram_2d(x, y, bins=32)
        assert hist.shape == (32, 32)
    
    def test_histogram_sum(self):
        """Histogramm-Summe sollte Anzahl Elemente sein."""
        n = 1000
        x = np.random.random(n)
        y = np.random.random(n)
        hist = compute_histogram_2d(x, y, bins=32)
        assert hist.sum() == n


class TestEdgeCases:
    """Tests for input validation and edge cases."""

    def test_empty_arrays_raise(self):
        """Empty arrays should raise ValueError."""
        x = np.array([])
        y = np.array([])
        with pytest.raises(ValueError, match="empty"):
            mutual_information(x, y)
        with pytest.raises(ValueError, match="empty"):
            normalized_mutual_information(x, y)
        with pytest.raises(ValueError, match="empty"):
            conditional_entropy(x, y)

    def test_all_nan_raises(self):
        """All-NaN arrays should raise ValueError."""
        x = np.full(200, np.nan)
        y = np.full(200, np.nan)
        with pytest.raises(ValueError, match="No valid"):
            mutual_information(x, y)
        with pytest.raises(ValueError, match="No valid"):
            normalized_mutual_information(x, y)
        with pytest.raises(ValueError, match="No valid"):
            conditional_entropy(x, y)

    def test_shape_mismatch_raises(self):
        """Different-sized arrays should raise ValueError."""
        x = np.ones(200)
        y = np.ones(300)
        with pytest.raises(ValueError, match="Shape mismatch"):
            mutual_information(x, y)
        with pytest.raises(ValueError, match="Shape mismatch"):
            normalized_mutual_information(x, y)
        with pytest.raises(ValueError, match="Shape mismatch"):
            conditional_entropy(x, y)

    def test_constant_array(self):
        """Constant values have zero entropy, so MI = 0."""
        x = np.ones(10000)
        y = np.ones(10000)
        mi = mutual_information(x, y)
        assert mi == 0.0

    def test_partial_nan(self):
        """Arrays with some NaN should work using valid subset."""
        rng = np.random.default_rng(42)
        x = rng.random(10000)
        y = x + rng.random(10000) * 0.1
        # Sprinkle NaNs into 10% of values
        nan_idx = rng.choice(10000, size=1000, replace=False)
        x[nan_idx] = np.nan
        mi = mutual_information(x, y)
        assert mi > 1.0  # Still correlated

    def test_named_constants_exist(self):
        """Named constants should be importable with expected values."""
        assert EPSILON == 1e-10
        assert MIN_SAMPLES == 100
        assert DEFAULT_BINS == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
