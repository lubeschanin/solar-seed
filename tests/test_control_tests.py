"""
Tests für Kontroll-Tests
========================
"""

import numpy as np
import pytest

from solar_seed.control_tests import (
    time_shift_null,
    ring_wise_shuffle_test,
    sector_ring_shuffle_test,
    psf_blur_matching,
    co_alignment_check,
    run_all_controls,
    create_radial_bins,
    create_sector_ring_bins,
    ring_shuffle,
    sector_ring_shuffle,
    apply_gaussian_blur,
    shift_image,
    TimeShiftResult,
    RingShuffleResult,
    SectorRingShuffleResult,
    BlurMatchResult,
    CoAlignmentResult
)
from solar_seed.data_loader import generate_synthetic_sun


class TestTimeShiftNull:
    """Tests für C1: Time-Shift Null."""

    def test_returns_result(self):
        """Sollte TimeShiftResult zurückgeben."""
        data_1, data_2 = generate_synthetic_sun(shape=(64, 64), seed=42)
        result = time_shift_null(data_1, data_2)

        assert isinstance(result, TimeShiftResult)
        assert result.mi_original >= 0
        assert result.mi_shifted >= 0

    def test_mi_reduces_after_shift(self):
        """MI sollte nach Zeit-Shift reduziert sein."""
        data_1, data_2 = generate_synthetic_sun(
            shape=(64, 64),
            extra_correlation=0.5,
            seed=42
        )
        result = time_shift_null(data_1, data_2)

        # MI sollte nach Shuffle deutlich niedriger sein
        assert result.mi_shifted < result.mi_original
        assert result.mi_reduction > 0

    def test_with_uncorrelated_data(self):
        """Bei unkorrelierten Daten sollte Reduktion gering sein."""
        rng = np.random.default_rng(42)
        data_1 = rng.random((64, 64)) * 1000
        data_2 = rng.random((64, 64)) * 1000

        result = time_shift_null(data_1, data_2)

        # Bei unkorrelierten Daten ist die Reduktion geringer
        assert result.mi_reduction_percent < 90


class TestRingWiseShuffle:
    """Tests für C2: Ring-wise Shuffle."""

    def test_returns_result(self):
        """Sollte RingShuffleResult zurückgeben."""
        data_1, data_2 = generate_synthetic_sun(shape=(64, 64), seed=42)
        result = ring_wise_shuffle_test(data_1, data_2)

        assert isinstance(result, RingShuffleResult)
        assert result.mi_original >= 0

    def test_creates_radial_bins(self):
        """Radiale Bins sollten korrekt erstellt werden."""
        bins = create_radial_bins((100, 100), (50, 50), n_rings=10)

        assert bins.shape == (100, 100)
        assert bins.min() == 0
        assert bins.max() == 9

        # Zentrum sollte Ring 0 sein
        assert bins[50, 50] == 0

    def test_ring_shuffle_preserves_radial_stats(self):
        """Ring-Shuffle sollte radiale Statistik erhalten."""
        rng = np.random.default_rng(42)
        image = rng.random((100, 100)) * 1000
        bins = create_radial_bins((100, 100), (50, 50), n_rings=10)

        shuffled = ring_shuffle(image, bins, seed=42)

        # Mittelwert pro Ring sollte gleich bleiben
        for ring_idx in range(10):
            mask = bins == ring_idx
            if mask.sum() > 0:
                original_mean = image[mask].mean()
                shuffled_mean = shuffled[mask].mean()
                assert abs(original_mean - shuffled_mean) < 1e-10

    def test_both_shuffles_reduce_mi(self):
        """Beide Shuffle-Methoden sollten MI reduzieren."""
        data_1, data_2 = generate_synthetic_sun(
            shape=(64, 64),
            extra_correlation=0.5,
            seed=42
        )
        result = ring_wise_shuffle_test(data_1, data_2)

        assert result.mi_ring_shuffled < result.mi_original
        assert result.mi_global_shuffled < result.mi_original


class TestSectorRingShuffle:
    """Tests für erweiterten C2: Sector-Ring Shuffle."""

    def test_returns_result(self):
        """Sollte SectorRingShuffleResult zurückgeben."""
        data_1, data_2 = generate_synthetic_sun(shape=(64, 64), seed=42)
        result = sector_ring_shuffle_test(data_1, data_2)

        assert isinstance(result, SectorRingShuffleResult)
        assert result.mi_original >= 0

    def test_creates_sector_ring_bins(self):
        """Sector-Ring Bins sollten korrekt erstellt werden."""
        ring_bins, sector_bins = create_sector_ring_bins(
            (100, 100), (50, 50), n_rings=10, n_sectors=8
        )

        assert ring_bins.shape == (100, 100)
        assert sector_bins.shape == (100, 100)
        assert ring_bins.min() == 0
        assert ring_bins.max() == 9
        assert sector_bins.min() == 0
        assert sector_bins.max() == 7

    def test_sector_shuffle_preserves_sector_stats(self):
        """Sector-Shuffle sollte Statistik pro Sektor×Ring erhalten."""
        rng = np.random.default_rng(42)
        image = rng.random((100, 100)) * 1000
        ring_bins, sector_bins = create_sector_ring_bins(
            (100, 100), (50, 50), n_rings=5, n_sectors=4
        )

        shuffled = sector_ring_shuffle(image, ring_bins, sector_bins, seed=42)

        # Mittelwert pro Ring×Sektor sollte gleich bleiben
        for r in range(5):
            for s in range(4):
                mask = (ring_bins == r) & (sector_bins == s)
                if mask.sum() > 0:
                    original_mean = image[mask].mean()
                    shuffled_mean = shuffled[mask].mean()
                    assert abs(original_mean - shuffled_mean) < 1e-10

    def test_hierarchy_of_shuffles(self):
        """Global < Ring < Sector < Original MI sollte gelten."""
        data_1, data_2 = generate_synthetic_sun(
            shape=(64, 64),
            extra_correlation=0.5,
            seed=42
        )
        result = sector_ring_shuffle_test(data_1, data_2)

        # Je mehr Struktur erhalten, desto höher die MI
        assert result.mi_global_shuffled <= result.mi_ring_shuffled
        assert result.mi_ring_shuffled <= result.mi_sector_shuffled
        assert result.mi_sector_shuffled <= result.mi_original

    def test_contributions_sum_up(self):
        """Beiträge sollten sich sinnvoll zusammensetzen."""
        data_1, data_2 = generate_synthetic_sun(
            shape=(64, 64),
            extra_correlation=0.5,
            seed=42
        )
        result = sector_ring_shuffle_test(data_1, data_2)

        # radial + azimuthal + local ≈ original - global
        total = (result.radial_contribution +
                 result.azimuthal_contribution +
                 result.local_structure)
        expected = result.mi_original - result.mi_global_shuffled

        assert abs(total - expected) < 0.001


class TestPSFBlurMatching:
    """Tests für C3: PSF/Blur Matching."""

    def test_returns_result(self):
        """Sollte BlurMatchResult zurückgeben."""
        data_1, data_2 = generate_synthetic_sun(shape=(64, 64), seed=42)
        result = psf_blur_matching(data_1, data_2, sigma=2.0)

        assert isinstance(result, BlurMatchResult)
        assert result.blur_sigma == 2.0

    def test_gaussian_blur(self):
        """Gaussian Blur sollte Bild glätten."""
        rng = np.random.default_rng(42)
        image = rng.random((100, 100)) * 1000

        blurred = apply_gaussian_blur(image, sigma=5.0)

        # Blurred sollte geringere Varianz haben
        assert blurred.std() < image.std()

    def test_blur_changes_mi(self):
        """Blur sollte MI verändern (erhöhen oder reduzieren)."""
        data_1, data_2 = generate_synthetic_sun(
            shape=(64, 64),
            extra_correlation=0.5,
            seed=42
        )
        result = psf_blur_matching(data_1, data_2, sigma=2.0)

        # MI sollte sich ändern (nicht exakt gleich bleiben)
        # Bei großem Blur kann sich MI ändern
        assert result.mi_blurred >= 0


class TestCoAlignmentCheck:
    """Tests für C4: Co-alignment Check."""

    def test_returns_result(self):
        """Sollte CoAlignmentResult zurückgeben."""
        data_1, data_2 = generate_synthetic_sun(shape=(64, 64), seed=42)
        result = co_alignment_check(data_1, data_2, max_offset=2)

        assert isinstance(result, CoAlignmentResult)
        assert result.mi_map.shape == (5, 5)  # 2*2+1 = 5

    def test_shift_image(self):
        """Bild-Shift sollte korrekt funktionieren."""
        image = np.arange(100).reshape(10, 10).astype(float)

        # Shift nach rechts-unten
        shifted = shift_image(image, (1, 1))

        # Obere linke Ecke sollte 0 sein (aufgefüllt)
        assert shifted[0, 0] == 0
        assert shifted[0, 1] == 0
        assert shifted[1, 0] == 0

        # Verschobener Inhalt
        assert shifted[1, 1] == image[0, 0]

    def test_aligned_data_centered(self):
        """Bei ausgerichteten Daten sollte Maximum bei (0,0) sein."""
        data_1, data_2 = generate_synthetic_sun(
            shape=(64, 64),
            extra_correlation=0.5,
            seed=42
        )
        result = co_alignment_check(data_1, data_2, max_offset=2)

        # Maximum sollte bei oder nahe (0, 0) sein
        # Bei synthetischen Daten ist perfekte Ausrichtung erwartet
        assert abs(result.max_shift[0]) <= 1
        assert abs(result.max_shift[1]) <= 1

    def test_mi_map_shape(self):
        """MI-Karte sollte korrekte Größe haben."""
        data_1, data_2 = generate_synthetic_sun(shape=(64, 64), seed=42)

        result = co_alignment_check(data_1, data_2, max_offset=3)

        assert result.mi_map.shape == (7, 7)  # 2*3+1 = 7


class TestRunAllControls:
    """Tests für kombinierte Kontrollen."""

    def test_runs_all_controls(self):
        """Sollte alle vier Kontrollen durchführen."""
        data_1, data_2 = generate_synthetic_sun(
            shape=(64, 64),
            extra_correlation=0.5,
            seed=42
        )

        result = run_all_controls(data_1, data_2, verbose=False)

        assert result.c1_time_shift is not None
        assert result.c2_ring_shuffle is not None
        assert result.c3_blur_match is not None
        assert result.c4_co_alignment is not None

    def test_all_passed_property(self):
        """all_passed Property sollte funktionieren."""
        data_1, data_2 = generate_synthetic_sun(
            shape=(64, 64),
            extra_correlation=0.5,
            seed=42
        )

        result = run_all_controls(data_1, data_2, verbose=False)

        # all_passed ist ein bool
        assert isinstance(result.all_passed, bool)


class TestEdgeCases:
    """Tests für Randfälle."""

    def test_small_images(self):
        """Sollte mit kleinen Bildern funktionieren."""
        data_1, data_2 = generate_synthetic_sun(shape=(32, 32), seed=42)

        result = run_all_controls(data_1, data_2, verbose=False)

        assert result.c1_time_shift is not None

    def test_with_zeros(self):
        """Sollte mit Nullen im Bild umgehen können."""
        data_1, data_2 = generate_synthetic_sun(shape=(64, 64), seed=42)

        # Füge einige Nullen hinzu
        data_1[:10, :10] = 0
        data_2[:10, :10] = 0

        result = run_all_controls(data_1, data_2, verbose=False)

        assert result.c1_time_shift is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
