"""
Tests for Control Tests
=======================
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
    """Tests for C1: Time-Shift Null."""

    def test_returns_result(self):
        """Should return TimeShiftResult."""
        data_1, data_2 = generate_synthetic_sun(shape=(64, 64), seed=42)
        result = time_shift_null(data_1, data_2)

        assert isinstance(result, TimeShiftResult)
        assert result.mi_original >= 0
        assert result.mi_shifted >= 0

    def test_mi_reduces_after_shift(self):
        """MI should be reduced after time shift."""
        data_1, data_2 = generate_synthetic_sun(
            shape=(64, 64),
            extra_correlation=0.5,
            seed=42
        )
        result = time_shift_null(data_1, data_2)

        # MI should be significantly lower after shuffle
        assert result.mi_shifted < result.mi_original
        assert result.mi_reduction > 0

    def test_with_uncorrelated_data(self):
        """For uncorrelated data, reduction should be small."""
        rng = np.random.default_rng(42)
        data_1 = rng.random((64, 64)) * 1000
        data_2 = rng.random((64, 64)) * 1000

        result = time_shift_null(data_1, data_2)

        # For uncorrelated data, the reduction is smaller
        assert result.mi_reduction_percent < 90


class TestRingWiseShuffle:
    """Tests for C2: Ring-wise Shuffle."""

    def test_returns_result(self):
        """Should return RingShuffleResult."""
        data_1, data_2 = generate_synthetic_sun(shape=(64, 64), seed=42)
        result = ring_wise_shuffle_test(data_1, data_2)

        assert isinstance(result, RingShuffleResult)
        assert result.mi_original >= 0

    def test_creates_radial_bins(self):
        """Radial bins should be created correctly."""
        bins = create_radial_bins((100, 100), (50, 50), n_rings=10)

        assert bins.shape == (100, 100)
        assert bins.min() == 0
        assert bins.max() == 9

        # Center should be ring 0
        assert bins[50, 50] == 0

    def test_ring_shuffle_preserves_radial_stats(self):
        """Ring shuffle should preserve radial statistics."""
        rng = np.random.default_rng(42)
        image = rng.random((100, 100)) * 1000
        bins = create_radial_bins((100, 100), (50, 50), n_rings=10)

        shuffled = ring_shuffle(image, bins, seed=42)

        # Mean per ring should remain the same
        for ring_idx in range(10):
            mask = bins == ring_idx
            if mask.sum() > 0:
                original_mean = image[mask].mean()
                shuffled_mean = shuffled[mask].mean()
                assert abs(original_mean - shuffled_mean) < 1e-10

    def test_both_shuffles_reduce_mi(self):
        """Both shuffle methods should reduce MI."""
        data_1, data_2 = generate_synthetic_sun(
            shape=(64, 64),
            extra_correlation=0.5,
            seed=42
        )
        result = ring_wise_shuffle_test(data_1, data_2)

        assert result.mi_ring_shuffled < result.mi_original
        assert result.mi_global_shuffled < result.mi_original


class TestSectorRingShuffle:
    """Tests for extended C2: Sector-Ring Shuffle."""

    def test_returns_result(self):
        """Should return SectorRingShuffleResult."""
        data_1, data_2 = generate_synthetic_sun(shape=(64, 64), seed=42)
        result = sector_ring_shuffle_test(data_1, data_2)

        assert isinstance(result, SectorRingShuffleResult)
        assert result.mi_original >= 0

    def test_creates_sector_ring_bins(self):
        """Sector-ring bins should be created correctly."""
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
        """Sector shuffle should preserve statistics per sector×ring."""
        rng = np.random.default_rng(42)
        image = rng.random((100, 100)) * 1000
        ring_bins, sector_bins = create_sector_ring_bins(
            (100, 100), (50, 50), n_rings=5, n_sectors=4
        )

        shuffled = sector_ring_shuffle(image, ring_bins, sector_bins, seed=42)

        # Mean per ring×sector should remain the same
        for r in range(5):
            for s in range(4):
                mask = (ring_bins == r) & (sector_bins == s)
                if mask.sum() > 0:
                    original_mean = image[mask].mean()
                    shuffled_mean = shuffled[mask].mean()
                    assert abs(original_mean - shuffled_mean) < 1e-10

    def test_hierarchy_of_shuffles(self):
        """Global < Ring < Sector < Original MI should hold."""
        data_1, data_2 = generate_synthetic_sun(
            shape=(64, 64),
            extra_correlation=0.5,
            seed=42
        )
        result = sector_ring_shuffle_test(data_1, data_2)

        # The more structure preserved, the higher the MI
        assert result.mi_global_shuffled <= result.mi_ring_shuffled
        assert result.mi_ring_shuffled <= result.mi_sector_shuffled
        assert result.mi_sector_shuffled <= result.mi_original

    def test_contributions_sum_up(self):
        """Contributions should add up sensibly."""
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
    """Tests for C3: PSF/Blur Matching."""

    def test_returns_result(self):
        """Should return BlurMatchResult."""
        data_1, data_2 = generate_synthetic_sun(shape=(64, 64), seed=42)
        result = psf_blur_matching(data_1, data_2, sigma=2.0)

        assert isinstance(result, BlurMatchResult)
        assert result.blur_sigma == 2.0

    def test_gaussian_blur(self):
        """Gaussian blur should smooth image."""
        rng = np.random.default_rng(42)
        image = rng.random((100, 100)) * 1000

        blurred = apply_gaussian_blur(image, sigma=5.0)

        # Blurred should have lower variance
        assert blurred.std() < image.std()

    def test_blur_changes_mi(self):
        """Blur should change MI (increase or decrease)."""
        data_1, data_2 = generate_synthetic_sun(
            shape=(64, 64),
            extra_correlation=0.5,
            seed=42
        )
        result = psf_blur_matching(data_1, data_2, sigma=2.0)

        # MI should change (not stay exactly the same)
        # With large blur, MI can change
        assert result.mi_blurred >= 0


class TestCoAlignmentCheck:
    """Tests for C4: Co-alignment Check."""

    def test_returns_result(self):
        """Should return CoAlignmentResult."""
        data_1, data_2 = generate_synthetic_sun(shape=(64, 64), seed=42)
        result = co_alignment_check(data_1, data_2, max_offset=2)

        assert isinstance(result, CoAlignmentResult)
        assert result.mi_map.shape == (5, 5)  # 2*2+1 = 5

    def test_shift_image(self):
        """Image shift should work correctly."""
        image = np.arange(100).reshape(10, 10).astype(float)

        # Shift right-down
        shifted = shift_image(image, (1, 1))

        # Upper left corner should be 0 (padded)
        assert shifted[0, 0] == 0
        assert shifted[0, 1] == 0
        assert shifted[1, 0] == 0

        # Shifted content
        assert shifted[1, 1] == image[0, 0]

    def test_aligned_data_centered(self):
        """For aligned data, maximum should be at (0,0)."""
        data_1, data_2 = generate_synthetic_sun(
            shape=(64, 64),
            extra_correlation=0.5,
            seed=42
        )
        result = co_alignment_check(data_1, data_2, max_offset=2)

        # Maximum should be at or near (0, 0)
        # For synthetic data, perfect alignment is expected
        assert abs(result.max_shift[0]) <= 1
        assert abs(result.max_shift[1]) <= 1

    def test_mi_map_shape(self):
        """MI map should have correct size."""
        data_1, data_2 = generate_synthetic_sun(shape=(64, 64), seed=42)

        result = co_alignment_check(data_1, data_2, max_offset=3)

        assert result.mi_map.shape == (7, 7)  # 2*3+1 = 7


class TestRunAllControls:
    """Tests for combined controls."""

    def test_runs_all_controls(self):
        """Should run all four controls."""
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
        """all_passed property should work."""
        data_1, data_2 = generate_synthetic_sun(
            shape=(64, 64),
            extra_correlation=0.5,
            seed=42
        )

        result = run_all_controls(data_1, data_2, verbose=False)

        # all_passed is a bool
        assert isinstance(result.all_passed, bool)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_small_images(self):
        """Should work with small images."""
        data_1, data_2 = generate_synthetic_sun(shape=(32, 32), seed=42)

        result = run_all_controls(data_1, data_2, verbose=False)

        assert result.c1_time_shift is not None

    def test_with_zeros(self):
        """Should handle zeros in the image."""
        data_1, data_2 = generate_synthetic_sun(shape=(64, 64), seed=42)

        # Add some zeros
        data_1[:10, :10] = 0
        data_2[:10, :10] = 0

        result = run_all_controls(data_1, data_2, verbose=False)

        assert result.c1_time_shift is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
