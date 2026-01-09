"""
Tests for Radial Profile Analysis
==================================
"""

import numpy as np
import pytest

from solar_seed.radial_profile import (
    find_disk_center,
    compute_radial_profile,
    reconstruct_from_profile,
    compute_residual,
    subtract_radial_geometry,
    prepare_pair_for_residual_mi
)
from solar_seed.data_loader import generate_synthetic_sun


class TestFindDiskCenter:
    """Tests for center detection."""

    def test_centered_disk(self):
        """Centered disk should yield image center."""
        shape = (100, 100)
        y, x = np.ogrid[:shape[0], :shape[1]]
        center = (50, 50)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        image = np.maximum(0, 1000 - r * 20)

        found_center = find_disk_center(image)

        assert abs(found_center[0] - 50) < 2
        assert abs(found_center[1] - 50) < 2

    def test_offset_disk(self):
        """Offset disk should find correct center."""
        shape = (100, 100)
        y, x = np.ogrid[:shape[0], :shape[1]]
        center = (60, 40)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        image = np.maximum(0, 1000 - r * 25)

        found_center = find_disk_center(image)

        assert abs(found_center[0] - 60) < 3
        assert abs(found_center[1] - 40) < 3

    def test_empty_image(self):
        """Empty image should return image center."""
        image = np.zeros((100, 100))
        center = find_disk_center(image)

        assert center == (50, 50)


class TestRadialProfile:
    """Tests for radial profile calculation."""

    def test_profile_shape(self):
        """Profile should have correct length."""
        image = np.random.random((100, 100)) * 1000
        n_bins = 50
        profile = compute_radial_profile(image, n_bins=n_bins)

        assert len(profile.radii) == n_bins
        assert len(profile.intensities) == n_bins
        assert len(profile.counts) == n_bins

    def test_limb_darkening_profile(self):
        """Limb darkening should show decreasing intensity."""
        shape = (100, 100)
        y, x = np.ogrid[:shape[0], :shape[1]]
        center = (50, 50)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        # Simuliere Limb Darkening: I(r) = I_0 * sqrt(1 - (r/R)^2)
        R = 40
        mu = np.sqrt(np.maximum(0, 1 - (r / R)**2))
        image = mu * 1000
        image[r > R] = 0

        profile = compute_radial_profile(image, n_bins=20, center=center)

        # Intensity should decrease with radius (up to the edge)
        inner_bins = profile.intensities[:10]
        # Check that the trend is decreasing
        assert inner_bins[0] > inner_bins[5]

    def test_custom_center(self):
        """Custom center should be used."""
        image = np.random.random((100, 100))
        custom_center = (30, 70)
        profile = compute_radial_profile(image, center=custom_center)

        assert profile.center == custom_center


class TestReconstruction:
    """Tests for profile reconstruction."""

    def test_reconstruction_shape(self):
        """Reconstruction should have correct shape."""
        image = np.random.random((100, 100)) * 1000
        profile = compute_radial_profile(image, n_bins=50)
        reconstructed = reconstruct_from_profile(profile, image.shape)

        assert reconstructed.shape == image.shape

    def test_reconstruction_circular_symmetry(self):
        """Reconstruction should be circularly symmetric."""
        image = np.random.random((100, 100)) * 1000
        profile = compute_radial_profile(image, n_bins=50, center=(50, 50))
        reconstructed = reconstruct_from_profile(profile, image.shape)

        # Points with equal distance from center should be equal
        assert abs(reconstructed[50, 60] - reconstructed[50, 40]) < 1e-10
        assert abs(reconstructed[60, 50] - reconstructed[40, 50]) < 1e-10


class TestResidual:
    """Tests for residual calculation."""

    def test_ratio_method(self):
        """Ratio method should perform division."""
        image = np.ones((10, 10)) * 100
        model = np.ones((10, 10)) * 50

        residual = compute_residual(image, model, method="ratio", epsilon=0)

        np.testing.assert_array_almost_equal(residual, np.ones((10, 10)) * 2)

    def test_difference_method(self):
        """Difference method should perform subtraction."""
        image = np.ones((10, 10)) * 100
        model = np.ones((10, 10)) * 30

        residual = compute_residual(image, model, method="difference")

        np.testing.assert_array_almost_equal(residual, np.ones((10, 10)) * 70)

    def test_epsilon_prevents_division_by_zero(self):
        """Epsilon should prevent division by zero."""
        image = np.ones((10, 10)) * 100
        model = np.zeros((10, 10))

        residual = compute_residual(image, model, method="ratio", epsilon=1.0)

        assert np.all(np.isfinite(residual))
        np.testing.assert_array_almost_equal(residual, np.ones((10, 10)) * 100)


class TestSubtractRadialGeometry:
    """Tests for the complete workflow."""

    def test_workflow_returns_correct_types(self):
        """Workflow should return correct types."""
        image = np.random.random((100, 100)) * 1000
        residual, profile, model = subtract_radial_geometry(image)

        assert residual.shape == image.shape
        assert model.shape == image.shape
        assert hasattr(profile, 'radii')

    def test_residual_reduces_geometry(self):
        """Residual should reduce geometric variation."""
        # Create image with strong limb darkening
        shape = (100, 100)
        y, x = np.ogrid[:shape[0], :shape[1]]
        center = (50, 50)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        R = 40
        mu = np.sqrt(np.maximum(0, 1 - (r / R)**2))
        image = mu * 1000 + np.random.normal(0, 10, shape)
        image = np.maximum(0, image)

        residual, _, _ = subtract_radial_geometry(image, method="ratio")

        # Residual should have lower variation (after normalization)
        # In the center, the residual should be close to 1
        center_region = residual[45:55, 45:55]
        assert np.std(center_region) < np.std(image[45:55, 45:55]) / 100


class TestPreparePairForResidualMI:
    """Tests for pair preparation."""

    def test_returns_correct_shapes(self):
        """Function should return two residuals and info."""
        image_1 = np.random.random((100, 100)) * 1000
        image_2 = np.random.random((100, 100)) * 1000

        res_1, res_2, info = prepare_pair_for_residual_mi(image_1, image_2)

        assert res_1.shape == image_1.shape
        assert res_2.shape == image_2.shape
        assert "profile_1" in info
        assert "profile_2" in info
        assert "model_1" in info
        assert "model_2" in info

    def test_shared_center(self):
        """With shared_center, the same center should be used."""
        image_1 = np.random.random((100, 100)) * 1000
        image_2 = np.random.random((100, 100)) * 1000

        _, _, info = prepare_pair_for_residual_mi(
            image_1, image_2, shared_center=True
        )

        assert info["profile_1"].center == info["profile_2"].center

    def test_with_synthetic_sun(self):
        """Test with synthetic sun data."""
        data_1, data_2 = generate_synthetic_sun(shape=(128, 128), seed=42)

        res_1, res_2, info = prepare_pair_for_residual_mi(data_1, data_2)

        # Residuals should not have negative values (with ratio method)
        assert res_1.min() >= 0
        assert res_2.min() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
