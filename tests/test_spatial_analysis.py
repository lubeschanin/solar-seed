"""
Tests für räumliche MI-Analyse
==============================
"""

import numpy as np
import pytest

from solar_seed.spatial_analysis import (
    compute_spatial_mi_map,
    compute_spatial_residual_mi_map,
    find_top_hotspots,
    get_region_coordinates,
    mi_map_to_ascii,
    create_disk_mask,
    SpatialMIResult
)
from solar_seed.data_loader import generate_synthetic_sun


class TestSpatialMIMap:
    """Tests für räumliche MI-Karten."""

    def test_map_shape(self):
        """MI-Karte sollte korrekte Shape haben."""
        image_1 = np.random.random((100, 100)) * 1000
        image_2 = np.random.random((100, 100)) * 1000

        result = compute_spatial_mi_map(image_1, image_2, grid_size=(5, 5))

        assert result.mi_map.shape == (5, 5)
        assert result.grid_size == (5, 5)

    def test_map_with_correlated_data(self):
        """Korrelierte Daten sollten höhere MI zeigen."""
        rng = np.random.default_rng(42)
        image_1 = rng.random((100, 100)) * 1000
        image_2 = image_1 + rng.random((100, 100)) * 100  # Korreliert

        result = compute_spatial_mi_map(image_1, image_2, grid_size=(4, 4))

        # MI sollte durchweg positiv sein
        valid_mi = result.mi_map[~np.isnan(result.mi_map)]
        assert len(valid_mi) > 0
        assert np.all(valid_mi >= 0)
        assert result.mi_mean > 0

    def test_hotspot_detection(self):
        """Hotspot sollte die Zelle mit maximaler MI sein."""
        rng = np.random.default_rng(42)
        image_1 = rng.random((100, 100)) * 1000
        image_2 = rng.random((100, 100)) * 1000

        result = compute_spatial_mi_map(image_1, image_2, grid_size=(4, 4))

        # Hotspot-Wert sollte dem Maximum entsprechen
        assert result.hotspot_value == np.nanmax(result.mi_map)

    def test_statistics(self):
        """Statistiken sollten korrekt berechnet werden."""
        rng = np.random.default_rng(42)
        image_1 = rng.random((100, 100)) * 1000
        image_2 = rng.random((100, 100)) * 1000

        result = compute_spatial_mi_map(image_1, image_2, grid_size=(4, 4))

        valid_mi = result.mi_map[~np.isnan(result.mi_map)]
        assert abs(result.mi_mean - np.mean(valid_mi)) < 1e-10
        assert abs(result.mi_std - np.std(valid_mi)) < 1e-10
        assert result.mi_min <= result.mi_mean <= result.mi_max


class TestSpatialResidualMIMap:
    """Tests für Residual-MI-Karten."""

    def test_returns_both_maps(self):
        """Sollte Original und Residual Maps zurückgeben."""
        image_1 = np.random.random((100, 100)) * 1000
        image_2 = np.random.random((100, 100)) * 1000

        result = compute_spatial_residual_mi_map(
            image_1, image_2, grid_size=(4, 4)
        )

        assert result.original.mi_map.shape == (4, 4)
        assert result.residual.mi_map.shape == (4, 4)
        assert result.mi_reduction_map.shape == (4, 4)

    def test_reduction_map(self):
        """Reduktions-Karte sollte Differenz sein."""
        data_1, data_2 = generate_synthetic_sun(shape=(128, 128), seed=42)

        result = compute_spatial_residual_mi_map(
            data_1, data_2, grid_size=(4, 4)
        )

        expected = result.original.mi_map - result.residual.mi_map
        np.testing.assert_array_almost_equal(
            result.mi_reduction_map, expected
        )

    def test_geometry_reduces_mi(self):
        """Geometrie-Subtraktion sollte MI im Schnitt reduzieren."""
        data_1, data_2 = generate_synthetic_sun(
            shape=(128, 128),
            extra_correlation=0.0,
            seed=42
        )

        result = compute_spatial_residual_mi_map(
            data_1, data_2, grid_size=(4, 4)
        )

        # Mittlere Original-MI sollte höher sein als Residual-MI
        assert result.original.mi_mean >= result.residual.mi_mean


class TestFindTopHotspots:
    """Tests für Hotspot-Erkennung."""

    def test_returns_correct_count(self):
        """Sollte n Hotspots zurückgeben."""
        rng = np.random.default_rng(42)
        image_1 = rng.random((100, 100)) * 1000
        image_2 = rng.random((100, 100)) * 1000

        result = compute_spatial_mi_map(image_1, image_2, grid_size=(4, 4))
        hotspots = find_top_hotspots(result, n=3)

        assert len(hotspots) <= 3

    def test_sorted_by_value(self):
        """Hotspots sollten absteigend nach MI sortiert sein."""
        rng = np.random.default_rng(42)
        image_1 = rng.random((100, 100)) * 1000
        image_2 = rng.random((100, 100)) * 1000

        result = compute_spatial_mi_map(image_1, image_2, grid_size=(4, 4))
        hotspots = find_top_hotspots(result, n=5)

        values = [v for _, v in hotspots]
        assert values == sorted(values, reverse=True)


class TestGetRegionCoordinates:
    """Tests für Koordinaten-Berechnung."""

    def test_correct_coordinates(self):
        """Sollte korrekte Pixelkoordinaten zurückgeben."""
        image_1 = np.random.random((100, 100))
        image_2 = np.random.random((100, 100))

        result = compute_spatial_mi_map(image_1, image_2, grid_size=(4, 4))
        coords = get_region_coordinates(result, (1, 2))

        y_start, y_end, x_start, x_end = coords

        # Zelle (1, 2) bei 4x4 Grid auf 100x100 Bild
        assert y_start == 25  # 1 * 25
        assert x_start == 50  # 2 * 25


class TestASCIIVisualization:
    """Tests für ASCII-Ausgabe."""

    def test_produces_output(self):
        """Sollte String produzieren."""
        mi_map = np.random.random((4, 4))

        ascii_output = mi_map_to_ascii(mi_map)

        assert isinstance(ascii_output, str)
        assert len(ascii_output) > 0

    def test_handles_nan(self):
        """Sollte NaN-Werte behandeln."""
        mi_map = np.array([[1.0, np.nan], [0.5, 0.8]])

        ascii_output = mi_map_to_ascii(mi_map)

        assert isinstance(ascii_output, str)


class TestDiskMask:
    """Tests für Sonnenscheiben-Maske."""

    def test_mask_shape(self):
        """Maske sollte korrekte Shape haben."""
        mask = create_disk_mask((100, 100))

        assert mask.shape == (100, 100)
        assert mask.dtype == np.bool_

    def test_mask_is_circular(self):
        """Maske sollte kreisförmig sein."""
        mask = create_disk_mask((100, 100), center=(50, 50))

        # Zentrum sollte True sein
        assert mask[50, 50] == True

        # Ecken sollten False sein
        assert mask[0, 0] == False
        assert mask[0, 99] == False
        assert mask[99, 0] == False
        assert mask[99, 99] == False


class TestWithSyntheticSun:
    """Integrationstests mit synthetischen Sonnendaten."""

    def test_full_workflow(self):
        """Kompletter Workflow sollte funktionieren."""
        data_1, data_2 = generate_synthetic_sun(
            shape=(128, 128),
            extra_correlation=0.5,
            seed=42
        )

        result = compute_spatial_residual_mi_map(
            data_1, data_2,
            grid_size=(4, 4),
            bins=32
        )

        # Alle Komponenten sollten vorhanden sein
        assert result.original is not None
        assert result.residual is not None
        assert result.residual_hotspot_idx is not None

        # Hotspots sollten gefunden werden
        hotspots = find_top_hotspots(result.residual, n=3)
        assert len(hotspots) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
