#!/usr/bin/env python3
"""Tests for the multichannel module."""

import pytest
import numpy as np
from itertools import combinations

from solar_seed.multichannel import (
    AIAChannel,
    AIA_CHANNELS,
    WAVELENGTHS,
    WAVELENGTH_TO_TEMP,
    PairResult,
    CouplingMatrix,
    generate_multichannel_sun,
    generate_multichannel_timeseries,
    analyze_pair,
    build_coupling_matrix,
    AIA_DATA_SOURCE,
    AIA_QUALITY_FLAGS,
    AIA_CRITICAL_FLAGS,
)


class TestAIAChannels:
    """Tests for AIA channel definitions."""

    def test_seven_channels(self):
        """Test that 7 channels are defined."""
        assert len(AIA_CHANNELS) == 7

    def test_channel_structure(self):
        """Test AIAChannel dataclass structure."""
        for ch in AIA_CHANNELS:
            assert isinstance(ch, AIAChannel)
            assert isinstance(ch.wavelength, int)
            assert isinstance(ch.temperature, float)
            assert isinstance(ch.description, str)
            assert isinstance(ch.color, str)

    def test_temperature_ordering(self):
        """Test that channels are ordered by temperature."""
        temps = [ch.temperature for ch in AIA_CHANNELS]
        # First 5 should be increasing, last 2 are flare channels
        assert temps[0] < temps[1] < temps[2] < temps[3] < temps[4]

    def test_wavelength_list_consistency(self):
        """Test WAVELENGTHS matches AIA_CHANNELS."""
        channel_wls = [ch.wavelength for ch in AIA_CHANNELS]
        assert WAVELENGTHS == channel_wls

    def test_wavelength_to_temp_mapping(self):
        """Test WAVELENGTH_TO_TEMP mapping."""
        for ch in AIA_CHANNELS:
            assert ch.wavelength in WAVELENGTH_TO_TEMP
            assert WAVELENGTH_TO_TEMP[ch.wavelength] == ch.temperature


class TestAIADataSource:
    """Tests for AIA data source metadata."""

    def test_required_fields(self):
        """Test that required metadata fields exist."""
        required = [
            "instrument", "operator", "data_provider",
            "data_url", "wavelengths_angstrom", "reference"
        ]
        for field in required:
            assert field in AIA_DATA_SOURCE

    def test_wavelengths_match(self):
        """Test that data source wavelengths match WAVELENGTHS."""
        assert set(AIA_DATA_SOURCE["wavelengths_angstrom"]) == set(WAVELENGTHS)


class TestAIAQualityFlags:
    """Tests for AIA quality flag definitions."""

    def test_flags_defined(self):
        """Test that quality flags are defined."""
        assert len(AIA_QUALITY_FLAGS) > 0

    def test_critical_flags_subset(self):
        """Test that critical flags are subset of all flags."""
        for flag in AIA_CRITICAL_FLAGS:
            assert flag in AIA_QUALITY_FLAGS


class TestPairResult:
    """Tests for PairResult dataclass."""

    def test_creation(self):
        """Test creating a PairResult."""
        pr = PairResult(
            wavelength_1=193,
            wavelength_2=211,
            mi_original=1.5,
            mi_residual=0.5,
            mi_ratio=0.33,
            delta_mi_ring=0.2,
            delta_mi_sector=0.15,
            z_score=100.0,
            temperature_diff=0.8
        )

        assert pr.wavelength_1 == 193
        assert pr.wavelength_2 == 211
        assert pr.mi_ratio == 0.33


class TestCouplingMatrix:
    """Tests for CouplingMatrix dataclass."""

    def test_creation(self):
        """Test creating a CouplingMatrix."""
        matrix = np.zeros((7, 7))
        matrix[0, 1] = 0.5
        matrix[1, 0] = 0.5

        cm = CouplingMatrix(
            wavelengths=WAVELENGTHS.copy(),
            matrix=matrix,
            metric="delta_mi_sector"
        )

        assert cm.metric == "delta_mi_sector"
        assert cm.matrix.shape == (7, 7)

    def test_get_value(self):
        """Test get_value method."""
        matrix = np.zeros((7, 7))
        matrix[2, 3] = 0.73  # 193-211

        cm = CouplingMatrix(
            wavelengths=WAVELENGTHS.copy(),
            matrix=matrix,
            metric="delta_mi_sector"
        )

        assert cm.get_value(193, 211) == 0.73
        assert cm.get_value(211, 193) == 0.0  # Not symmetric in this test

    def test_to_ascii(self):
        """Test ASCII representation."""
        matrix = np.eye(7) * 0.5

        cm = CouplingMatrix(
            wavelengths=WAVELENGTHS.copy(),
            matrix=matrix,
            metric="test"
        )

        ascii_repr = cm.to_ascii()

        # Should contain wavelength headers
        assert "304" in ascii_repr
        assert "171" in ascii_repr
        assert "193" in ascii_repr

        # Should contain separator
        assert "-" in ascii_repr


class TestGenerateMultichannelSun:
    """Tests for generate_multichannel_sun function."""

    def test_returns_all_channels(self):
        """Test that all 7 channels are generated."""
        channels = generate_multichannel_sun(shape=(64, 64))

        assert len(channels) == 7
        for wl in WAVELENGTHS:
            assert wl in channels

    def test_correct_shape(self):
        """Test that images have correct shape."""
        shape = (128, 128)
        channels = generate_multichannel_sun(shape=shape)

        for wl, data in channels.items():
            assert data.shape == shape

    def test_non_negative_values(self):
        """Test that all values are non-negative."""
        channels = generate_multichannel_sun(shape=(64, 64))

        for wl, data in channels.items():
            assert data.min() >= 0

    def test_reproducibility(self):
        """Test that same seed produces same result."""
        ch1 = generate_multichannel_sun(shape=(64, 64), seed=123)
        ch2 = generate_multichannel_sun(shape=(64, 64), seed=123)

        for wl in WAVELENGTHS:
            np.testing.assert_array_equal(ch1[wl], ch2[wl])

    def test_different_seeds(self):
        """Test that different seeds produce different results."""
        ch1 = generate_multichannel_sun(shape=(64, 64), seed=1)
        ch2 = generate_multichannel_sun(shape=(64, 64), seed=2)

        # At least one channel should be different
        any_different = False
        for wl in WAVELENGTHS:
            if not np.allclose(ch1[wl], ch2[wl]):
                any_different = True
                break
        assert any_different

    def test_disk_masking(self):
        """Test that disk mask is applied (zeros outside disk)."""
        channels = generate_multichannel_sun(shape=(100, 100))

        # Corners should be zero (outside solar disk)
        for wl, data in channels.items():
            assert data[0, 0] == 0
            assert data[0, -1] == 0
            assert data[-1, 0] == 0
            assert data[-1, -1] == 0


class TestGenerateMultichannelTimeseries:
    """Tests for generate_multichannel_timeseries function."""

    def test_correct_length(self):
        """Test that correct number of timepoints is generated."""
        n_points = 5
        ts = generate_multichannel_timeseries(n_points=n_points, shape=(32, 32))

        assert len(ts) == n_points

    def test_tuple_structure(self):
        """Test that each element is (channels, timestamp) tuple."""
        ts = generate_multichannel_timeseries(n_points=3, shape=(32, 32))

        for channels, timestamp in ts:
            assert isinstance(channels, dict)
            assert isinstance(timestamp, str)
            assert len(channels) == 7

    def test_timestamps_increase(self):
        """Test that timestamps are increasing."""
        ts = generate_multichannel_timeseries(n_points=5, shape=(32, 32), cadence_minutes=12)

        timestamps = [t for _, t in ts]
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i-1]


class TestAnalyzePair:
    """Tests for analyze_pair function."""

    def test_returns_pair_result(self):
        """Test that function returns PairResult."""
        img1 = np.random.uniform(100, 1000, (64, 64))
        img2 = np.random.uniform(100, 1000, (64, 64))

        result = analyze_pair(img1, img2, 193, 211, bins=32, seed=42)

        assert isinstance(result, PairResult)

    def test_wavelengths_preserved(self):
        """Test that wavelengths are correctly stored."""
        img1 = np.random.uniform(100, 1000, (64, 64))
        img2 = np.random.uniform(100, 1000, (64, 64))

        result = analyze_pair(img1, img2, 171, 304, bins=32, seed=42)

        assert result.wavelength_1 == 171
        assert result.wavelength_2 == 304

    def test_mi_values_positive(self):
        """Test that MI values are non-negative."""
        # Create correlated images
        base = np.random.uniform(100, 1000, (64, 64))
        img1 = base + np.random.normal(0, 50, (64, 64))
        img2 = base + np.random.normal(0, 50, (64, 64))

        result = analyze_pair(img1, img2, 193, 211, bins=32, seed=42)

        assert result.mi_original >= 0
        assert result.mi_residual >= 0

    def test_temperature_diff_calculated(self):
        """Test that temperature difference is calculated."""
        img1 = np.random.uniform(100, 1000, (64, 64))
        img2 = np.random.uniform(100, 1000, (64, 64))

        # 193 Å (1.2 MK) and 211 Å (2.0 MK)
        result = analyze_pair(img1, img2, 193, 211, bins=32, seed=42)

        expected_diff = abs(1.2 - 2.0)
        assert abs(result.temperature_diff - expected_diff) < 0.01


class TestBuildCouplingMatrix:
    """Tests for build_coupling_matrix function."""

    def test_symmetric_matrix(self):
        """Test that resulting matrix is symmetric."""
        # Create sample pair results
        pair_results = []
        for wl1, wl2 in combinations(WAVELENGTHS, 2):
            pr = PairResult(
                wavelength_1=wl1,
                wavelength_2=wl2,
                mi_original=1.0,
                mi_residual=0.3,
                mi_ratio=0.3,
                delta_mi_ring=0.1,
                delta_mi_sector=0.15,
                z_score=50.0,
                temperature_diff=1.0
            )
            pair_results.append(pr)

        cm = build_coupling_matrix(pair_results, "delta_mi_sector")

        # Check symmetry
        for i in range(7):
            for j in range(7):
                assert cm.matrix[i, j] == cm.matrix[j, i]

    def test_correct_metric(self):
        """Test that correct metric is used."""
        pair_results = [
            PairResult(
                wavelength_1=304, wavelength_2=171,
                mi_original=1.0, mi_residual=0.3, mi_ratio=0.3,
                delta_mi_ring=0.1, delta_mi_sector=0.2,
                z_score=50.0, temperature_diff=0.55
            )
        ]

        cm_sector = build_coupling_matrix(pair_results, "delta_mi_sector")
        cm_ratio = build_coupling_matrix(pair_results, "mi_ratio")

        idx_304 = WAVELENGTHS.index(304)
        idx_171 = WAVELENGTHS.index(171)

        assert cm_sector.matrix[idx_304, idx_171] == 0.2
        assert cm_ratio.matrix[idx_304, idx_171] == 0.3

    def test_all_pairs_included(self):
        """Test that all 21 pairs are represented."""
        pair_results = []
        for wl1, wl2 in combinations(WAVELENGTHS, 2):
            pr = PairResult(
                wavelength_1=wl1, wavelength_2=wl2,
                mi_original=1.0, mi_residual=0.3, mi_ratio=0.3,
                delta_mi_ring=0.1, delta_mi_sector=0.15 + (wl1 + wl2) / 1000,
                z_score=50.0, temperature_diff=1.0
            )
            pair_results.append(pr)

        cm = build_coupling_matrix(pair_results, "delta_mi_sector")

        # Count non-zero off-diagonal elements (should be 42 = 21 pairs × 2 for symmetry)
        n_pairs = 0
        for i in range(7):
            for j in range(7):
                if i != j and cm.matrix[i, j] != 0:
                    n_pairs += 1

        assert n_pairs == 42  # 21 pairs × 2 (symmetric)
