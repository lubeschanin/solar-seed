#!/usr/bin/env python3
"""Tests for the render_sun module."""

import pytest
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

from solar_seed.render_sun import (
    parse_local_datetime,
    format_datetime_label,
    normalize_image,
    COMMON_TIMEZONES,
    AIA_COLORMAPS,
    WAVELENGTHS,
)


class TestCommonTimezones:
    """Tests for COMMON_TIMEZONES constant."""

    def test_timezones_exist(self):
        """Test that common timezones are defined."""
        assert len(COMMON_TIMEZONES) > 0
        assert "Berlin" in COMMON_TIMEZONES
        assert "New York" in COMMON_TIMEZONES
        assert "Tokyo" in COMMON_TIMEZONES

    def test_timezone_format(self):
        """Test that timezone values are valid format."""
        for city, tz in COMMON_TIMEZONES.items():
            assert "/" in tz, f"Timezone {tz} should contain '/'"
            # Verify it's a valid timezone
            ZoneInfo(tz)  # Will raise if invalid


class TestParseLocalDatetime:
    """Tests for parse_local_datetime function."""

    def test_dd_mm_yyyy_format(self):
        """Test parsing DD.MM.YYYY format."""
        local_dt, utc_dt = parse_local_datetime("08.03.2012", "14:00", "Europe/Berlin")

        assert local_dt.day == 8
        assert local_dt.month == 3
        assert local_dt.year == 2012
        assert local_dt.hour == 14
        assert local_dt.minute == 0

    def test_yyyy_mm_dd_format(self):
        """Test parsing YYYY-MM-DD format."""
        local_dt, utc_dt = parse_local_datetime("2024-01-15", "12:00", "Europe/Berlin")

        assert local_dt.day == 15
        assert local_dt.month == 1
        assert local_dt.year == 2024

    def test_utc_conversion_winter(self):
        """Test UTC conversion in winter (CET = UTC+1)."""
        local_dt, utc_dt = parse_local_datetime("15.01.2024", "14:00", "Europe/Berlin")

        # Berlin in January is UTC+1
        assert utc_dt.hour == 13  # 14:00 CET = 13:00 UTC

    def test_utc_conversion_summer(self):
        """Test UTC conversion in summer (CEST = UTC+2)."""
        local_dt, utc_dt = parse_local_datetime("15.07.2024", "14:00", "Europe/Berlin")

        # Berlin in July is UTC+2
        assert utc_dt.hour == 12  # 14:00 CEST = 12:00 UTC

    def test_different_timezones(self):
        """Test conversion for different timezones."""
        # New York (EST = UTC-5 in winter)
        local_dt, utc_dt = parse_local_datetime("15.01.2024", "12:00", "America/New_York")
        assert utc_dt.hour == 17  # 12:00 EST = 17:00 UTC

        # Tokyo (JST = UTC+9)
        local_dt, utc_dt = parse_local_datetime("15.01.2024", "12:00", "Asia/Tokyo")
        assert utc_dt.hour == 3  # 12:00 JST = 03:00 UTC

    def test_returns_tuple(self):
        """Test that function returns correct tuple types."""
        result = parse_local_datetime("01.01.2024", "12:00", "UTC")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], datetime)
        assert isinstance(result[1], datetime)

    def test_timezone_info_preserved(self):
        """Test that timezone info is set correctly."""
        local_dt, utc_dt = parse_local_datetime("01.01.2024", "12:00", "Europe/Berlin")

        assert local_dt.tzinfo is not None
        assert utc_dt.tzinfo is not None
        assert str(utc_dt.tzinfo) == "UTC"


class TestFormatDatetimeLabel:
    """Tests for format_datetime_label function."""

    def test_format_output(self):
        """Test label format."""
        local_dt = datetime(2024, 3, 8, 14, 0, tzinfo=ZoneInfo("Europe/Berlin"))
        utc_dt = datetime(2024, 3, 8, 13, 0, tzinfo=ZoneInfo("UTC"))

        result = format_datetime_label(local_dt, utc_dt, "Europe/Berlin")

        assert "08.03.2024" in result
        assert "14:00" in result
        assert "13:00 UTC" in result
        assert "Berlin" in result

    def test_underscore_replacement(self):
        """Test that underscores in timezone are replaced with spaces."""
        local_dt = datetime(2024, 1, 1, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        utc_dt = datetime(2024, 1, 1, 17, 0, tzinfo=ZoneInfo("UTC"))

        result = format_datetime_label(local_dt, utc_dt, "America/New_York")

        assert "New York" in result
        assert "New_York" not in result


class TestNormalizeImage:
    """Tests for normalize_image function."""

    def test_output_range(self):
        """Test that output is in [0, 1] range."""
        data = np.random.uniform(100, 10000, (256, 256))
        result = normalize_image(data)

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preserves_shape(self):
        """Test that output shape matches input."""
        data = np.random.uniform(100, 10000, (128, 128))
        result = normalize_image(data)

        assert result.shape == data.shape

    def test_handles_zeros(self):
        """Test handling of zero values (disk mask)."""
        data = np.ones((100, 100)) * 1000
        data[:50, :] = 0  # Half is masked

        result = normalize_image(data)

        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

    def test_all_zeros(self):
        """Test handling of all-zero input."""
        data = np.zeros((100, 100))
        result = normalize_image(data)

        assert np.allclose(result, 0)

    def test_sqrt_scaling(self):
        """Test that sqrt scaling is applied (dynamic range compression)."""
        data = np.array([[100, 10000]])
        result = normalize_image(data, vmin_pct=0, vmax_pct=100)

        # With sqrt scaling, difference should be compressed
        # sqrt(10000)/sqrt(100) = 100/10 = 10, not 100
        # So normalized range is more compressed than linear
        assert result[0, 1] > result[0, 0]


class TestAIAColormaps:
    """Tests for AIA_COLORMAPS constant."""

    def test_all_wavelengths_defined(self):
        """Test that all wavelengths have colormap definitions."""
        for wl in WAVELENGTHS:
            assert wl in AIA_COLORMAPS, f"Missing colormap for {wl} Å"

    def test_colormap_structure(self):
        """Test colormap dictionary structure."""
        for wl, cmap_info in AIA_COLORMAPS.items():
            assert "cmap" in cmap_info
            assert "color" in cmap_info
            assert isinstance(cmap_info["color"], tuple)
            assert len(cmap_info["color"]) == 3


class TestWavelengths:
    """Tests for WAVELENGTHS constant."""

    def test_seven_wavelengths(self):
        """Test that all 7 AIA EUV wavelengths are defined."""
        assert len(WAVELENGTHS) == 7

    def test_expected_wavelengths(self):
        """Test that expected wavelengths are present."""
        expected = [304, 171, 193, 211, 335, 94, 131]
        for wl in expected:
            assert wl in WAVELENGTHS
