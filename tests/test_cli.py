#!/usr/bin/env python3
"""Tests for the CLI module."""

import pytest
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from solar_seed.cli import (
    show_status_bar,
    check_checkpoint,
    get_number,
    get_date,
    get_time,
)


class TestShowStatusBar:
    """Tests for show_status_bar function."""

    def test_empty_progress(self):
        """Test 0% progress."""
        result = show_status_bar(0, 100)
        assert "0/100" in result
        assert "0.0%" in result
        assert "░" * 40 in result

    def test_full_progress(self):
        """Test 100% progress."""
        result = show_status_bar(100, 100)
        assert "100/100" in result
        assert "100.0%" in result
        assert "█" * 40 in result

    def test_half_progress(self):
        """Test 50% progress."""
        result = show_status_bar(50, 100)
        assert "50/100" in result
        assert "50.0%" in result
        assert "█" * 20 in result

    def test_zero_total(self):
        """Test with zero total (edge case)."""
        result = show_status_bar(0, 0)
        assert "0/0" in result
        assert "0.0%" in result

    def test_custom_width(self):
        """Test with custom width."""
        result = show_status_bar(50, 100, width=20)
        # Should have 10 filled + 10 empty = 20 total
        assert len(result.split("[")[1].split("]")[0]) == 20


class TestCheckCheckpoint:
    """Tests for check_checkpoint function."""

    def test_no_checkpoint(self, tmp_path, monkeypatch):
        """Test when no checkpoint exists."""
        monkeypatch.chdir(tmp_path)

        result = check_checkpoint()

        assert result["exists"] is False
        assert result["processed"] == 0
        assert result["start_date"] is None

    def test_with_checkpoint(self, tmp_path, monkeypatch):
        """Test when checkpoint exists."""
        monkeypatch.chdir(tmp_path)

        # Create checkpoint directory and file
        checkpoint_dir = tmp_path / "results" / "rotation"
        checkpoint_dir.mkdir(parents=True)

        checkpoint_data = {
            "last_index": 50,
            "timestamps": ["2024-01-01T00:00:00"] * 100
        }
        with open(checkpoint_dir / "checkpoint.json", "w") as f:
            json.dump(checkpoint_data, f)

        result = check_checkpoint()

        assert result["exists"] is True
        assert result["processed"] == 50
        assert result["timestamps"] == 100

    def test_with_result_file(self, tmp_path, monkeypatch):
        """Test when result file exists with metadata."""
        monkeypatch.chdir(tmp_path)

        checkpoint_dir = tmp_path / "results" / "rotation"
        checkpoint_dir.mkdir(parents=True)

        result_data = {
            "metadata": {
                "start_time": "2024-01-01T00:00:00",
                "hours": 648,
                "cadence_minutes": 60
            }
        }
        with open(checkpoint_dir / "rotation_analysis.json", "w") as f:
            json.dump(result_data, f)

        result = check_checkpoint()

        assert result["start_date"] == "2024-01-01"
        assert result["hours"] == 648
        assert result["cadence"] == 60


class TestGetNumber:
    """Tests for get_number function."""

    def test_default_value(self):
        """Test that empty input returns default."""
        with patch("builtins.input", return_value=""):
            result = get_number("Test:", 42.0)
            assert result == 42.0

    def test_valid_input(self):
        """Test valid numeric input."""
        with patch("builtins.input", return_value="100"):
            result = get_number("Test:", 42.0)
            assert result == 100.0

    def test_invalid_then_valid(self):
        """Test invalid input followed by valid."""
        with patch("builtins.input", side_effect=["abc", "50"]):
            result = get_number("Test:", 42.0)
            assert result == 50.0

    def test_out_of_range_then_valid(self):
        """Test out of range value followed by valid."""
        with patch("builtins.input", side_effect=["99999", "50"]):
            result = get_number("Test:", 42.0, min_val=0, max_val=100)
            assert result == 50.0


class TestGetDate:
    """Tests for get_date function."""

    def test_default_value(self):
        """Test that empty input returns default date."""
        with patch("builtins.input", return_value=""):
            result = get_date("Test:", default_days_ago=7)
            expected = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            assert result == f"{expected}T00:00:00"

    def test_valid_date(self):
        """Test valid date input."""
        with patch("builtins.input", return_value="2024-06-15"):
            result = get_date("Test:")
            assert result == "2024-06-15T00:00:00"

    def test_invalid_then_valid(self):
        """Test invalid date followed by valid."""
        with patch("builtins.input", side_effect=["invalid", "2024-01-01"]):
            result = get_date("Test:")
            assert result == "2024-01-01T00:00:00"


class TestGetTime:
    """Tests for get_time function."""

    def test_default_value(self):
        """Test that empty input returns default time."""
        with patch("builtins.input", return_value=""):
            result = get_time("Test:", default="14:30")
            assert result == "14:30"

    def test_valid_time(self):
        """Test valid time input."""
        with patch("builtins.input", return_value="09:45"):
            result = get_time("Test:")
            assert result == "09:45"

    def test_invalid_then_valid(self):
        """Test invalid time followed by valid."""
        with patch("builtins.input", side_effect=["25:00", "12:00"]):
            result = get_time("Test:")
            assert result == "12:00"
