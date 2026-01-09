#!/usr/bin/env python3
"""Tests for segment-based rotation analysis."""

import pytest
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from solar_seed.final_analysis import (
    SegmentResult,
    save_segment,
    load_segment,
    aggregate_segments,
    convert_checkpoint_to_segments,
)


class TestSegmentResult:
    """Tests for SegmentResult dataclass."""

    def test_create_segment(self):
        """Test creating a SegmentResult."""
        result = SegmentResult(
            date="2025-12-01",
            start_time="2025-12-01T00:00:00",
            end_time="2025-12-02T00:00:00",
            n_points=120,
            cadence_minutes=12,
            timestamps=["2025-12-01T00:00:00", "2025-12-01T00:12:00"],
            pair_values={"304-171": [0.1, 0.2], "304-193": [0.15, 0.25]},
            pair_means={"304-171": 0.15, "304-193": 0.2},
            pair_stds={"304-171": 0.05, "304-193": 0.05}
        )

        assert result.date == "2025-12-01"
        assert result.n_points == 120
        assert len(result.pair_values["304-171"]) == 2


class TestSaveLoadSegment:
    """Tests for save_segment and load_segment functions."""

    def test_save_and_load(self, tmp_path):
        """Test saving and loading a segment."""
        segment = SegmentResult(
            date="2025-12-01",
            start_time="2025-12-01T00:00:00",
            end_time="2025-12-02T00:00:00",
            n_points=5,
            cadence_minutes=12,
            timestamps=["2025-12-01T00:00:00"],
            pair_values={"304-171": [0.1, 0.2, 0.3, 0.4, 0.5]},
            pair_means={"304-171": 0.3},
            pair_stds={"304-171": 0.141}
        )

        path = tmp_path / "test_segment.json"
        save_segment(segment, path)

        assert path.exists()

        loaded = load_segment(path)

        assert loaded.date == segment.date
        assert loaded.n_points == segment.n_points
        assert loaded.pair_values == segment.pair_values
        assert loaded.pair_means == segment.pair_means

    def test_segment_json_structure(self, tmp_path):
        """Test that saved JSON has correct structure."""
        segment = SegmentResult(
            date="2025-12-01",
            start_time="2025-12-01T00:00:00",
            end_time="2025-12-02T00:00:00",
            n_points=10,
            cadence_minutes=12,
            timestamps=["2025-12-01T00:00:00"],
            pair_values={"304-171": [0.1]},
            pair_means={"304-171": 0.1},
            pair_stds={"304-171": 0.0}
        )

        path = tmp_path / "test.json"
        save_segment(segment, path)

        with open(path) as f:
            data = json.load(f)

        assert "date" in data
        assert "n_points" in data
        assert "pair_values" in data
        assert "pair_means" in data
        assert "pair_stds" in data


class TestAggregateSegments:
    """Tests for aggregate_segments function."""

    def test_aggregate_multiple_segments(self, tmp_path):
        """Test aggregating multiple segments."""
        segment_dir = tmp_path / "segments"
        segment_dir.mkdir()

        # Create test segments
        for i, date in enumerate(["2025-12-01", "2025-12-02", "2025-12-03"]):
            segment = SegmentResult(
                date=date,
                start_time=f"{date}T00:00:00",
                end_time=f"{date}T23:59:59",
                n_points=10,
                cadence_minutes=12,
                timestamps=[f"{date}T{h:02d}:00:00" for h in range(10)],
                pair_values={"304-171": [0.1 + i * 0.01] * 10},
                pair_means={"304-171": 0.1 + i * 0.01},
                pair_stds={"304-171": 0.0}
            )
            save_segment(segment, segment_dir / f"{date}.json")

        result = aggregate_segments(
            segment_dir=str(segment_dir),
            output_dir=str(tmp_path / "output"),
            verbose=False
        )

        assert result is not None
        assert result.n_points == 30  # 3 segments * 10 points
        assert result.hours == 72  # 3 days * 24 hours
        assert (304, 171) in result.pair_means

    def test_aggregate_empty_dir(self, tmp_path):
        """Test aggregating from empty directory."""
        segment_dir = tmp_path / "empty_segments"
        segment_dir.mkdir()

        result = aggregate_segments(
            segment_dir=str(segment_dir),
            output_dir=str(tmp_path / "output"),
            verbose=False
        )

        assert result is None

    def test_aggregate_creates_output_files(self, tmp_path):
        """Test that aggregation creates output files."""
        segment_dir = tmp_path / "segments"
        segment_dir.mkdir()
        output_dir = tmp_path / "output"

        # Create one test segment
        segment = SegmentResult(
            date="2025-12-01",
            start_time="2025-12-01T00:00:00",
            end_time="2025-12-01T23:59:59",
            n_points=5,
            cadence_minutes=12,
            timestamps=["2025-12-01T00:00:00"],
            pair_values={"304-171": [0.1, 0.2, 0.3, 0.4, 0.5]},
            pair_means={"304-171": 0.3},
            pair_stds={"304-171": 0.141}
        )
        save_segment(segment, segment_dir / "2025-12-01.json")

        aggregate_segments(
            segment_dir=str(segment_dir),
            output_dir=str(output_dir),
            verbose=False
        )

        assert (output_dir / "rotation_analysis.json").exists()
        assert (output_dir / "rotation_analysis.txt").exists()


class TestConvertCheckpointToSegments:
    """Tests for convert_checkpoint_to_segments function."""

    def test_convert_checkpoint(self, tmp_path):
        """Test converting a checkpoint to segments."""
        # Create a mock checkpoint
        checkpoint_data = {
            "timestamps": [
                "2025-12-01T00:00:00",
                "2025-12-01T00:12:00",
                "2025-12-01T00:24:00",
                "2025-12-02T00:00:00",
                "2025-12-02T00:12:00",
            ],
            "pair_timeseries": {
                "304-171": [0.1, 0.2, 0.3, 0.4, 0.5],
                "304-193": [0.15, 0.25, 0.35, 0.45, 0.55]
            },
            "last_index": 5
        }

        checkpoint_path = tmp_path / "checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

        output_dir = tmp_path / "segments"

        count = convert_checkpoint_to_segments(
            checkpoint_path=str(checkpoint_path),
            output_dir=str(output_dir),
            verbose=False
        )

        assert count == 2  # Two days
        assert (output_dir / "2025-12-01.json").exists()
        assert (output_dir / "2025-12-02.json").exists()

        # Verify segment content
        with open(output_dir / "2025-12-01.json") as f:
            seg1 = json.load(f)
        assert seg1["n_points"] == 3
        assert len(seg1["pair_values"]["304-171"]) == 3

    def test_convert_nonexistent_checkpoint(self, tmp_path):
        """Test converting a non-existent checkpoint."""
        count = convert_checkpoint_to_segments(
            checkpoint_path=str(tmp_path / "nonexistent.json"),
            output_dir=str(tmp_path / "segments"),
            verbose=False
        )

        assert count == 0

    def test_convert_empty_checkpoint(self, tmp_path):
        """Test converting an empty checkpoint."""
        checkpoint_data = {
            "timestamps": [],
            "pair_timeseries": {}
        }

        checkpoint_path = tmp_path / "checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

        count = convert_checkpoint_to_segments(
            checkpoint_path=str(checkpoint_path),
            output_dir=str(tmp_path / "segments"),
            verbose=False
        )

        assert count == 0

    def test_skip_existing_segments(self, tmp_path):
        """Test that existing segments are skipped."""
        # Create checkpoint
        checkpoint_data = {
            "timestamps": [
                "2025-12-01T00:00:00",
                "2025-12-02T00:00:00",
            ],
            "pair_timeseries": {
                "304-171": [0.1, 0.2]
            },
            "last_index": 2
        }

        checkpoint_path = tmp_path / "checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

        output_dir = tmp_path / "segments"
        output_dir.mkdir()

        # Pre-create one segment
        existing = {
            "date": "2025-12-01",
            "n_points": 1,
            "timestamps": ["2025-12-01T00:00:00"],
            "pair_values": {"304-171": [0.999]},  # Different value
            "pair_means": {"304-171": 0.999},
            "pair_stds": {"304-171": 0.0}
        }
        with open(output_dir / "2025-12-01.json", "w") as f:
            json.dump(existing, f)

        count = convert_checkpoint_to_segments(
            checkpoint_path=str(checkpoint_path),
            output_dir=str(output_dir),
            verbose=False
        )

        assert count == 1  # Only one new segment created

        # Verify existing segment was not overwritten
        with open(output_dir / "2025-12-01.json") as f:
            seg = json.load(f)
        assert seg["pair_values"]["304-171"][0] == 0.999  # Original value
