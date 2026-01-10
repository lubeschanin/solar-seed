"""
Tests for Solar Monitoring Database
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from monitoring_db import MonitoringDB


class TestDatabaseSchema:
    """Test database creation and schema."""

    @pytest.fixture
    def db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        db = MonitoringDB(db_path=db_path)
        yield db
        db.close()
        db_path.unlink()

    def test_database_creation(self, db):
        """Database is created with all tables."""
        cursor = db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        expected = {'goes_xray', 'solar_wind', 'coupling_measurements',
                    'flare_events', 'predictions', 'noaa_alerts'}
        assert expected.issubset(tables)

    def test_indices_created(self, db):
        """Indices are created for fast queries."""
        cursor = db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indices = {row[0] for row in cursor.fetchall()}

        assert 'idx_goes_timestamp' in indices
        assert 'idx_coupling_pair' in indices


class TestInsertOperations:
    """Test data insertion."""

    @pytest.fixture
    def db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        db = MonitoringDB(db_path=db_path)
        yield db
        db.close()
        db_path.unlink()

    def test_insert_goes_xray(self, db):
        """Insert GOES X-ray measurement."""
        row_id = db.insert_goes_xray(
            timestamp="2026-01-10T12:00:00",
            flux=5.4e-7,
            flare_class="B",
            magnitude=5.4
        )
        assert row_id > 0

        # Verify data
        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM goes_xray WHERE id = ?", (row_id,))
        row = cursor.fetchone()
        assert row['flux'] == pytest.approx(5.4e-7)
        assert row['flare_class'] == 'B'

    def test_insert_solar_wind(self, db):
        """Insert solar wind measurement."""
        row_id = db.insert_solar_wind(
            timestamp="2026-01-10T12:00:00",
            speed=450.0,
            density=8.5,
            bz=-5.2,
            bt=7.1,
            risk_level=1
        )
        assert row_id > 0

        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM solar_wind WHERE id = ?", (row_id,))
        row = cursor.fetchone()
        assert row['speed'] == pytest.approx(450.0)
        assert row['bz'] == pytest.approx(-5.2)

    def test_insert_coupling(self, db):
        """Insert coupling measurement."""
        row_id = db.insert_coupling(
            timestamp="2026-01-10T12:00:00",
            pair="193-211",
            delta_mi=0.59,
            residual=0.5,
            deviation_pct=-0.05,
            status="NORMAL",
            trend="STABLE",
            confidence="high"
        )
        assert row_id > 0

        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM coupling_measurements WHERE id = ?", (row_id,))
        row = cursor.fetchone()
        assert row['pair'] == '193-211'
        assert row['delta_mi'] == pytest.approx(0.59)
        assert row['status'] == 'NORMAL'

    def test_insert_flare_event(self, db):
        """Insert flare event."""
        row_id = db.insert_flare_event(
            start_time="2026-01-10T12:00:00",
            flare_class="M",
            magnitude=2.5,
            peak_time="2026-01-10T12:15:00",
            peak_flux=2.5e-5
        )
        assert row_id > 0

        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM flare_events WHERE id = ?", (row_id,))
        row = cursor.fetchone()
        assert row['class'] == 'M'
        assert row['magnitude'] == pytest.approx(2.5)

    def test_insert_prediction(self, db):
        """Insert prediction."""
        row_id = db.insert_prediction(
            prediction_time="2026-01-10T12:00:00",
            predicted_class="M",
            predicted_probability=0.75,
            trigger_pair="193-211",
            trigger_status="ALERT"
        )
        assert row_id > 0

    def test_duplicate_handling(self, db):
        """Duplicate timestamps are handled (upsert)."""
        db.insert_goes_xray("2026-01-10T12:00:00", 5.0e-7, "B", 5.0)
        db.insert_goes_xray("2026-01-10T12:00:00", 6.0e-7, "B", 6.0)

        cursor = db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM goes_xray")
        assert cursor.fetchone()[0] == 1  # Only one entry

        cursor.execute("SELECT flux FROM goes_xray")
        assert cursor.fetchone()[0] == pytest.approx(6.0e-7)  # Updated value


class TestQueryOperations:
    """Test data querying."""

    @pytest.fixture
    def db_with_data(self):
        """Create database with sample data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        db = MonitoringDB(db_path=db_path)

        # Insert sample data
        now = datetime.now(timezone.utc)
        for i in range(24):
            ts = (now - timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S")
            db.insert_goes_xray(ts, 5e-7 + i * 1e-8, "B", 5.0 + i * 0.1)
            db.insert_coupling(ts, "193-211", 0.59 - i * 0.01, status="NORMAL" if i < 20 else "WARNING")

        # Add a flare
        flare_time = (now - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%S")
        db.insert_flare_event(flare_time, "C", 3.5)

        yield db
        db.close()
        db_path.unlink()

    def test_get_recent_goes(self, db_with_data):
        """Query recent GOES data."""
        recent = db_with_data.get_recent_goes(hours=24)
        assert len(recent) >= 20  # Should have most of the 24 records
        assert all('flux' in r for r in recent)

    def test_get_recent_coupling(self, db_with_data):
        """Query recent coupling data."""
        recent = db_with_data.get_recent_coupling(hours=24)
        assert len(recent) >= 20  # Should have most of the 24 records

        # Filter by pair
        pair_data = db_with_data.get_recent_coupling(hours=24, pair="193-211")
        assert len(pair_data) >= 20

    def test_get_flare_events(self, db_with_data):
        """Query flare events."""
        flares = db_with_data.get_flare_events(days=1, min_class='B')
        assert len(flares) == 1
        assert flares[0]['class'] == 'C'

    def test_get_coupling_statistics(self, db_with_data):
        """Calculate coupling statistics."""
        stats = db_with_data.get_coupling_statistics("193-211", days=1)
        assert stats['n_measurements'] == 24
        assert 'mean_delta_mi' in stats
        assert stats['n_warnings'] == 4


class TestAnalysisMethods:
    """Test analysis methods."""

    @pytest.fixture
    def db_with_flares(self):
        """Create database with flares and coupling data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        db = MonitoringDB(db_path=db_path)

        # Insert coupling data before and after flare
        flare_time = datetime(2026, 1, 10, 12, 0, 0)

        # Coupling before flare (decreasing - pre-flare signature)
        for i in range(12):
            ts = (flare_time - timedelta(hours=i+1)).strftime("%Y-%m-%dT%H:%M:%S")
            status = "WARNING" if i < 3 else "NORMAL"
            db.insert_coupling(ts, "193-211", 0.59 - i * 0.03, status=status)

        # Flare event
        db.insert_flare_event(flare_time.strftime("%Y-%m-%dT%H:%M:%S"), "M", 2.5)

        # Prediction before flare
        pred_time = (flare_time - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
        db.insert_prediction(pred_time, "M", 0.6, "193-211", "WARNING")

        yield db
        db.close()
        db_path.unlink()

    def test_coupling_before_flares(self, db_with_flares):
        """Get coupling measurements before flares."""
        correlations = db_with_flares.get_coupling_before_flares(hours_before=6)
        assert len(correlations) > 0
        # All results should be within reasonable pre-flare window (allowing for rounding)
        assert all(r['hours_before_flare'] >= 0 for r in correlations)
        assert all(r['flare_class'] == 'M' for r in correlations)

    def test_database_stats(self, db_with_flares):
        """Get database statistics."""
        stats = db_with_flares.get_database_stats()
        assert stats['coupling_measurements'] == 12
        assert stats['flare_events'] == 1
        assert stats['predictions'] == 1


class TestMigration:
    """Test data migration."""

    @pytest.fixture
    def db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        db = MonitoringDB(db_path=db_path)
        yield db
        db.close()
        db_path.unlink()

    def test_import_from_json(self, db, tmp_path):
        """Import from JSON history file."""
        import json

        # Create sample JSON history
        json_file = tmp_path / "history.json"
        history = [
            {
                "timestamp": "2026-01-10T12:00:00",
                "coupling": {
                    "193-211": {"delta_mi": 0.59, "status": "NORMAL", "trend": "STABLE"},
                    "193-304": {"delta_mi": 0.07, "status": "NORMAL", "trend": "STABLE"}
                }
            },
            {
                "timestamp": "2026-01-10T12:10:00",
                "coupling": {
                    "193-211": {"delta_mi": 0.55, "status": "WARNING", "trend": "DECLINING"}
                }
            }
        ]
        with open(json_file, 'w') as f:
            json.dump(history, f)

        # Import
        count = db.import_from_json_history(json_file)
        assert count == 3  # 2 + 1 measurements

        # Verify
        recent = db.get_recent_coupling(hours=1)
        assert len(recent) == 3


class TestPredictionVerification:
    """Test prediction verification workflow."""

    @pytest.fixture
    def db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        db = MonitoringDB(db_path=db_path)
        yield db
        db.close()
        db_path.unlink()

    def test_verify_prediction(self, db):
        """Verify prediction with actual flare."""
        # Create prediction
        pred_id = db.insert_prediction(
            prediction_time="2026-01-10T10:00:00",
            predicted_class="M",
            trigger_status="ALERT"
        )

        # Create flare (happened 2 hours later)
        flare_id = db.insert_flare_event(
            start_time="2026-01-10T12:00:00",
            flare_class="M",
            magnitude=2.5
        )

        # Verify prediction
        success = db.verify_prediction(pred_id, flare_id, lead_time_hours=2.0)
        assert success

        # Check verification
        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,))
        row = cursor.fetchone()
        assert row['verified'] == 1
        assert row['actual_flare_id'] == flare_id
        assert row['lead_time_hours'] == pytest.approx(2.0)

    def test_prediction_accuracy(self, db):
        """Calculate prediction accuracy."""
        # Add some predictions
        pred1 = db.insert_prediction("2026-01-10T10:00:00", "M", 0.7)
        pred2 = db.insert_prediction("2026-01-10T14:00:00", "C", 0.5)

        # Add flare and verify first prediction
        flare_id = db.insert_flare_event("2026-01-10T12:00:00", "M", 2.5)
        db.verify_prediction(pred1, flare_id, 2.0)

        # Get accuracy
        accuracy = db.get_prediction_accuracy()
        assert accuracy['overall']['total_predictions'] == 2
        assert accuracy['overall']['verified_correct'] == 1
