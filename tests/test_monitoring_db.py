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

    def test_insert_coupling_quality_fields(self, db):
        """Quality fields (quality_ok, robustness_score, sync_delta_s) are stored."""
        row_id = db.insert_coupling(
            timestamp="2026-01-10T12:00:00",
            pair="193-211",
            delta_mi=0.59,
            status="NORMAL",
            quality_ok=True,
            robustness_score=3.5,
            sync_delta_s=12.0,
        )

        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM coupling_measurements WHERE id = ?", (row_id,))
        row = cursor.fetchone()
        assert row['quality_ok'] == 1
        assert row['robustness_score'] == pytest.approx(3.5)
        assert row['sync_delta_s'] == pytest.approx(12.0)

    def test_insert_coupling_quality_ok_false(self, db):
        """quality_ok=False for failed measurements."""
        row_id = db.insert_coupling(
            timestamp="2026-01-10T12:05:00",
            pair="193-304",
            delta_mi=0.07,
            status="DATA_ERROR",
            quality_ok=False,
            robustness_score=35.0,
        )

        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM coupling_measurements WHERE id = ?", (row_id,))
        row = cursor.fetchone()
        assert row['quality_ok'] == 0
        assert row['robustness_score'] == pytest.approx(35.0)

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
        from datetime import datetime, timezone, timedelta

        # Use dynamic timestamps relative to now (within last hour)
        now = datetime.now(timezone.utc)
        ts1 = (now - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%S")
        ts2 = (now - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%S")

        # Create sample JSON history
        json_file = tmp_path / "history.json"
        history = [
            {
                "timestamp": ts1,
                "coupling": {
                    "193-211": {"delta_mi": 0.59, "status": "NORMAL", "trend": "STABLE"},
                    "193-304": {"delta_mi": 0.07, "status": "NORMAL", "trend": "STABLE"}
                }
            },
            {
                "timestamp": ts2,
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


class TestNoaaAlerts:
    """Test NOAA alert parsing and storage."""

    @pytest.fixture
    def db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        db = MonitoringDB(db_path=db_path)
        yield db
        db.close()
        db_path.unlink()

    def test_insert_noaa_alert(self, db):
        """Insert NOAA alert with all fields."""
        row_id = db.insert_noaa_alert(
            alert_id="WARK04_2026-01-12T09:00:00",
            issue_time="2026-01-12T09:00:00",
            message_code="WARK04",
            alert_type="WARNING",
            kp_observed=3,
            kp_predicted=4,
            g_scale=2,
            source_region="AR3842",
            message="Geomagnetic K-index of 4 expected"
        )
        assert row_id > 0

        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM noaa_alerts WHERE id = ?", (row_id,))
        row = cursor.fetchone()
        assert row['message_code'] == 'WARK04'
        assert row['alert_type'] == 'WARNING'
        assert row['kp_predicted'] == 4
        assert row['g_scale'] == 2

    def test_noaa_alert_duplicate_prevention(self, db):
        """Duplicate alerts are ignored."""
        db.insert_noaa_alert(
            alert_id="ALTEF3_2026-01-12T11:00:00",
            issue_time="2026-01-12T11:00:00",
            message_code="ALTEF3",
            alert_type="ALERT"
        )
        # Try inserting same alert_id
        db.insert_noaa_alert(
            alert_id="ALTEF3_2026-01-12T11:00:00",
            issue_time="2026-01-12T11:00:00",
            message_code="ALTEF3_modified",
            alert_type="ALERT"
        )

        cursor = db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM noaa_alerts")
        assert cursor.fetchone()[0] == 1  # Only one entry

    def test_noaa_alert_scales(self, db):
        """Test G/S/R scale storage."""
        db.insert_noaa_alert(
            alert_id="test_scales",
            issue_time="2026-01-12T10:00:00",
            g_scale=3,
            s_scale=2,
            r_scale=1
        )

        cursor = db.conn.cursor()
        cursor.execute("SELECT g_scale, s_scale, r_scale FROM noaa_alerts WHERE alert_id = ?",
                       ("test_scales",))
        row = cursor.fetchone()
        assert row['g_scale'] == 3
        assert row['s_scale'] == 2
        assert row['r_scale'] == 1


class TestNoaaAlertParsing:
    """Test NOAA alert message parsing."""

    def test_parse_kp_warning(self):
        """Parse Kp warning message."""
        from scripts.early_warning import parse_noaa_alert

        alert = {
            'product_id': 'K04W',
            'message': '''Space Weather Message Code: WARK04
Serial Number: 5218
Issue Time: 2026 Jan 12 0902 UTC

EXTENDED WARNING: Geomagnetic K-index of 4 expected
Extension to Serial Number: 5217
Valid From: 2026 Jan 11 1800 UTC
Valid To: 2026 Jan 12 0900 UTC'''
        }
        result = parse_noaa_alert(alert)

        assert result['message_code'] == 'WARK04'
        assert result['alert_type'] == 'WARNING'
        assert result['kp_predicted'] == 4

    def test_parse_electron_flux_alert(self):
        """Parse electron flux alert message."""
        from scripts.early_warning import parse_noaa_alert

        alert = {
            'product_id': 'EF3A',
            'message': '''Space Weather Message Code: ALTEF3
Serial Number: 3598
Issue Time: 2026 Jan 12 1101 UTC

ALERT: Electron 2MeV Integral Flux exceeded 1000pfu
Threshold Reached: 2026 Jan 12 1040 UTC'''
        }
        result = parse_noaa_alert(alert)

        assert result['message_code'] == 'ALTEF3'
        assert result['alert_type'] == 'ALERT'
        assert result['kp_predicted'] == 3  # Extracted from ALTEF3

    def test_parse_geomagnetic_storm_scales(self):
        """Parse G-scale from storm message."""
        from scripts.early_warning import parse_noaa_alert

        alert = {
            'product_id': 'WATA50',
            'message': '''Space Weather Message Code: WATA50
WATCH: Geomagnetic Storm Category G2 - Moderate
Expected: 2026 Jan 12'''
        }
        result = parse_noaa_alert(alert)

        assert result['g_scale'] == 2
        assert result['alert_type'] == 'WATCH'

    def test_parse_radio_blackout(self):
        """Parse R-scale from radio blackout."""
        from scripts.early_warning import parse_noaa_alert

        alert = {
            'product_id': 'ALTTP2',
            'message': '''Space Weather Message Code: ALTTP2
ALERT: Type II Radio Emission
R1 - Minor Radio Blackout observed'''
        }
        result = parse_noaa_alert(alert)

        assert result['r_scale'] == 1
        assert result['alert_type'] == 'ALERT'

    def test_parse_active_region(self):
        """Parse active region from message."""
        from scripts.early_warning import parse_noaa_alert

        alert = {
            'product_id': 'SUMX01',
            'message': '''Space Weather Message Code: SUMX01
Summary: X1.0 Flare from Region 3842
Peak Time: 2026 Jan 12 0530 UTC'''
        }
        result = parse_noaa_alert(alert)

        assert result['source_region'] == 'AR3842'

    def test_parse_empty_message(self):
        """Handle empty/missing fields gracefully."""
        from scripts.early_warning import parse_noaa_alert

        alert = {'product_id': 'TEST', 'message': ''}
        result = parse_noaa_alert(alert)

        assert result['message_code'] == 'TEST'  # Falls back to product_id
        assert result['kp_observed'] is None
        assert result['kp_predicted'] is None
        assert result['g_scale'] is None


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


class TestDonkiFields:
    """Test DONKI-specific flare event fields."""

    @pytest.fixture
    def db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        db = MonitoringDB(db_path=db_path)
        yield db
        db.close()
        db_path.unlink()

    def test_flare_with_donki_fields(self, db):
        """Insert flare with active region, CME links, and DONKI metadata."""
        cursor = db.conn.cursor()

        # Insert flare with all DONKI fields
        cursor.execute("""
            INSERT INTO flare_events (
                start_time, peak_time, class, magnitude, source, location,
                active_region_num, linked_cme_ids, donki_link, donki_flr_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "2026-01-18T17:27:00+00:00",
            "2026-01-18T18:09:00+00:00",
            "X",
            1.9,
            "DONKI",
            "S15E20",
            14341,
            '["2026-01-18T18:09:00-CME-001"]',
            "https://kauai.ccmc.gsfc.nasa.gov/DONKI/view/FLR/44029/-1",
            "2026-01-18T17:27:00-FLR-001"
        ))
        db.conn.commit()

        # Verify storage
        cursor.execute("SELECT * FROM flare_events WHERE class = 'X'")
        row = cursor.fetchone()

        assert row['active_region_num'] == 14341
        assert '2026-01-18T18:09:00-CME-001' in row['linked_cme_ids']
        assert 'DONKI' in row['donki_link']
        assert row['donki_flr_id'] == '2026-01-18T17:27:00-FLR-001'

    def test_flare_with_multiple_cmes(self, db):
        """Flare with multiple linked CMEs."""
        import json
        cursor = db.conn.cursor()

        cme_ids = ["2026-01-10T12:00:00-CME-001", "2026-01-10T14:00:00-CME-002"]

        cursor.execute("""
            INSERT INTO flare_events (
                start_time, class, magnitude, linked_cme_ids
            ) VALUES (?, ?, ?, ?)
        """, (
            "2026-01-10T11:30:00+00:00",
            "X",
            5.0,
            json.dumps(cme_ids)
        ))
        db.conn.commit()

        cursor.execute("SELECT linked_cme_ids FROM flare_events WHERE magnitude = 5.0")
        row = cursor.fetchone()

        parsed = json.loads(row['linked_cme_ids'])
        assert len(parsed) == 2
        assert "CME-001" in parsed[0]
        assert "CME-002" in parsed[1]

    def test_query_by_active_region(self, db):
        """Query flares by active region number."""
        cursor = db.conn.cursor()

        # Insert multiple flares from same AR
        for i, mag in enumerate([1.5, 2.0, 3.3]):
            cursor.execute("""
                INSERT INTO flare_events (
                    start_time, class, magnitude, active_region_num
                ) VALUES (?, ?, ?, ?)
            """, (
                f"2026-01-1{i}T12:00:00+00:00",
                "M",
                mag,
                14341
            ))

        # Insert flare from different AR
        cursor.execute("""
            INSERT INTO flare_events (
                start_time, class, magnitude, active_region_num
            ) VALUES (?, ?, ?, ?)
        """, ("2026-01-15T12:00:00+00:00", "M", 1.0, 14298))

        db.conn.commit()

        # Query by AR
        cursor.execute("""
            SELECT COUNT(*) as count, SUM(magnitude) as total_mag
            FROM flare_events WHERE active_region_num = 14341
        """)
        row = cursor.fetchone()

        assert row['count'] == 3
        assert row['total_mag'] == pytest.approx(6.8)

    def test_flare_without_donki_fields(self, db):
        """Flares without DONKI fields should still work."""
        row_id = db.insert_flare_event(
            start_time="2026-01-10T12:00:00",
            flare_class="C",
            magnitude=5.0
        )
        assert row_id > 0

        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM flare_events WHERE id = ?", (row_id,))
        row = cursor.fetchone()

        assert row['class'] == 'C'
        assert row['active_region_num'] is None
        assert row['linked_cme_ids'] is None
