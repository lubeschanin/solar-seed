#!/usr/bin/env python3
"""
Solar Monitoring Database
=========================

SQLite database for persistent storage of monitoring data.
Enables correlation analysis between coupling anomalies and flares.

Usage:
    from monitoring_db import MonitoringDB

    db = MonitoringDB()
    db.insert_goes_xray(timestamp, flux, flare_class, magnitude)
    db.insert_coupling(timestamp, pair, delta_mi, residual, status, trend)

    # Query recent data
    recent = db.get_recent_coupling(hours=24)

    # Correlation analysis
    correlations = db.get_coupling_before_flares(hours_before=6)
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional
import json


class MonitoringDB:
    """SQLite database for solar monitoring data."""

    DEFAULT_PATH = Path("results/early_warning/monitoring.db")

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or self.DEFAULT_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Connect to database."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON")

    def _create_tables(self):
        """Create database schema."""
        cursor = self.conn.cursor()

        # GOES X-ray measurements
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS goes_xray (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL UNIQUE,
                flux REAL NOT NULL,
                flare_class TEXT,
                magnitude REAL,
                energy TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Solar wind data (DSCOVR)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS solar_wind (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL UNIQUE,
                speed REAL,
                density REAL,
                temperature REAL,
                bx REAL,
                by REAL,
                bz REAL,
                bt REAL,
                risk_level INTEGER,
                risk_description TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # =================================================================
        # DIMENSION TABLES (Normalization)
        # =================================================================

        # Channels lookup table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS channels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wavelength INTEGER NOT NULL,
                instrument TEXT NOT NULL DEFAULT 'AIA',
                name TEXT,
                temperature_mk REAL,
                region TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(wavelength, instrument)
            )
        """)

        # Pairs lookup table (normalized)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ch_a_id INTEGER NOT NULL REFERENCES channels(id),
                ch_b_id INTEGER NOT NULL REFERENCES channels(id),
                pair_name TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ch_a_id, ch_b_id),
                CHECK(ch_a_id < ch_b_id)
            )
        """)

        # Seed channels if empty
        cursor.execute("SELECT COUNT(*) FROM channels")
        if cursor.fetchone()[0] == 0:
            aia_channels = [
                (94, 'AIA', '94 Å', 6.3, 'Flares'),
                (131, 'AIA', '131 Å', 10.0, 'Flares (hot)'),
                (171, 'AIA', '171 Å', 0.6, 'Quiet Corona'),
                (193, 'AIA', '193 Å', 1.2, 'Corona'),
                (211, 'AIA', '211 Å', 2.0, 'Active Regions'),
                (304, 'AIA', '304 Å', 0.05, 'Chromosphere'),
                (335, 'AIA', '335 Å', 2.5, 'Active Regions (hot)'),
            ]
            cursor.executemany("""
                INSERT INTO channels (wavelength, instrument, name, temperature_mk, region)
                VALUES (?, ?, ?, ?, ?)
            """, aia_channels)

        # Seed pairs if empty
        cursor.execute("SELECT COUNT(*) FROM pairs")
        if cursor.fetchone()[0] == 0:
            cursor.execute("SELECT id, wavelength FROM channels WHERE instrument = 'AIA'")
            ch_map = {row[1]: row[0] for row in cursor.fetchall()}
            common_pairs = [
                (171, 193), (171, 211), (171, 304),
                (193, 211), (193, 304), (193, 335),
                (211, 304), (211, 335),
                (94, 131), (94, 193), (131, 193),
            ]
            for a, b in common_pairs:
                if a in ch_map and b in ch_map:
                    ch_a = min(ch_map[a], ch_map[b])
                    ch_b = max(ch_map[a], ch_map[b])
                    pair_name = f"{min(a,b)}-{max(a,b)}"
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO pairs (ch_a_id, ch_b_id, pair_name)
                            VALUES (?, ?, ?)
                        """, (ch_a, ch_b, pair_name))
                    except:
                        pass

        # =================================================================
        # OBSERVATION TABLES
        # =================================================================

        # Coupling measurements
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coupling_measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                pair TEXT NOT NULL,
                pair_id INTEGER REFERENCES pairs(id),
                delta_mi REAL NOT NULL,
                mi_original REAL,
                residual REAL,
                deviation_pct REAL,
                status TEXT,
                trend TEXT,
                slope_pct_per_hour REAL,
                acceleration REAL,
                confidence TEXT,
                n_points INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(timestamp, pair)
            )
        """)

        # Detected flare events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flare_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time DATETIME NOT NULL,
                peak_time DATETIME,
                end_time DATETIME,
                class TEXT NOT NULL,
                magnitude REAL NOT NULL,
                peak_flux REAL,
                location TEXT,
                source TEXT DEFAULT 'auto',
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Predictions and their outcomes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_time DATETIME NOT NULL,
                predicted_class TEXT,
                predicted_probability REAL,
                trigger_pair TEXT,
                trigger_status TEXT,
                trigger_residual REAL,
                trigger_trend TEXT,
                actual_flare_id INTEGER REFERENCES flare_events(id),
                verified BOOLEAN DEFAULT FALSE,
                lead_time_hours REAL,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # NOAA alerts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS noaa_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE,
                issue_time DATETIME NOT NULL,
                message_code TEXT,
                alert_type TEXT CHECK(alert_type IN ('WARNING', 'ALERT', 'WATCH', 'SUMMARY', 'FORECAST')),
                kp_observed REAL,
                kp_predicted REAL,
                kp_max_24h REAL,
                valid_from DATETIME,
                valid_to DATETIME,
                g_scale INTEGER CHECK(g_scale BETWEEN 0 AND 5),
                s_scale INTEGER CHECK(s_scale BETWEEN 0 AND 5),
                r_scale INTEGER CHECK(r_scale BETWEEN 0 AND 5),
                source_region TEXT,
                message TEXT,
                raw_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Pipeline runs (reproducibility)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                ended_at DATETIME,
                git_commit TEXT,
                config_hash TEXT,
                pipeline_version TEXT DEFAULT 'v0.5',
                status TEXT CHECK(status IN ('running', 'completed', 'failed', 'interrupted')) DEFAULT 'running',
                n_measurements INTEGER DEFAULT 0,
                n_predictions INTEGER DEFAULT 0,
                n_divergences INTEGER DEFAULT 0,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Run configuration (reproducibility)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS run_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_hash TEXT UNIQUE NOT NULL,
                config_json TEXT NOT NULL,
                description TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migrations for new columns
        for col, dtype in [
            ('quality_ok', 'BOOLEAN'),
            ('robustness_score', 'REAL'),
            ('sync_delta_s', 'REAL'),
            ('run_id', 'INTEGER'),
            ('pair_id', 'INTEGER'),
        ]:
            try:
                cursor.execute(f"ALTER TABLE coupling_measurements ADD COLUMN {col} {dtype}")
            except sqlite3.OperationalError:
                pass

        # Migrate existing pair strings to pair_id
        try:
            cursor.execute("""
                UPDATE coupling_measurements
                SET pair_id = (SELECT p.id FROM pairs p WHERE p.pair_name = coupling_measurements.pair)
                WHERE pair_id IS NULL AND pair IS NOT NULL
            """)
        except sqlite3.OperationalError:
            pass

        # Create indices for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_goes_timestamp ON goes_xray(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_wind_timestamp ON solar_wind(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_coupling_timestamp ON coupling_measurements(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_coupling_pair ON coupling_measurements(pair)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_coupling_status ON coupling_measurements(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_coupling_pair_id ON coupling_measurements(pair_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_coupling_run_time ON coupling_measurements(run_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_flares_time ON flare_events(start_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_flares_class ON flare_events(class)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(prediction_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_channels_wavelength ON channels(wavelength)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pairs_name ON pairs(pair_name)")

        self.conn.commit()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # =========================================================================
    # INSERT METHODS
    # =========================================================================

    def insert_goes_xray(self, timestamp: str, flux: float,
                         flare_class: str = None, magnitude: float = None,
                         energy: str = None) -> int:
        """Insert GOES X-ray measurement."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO goes_xray
                (timestamp, flux, flare_class, magnitude, energy)
                VALUES (?, ?, ?, ?, ?)
            """, (timestamp, flux, flare_class, magnitude, energy))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error inserting GOES data: {e}")
            return -1

    def insert_solar_wind(self, timestamp: str, speed: float = None,
                          density: float = None, temperature: float = None,
                          bx: float = None, by: float = None, bz: float = None,
                          bt: float = None, risk_level: int = None,
                          risk_description: str = None) -> int:
        """Insert solar wind measurement."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO solar_wind
                (timestamp, speed, density, temperature, bx, by, bz, bt,
                 risk_level, risk_description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, speed, density, temperature, bx, by, bz, bt,
                  risk_level, risk_description))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error inserting solar wind data: {e}")
            return -1

    def insert_coupling(self, timestamp: str, pair: str, delta_mi: float,
                        mi_original: float = None, residual: float = None,
                        deviation_pct: float = None, status: str = None,
                        trend: str = None, slope_pct_per_hour: float = None,
                        acceleration: float = None, confidence: str = None,
                        n_points: int = None, quality_ok: bool = None,
                        robustness_score: float = None, sync_delta_s: float = None,
                        run_id: int = None, pair_id: int = None) -> int:
        """Insert coupling measurement with quality fields and pair normalization."""
        cursor = self.conn.cursor()

        # Auto-resolve pair_id from pair string if not provided
        if pair_id is None and pair:
            cursor.execute("SELECT id FROM pairs WHERE pair_name = ?", (pair,))
            row = cursor.fetchone()
            if row:
                pair_id = row[0]

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO coupling_measurements
                (timestamp, pair, pair_id, delta_mi, mi_original, residual, deviation_pct,
                 status, trend, slope_pct_per_hour, acceleration, confidence, n_points,
                 quality_ok, robustness_score, sync_delta_s, run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, pair, pair_id, delta_mi, mi_original, residual, deviation_pct,
                  status, trend, slope_pct_per_hour, acceleration, confidence, n_points,
                  quality_ok, robustness_score, sync_delta_s, run_id))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error inserting coupling data: {e}")
            return -1

    def insert_flare_event(self, start_time: str, flare_class: str,
                           magnitude: float, peak_time: str = None,
                           end_time: str = None, peak_flux: float = None,
                           location: str = None, source: str = 'auto',
                           notes: str = None) -> int:
        """Insert detected flare event."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO flare_events
                (start_time, peak_time, end_time, class, magnitude, peak_flux,
                 location, source, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (start_time, peak_time, end_time, flare_class, magnitude,
                  peak_flux, location, source, notes))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error inserting flare event: {e}")
            return -1

    def insert_prediction(self, prediction_time: str, predicted_class: str = None,
                          predicted_probability: float = None,
                          trigger_pair: str = None, trigger_status: str = None,
                          trigger_residual: float = None, trigger_trend: str = None,
                          notes: str = None) -> int:
        """Insert a prediction."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO predictions
                (prediction_time, predicted_class, predicted_probability,
                 trigger_pair, trigger_status, trigger_residual, trigger_trend, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (prediction_time, predicted_class, predicted_probability,
                  trigger_pair, trigger_status, trigger_residual, trigger_trend, notes))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error inserting prediction: {e}")
            return -1

    def insert_noaa_alert(self, alert_id: str, issue_time: str,
                          message_code: str = None, alert_type: str = None,
                          kp_observed: float = None, kp_predicted: float = None,
                          kp_max_24h: float = None,
                          valid_from: str = None, valid_to: str = None,
                          g_scale: int = None, s_scale: int = None, r_scale: int = None,
                          source_region: str = None, message: str = None,
                          raw_json: str = None) -> int:
        """Insert NOAA Space Weather alert with Kp and scale tracking."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO noaa_alerts
                (alert_id, issue_time, message_code, alert_type,
                 kp_observed, kp_predicted, kp_max_24h,
                 valid_from, valid_to,
                 g_scale, s_scale, r_scale,
                 source_region, message, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (alert_id, issue_time, message_code, alert_type,
                  kp_observed, kp_predicted, kp_max_24h,
                  valid_from, valid_to,
                  g_scale, s_scale, r_scale,
                  source_region, message, raw_json))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error inserting NOAA alert: {e}")
            return -1

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_recent_goes(self, hours: int = 24) -> list[dict]:
        """Get recent GOES X-ray data."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM goes_xray
            WHERE timestamp >= datetime('now', ?)
            ORDER BY timestamp DESC
        """, (f'-{hours} hours',))
        return [dict(row) for row in cursor.fetchall()]

    def get_recent_solar_wind(self, hours: int = 24) -> list[dict]:
        """Get recent solar wind data."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM solar_wind
            WHERE timestamp >= datetime('now', ?)
            ORDER BY timestamp DESC
        """, (f'-{hours} hours',))
        return [dict(row) for row in cursor.fetchall()]

    def get_recent_coupling(self, hours: int = 24, pair: str = None) -> list[dict]:
        """Get recent coupling measurements."""
        cursor = self.conn.cursor()
        if pair:
            cursor.execute("""
                SELECT * FROM coupling_measurements
                WHERE timestamp >= datetime('now', ?)
                AND pair = ?
                ORDER BY timestamp DESC
            """, (f'-{hours} hours', pair))
        else:
            cursor.execute("""
                SELECT * FROM coupling_measurements
                WHERE timestamp >= datetime('now', ?)
                ORDER BY timestamp DESC
            """, (f'-{hours} hours',))
        return [dict(row) for row in cursor.fetchall()]

    def get_flare_events(self, days: int = 30, min_class: str = 'C') -> list[dict]:
        """Get flare events, optionally filtered by minimum class."""
        class_order = {'A': 0, 'B': 1, 'C': 2, 'M': 3, 'X': 4}
        min_order = class_order.get(min_class, 0)

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM flare_events
            WHERE start_time >= datetime('now', ?)
            ORDER BY start_time DESC
        """, (f'-{days} days',))

        results = []
        for row in cursor.fetchall():
            row_dict = dict(row)
            if class_order.get(row_dict['class'], 0) >= min_order:
                results.append(row_dict)
        return results

    def get_coupling_before_flares(self, hours_before: int = 6) -> list[dict]:
        """Get coupling measurements in the hours before each flare."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT
                f.id as flare_id,
                f.class as flare_class,
                f.magnitude as flare_magnitude,
                f.start_time as flare_time,
                c.pair,
                c.delta_mi,
                c.residual,
                c.status,
                c.trend,
                c.timestamp as coupling_time,
                (julianday(f.start_time) - julianday(c.timestamp)) * 24 as hours_before_flare
            FROM flare_events f
            JOIN coupling_measurements c
                ON c.timestamp BETWEEN datetime(f.start_time, ?)
                                   AND f.start_time
            ORDER BY f.start_time DESC, hours_before_flare ASC
        """, (f'-{hours_before} hours',))
        return [dict(row) for row in cursor.fetchall()]

    def get_prediction_accuracy(self) -> dict:
        """Calculate prediction accuracy statistics."""
        cursor = self.conn.cursor()

        # Overall stats
        cursor.execute("""
            SELECT
                COUNT(*) as total_predictions,
                SUM(CASE WHEN actual_flare_id IS NOT NULL THEN 1 ELSE 0 END) as with_flare,
                SUM(CASE WHEN verified THEN 1 ELSE 0 END) as verified_correct,
                AVG(CASE WHEN actual_flare_id IS NOT NULL THEN lead_time_hours END) as avg_lead_time
            FROM predictions
        """)
        overall = dict(cursor.fetchone())

        # By predicted class
        cursor.execute("""
            SELECT
                predicted_class,
                COUNT(*) as total,
                SUM(CASE WHEN actual_flare_id IS NOT NULL THEN 1 ELSE 0 END) as hits,
                AVG(lead_time_hours) as avg_lead_time
            FROM predictions
            GROUP BY predicted_class
        """)
        by_class = [dict(row) for row in cursor.fetchall()]

        return {
            'overall': overall,
            'by_class': by_class
        }

    def verify_prediction(self, prediction_id: int, flare_id: int,
                          lead_time_hours: float) -> bool:
        """Link a prediction to an actual flare event."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                UPDATE predictions
                SET actual_flare_id = ?, verified = TRUE, lead_time_hours = ?
                WHERE id = ?
            """, (flare_id, lead_time_hours, prediction_id))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error verifying prediction: {e}")
            return False

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================

    def get_coupling_statistics(self, pair: str, days: int = 7) -> dict:
        """Calculate coupling statistics for a pair."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT
                COUNT(*) as n_measurements,
                AVG(delta_mi) as mean_delta_mi,
                MIN(delta_mi) as min_delta_mi,
                MAX(delta_mi) as max_delta_mi,
                AVG(residual) as mean_residual,
                SUM(CASE WHEN status = 'ALERT' THEN 1 ELSE 0 END) as n_alerts,
                SUM(CASE WHEN status = 'WARNING' THEN 1 ELSE 0 END) as n_warnings
            FROM coupling_measurements
            WHERE pair = ?
            AND timestamp >= datetime('now', ?)
        """, (pair, f'-{days} days'))
        return dict(cursor.fetchone())

    def detect_flare_from_goes(self, threshold_class: str = 'C') -> list[dict]:
        """Auto-detect flare events from GOES X-ray data."""
        thresholds = {'B': 1e-7, 'C': 1e-6, 'M': 1e-5, 'X': 1e-4}
        min_flux = thresholds.get(threshold_class, 1e-6)

        cursor = self.conn.cursor()
        # Find peaks above threshold that aren't already recorded
        cursor.execute("""
            SELECT g.timestamp, g.flux, g.flare_class, g.magnitude
            FROM goes_xray g
            WHERE g.flux >= ?
            AND NOT EXISTS (
                SELECT 1 FROM flare_events f
                WHERE abs(julianday(f.start_time) - julianday(g.timestamp)) < 0.02
            )
            ORDER BY g.timestamp
        """, (min_flux,))

        return [dict(row) for row in cursor.fetchall()]

    def get_alert_rate(self, days: int = 7) -> dict:
        """Calculate alert rates over time."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT
                date(timestamp) as date,
                COUNT(*) as total_measurements,
                SUM(CASE WHEN status = 'ALERT' THEN 1 ELSE 0 END) as alerts,
                SUM(CASE WHEN status = 'WARNING' THEN 1 ELSE 0 END) as warnings,
                AVG(residual) as avg_residual
            FROM coupling_measurements
            WHERE timestamp >= datetime('now', ?)
            GROUP BY date(timestamp)
            ORDER BY date DESC
        """, (f'-{days} days',))
        return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # MIGRATION / IMPORT
    # =========================================================================

    def import_from_json_history(self, json_path: Path) -> int:
        """Import data from existing JSON coupling history."""
        if not json_path.exists():
            print(f"JSON file not found: {json_path}")
            return 0

        with open(json_path) as f:
            history = json.load(f)

        count = 0
        for entry in history:
            timestamp = entry.get('timestamp')
            coupling = entry.get('coupling', {})

            for pair, data in coupling.items():
                self.insert_coupling(
                    timestamp=timestamp,
                    pair=pair,
                    delta_mi=data.get('delta_mi'),
                    mi_original=data.get('mi_original'),
                    residual=data.get('residual'),
                    deviation_pct=data.get('deviation_pct'),
                    status=data.get('status'),
                    trend=data.get('trend'),
                    slope_pct_per_hour=data.get('slope_pct_per_hour'),
                    acceleration=data.get('acceleration'),
                    confidence=data.get('confidence'),
                    n_points=data.get('n_points')
                )
                count += 1

        print(f"Imported {count} coupling measurements from {json_path}")
        return count

    def get_database_stats(self) -> dict:
        """Get database statistics."""
        cursor = self.conn.cursor()

        stats = {}
        tables = ['goes_xray', 'solar_wind', 'coupling_measurements',
                  'flare_events', 'predictions', 'noaa_alerts']

        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]

        # Date range
        cursor.execute("""
            SELECT MIN(timestamp), MAX(timestamp) FROM goes_xray
        """)
        row = cursor.fetchone()
        stats['goes_range'] = {'min': row[0], 'max': row[1]}

        cursor.execute("""
            SELECT MIN(timestamp), MAX(timestamp) FROM coupling_measurements
        """)
        row = cursor.fetchone()
        stats['coupling_range'] = {'min': row[0], 'max': row[1]}

        return stats


# =============================================================================
# CLI for database management
# =============================================================================

def main():
    """Database management CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Solar Monitoring Database")
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--import-json', type=str, help='Import from JSON history file')
    parser.add_argument('--accuracy', action='store_true', help='Show prediction accuracy')
    parser.add_argument('--alerts', type=int, default=0, help='Show alert rate for N days')
    args = parser.parse_args()

    with MonitoringDB() as db:
        if args.stats:
            stats = db.get_database_stats()
            print("\n  Database Statistics:")
            print("  " + "-" * 40)
            for table, count in stats.items():
                if isinstance(count, dict):
                    print(f"  {table}: {count}")
                else:
                    print(f"  {table}: {count} rows")

        if args.import_json:
            db.import_from_json_history(Path(args.import_json))

        if args.accuracy:
            accuracy = db.get_prediction_accuracy()
            print("\n  Prediction Accuracy:")
            print("  " + "-" * 40)
            print(f"  Overall: {accuracy['overall']}")
            for cls in accuracy['by_class']:
                print(f"  {cls['predicted_class']}: {cls}")

        if args.alerts > 0:
            rates = db.get_alert_rate(days=args.alerts)
            print(f"\n  Alert Rate (last {args.alerts} days):")
            print("  " + "-" * 40)
            for day in rates:
                print(f"  {day['date']}: {day['alerts']} alerts, {day['warnings']} warnings")


if __name__ == "__main__":
    main()
