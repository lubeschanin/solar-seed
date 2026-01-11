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

        # Coupling measurements
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coupling_measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                pair TEXT NOT NULL,
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
                alert_type TEXT,
                message TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Phase classifier divergence events
        # Tracks when GOES-only and ΔMI-integrated classifiers disagree
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS phase_divergence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                phase_goes TEXT NOT NULL,
                phase_experimental TEXT NOT NULL,
                reason_goes TEXT,
                reason_experimental TEXT,
                divergence_type TEXT,
                goes_flux REAL,
                goes_class TEXT,
                goes_rising INTEGER,
                max_z_score REAL,
                trigger_pair TEXT,
                validated INTEGER DEFAULT NULL,
                validation_type TEXT DEFAULT NULL,
                flare_within_24h INTEGER DEFAULT NULL,
                flare_class TEXT DEFAULT NULL,
                flare_time DATETIME DEFAULT NULL,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add new columns to existing table (migration for existing DBs)
        try:
            cursor.execute("ALTER TABLE phase_divergence ADD COLUMN divergence_type TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            cursor.execute("ALTER TABLE phase_divergence ADD COLUMN goes_rising INTEGER")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE phase_divergence ADD COLUMN validated INTEGER")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE phase_divergence ADD COLUMN validation_type TEXT")
        except sqlite3.OperationalError:
            pass

        # Create indices for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_goes_timestamp ON goes_xray(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_wind_timestamp ON solar_wind(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_coupling_timestamp ON coupling_measurements(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_coupling_pair ON coupling_measurements(pair)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_coupling_status ON coupling_measurements(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_flares_time ON flare_events(start_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_flares_class ON flare_events(class)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(prediction_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_divergence_timestamp ON phase_divergence(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_divergence_phases ON phase_divergence(phase_goes, phase_experimental)")

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
                        n_points: int = None) -> int:
        """Insert coupling measurement."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO coupling_measurements
                (timestamp, pair, delta_mi, mi_original, residual, deviation_pct,
                 status, trend, slope_pct_per_hour, acceleration, confidence, n_points)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, pair, delta_mi, mi_original, residual, deviation_pct,
                  status, trend, slope_pct_per_hour, acceleration, confidence, n_points))
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
                          alert_type: str = None, message: str = None) -> int:
        """Insert NOAA alert."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO noaa_alerts
                (alert_id, issue_time, alert_type, message)
                VALUES (?, ?, ?, ?)
            """, (alert_id, issue_time, alert_type, message))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error inserting NOAA alert: {e}")
            return -1

    def insert_phase_divergence(
        self,
        timestamp: str,
        phase_goes: str,
        phase_experimental: str,
        reason_goes: str = None,
        reason_experimental: str = None,
        divergence_type: str = None,
        goes_flux: float = None,
        goes_class: str = None,
        goes_rising: bool = None,
        max_z_score: float = None,
        trigger_pair: str = None,
        notes: str = None,
    ) -> int:
        """
        Insert a phase divergence event.

        Called when GOES-only and ΔMI-integrated classifiers disagree.
        This data enables empirical validation of which classifier is more predictive.

        Args:
            divergence_type: PRECURSOR, POST_EVENT, or STRUCTURAL_EVENT
            goes_rising: Whether GOES was trending up at time of divergence
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO phase_divergence
                (timestamp, phase_goes, phase_experimental, reason_goes,
                 reason_experimental, divergence_type, goes_flux, goes_class,
                 goes_rising, max_z_score, trigger_pair, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, phase_goes, phase_experimental, reason_goes,
                  reason_experimental, divergence_type, goes_flux, goes_class,
                  1 if goes_rising else 0 if goes_rising is not None else None,
                  max_z_score, trigger_pair, notes))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error inserting phase divergence: {e}")
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

    def extract_flare_events_from_goes(self, min_class: str = 'C', gap_minutes: int = 30) -> int:
        """
        Extract discrete flare events from GOES X-ray measurements.

        Identifies periods of elevated flux, groups consecutive readings,
        and stores as flare_events.

        Args:
            min_class: Minimum flare class to extract ('C', 'M', or 'X')
            gap_minutes: Max gap between readings to consider same event

        Returns:
            Number of flare events extracted
        """
        from datetime import datetime, timedelta

        # Flux thresholds by class
        thresholds = {'C': 1e-6, 'M': 1e-5, 'X': 1e-4}
        min_flux = thresholds.get(min_class, 1e-6)

        cursor = self.conn.cursor()

        # Get all GOES readings above threshold, ordered by time
        cursor.execute("""
            SELECT timestamp, flux, flare_class
            FROM goes_xray
            WHERE flux >= ?
            ORDER BY timestamp
        """, (min_flux,))

        readings = cursor.fetchall()
        if not readings:
            return 0

        # Group into events based on time gaps
        events = []
        current_event = []

        for i, row in enumerate(readings):
            ts = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))

            if not current_event:
                current_event = [(ts, row['flux'], row['flare_class'])]
            else:
                last_ts = current_event[-1][0]
                gap = (ts - last_ts).total_seconds() / 60

                if gap <= gap_minutes:
                    current_event.append((ts, row['flux'], row['flare_class']))
                else:
                    # Save current event and start new one
                    events.append(current_event)
                    current_event = [(ts, row['flux'], row['flare_class'])]

        # Don't forget last event
        if current_event:
            events.append(current_event)

        # Extract event properties and insert
        inserted = 0
        for event in events:
            if len(event) < 1:
                continue

            # Find peak
            peak_idx = max(range(len(event)), key=lambda i: event[i][1])
            peak_ts, peak_flux, peak_class = event[peak_idx]

            start_ts = event[0][0]
            end_ts = event[-1][0]

            # Parse class and magnitude
            flare_class = peak_class[0] if peak_class else 'C'
            try:
                magnitude = float(peak_class[1:]) if len(peak_class) > 1 else 1.0
            except ValueError:
                magnitude = 1.0

            # Check if already exists (within 5 minutes of peak)
            cursor.execute("""
                SELECT id FROM flare_events
                WHERE peak_time BETWEEN datetime(?, '-5 minutes') AND datetime(?, '+5 minutes')
            """, (peak_ts.isoformat(), peak_ts.isoformat()))

            if cursor.fetchone():
                continue  # Already exists

            # Insert
            cursor.execute("""
                INSERT INTO flare_events (start_time, peak_time, end_time, class, magnitude, peak_flux, source)
                VALUES (?, ?, ?, ?, ?, ?, 'auto_extracted')
            """, (
                start_ts.isoformat(),
                peak_ts.isoformat(),
                end_ts.isoformat(),
                flare_class,
                magnitude,
                peak_flux
            ))
            inserted += 1

        self.conn.commit()
        return inserted

    def get_recent_divergences(self, hours: int = 24) -> list[dict]:
        """Get recent phase divergence events."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM phase_divergence
            WHERE timestamp >= datetime('now', ?)
            ORDER BY timestamp DESC
        """, (f'-{hours} hours',))
        return [dict(row) for row in cursor.fetchall()]

    def get_divergence_statistics(self, days: int = 7) -> dict:
        """
        Calculate divergence statistics for classifier comparison.

        Returns counts of divergence types and correlation with flares.
        """
        cursor = self.conn.cursor()

        # Overall counts
        cursor.execute("""
            SELECT
                COUNT(*) as total_divergences,
                COUNT(DISTINCT date(timestamp)) as days_with_divergence,
                SUM(CASE WHEN flare_within_24h = 1 THEN 1 ELSE 0 END) as followed_by_flare,
                SUM(CASE WHEN flare_within_24h = 0 THEN 1 ELSE 0 END) as no_flare
            FROM phase_divergence
            WHERE timestamp >= datetime('now', ?)
        """, (f'-{days} days',))
        overall = dict(cursor.fetchone())

        # By divergence type (PRECURSOR, POST_EVENT, UNCONFIRMED)
        cursor.execute("""
            SELECT
                COALESCE(divergence_type, 'UNKNOWN') as divergence_type,
                COUNT(*) as count,
                SUM(CASE WHEN flare_within_24h = 1 THEN 1 ELSE 0 END) as followed_by_flare
            FROM phase_divergence
            WHERE timestamp >= datetime('now', ?)
            GROUP BY divergence_type
            ORDER BY count DESC
        """, (f'-{days} days',))
        by_divergence_type = [dict(row) for row in cursor.fetchall()]

        # By phase pair (which phases disagreed)
        cursor.execute("""
            SELECT
                phase_goes,
                phase_experimental,
                COUNT(*) as count,
                SUM(CASE WHEN flare_within_24h = 1 THEN 1 ELSE 0 END) as followed_by_flare
            FROM phase_divergence
            WHERE timestamp >= datetime('now', ?)
            GROUP BY phase_goes, phase_experimental
            ORDER BY count DESC
        """, (f'-{days} days',))
        by_phase_pair = [dict(row) for row in cursor.fetchall()]

        return {
            'overall': overall,
            'by_divergence_type': by_divergence_type,
            'by_phase_pair': by_phase_pair,
            'days_analyzed': days,
        }

    def get_divergences_before_flares(self, hours_before: int = 24) -> list[dict]:
        """
        Find divergence events that occurred before flares.

        This is the KEY analysis: do divergences predict flares?
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT
                d.id as divergence_id,
                d.timestamp as divergence_time,
                d.phase_goes,
                d.phase_experimental,
                d.max_z_score,
                d.goes_class as goes_at_divergence,
                f.id as flare_id,
                f.class as flare_class,
                f.magnitude as flare_magnitude,
                f.start_time as flare_time,
                (julianday(f.start_time) - julianday(d.timestamp)) * 24 as hours_before_flare
            FROM phase_divergence d
            JOIN flare_events f
                ON f.start_time BETWEEN d.timestamp
                                    AND datetime(d.timestamp, ?)
            WHERE f.class IN ('C', 'M', 'X')
            ORDER BY d.timestamp DESC
        """, (f'+{hours_before} hours',))
        return [dict(row) for row in cursor.fetchall()]

    def update_divergence_with_flare(self, divergence_id: int, flare_class: str,
                                      flare_time: str) -> bool:
        """Update a divergence record with subsequent flare info."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                UPDATE phase_divergence
                SET flare_within_24h = 1, flare_class = ?, flare_time = ?
                WHERE id = ?
            """, (flare_class, flare_time, divergence_id))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error updating divergence: {e}")
            return False

    def mark_divergence_no_flare(self, divergence_id: int) -> bool:
        """Mark a divergence as having no subsequent flare (for validation)."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                UPDATE phase_divergence
                SET flare_within_24h = 0
                WHERE id = ?
            """, (divergence_id,))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error updating divergence: {e}")
            return False

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
                  'flare_events', 'predictions', 'noaa_alerts', 'phase_divergence']

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

    # =========================================================================
    # DIVERGENCE VALIDATION
    # =========================================================================

    def validate_divergences_against_flares(self, window_hours: int = 24) -> dict:
        """
        Retrospectively validate divergences against GOES flare history.

        Classification:
        - POST_EVENT: Flare within window_hours BEFORE divergence
        - PRECURSOR: Flare within window_hours AFTER divergence (predictive!)
        - BETWEEN_EVENTS: Flares both before and after
        - ISOLATED: No flares in either window

        Returns:
            dict with validation statistics and updated records
        """
        cursor = self.conn.cursor()

        # Get all unvalidated divergences
        cursor.execute("""
            SELECT id, timestamp, divergence_type
            FROM phase_divergence
            WHERE validated IS NULL OR validated = 0
            ORDER BY timestamp
        """)
        divergences = cursor.fetchall()

        stats = {
            'total_checked': 0,
            'post_event': 0,
            'precursor': 0,
            'between_events': 0,
            'isolated': 0,
            'updated_ids': []
        }

        for div in divergences:
            div_id, div_time, current_type = div

            # Find flares BEFORE this divergence (within window)
            cursor.execute("""
                SELECT timestamp, flare_class, magnitude
                FROM goes_xray
                WHERE flux >= 1e-6
                AND timestamp < ?
                AND timestamp >= datetime(?, ?)
                ORDER BY timestamp DESC
                LIMIT 1
            """, (div_time, div_time, f'-{window_hours} hours'))
            flare_before = cursor.fetchone()

            # Find flares AFTER this divergence (within window)
            cursor.execute("""
                SELECT timestamp, flare_class, magnitude
                FROM goes_xray
                WHERE flux >= 1e-6
                AND timestamp > ?
                AND timestamp <= datetime(?, ?)
                ORDER BY timestamp ASC
                LIMIT 1
            """, (div_time, div_time, f'+{window_hours} hours'))
            flare_after = cursor.fetchone()

            # Classify
            has_before = flare_before is not None
            has_after = flare_after is not None

            if has_before and has_after:
                new_type = 'BETWEEN_EVENTS'
                stats['between_events'] += 1
            elif has_before and not has_after:
                new_type = 'POST_EVENT'
                stats['post_event'] += 1
            elif has_after and not has_before:
                new_type = 'PRECURSOR'
                stats['precursor'] += 1
            else:
                new_type = 'ISOLATED'
                stats['isolated'] += 1

            # Update record
            flare_class = None
            flare_time = None
            if flare_after:
                flare_class = f"{flare_after[1]}{flare_after[2]:.1f}"
                flare_time = flare_after[0]
                flare_within = 1
            elif flare_before:
                flare_class = f"{flare_before[1]}{flare_before[2]:.1f}"
                flare_time = flare_before[0]
                flare_within = 0
            else:
                flare_within = 0

            cursor.execute("""
                UPDATE phase_divergence
                SET divergence_type = ?,
                    validated = 1,
                    validation_type = 'auto_goes',
                    flare_within_24h = ?,
                    flare_class = ?,
                    flare_time = ?
                WHERE id = ?
            """, (new_type, flare_within, flare_class, flare_time, div_id))

            stats['total_checked'] += 1
            stats['updated_ids'].append(div_id)

        self.conn.commit()
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
    parser.add_argument('--divergence', type=int, default=0,
                        help='Show phase divergence analysis for N days')
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

        if args.divergence > 0:
            div_stats = db.get_divergence_statistics(days=args.divergence)
            print(f"\n  Phase Divergence Analysis (last {args.divergence} days):")
            print("  " + "-" * 50)
            overall = div_stats['overall']
            print(f"  Total divergences:     {overall['total_divergences']}")
            print(f"  Days with divergence:  {overall['days_with_divergence']}")
            print(f"  Followed by flare:     {overall['followed_by_flare']}")
            print(f"  No flare after:        {overall['no_flare']}")

            print("\n  By divergence type:")
            for dt in div_stats['by_divergence_type']:
                hit_rate = (dt['followed_by_flare'] / dt['count'] * 100
                            if dt['count'] > 0 else 0)
                dtype = dt['divergence_type']
                marker = "⚡" if dtype == "PRECURSOR" else "🔄" if dtype == "POST_EVENT" else "❓"
                print(f"    {marker} {dtype:12} ({dt['count']:3}x, {hit_rate:.0f}% flare hit rate)")

            print("\n  By phase pair:")
            for dt in div_stats['by_phase_pair']:
                hit_rate = (dt['followed_by_flare'] / dt['count'] * 100
                            if dt['count'] > 0 else 0)
                print(f"    {dt['phase_goes']:12} → {dt['phase_experimental']:15} "
                      f"({dt['count']:3}x, {hit_rate:.0f}% flare hit rate)")


if __name__ == "__main__":
    main()
