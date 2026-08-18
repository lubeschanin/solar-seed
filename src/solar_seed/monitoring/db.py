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

from .constants import (
    Z_ALERT,
    Z_ELEVATED,
    Z_SUDDEN_DROP_MODERATE,
    Z_SUDDEN_DROP_SEVERE,
    Z_WARNING,
    classify_status,
)
import json


def classify_trigger_kind(data: dict) -> tuple[str, float | None, float | None]:
    """
    Decide what fired a prediction, from a coupling reading.

    Shared by the live path (store_coupling_reading) and the batch path
    (extract_predictions_from_coupling) so the two cannot drift apart - the
    batch path used to insert rows with trigger_kind NULL entirely.

    Args:
        data: Mapping with any of sudden_drop_severity, sudden_drop_pct,
            is_break, z_mad, deviation_pct, residual, trend.

    Returns:
        (kind, value, threshold). Falls back to STATUS_ONLY rather than None,
        so every prediction stays attributable.
    """
    sudden_drop = data.get('sudden_drop_severity')
    residual = data.get('residual')

    if sudden_drop:
        return ('SUDDEN_DROP', data.get('sudden_drop_pct'),
                Z_SUDDEN_DROP_MODERATE if sudden_drop == 'MODERATE'
                else Z_SUDDEN_DROP_SEVERE)

    if data.get('is_break'):
        return 'BREAK', data.get('z_mad'), 2.0

    # Thresholds are in sigma, matching classify_status()
    if residual is not None:
        for threshold in (Z_ALERT, Z_WARNING, Z_ELEVATED):
            if residual < threshold:
                return 'THRESHOLD', residual, threshold

    if residual is not None and abs(residual) > 4.0:
        return 'Z_SCORE_SPIKE', residual, 4.0

    if data.get('trend') in ('DECLINING', 'ACCELERATING_DOWN'):
        return 'TREND', data.get('slope_pct_per_hour'), None

    return 'STATUS_ONLY', data.get('deviation_pct'), None


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
        # WAL mode: readers don't block the writer and vice versa.
        # Needed because cron jobs (import-flares, backfill, extract-predictions)
        # and the monitor loop can access the DB concurrently.
        self.conn.execute("PRAGMA journal_mode=WAL")
        # Wait up to 30s for a lock instead of failing immediately
        self.conn.execute("PRAGMA busy_timeout=30000")

    @staticmethod
    def _sql_dt(base: str = "'now'", offset: str = "?") -> str:
        """Build a SQL datetime expression normalized to stored timestamp format.

        Stored timestamps use the ISO 'T' separator ('YYYY-MM-DDTHH:MM:SS'),
        but SQLite's datetime() emits a space separator. SQLite compares
        datetimes as plain strings and ASCII 'T' > ' ', so an un-normalized
        datetime() in a comparison silently shifts results.

        Args:
            base: SQL expression for the base time (default "'now'").
            offset: SQL expression for the modifier (default '?' placeholder,
                    bind e.g. '-24 hours' as a parameter).

        Returns:
            SQL snippet like "REPLACE(datetime('now', ?), ' ', 'T')".
        """
        return f"REPLACE(datetime({base}, {offset}), ' ', 'T')"

    @staticmethod
    def _add_missing_columns(cursor, table: str, columns: list) -> None:
        """Add columns to a table if they don't exist yet (idempotent migration).

        Checks PRAGMA table_info first instead of catching OperationalError,
        so unrelated OperationalErrors (locked DB, malformed schema) are not
        silently swallowed.

        Args:
            columns: list of (column_name, column_type_sql) tuples.
        """
        cursor.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cursor.fetchall()}
        for col, dtype in columns:
            if col not in existing:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {dtype}")

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

        # Channels lookup table (AIA, EUVI wavelengths)
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

        # Pairs lookup table (normalized, ch_a < ch_b guaranteed)
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

        # Seed pairs if empty (common AIA pairs)
        cursor.execute("SELECT COUNT(*) FROM pairs")
        if cursor.fetchone()[0] == 0:
            # Get channel IDs
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
                    except Exception:
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
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                active_region_num INTEGER,
                linked_cme_ids TEXT,
                donki_link TEXT,
                donki_flr_id TEXT
            )
        """)

        # Predictions and their outcomes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_time DATETIME NOT NULL,
                issued_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                valid_from DATETIME,
                valid_to DATETIME,
                predicted_class TEXT,
                predicted_probability REAL,
                trigger_pair TEXT,
                trigger_pair_id INTEGER REFERENCES pairs(id),
                trigger_measurement_id INTEGER REFERENCES coupling_measurements(id),
                trigger_status TEXT,
                trigger_residual REAL,
                trigger_trend TEXT,
                -- STATUS_ONLY is the catch-all from classify_trigger_kind().
                -- It was missing here, so a freshly created database rejected
                -- those rows outright while an existing one accepted them (the
                -- column was added by ALTER TABLE, which carries no CHECK) -
                -- the same code silently kept or dropped predictions depending
                -- on how old the database was.
                trigger_kind TEXT CHECK(trigger_kind IN ('Z_SCORE_SPIKE', 'SUDDEN_DROP', 'BREAK', 'TREND', 'THRESHOLD', 'TRANSFER_STATE', 'STATUS_ONLY')),
                trigger_value REAL,
                trigger_threshold REAL,
                actual_flare_id INTEGER REFERENCES flare_events(id),
                verified BOOLEAN DEFAULT FALSE,
                lead_time_hours REAL,
                notes TEXT,
                pipeline_version TEXT DEFAULT 'v0.5',
                run_id INTEGER REFERENCES runs(id),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Many-to-many: Prediction ↔ Flare matches (for proper evaluation)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL REFERENCES predictions(id),
                flare_event_id INTEGER NOT NULL REFERENCES flare_events(id),
                match_type TEXT CHECK(match_type IN ('hit', 'near_miss', 'post_event', 'ambiguous')),
                time_to_peak_min REAL,
                distance_score REAL,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(prediction_id, flare_event_id)
            )
        """)

        # NOAA Space Weather Alerts (enhanced for Kp tracking)
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
                trigger_was_vetoed BOOLEAN DEFAULT NULL,
                trigger_robustness_score REAL DEFAULT NULL,
                trigger_quality_ok BOOLEAN DEFAULT NULL,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # =================================================================
        # REPRODUCIBILITY TABLES (Run tracking)
        # =================================================================

        # Pipeline runs - captures each monitoring session
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

        # Run configuration - JSON blob for full reproducibility
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS run_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_hash TEXT UNIQUE NOT NULL,
                config_json TEXT NOT NULL,
                description TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add new columns to existing tables (migration for existing DBs).
        # Missing columns are detected via PRAGMA table_info instead of
        # swallowing OperationalErrors, so real errors (locked DB, malformed
        # schema, typos) propagate instead of being silently ignored.
        self._add_missing_columns(cursor, 'phase_divergence', [
            ('divergence_type', 'TEXT'),
            ('goes_rising', 'INTEGER'),
            ('validated', 'INTEGER'),
            ('validation_type', 'TEXT'),
            # Trigger quality fields (v0.5)
            ('trigger_was_vetoed', 'BOOLEAN'),
            ('trigger_robustness_score', 'REAL'),
            ('trigger_quality_ok', 'BOOLEAN'),
            ('run_id', 'INTEGER'),
        ])

        # Add new columns to coupling_measurements
        self._add_missing_columns(cursor, 'coupling_measurements', [
            ('sudden_drop_pct', 'REAL'),
            ('sudden_drop_severity', 'TEXT'),
            ('pipeline_version', 'TEXT'),
            ('veto_reason', 'TEXT'),
            # Structured quality fields (v0.5)
            ('quality_ok', 'BOOLEAN'),
            ('robustness_score', 'REAL'),
            ('sync_delta_s', 'REAL'),
            ('run_id', 'INTEGER'),
            # Normalized pair reference (v0.6)
            ('pair_id', 'INTEGER'),
            # Backfill support (v0.7)
            ('resolution', 'TEXT'),           # '1k' or '4k'
            ('backfilled_at', 'DATETIME'),    # when 4k data was backfilled
            ('original_delta_mi', 'REAL'),    # original 1k value before backfill
        ])

        # Add columns to predictions (v0.6 - enhanced prediction tracking)
        self._add_missing_columns(cursor, 'predictions', [
            ('pipeline_version', "TEXT DEFAULT 'v0.5'"),
            ('run_id', 'INTEGER'),
            ('issued_at', 'DATETIME'),
            ('valid_from', 'DATETIME'),
            ('valid_to', 'DATETIME'),
            ('trigger_pair_id', 'INTEGER'),
            ('trigger_measurement_id', 'INTEGER'),
            ('trigger_kind', 'TEXT'),
            ('trigger_value', 'REAL'),
            ('trigger_threshold', 'REAL'),
            # Episode tracking: the most recent trigger folded into this
            # prediction. insert_or_extend_prediction measures the episode gap
            # from here, not from prediction_time - otherwise a sustained
            # deviation rolls past the gap tolerance and splits into a new row
            # every few readings, which is what it is meant to prevent.
            ('last_trigger_time', 'DATETIME'),
            ('n_triggers', 'INTEGER DEFAULT 1'),
        ])

        # Migrate existing pair strings to pair_id (one-time migration)
        cursor.execute("""
            UPDATE coupling_measurements
            SET pair_id = (SELECT p.id FROM pairs p WHERE p.pair_name = coupling_measurements.pair)
            WHERE pair_id IS NULL AND pair IS NOT NULL
        """)

        # Migrate predictions trigger_pair to trigger_pair_id
        cursor.execute("""
            UPDATE predictions
            SET trigger_pair_id = (SELECT p.id FROM pairs p WHERE p.pair_name = predictions.trigger_pair)
            WHERE trigger_pair_id IS NULL AND trigger_pair IS NOT NULL
        """)

        # Add new columns to noaa_alerts (migration for existing DBs)
        self._add_missing_columns(cursor, 'noaa_alerts', [
            ('message_code', 'TEXT'),
            ('kp_observed', 'REAL'),
            ('kp_predicted', 'REAL'),
            ('kp_max_24h', 'REAL'),
            ('valid_from', 'DATETIME'),
            ('valid_to', 'DATETIME'),
            ('g_scale', 'INTEGER'),
            ('s_scale', 'INTEGER'),
            ('r_scale', 'INTEGER'),
            ('source_region', 'TEXT'),
            ('raw_json', 'TEXT'),
        ])

        # Add new columns to flare_events for DONKI data
        self._add_missing_columns(cursor, 'flare_events', [
            ('active_region_num', 'INTEGER'),
            ('linked_cme_ids', 'TEXT'),  # JSON array of CME IDs
            ('donki_link', 'TEXT'),
            ('donki_flr_id', 'TEXT'),    # DONKI flare ID for deduplication
        ])

        # Create indices for fast queries
        # Time series: timestamp is primary query axis
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_goes_timestamp ON goes_xray(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_wind_timestamp ON solar_wind(timestamp)")

        # Coupling: composite index for (pair, timestamp) queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_coupling_timestamp ON coupling_measurements(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_coupling_pair ON coupling_measurements(pair)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_coupling_pair_time ON coupling_measurements(pair, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_coupling_status ON coupling_measurements(status)")
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_coupling_unique ON coupling_measurements(timestamp, pair)")

        # Flare events: time-based queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_flares_time ON flare_events(start_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_flares_peak ON flare_events(peak_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_flares_class ON flare_events(class)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_flares_source ON flare_events(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_flares_ar ON flare_events(active_region_num)")

        # Predictions: time and verification status
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(prediction_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_verified ON predictions(verified)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_pair ON predictions(trigger_pair)")
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_unique ON predictions(prediction_time, trigger_pair)")

        # Prediction matches: for evaluation queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_prediction ON prediction_matches(prediction_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_flare ON prediction_matches(flare_event_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_type ON prediction_matches(match_type)")

        # Phase divergence
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_divergence_timestamp ON phase_divergence(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_divergence_phases ON phase_divergence(phase_goes, phase_experimental)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_divergence_type ON phase_divergence(divergence_type)")

        # NOAA alerts
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_noaa_issue_time ON noaa_alerts(issue_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_noaa_code ON noaa_alerts(message_code)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_noaa_type ON noaa_alerts(alert_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_noaa_kp ON noaa_alerts(kp_observed)")

        # Runs and config (reproducibility)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_version ON runs(pipeline_version)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_config_hash ON run_config(config_hash)")

        # Run FK indexes (for queries by run)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_coupling_run ON coupling_measurements(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_run ON predictions(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_divergence_run ON phase_divergence(run_id)")

        # Quality filtering indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_coupling_quality ON coupling_measurements(quality_ok)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_divergence_vetoed ON phase_divergence(trigger_was_vetoed)")

        # Dimension tables (channels, pairs)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_channels_wavelength ON channels(wavelength)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_channels_instrument ON channels(instrument)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pairs_name ON pairs(pair_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pairs_channels ON pairs(ch_a_id, ch_b_id)")

        # Normalized pair_id indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_coupling_pair_id ON coupling_measurements(pair_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_pair_id ON predictions(trigger_pair_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_measurement ON predictions(trigger_measurement_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_kind ON predictions(trigger_kind)")

        # Composite indexes for run-scoped queries (critical for performance)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_coupling_run_time ON coupling_measurements(run_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_run_time ON predictions(run_id, prediction_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_divergence_run_time ON phase_divergence(run_id, timestamp)")

        # Composite indexes for class-time queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_flares_class_time ON flare_events(class, start_time)")

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
                        n_points: int = None, sudden_drop_pct: float = None,
                        sudden_drop_severity: str = None, veto_reason: str = None,
                        pipeline_version: str = 'v0.6',
                        quality_ok: bool = None, robustness_score: float = None,
                        sync_delta_s: float = None, run_id: int = None,
                        pair_id: int = None, resolution: str = None) -> int:
        """Insert coupling measurement with quality fields and run tracking.

        Args:
            resolution: '1k' for synoptic, '4k' for full-res. Enables backfill tracking.
        """
        cursor = self.conn.cursor()

        # Auto-resolve pair_id from pair string if not provided
        if pair_id is None and pair:
            cursor.execute("SELECT id FROM pairs WHERE pair_name = ?", (pair,))
            row = cursor.fetchone()
            if row:
                pair_id = row[0]

        try:
            # Upsert instead of INSERT OR REPLACE: REPLACE deletes the row and
            # re-inserts it with a NEW id, breaking existing FK references
            # (predictions.trigger_measurement_id). DO UPDATE keeps the id.
            cursor.execute("""
                INSERT INTO coupling_measurements
                (timestamp, pair, pair_id, delta_mi, mi_original, residual, deviation_pct,
                 status, trend, slope_pct_per_hour, acceleration, confidence, n_points,
                 sudden_drop_pct, sudden_drop_severity, veto_reason, pipeline_version,
                 quality_ok, robustness_score, sync_delta_s, run_id, resolution)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(timestamp, pair) DO UPDATE SET
                    pair_id = excluded.pair_id,
                    delta_mi = excluded.delta_mi,
                    mi_original = excluded.mi_original,
                    residual = excluded.residual,
                    deviation_pct = excluded.deviation_pct,
                    status = excluded.status,
                    trend = excluded.trend,
                    slope_pct_per_hour = excluded.slope_pct_per_hour,
                    acceleration = excluded.acceleration,
                    confidence = excluded.confidence,
                    n_points = excluded.n_points,
                    sudden_drop_pct = excluded.sudden_drop_pct,
                    sudden_drop_severity = excluded.sudden_drop_severity,
                    veto_reason = excluded.veto_reason,
                    pipeline_version = excluded.pipeline_version,
                    quality_ok = excluded.quality_ok,
                    robustness_score = excluded.robustness_score,
                    sync_delta_s = excluded.sync_delta_s,
                    run_id = excluded.run_id,
                    resolution = excluded.resolution
            """, (timestamp, pair, pair_id, delta_mi, mi_original, residual, deviation_pct,
                  status, trend, slope_pct_per_hour, acceleration, confidence, n_points,
                  sudden_drop_pct, sudden_drop_severity, veto_reason, pipeline_version,
                  quality_ok, robustness_score, sync_delta_s, run_id, resolution))
            self.conn.commit()
            # lastrowid is unreliable after DO UPDATE - fetch the stable row id
            cursor.execute(
                "SELECT id FROM coupling_measurements WHERE timestamp = ? AND pair = ?",
                (timestamp, pair))
            row = cursor.fetchone()
            return row[0] if row else cursor.lastrowid
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
                          trigger_kind: str = None, trigger_value: float = None,
                          trigger_threshold: float = None,
                          trigger_measurement_id: int = None,
                          valid_from: str = None, valid_to: str = None,
                          run_id: int = None, pipeline_version: str = 'v0.6',
                          notes: str = None) -> int:
        """
        Insert a prediction (skips if same time+pair exists).

        Args:
            trigger_kind: Z_SCORE_SPIKE, SUDDEN_DROP, BREAK, TREND, THRESHOLD, TRANSFER_STATE, STATUS_ONLY
            trigger_value: The actual value that triggered the prediction
            trigger_threshold: The threshold that was exceeded
            trigger_measurement_id: FK to the coupling_measurement that triggered this
            valid_from, valid_to: Forecast window (defaults to prediction_time, +90min)
        """
        cursor = self.conn.cursor()

        # Auto-resolve trigger_pair_id from trigger_pair string
        trigger_pair_id = None
        if trigger_pair:
            cursor.execute("SELECT id FROM pairs WHERE pair_name = ?", (trigger_pair,))
            row = cursor.fetchone()
            if row:
                trigger_pair_id = row[0]

        # Default valid_from/to if not provided
        if valid_from is None:
            valid_from = prediction_time
        if valid_to is None and prediction_time:
            # Default 90 min forecast window
            try:
                from datetime import datetime, timedelta
                dt = datetime.fromisoformat(prediction_time.replace('Z', '+00:00'))
                valid_to = (dt + timedelta(minutes=90)).isoformat()
            except (ValueError, AttributeError):
                pass

        try:
            # Check for existing prediction with same time and pair
            cursor.execute("""
                SELECT id FROM predictions
                WHERE prediction_time = ? AND trigger_pair = ?
            """, (prediction_time, trigger_pair))
            if cursor.fetchone():
                return -1  # Already exists

            cursor.execute("""
                INSERT INTO predictions
                (prediction_time, issued_at, valid_from, valid_to,
                 predicted_class, predicted_probability,
                 trigger_pair, trigger_pair_id, trigger_measurement_id,
                 trigger_status, trigger_residual, trigger_trend,
                 trigger_kind, trigger_value, trigger_threshold,
                 run_id, pipeline_version, notes)
                VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (prediction_time, valid_from, valid_to,
                  predicted_class, predicted_probability,
                  trigger_pair, trigger_pair_id, trigger_measurement_id,
                  trigger_status, trigger_residual, trigger_trend,
                  trigger_kind, trigger_value, trigger_threshold,
                  run_id, pipeline_version, notes))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error inserting prediction: {e}")
            return -1

    #: Severity ordering used to decide whether an ongoing alarm episode has
    #: escalated. Higher wins.
    STATUS_SEVERITY = {'ELEVATED': 1, 'WARNING': 2, 'ALERT': 3}

    #: A trigger this close to the previous one continues the same episode
    #: rather than opening a new prediction.
    EPISODE_GAP_MINUTES = 30

    #: Forecast horizon of a prediction: valid_to = last trigger + this.
    PREDICTION_WINDOW_MINUTES = 90

    def insert_or_extend_prediction(self, prediction_time: str, trigger_pair: str,
                                    trigger_status: str, episode_gap_minutes: int = None,
                                    **kwargs) -> tuple[int, str]:
        """
        Record a prediction as part of an alarm EPISODE, not per measurement.

        The monitor samples every ~10 min and used to emit one prediction per
        pair per reading, so a single sustained deviation produced dozens of
        rows: 14708 predictions accumulated over six months, which makes any
        hit rate meaningless (with that many alarms, hitting M/X flares is
        arithmetic, not skill). One episode = one prediction, so precision and
        recall become computable.

        An episode continues while triggers keep arriving within
        `episode_gap_minutes` of each other. Within an episode:
          - a MORE severe status escalates the existing row in place
          - an equal or milder status is absorbed (no new row)
        A gap longer than the tolerance opens a new episode.

        Returns:
            (prediction_id, action) where action is 'created', 'escalated'
            or 'absorbed'. prediction_id is -1 on error.
        """
        gap = episode_gap_minutes if episode_gap_minutes is not None else self.EPISODE_GAP_MINUTES
        cursor = self.conn.cursor()

        try:
            pred_dt = datetime.fromisoformat(str(prediction_time).replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return self.insert_prediction(
                prediction_time=prediction_time, trigger_pair=trigger_pair,
                trigger_status=trigger_status, **kwargs), 'created'

        lo = (pred_dt - timedelta(minutes=gap)).strftime("%Y-%m-%dT%H:%M:%S")
        hi = (pred_dt + timedelta(minutes=gap)).strftime("%Y-%m-%dT%H:%M:%S")

        # A trigger belongs to an episode if it falls within `gap` of that
        # episode's span [prediction_time .. last_trigger_time] - on EITHER
        # side, not just after its end. Matching only backwards works for a
        # live stream but breaks when re-processing stored measurements
        # (extract_predictions_from_coupling): a timestamp in the middle of an
        # already-recorded episode found no match and opened a duplicate.
        cursor.execute("""
            SELECT id, trigger_status, prediction_time,
                   COALESCE(last_trigger_time, prediction_time) AS last_seen
            FROM predictions
            WHERE trigger_pair = ?
              AND prediction_time <= ?
              AND COALESCE(last_trigger_time, prediction_time) >= ?
            ORDER BY prediction_time DESC
            LIMIT 1
        """, (trigger_pair, hi, lo))
        open_row = cursor.fetchone()

        if open_row is None:
            new_id = self.insert_prediction(
                prediction_time=prediction_time, trigger_pair=trigger_pair,
                trigger_status=trigger_status, **kwargs)
            if new_id > 0:
                cursor.execute(
                    "UPDATE predictions SET last_trigger_time = ?, n_triggers = 1 WHERE id = ?",
                    (prediction_time, new_id))
                self.conn.commit()
            return new_id, 'created'

        pred_id = open_row['id']
        prev_severity = self.STATUS_SEVERITY.get(open_row['trigger_status'], 0)
        new_severity = self.STATUS_SEVERITY.get(trigger_status, 0)

        # Every trigger extends the episode, whether or not it escalates.
        # MAX so a replay in a different order cannot shrink the span.
        # prediction_time deliberately stays put: it is part of a UNIQUE
        # (prediction_time, trigger_pair) key, so moving the anchor backwards
        # can collide with an existing row - and the anchor means "when this
        # episode was first recorded" anyway.
        #
        # valid_to moves with the episode too, not only on escalation. Only
        # escalating triggers used to push it, so a sustained alarm at one
        # level ran on with an expired forecast window: id 15352 (193-211,
        # ALERT) collected 171 triggers over 22h while valid_to still said
        # the window had closed after 90 minutes. Anything reading valid_to
        # as "is this prediction live" saw a dead row for an active alarm.
        window_end = (pred_dt + timedelta(minutes=self.PREDICTION_WINDOW_MINUTES)).isoformat()
        cursor.execute("""
            UPDATE predictions
            SET last_trigger_time = MAX(COALESCE(last_trigger_time, prediction_time), ?),
                n_triggers = COALESCE(n_triggers, 1) + 1,
                valid_to = MAX(COALESCE(valid_to, ''), ?)
            WHERE id = ?
        """, (prediction_time, window_end, pred_id))

        if new_severity <= prev_severity:
            self.conn.commit()
            return pred_id, 'absorbed'

        # Escalation: upgrade the episode in place. An explicit valid_to from
        # the caller overrides the rolling window; otherwise the update above
        # already moved it, so leave it alone rather than recomputing it.
        valid_to = kwargs.get('valid_to') or window_end

        try:
            cursor.execute("""
                UPDATE predictions
                SET trigger_status = ?,
                    predicted_class = COALESCE(?, predicted_class),
                    trigger_residual = COALESCE(?, trigger_residual),
                    trigger_trend = COALESCE(?, trigger_trend),
                    trigger_kind = COALESCE(?, trigger_kind),
                    trigger_value = COALESCE(?, trigger_value),
                    trigger_threshold = COALESCE(?, trigger_threshold),
                    valid_to = MAX(COALESCE(valid_to, ''), ?),
                    notes = COALESCE(notes, '') || ?
                WHERE id = ?
            """, (
                trigger_status,
                kwargs.get('predicted_class'),
                kwargs.get('trigger_residual'),
                kwargs.get('trigger_trend'),
                kwargs.get('trigger_kind'),
                kwargs.get('trigger_value'),
                kwargs.get('trigger_threshold'),
                valid_to,
                f' | escalated to {trigger_status} at {prediction_time}',
                pred_id,
            ))
            self.conn.commit()
            return pred_id, 'escalated'
        except sqlite3.Error as e:
            print(f"Error escalating prediction: {e}")
            return -1, 'error'

    def merge_duplicate_prediction_episodes(self, since: str = None,
                                            dry_run: bool = False) -> dict:
        """
        Fold prediction rows that sit inside another episode's span into it.

        Cleans up after the batch path that bypassed the episode logic: it
        wrote one row per measurement alongside the episode the live path had
        already recorded, so the same alarm is counted several times and any
        precision figure over that period is wrong.

        A row is a duplicate when another row for the SAME pair starts earlier
        and its span [prediction_time .. last_trigger_time] covers it. The
        surviving row keeps the earliest anchor, the latest end, the summed
        trigger count and the most severe status.

        Rows that were already verified or carry prediction_matches are left
        alone and reported - merging them would silently rewrite evaluated
        results.

        Args:
            since: Only consider rows at or after this ISO timestamp
            dry_run: Report what would happen without writing

        Returns:
            {'merged': int, 'skipped_verified': int, 'remaining': int}
        """
        cursor = self.conn.cursor()
        where_since = "AND b.prediction_time >= ?" if since else ""
        params = (since,) if since else ()

        merged = 0
        skipped = 0

        # Containment can chain (b inside a, c inside b), and each merge widens
        # the surviving span, so iterate until the query comes back empty.
        # Bounded to keep a pathological dataset from looping forever.
        for _ in range(1000):
            cursor.execute(f"""
                SELECT b.id AS dup_id, a.id AS keep_id
                FROM predictions b
                JOIN predictions a
                  ON a.trigger_pair = b.trigger_pair
                 AND a.id <> b.id
                 AND a.prediction_time < b.prediction_time
                 AND b.prediction_time <= COALESCE(a.last_trigger_time, a.prediction_time)
                WHERE COALESCE(b.verified, 0) = 0
                  AND NOT EXISTS (SELECT 1 FROM prediction_matches m
                                  WHERE m.prediction_id = b.id)
                  {where_since}
                ORDER BY a.prediction_time ASC, b.prediction_time ASC
                LIMIT 1
            """, params)
            row = cursor.fetchone()
            if row is None:
                break

            dup_id, keep_id = row['dup_id'], row['keep_id']
            if dry_run:
                # Cannot iterate without writing, so count the candidates
                # directly instead of reporting just the first one.
                cursor.execute(f"""
                    SELECT COUNT(*) FROM predictions b
                    JOIN predictions a
                      ON a.trigger_pair = b.trigger_pair AND a.id <> b.id
                     AND a.prediction_time < b.prediction_time
                     AND b.prediction_time <= COALESCE(a.last_trigger_time, a.prediction_time)
                    WHERE COALESCE(b.verified, 0) = 0
                      AND NOT EXISTS (SELECT 1 FROM prediction_matches m
                                      WHERE m.prediction_id = b.id)
                      {where_since}
                """, params)
                merged = cursor.fetchone()[0]
                break

            dup = cursor.execute(
                "SELECT * FROM predictions WHERE id = ?", (dup_id,)).fetchone()
            keep = cursor.execute(
                "SELECT * FROM predictions WHERE id = ?", (keep_id,)).fetchone()

            severity = max(
                self.STATUS_SEVERITY.get(keep['trigger_status'], 0),
                self.STATUS_SEVERITY.get(dup['trigger_status'], 0),
            )
            status = next(
                (s for s, v in self.STATUS_SEVERITY.items() if v == severity),
                keep['trigger_status'])

            cursor.execute("""
                UPDATE predictions
                SET last_trigger_time = MAX(
                        COALESCE(last_trigger_time, prediction_time),
                        COALESCE(?, ?)),
                    valid_to = MAX(COALESCE(valid_to, ''), COALESCE(?, '')),
                    n_triggers = COALESCE(n_triggers, 1) + COALESCE(?, 1),
                    trigger_status = ?,
                    predicted_class = ?,
                    trigger_kind = COALESCE(trigger_kind, ?)
                WHERE id = ?
            """, (
                dup['last_trigger_time'], dup['prediction_time'],
                dup['valid_to'],
                dup['n_triggers'],
                status,
                'M' if status == 'ALERT' else 'C',
                dup['trigger_kind'],
                keep_id,
            ))
            cursor.execute("DELETE FROM predictions WHERE id = ?", (dup_id,))
            merged += 1

        if not dry_run:
            self.conn.commit()

        # Anything still contained but untouchable (verified / has matches)
        cursor.execute(f"""
            SELECT COUNT(*) FROM predictions b
            JOIN predictions a
              ON a.trigger_pair = b.trigger_pair AND a.id <> b.id
             AND a.prediction_time < b.prediction_time
             AND b.prediction_time <= COALESCE(a.last_trigger_time, a.prediction_time)
            WHERE 1=1 {where_since}
        """, params)
        remaining = cursor.fetchone()[0]
        skipped = remaining if not dry_run else 0

        return {'merged': merged, 'skipped_verified': skipped, 'remaining': remaining}

    def insert_prediction_match(self, prediction_id: int, flare_event_id: int,
                                match_type: str, time_to_peak_min: float = None,
                                distance_score: float = None, notes: str = None) -> int:
        """
        Insert prediction ↔ flare match (many-to-many relationship).

        Args:
            prediction_id: ID from predictions table
            flare_event_id: ID from flare_events table
            match_type: 'hit', 'near_miss', 'post_event', or 'ambiguous'
            time_to_peak_min: Time from prediction to flare peak in minutes
            distance_score: Optional matching score (0-1, higher = better match)
            notes: Optional notes

        Returns:
            Row ID or -1 on error/duplicate
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO prediction_matches
                (prediction_id, flare_event_id, match_type, time_to_peak_min,
                 distance_score, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (prediction_id, flare_event_id, match_type, time_to_peak_min,
                  distance_score, notes))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return -1  # Duplicate match
        except sqlite3.Error as e:
            print(f"Error inserting prediction match: {e}")
            return -1

    def insert_noaa_alert(self, alert_id: str, issue_time: str,
                          message_code: str = None, alert_type: str = None,
                          kp_observed: float = None, kp_predicted: float = None,
                          kp_max_24h: float = None,
                          valid_from: str = None, valid_to: str = None,
                          g_scale: int = None, s_scale: int = None, r_scale: int = None,
                          source_region: str = None, message: str = None,
                          raw_json: str = None) -> int:
        """
        Insert NOAA Space Weather alert with Kp and scale tracking.

        Args:
            alert_id: Unique alert identifier
            issue_time: When alert was issued (ISO datetime)
            message_code: NOAA code (WARK05, ALTK06, etc.)
            alert_type: WARNING, ALERT, WATCH, SUMMARY, or FORECAST
            kp_observed: Current observed Kp index
            kp_predicted: Predicted Kp index
            kp_max_24h: Maximum Kp in last 24 hours
            valid_from: Alert validity start time
            valid_to: Alert validity end time
            g_scale: Geomagnetic storm scale (G0-G5)
            s_scale: Solar radiation scale (S0-S5)
            r_scale: Radio blackout scale (R0-R5)
            source_region: Active region number (e.g., AR3842)
            message: Full message text
            raw_json: Original JSON for debugging

        Returns:
            Row ID or -1 on error
        """
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
        trigger_was_vetoed: bool = None,
        trigger_robustness_score: float = None,
        trigger_quality_ok: bool = None,
        run_id: int = None,
        notes: str = None,
    ) -> int:
        """
        Insert a phase divergence event.

        Called when GOES-only and ΔMI-integrated classifiers disagree.
        This data enables empirical validation of which classifier is more predictive.

        Args:
            divergence_type: PRECURSOR, POST_EVENT, or STRUCTURAL_EVENT
            goes_rising: Whether GOES was trending up at time of divergence
            trigger_was_vetoed: Whether the trigger measurement was vetoed
            trigger_robustness_score: Robustness score (0-1) of trigger measurement
            trigger_quality_ok: Whether trigger passed all quality checks
            run_id: FK to runs table for reproducibility
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO phase_divergence
                (timestamp, phase_goes, phase_experimental, reason_goes,
                 reason_experimental, divergence_type, goes_flux, goes_class,
                 goes_rising, max_z_score, trigger_pair,
                 trigger_was_vetoed, trigger_robustness_score, trigger_quality_ok,
                 run_id, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, phase_goes, phase_experimental, reason_goes,
                  reason_experimental, divergence_type, goes_flux, goes_class,
                  1 if goes_rising else 0 if goes_rising is not None else None,
                  max_z_score, trigger_pair,
                  trigger_was_vetoed, trigger_robustness_score, trigger_quality_ok,
                  run_id, notes))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error inserting phase divergence: {e}")
            return -1

    # =========================================================================
    # RUN MANAGEMENT (Reproducibility)
    # =========================================================================

    def start_run(self, git_commit: str = None, config_hash: str = None,
                  pipeline_version: str = 'v0.5', notes: str = None) -> int:
        """
        Start a new monitoring run.

        Args:
            git_commit: Git commit hash for code version
            config_hash: Hash of configuration (FK to run_config)
            pipeline_version: Pipeline version string
            notes: Optional notes about this run

        Returns:
            Run ID for use in subsequent inserts
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO runs (git_commit, config_hash, pipeline_version, notes)
                VALUES (?, ?, ?, ?)
            """, (git_commit, config_hash, pipeline_version, notes))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error starting run: {e}")
            return -1

    def complete_run(self, run_id: int, status: str = 'completed',
                     n_measurements: int = None, n_predictions: int = None,
                     n_divergences: int = None) -> bool:
        """
        Mark a run as completed and update statistics.

        Args:
            run_id: ID of the run to complete
            status: Final status ('completed', 'failed', 'interrupted')
            n_measurements: Total coupling measurements in this run
            n_predictions: Total predictions generated
            n_divergences: Total phase divergences logged
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                UPDATE runs SET
                    ended_at = CURRENT_TIMESTAMP,
                    status = ?,
                    n_measurements = COALESCE(?, n_measurements),
                    n_predictions = COALESCE(?, n_predictions),
                    n_divergences = COALESCE(?, n_divergences)
                WHERE id = ?
            """, (status, n_measurements, n_predictions, n_divergences, run_id))
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Error completing run: {e}")
            return False

    def save_config(self, config_dict: dict, description: str = None) -> str:
        """
        Save a configuration and return its hash.

        Args:
            config_dict: Configuration as dictionary
            description: Human-readable description

        Returns:
            Config hash (use as FK in runs.config_hash)
        """
        import hashlib
        config_json = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]

        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO run_config (config_hash, config_json, description)
                VALUES (?, ?, ?)
            """, (config_hash, config_json, description))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error saving config: {e}")

        return config_hash

    def get_run(self, run_id: int) -> dict:
        """Get run details by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_active_run(self) -> dict:
        """Get the most recent running run (if any)."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM runs
            WHERE status = 'running'
            ORDER BY started_at DESC LIMIT 1
        """)
        row = cursor.fetchone()
        return dict(row) if row else None

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_recent_goes(self, hours: int = 24) -> list[dict]:
        """Get recent GOES X-ray data."""
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT * FROM goes_xray
            WHERE timestamp >= {self._sql_dt()}
            ORDER BY timestamp DESC
        """, (f'-{hours} hours',))
        return [dict(row) for row in cursor.fetchall()]

    def get_recent_solar_wind(self, hours: int = 24) -> list[dict]:
        """Get recent solar wind data."""
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT * FROM solar_wind
            WHERE timestamp >= {self._sql_dt()}
            ORDER BY timestamp DESC
        """, (f'-{hours} hours',))
        return [dict(row) for row in cursor.fetchall()]

    def get_recent_coupling(self, hours: int = 24, pair: str = None) -> list[dict]:
        """Get recent coupling measurements."""
        cursor = self.conn.cursor()
        if pair:
            cursor.execute(f"""
                SELECT * FROM coupling_measurements
                WHERE timestamp >= {self._sql_dt()}
                AND pair = ?
                ORDER BY timestamp DESC
            """, (f'-{hours} hours', pair))
        else:
            cursor.execute(f"""
                SELECT * FROM coupling_measurements
                WHERE timestamp >= {self._sql_dt()}
                ORDER BY timestamp DESC
            """, (f'-{hours} hours',))
        return [dict(row) for row in cursor.fetchall()]

    def get_flare_events(self, days: int = 30, min_class: str = 'C') -> list[dict]:
        """Get flare events, optionally filtered by minimum class."""
        class_order = {'A': 0, 'B': 1, 'C': 2, 'M': 3, 'X': 4}
        min_order = class_order.get(min_class, 0)

        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT * FROM flare_events
            WHERE start_time >= {self._sql_dt()}
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
        cursor.execute(f"""
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
                ON c.timestamp BETWEEN {self._sql_dt('f.start_time')}
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

            # Check if already exists (exact peak time match)
            # Use string comparison since SQLite datetime() doesn't handle ISO8601 with TZ well
            peak_str = peak_ts.isoformat()
            cursor.execute("""
                SELECT id FROM flare_events
                WHERE peak_time = ?
            """, (peak_str,))

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

    def import_flares_from_donki(self, start_date: str, end_date: str, min_class: str = 'M') -> int:
        """
        Import historical flares from NASA DONKI API.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_class: Minimum flare class ('C', 'M', or 'X')

        Returns:
            Number of flares imported
        """
        import urllib.request
        import json
        from datetime import datetime

        from solar_seed.endpoints import endpoint

        url = f"{endpoint('donki_flr')}?startDate={start_date}&endDate={end_date}"

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode())
        except Exception as e:
            print(f"Error fetching DONKI data: {e}")
            return 0

        # Filter by class
        class_order = {'C': 1, 'M': 2, 'X': 3}
        min_order = class_order.get(min_class, 2)

        cursor = self.conn.cursor()
        imported = 0

        for flare in data:
            class_type = flare.get('classType', '')
            if not class_type:
                continue

            flare_class = class_type[0]
            if class_order.get(flare_class, 0) < min_order:
                continue

            # Parse magnitude
            try:
                magnitude = float(class_type[1:])
            except ValueError:
                magnitude = 1.0

            # Parse times
            begin_time = flare.get('beginTime', '')
            peak_time = flare.get('peakTime', '')
            end_time = flare.get('endTime', '')

            if not begin_time:
                continue

            # Normalize timestamps to bare ISO (no timezone suffix)
            # All DONKI times are UTC; strip Z/+00:00 for consistent DB storage
            def _normalize_ts(ts: str) -> str:
                return ts.replace('Z', '').replace('+00:00', '').strip()

            begin_time = _normalize_ts(begin_time)

            # Extract new DONKI fields
            donki_flr_id = flare.get('flrID', '')
            active_region_num = flare.get('activeRegionNum')
            donki_link = flare.get('link', '')

            # Extract linked CME IDs
            linked_events = flare.get('linkedEvents') or []
            cme_ids = [e.get('activityID', '') for e in linked_events
                       if 'CME' in e.get('activityID', '')]
            linked_cme_ids = json.dumps(cme_ids) if cme_ids else None

            # Check if already exists (by DONKI ID or by class/magnitude/time)
            if donki_flr_id:
                cursor.execute("SELECT id FROM flare_events WHERE donki_flr_id = ?", (donki_flr_id,))
                existing = cursor.fetchone()
                if existing:
                    # Update existing record with new fields
                    cursor.execute("""
                        UPDATE flare_events
                        SET active_region_num = ?, linked_cme_ids = ?, donki_link = ?
                        WHERE id = ?
                    """, (active_region_num, linked_cme_ids, donki_link, existing[0]))
                    continue

            # Fallback: check by class/magnitude/date
            cursor.execute("""
                SELECT id FROM flare_events
                WHERE class = ? AND magnitude = ?
                AND peak_time LIKE ?
            """, (flare_class, magnitude, f"{begin_time[:10]}%"))

            existing = cursor.fetchone()
            if existing:
                # Update existing record with new fields
                cursor.execute("""
                    UPDATE flare_events
                    SET active_region_num = ?, linked_cme_ids = ?, donki_link = ?, donki_flr_id = ?
                    WHERE id = ?
                """, (active_region_num, linked_cme_ids, donki_link, donki_flr_id, existing[0]))
                continue

            # Insert new flare
            cursor.execute("""
                INSERT INTO flare_events (
                    start_time, peak_time, end_time, class, magnitude, source, location,
                    active_region_num, linked_cme_ids, donki_link, donki_flr_id
                )
                VALUES (?, ?, ?, ?, ?, 'DONKI', ?, ?, ?, ?, ?)
            """, (
                begin_time,
                _normalize_ts(peak_time) if peak_time else begin_time,
                _normalize_ts(end_time) if end_time else None,
                flare_class,
                magnitude,
                flare.get('sourceLocation', ''),
                active_region_num,
                linked_cme_ids,
                donki_link,
                donki_flr_id
            ))
            imported += 1

        self.conn.commit()
        return imported

    def get_recent_divergences(self, hours: int = 24) -> list[dict]:
        """Get recent phase divergence events."""
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT * FROM phase_divergence
            WHERE timestamp >= {self._sql_dt()}
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
        cursor.execute(f"""
            SELECT
                COUNT(*) as total_divergences,
                COUNT(DISTINCT date(timestamp)) as days_with_divergence,
                SUM(CASE WHEN flare_within_24h = 1 THEN 1 ELSE 0 END) as followed_by_flare,
                SUM(CASE WHEN flare_within_24h = 0 THEN 1 ELSE 0 END) as no_flare
            FROM phase_divergence
            WHERE timestamp >= {self._sql_dt()}
        """, (f'-{days} days',))
        overall = dict(cursor.fetchone())

        # By divergence type (PRECURSOR, POST_EVENT, UNCONFIRMED)
        cursor.execute(f"""
            SELECT
                COALESCE(divergence_type, 'UNKNOWN') as divergence_type,
                COUNT(*) as count,
                SUM(CASE WHEN flare_within_24h = 1 THEN 1 ELSE 0 END) as followed_by_flare
            FROM phase_divergence
            WHERE timestamp >= {self._sql_dt()}
            GROUP BY divergence_type
            ORDER BY count DESC
        """, (f'-{days} days',))
        by_divergence_type = [dict(row) for row in cursor.fetchall()]

        # By phase pair (which phases disagreed)
        cursor.execute(f"""
            SELECT
                phase_goes,
                phase_experimental,
                COUNT(*) as count,
                SUM(CASE WHEN flare_within_24h = 1 THEN 1 ELSE 0 END) as followed_by_flare
            FROM phase_divergence
            WHERE timestamp >= {self._sql_dt()}
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
        cursor.execute(f"""
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
                                    AND {self._sql_dt('d.timestamp')}
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
        cursor.execute(f"""
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
            AND timestamp >= {self._sql_dt()}
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
        cursor.execute(f"""
            SELECT
                date(timestamp) as date,
                COUNT(*) as total_measurements,
                SUM(CASE WHEN status = 'ALERT' THEN 1 ELSE 0 END) as alerts,
                SUM(CASE WHEN status = 'WARNING' THEN 1 ELSE 0 END) as warnings,
                AVG(residual) as avg_residual
            FROM coupling_measurements
            WHERE timestamp >= {self._sql_dt()}
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

    def extract_predictions_from_coupling(self, statuses: list[str] = None) -> int:
        """
        Extract predictions from coupling_measurements into predictions table.

        Converts ALERT/WARNING/ELEVATED coupling entries into prediction
        records for separate tracking and validation.

        Routes through insert_or_extend_prediction, so a run of consecutive
        anomalous measurements becomes ONE episode - exactly as the live path
        records it. This used to be a raw INSERT that bypassed the episode
        logic, trigger_kind classification and valid_from/to entirely, and
        deduplicated on an exact prediction_time match. Since the live path
        folds a range of timestamps into a single row anchored at the episode
        start, every later measurement in that episode looked "not yet
        extracted" and got a duplicate row with NULL trigger fields.

        Args:
            statuses: List of statuses to extract
                      (default: ['ALERT', 'WARNING', 'ELEVATED'])

        Returns:
            Number of NEW prediction episodes created (escalations and
            absorptions into existing episodes are not counted)
        """
        if statuses is None:
            # Status hierarchy is ALERT > WARNING > ELEVATED - all three are
            # anomalous and must produce prediction records
            statuses = ['ALERT', 'WARNING', 'ELEVATED']

        cursor = self.conn.cursor()

        # Get coupling measurements with alert/elevated status
        placeholders = ','.join('?' * len(statuses))
        cursor.execute(f"""
            SELECT DISTINCT timestamp, pair, delta_mi, residual, deviation_pct,
                   status, trend, slope_pct_per_hour,
                   sudden_drop_pct, sudden_drop_severity
            FROM coupling_measurements
            WHERE status IN ({placeholders})
            ORDER BY timestamp
        """, statuses)

        measurements = [dict(row) for row in cursor.fetchall()]
        inserted = 0

        for m in measurements:
            status = m['status']
            # Estimate predicted class based on status
            predicted_class = 'M' if status == 'ALERT' else 'C'

            kind, value, threshold = classify_trigger_kind(m)

            _, action = self.insert_or_extend_prediction(
                prediction_time=m['timestamp'],
                trigger_pair=m['pair'],
                trigger_status=status,
                predicted_class=predicted_class,
                trigger_residual=m['residual'],
                trigger_trend=m['trend'],
                trigger_kind=kind,
                trigger_value=value,
                trigger_threshold=threshold,
                notes='Auto-extracted from coupling_measurements',
            )
            if action == 'created':
                inserted += 1

        self.conn.commit()
        return inserted

    def verify_predictions_against_flares(self, window_hours: float = 6.0) -> dict:
        """
        Match predictions with subsequent flares.

        Args:
            window_hours: Time window to look for flares after prediction

        Returns:
            dict with verification statistics
        """
        cursor = self.conn.cursor()

        # Get unverified predictions
        cursor.execute("""
            SELECT id, prediction_time, predicted_class, trigger_pair
            FROM predictions
            WHERE verified = 0 OR verified IS NULL
            ORDER BY prediction_time
        """)
        predictions = cursor.fetchall()

        stats = {
            'total_checked': 0,
            'matched': 0,
            'unmatched': 0,
            'avg_lead_time': None,
            'lead_times': []
        }

        for pred in predictions:
            pred_id, pred_time, pred_class, pair = pred

            # Find flare within window AFTER prediction
            # _sql_dt normalizes datetime() output (space→T) for consistent comparison
            cursor.execute(f"""
                SELECT id, start_time, peak_time, class, magnitude
                FROM flare_events
                WHERE start_time > ?
                AND start_time <= {self._sql_dt('?')}
                AND class IN ('C', 'M', 'X')
                ORDER BY start_time ASC
                LIMIT 1
            """, (pred_time, pred_time, f'+{window_hours} hours'))

            flare = cursor.fetchone()
            stats['total_checked'] += 1

            if flare:
                flare_id, flare_time, peak_time, flare_class, magnitude = flare

                # Calculate lead time (handle timezone-naive vs aware)
                from datetime import datetime, timezone as tz
                pred_time_str = pred_time.replace('Z', '+00:00')
                flare_time_str = flare_time.replace('Z', '+00:00')

                pred_dt = datetime.fromisoformat(pred_time_str)
                flare_dt = datetime.fromisoformat(flare_time_str)

                # Make both timezone-aware (UTC) if not already
                if pred_dt.tzinfo is None:
                    pred_dt = pred_dt.replace(tzinfo=tz.utc)
                if flare_dt.tzinfo is None:
                    flare_dt = flare_dt.replace(tzinfo=tz.utc)

                lead_time = (flare_dt - pred_dt).total_seconds() / 3600

                cursor.execute("""
                    UPDATE predictions
                    SET verified = 1, actual_flare_id = ?, lead_time_hours = ?
                    WHERE id = ?
                """, (flare_id, lead_time, pred_id))

                # Also populate v0.6 prediction_matches table (many-to-many)
                peak_ref = peak_time or flare_time
                peak_dt = datetime.fromisoformat(peak_ref.replace('Z', '+00:00'))
                if peak_dt.tzinfo is None:
                    peak_dt = peak_dt.replace(tzinfo=tz.utc)
                time_to_peak_min = (peak_dt - pred_dt).total_seconds() / 60.0
                # Classify: hit if lead ≤ 2h (paper precursor definition), else near_miss
                match_type = 'hit' if lead_time <= 2.0 else 'near_miss'
                self.insert_prediction_match(
                    prediction_id=pred_id,
                    flare_event_id=flare_id,
                    match_type=match_type,
                    time_to_peak_min=time_to_peak_min,
                )

                stats['matched'] += 1
                stats['lead_times'].append(lead_time)
            else:
                cursor.execute("""
                    UPDATE predictions
                    SET verified = 0
                    WHERE id = ?
                """, (pred_id,))
                stats['unmatched'] += 1

        self.conn.commit()

        if stats['lead_times']:
            stats['avg_lead_time'] = sum(stats['lead_times']) / len(stats['lead_times'])

        return stats

    def get_prediction_summary(self) -> dict:
        """Get summary of predictions vs NOAA flares with post-flare classification."""
        cursor = self.conn.cursor()

        # Classify each prediction as PRE-flare, POST-flare, or unrelated
        cursor.execute("""
            SELECT p.id, p.prediction_time, p.actual_flare_id, p.verified, p.lead_time_hours
            FROM predictions p
            ORDER BY p.prediction_time
        """)
        predictions = cursor.fetchall()

        true_positives = 0
        false_positives = 0
        post_flare = 0
        lead_times = []

        for pred in predictions:
            pred_id, pred_time, flare_id, verified, lead_time = pred

            if flare_id and lead_time and lead_time > 0:
                # Prediction followed by flare = True Positive
                true_positives += 1
                lead_times.append(lead_time)
            else:
                # Check if this prediction is AFTER a recent flare (post-flare effect)
                cursor.execute(f"""
                    SELECT id, peak_time, class, magnitude
                    FROM flare_events
                    WHERE peak_time < ?
                    AND peak_time >= {self._sql_dt('?', "'-2 hours'")}
                    ORDER BY peak_time DESC
                    LIMIT 1
                """, (pred_time, pred_time))
                recent_flare = cursor.fetchone()

                if recent_flare:
                    # This is a post-flare alert, not a false positive
                    post_flare += 1
                else:
                    false_positives += 1

        # NOAA flares summary (only recent, matching our monitoring period)
        cursor.execute("""
            SELECT
                COUNT(*) as total_flares,
                SUM(CASE WHEN class = 'X' THEN 1 ELSE 0 END) as x_class,
                SUM(CASE WHEN class = 'M' THEN 1 ELSE 0 END) as m_class,
                SUM(CASE WHEN class = 'C' THEN 1 ELSE 0 END) as c_class
            FROM flare_events
        """)
        flare_stats = dict(cursor.fetchone())

        # FN = flares in monitoring period without predictions
        cursor.execute("""
            SELECT MIN(prediction_time), MAX(prediction_time) FROM predictions
        """)
        time_range = cursor.fetchone()

        if time_range[0]:
            cursor.execute(f"""
                SELECT COUNT(DISTINCT f.id) as missed_flares
                FROM flare_events f
                WHERE f.class IN ('C', 'M', 'X')
                AND f.start_time >= ?
                AND f.start_time <= {self._sql_dt('?', "'+1 day'")}
                AND NOT EXISTS (
                    SELECT 1 FROM predictions p
                    WHERE p.actual_flare_id = f.id
                )
            """, (time_range[0], time_range[1]))
            fn = cursor.fetchone()[0]
        else:
            fn = 0

        # Calculate metrics (excluding post-flare from FP)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + fn) if (true_positives + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        avg_lead = sum(lead_times) / len(lead_times) if lead_times else None

        return {
            'predictions': {
                'total_predictions': len(predictions),
                'true_positives': true_positives,
                'false_positives': false_positives,
                'post_flare': post_flare,
                'avg_lead_time': avg_lead
            },
            'noaa_flares': flare_stats,
            'metrics': {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'post_flare_alerts': post_flare,
                'false_negatives': fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'avg_lead_time_hours': avg_lead
            }
        }

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
            cursor.execute(f"""
                SELECT timestamp, flare_class, magnitude
                FROM goes_xray
                WHERE flux >= 1e-6
                AND timestamp < ?
                AND timestamp >= {self._sql_dt('?')}
                ORDER BY timestamp DESC
                LIMIT 1
            """, (div_time, div_time, f'-{window_hours} hours'))
            flare_before = cursor.fetchone()

            # Find flares AFTER this divergence (within window)
            cursor.execute(f"""
                SELECT timestamp, flare_class, magnitude
                FROM goes_xray
                WHERE flux >= 1e-6
                AND timestamp > ?
                AND timestamp <= {self._sql_dt('?')}
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

    # =========================================================================
    # EXPORT METHODS
    # =========================================================================

    def export_to_csv(self, table: str, output_path: Path, days: int = None) -> int:
        """
        Export table data to CSV file.

        Args:
            table: Table name (predictions, flare_events, coupling_measurements, etc.)
            output_path: Output CSV file path
            days: Limit to last N days (None = all data)

        Returns:
            Number of rows exported
        """
        import csv

        valid_tables = [
            'predictions', 'flare_events', 'coupling_measurements',
            'goes_xray', 'solar_wind', 'phase_divergence'
        ]

        if table not in valid_tables:
            raise ValueError(f"Invalid table: {table}. Valid: {valid_tables}")

        cursor = self.conn.cursor()

        # Build query with optional date filter
        if days:
            # Find timestamp column
            ts_col = 'timestamp' if table != 'predictions' else 'prediction_time'
            if table == 'flare_events':
                ts_col = 'start_time'
            offset_sql = f"'-{days} days'"
            query = (f"SELECT * FROM {table} "
                     f"WHERE {ts_col} >= {self._sql_dt(offset=offset_sql)} "
                     f"ORDER BY {ts_col}")
        else:
            query = f"SELECT * FROM {table}"

        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            return 0

        # Get column names
        columns = [desc[0] for desc in cursor.description]

        # Write CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for row in rows:
                writer.writerow(row)

        return len(rows)

    def export_all_csv(self, output_dir: Path, days: int = None) -> dict:
        """
        Export all tables to CSV files.

        Args:
            output_dir: Output directory
            days: Limit to last N days (None = all data)

        Returns:
            Dict with table names and row counts
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tables = [
            'predictions', 'flare_events', 'coupling_measurements',
            'goes_xray', 'solar_wind'
        ]

        results = {}
        for table in tables:
            output_path = output_dir / f"{table}.csv"
            count = self.export_to_csv(table, output_path, days)
            results[table] = count

        return results

    # =========================================================================
    # BACKFILL SUPPORT
    # =========================================================================

    def get_measurements_for_backfill(
        self,
        min_age_days: int = 3,
        max_age_days: int = 30,
        limit: int = 0
    ) -> list[dict]:
        """
        Get 1k measurements that could be backfilled with 4k data.

        Args:
            min_age_days: Minimum age (SDAC needs ~3 days)
            max_age_days: Maximum age to consider
            limit: Maximum number of results (0=unlimited)

        Returns:
            List of measurements with timestamp, pair, delta_mi
        """
        cursor = self.conn.cursor()
        cutoff_min = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        cutoff_max = (datetime.now(timezone.utc) - timedelta(days=min_age_days)).isoformat()

        if limit > 0:
            cursor.execute("""
                SELECT timestamp, pair, delta_mi, resolution
                FROM coupling_measurements
                WHERE (resolution = '1k' OR resolution IS NULL)
                  AND backfilled_at IS NULL
                  AND timestamp >= ?
                  AND timestamp <= ?
                ORDER BY timestamp ASC
                LIMIT ?
            """, (cutoff_min, cutoff_max, limit))
        else:
            cursor.execute("""
                SELECT timestamp, pair, delta_mi, resolution
                FROM coupling_measurements
                WHERE (resolution = '1k' OR resolution IS NULL)
                  AND backfilled_at IS NULL
                  AND timestamp >= ?
                  AND timestamp <= ?
                ORDER BY timestamp ASC
            """, (cutoff_min, cutoff_max))

        return [dict(row) for row in cursor.fetchall()]

    def update_measurement_backfill(
        self,
        timestamp: str,
        pair: str,
        new_delta_mi: float,
        original_delta_mi: float = None,
        new_mi_original: float = None,
        baselines: dict = None,
        sync_delta_s: float = None,
    ) -> bool:
        """
        Update a measurement with backfilled 4k data.

        Derived fields (residual, deviation_pct, status) are recomputed against
        the 4k baseline, or set to NULL when no baseline is available. Leaving
        them at their 1k values - as this method used to - produces rows that
        carry a 4k ΔMI next to a 1k-derived label, so any analysis joining ΔMI
        against status silently mixes two computations. `mi_original` is
        likewise cleared unless the caller supplies the recomputed value.

        Args:
            timestamp: Measurement timestamp
            pair: Channel pair (e.g., '193-304')
            new_delta_mi: New MI value from 4k data
            original_delta_mi: Original 1k value (for audit)
            new_mi_original: Recomputed MI_original at 4k, if available
            baselines: {pair: {'mean', 'std'}} at 4k. When omitted, the
                measured 4k baselines are loaded from disk.
            sync_delta_s: Channel time spread measured on the 4k frames. Pass
                None when it could not be determined - the column is then
                cleared rather than left holding the 1k value of the replaced
                row. Keeping the old value made every 4k row echo the 1k spread
                distribution, which is not a measurement of anything.

        Returns:
            True if updated, False if not found
        """
        from .baselines import load_baselines
        from .validation import assess_measurement_quality

        cursor = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()

        if baselines is None:
            baselines = load_baselines().get('4k', {})
        base = (baselines or {}).get(pair)

        residual = deviation_pct = status = None
        if base and base.get('std'):
            residual = (new_delta_mi - base['mean']) / base['std']
            deviation_pct = (new_delta_mi - base['mean']) / base['mean'] if base['mean'] else None
            status = classify_status(residual)

        # Re-derive the quality verdict for the 4k frames. The time-sync test is
        # recomputed from the new spread; any reason that was NOT recomputed
        # (robustness, measured on the 1k image pair) is carried forward rather
        # than silently dropped.
        prior = cursor.execute(
            "SELECT veto_reason FROM coupling_measurements WHERE timestamp = ? AND pair = ?",
            (timestamp, pair)
        ).fetchone()
        inherited = None
        if prior and prior[0]:
            inherited = '+'.join(
                r for r in (x.strip() for x in str(prior[0]).split('+'))
                if r and not r.startswith('time_sync')
            ) or None

        quality = assess_measurement_quality(
            time_spread_sec=sync_delta_s,
            inherited_reasons=inherited,
        )

        cursor.execute("""
            UPDATE coupling_measurements
            SET delta_mi = ?,
                mi_original = ?,
                residual = ?,
                deviation_pct = ?,
                status = ?,
                resolution = '4k',
                backfilled_at = ?,
                sync_delta_s = ?,
                quality_ok = ?,
                veto_reason = ?,
                original_delta_mi = COALESCE(original_delta_mi, ?)
            WHERE timestamp = ? AND pair = ?
        """, (new_delta_mi, new_mi_original, residual, deviation_pct, status,
              now, sync_delta_s, quality['quality_ok'], quality['veto_reason'],
              original_delta_mi, timestamp, pair))

        self.conn.commit()
        return cursor.rowcount > 0

    def repair_backfilled_rows(self, baselines: dict = None, dry_run: bool = False) -> dict:
        """
        Recompute derived fields on rows backfilled before the fix above.

        Those rows hold a 4k delta_mi next to residual/deviation_pct/status
        computed from the pre-backfill 1k value. Sudden-drop fields and
        mi_original cannot be reconstructed and are cleared.

        Returns:
            {'examined': int, 'updated': int, 'skipped_no_baseline': int}
        """
        from .baselines import load_baselines

        if baselines is None:
            baselines = load_baselines().get('4k', {})

        cursor = self.conn.cursor()
        rows = cursor.execute("""
            SELECT id, pair, delta_mi FROM coupling_measurements
            WHERE resolution = '4k' AND backfilled_at IS NOT NULL
        """).fetchall()

        updated = skipped = 0
        for row in rows:
            base = (baselines or {}).get(row['pair'])
            if not base or not base.get('std') or not base.get('mean'):
                skipped += 1
                continue
            delta_mi = row['delta_mi']
            if delta_mi is None:
                skipped += 1
                continue
            residual = (delta_mi - base['mean']) / base['std']
            deviation_pct = (delta_mi - base['mean']) / base['mean']
            status = classify_status(residual)

            if not dry_run:
                cursor.execute("""
                    UPDATE coupling_measurements
                    SET residual = ?, deviation_pct = ?, status = ?,
                        mi_original = NULL,
                        sudden_drop_pct = NULL, sudden_drop_severity = NULL
                    WHERE id = ?
                """, (residual, deviation_pct, status, row['id']))
            updated += 1

        if not dry_run:
            self.conn.commit()

        return {'examined': len(rows), 'updated': updated, 'skipped_no_baseline': skipped}

    def get_backfill_stats(self) -> dict:
        """Get backfill statistics."""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN resolution = '4k' THEN 1 ELSE 0 END) as backfilled,
                SUM(CASE WHEN resolution = '1k' OR resolution IS NULL THEN 1 ELSE 0 END) as pending_1k
            FROM coupling_measurements
        """)
        row = cursor.fetchone()

        cursor.execute(f"""
            SELECT COUNT(DISTINCT timestamp) as unique_timestamps
            FROM coupling_measurements
            WHERE (resolution = '1k' OR resolution IS NULL)
              AND backfilled_at IS NULL
              AND timestamp <= {self._sql_dt(offset="'-3 days'")}
        """)
        eligible = cursor.fetchone()[0]

        return {
            'total': row[0] if row[0] else 0,
            'backfilled_4k': row[1] if row[1] else 0,
            'pending_1k': row[2] if row[2] else 0,
            'eligible_for_backfill': eligible if eligible else 0,
        }


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
