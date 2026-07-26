"""
Tests for data-derived ΔMI baselines and the fields that depend on them.

Baselines feed every residual, deviation_pct and status. The hardcoded table
they replaced had drifted far from the measured distributions (193-304 at 4k:
coded 0.32, measured 0.107 - which made 65% of all 4k readings fire ALERT), so
these tests pin down that the values now come from the data and that the
backfill path keeps ΔMI and its derived fields in the same computation.
"""

import json

import pytest

from solar_seed.monitoring.baselines import (
    PROVISIONAL_BASELINES,
    compute_baselines_from_db,
    load_baselines,
    save_baselines,
)
from solar_seed.monitoring.coupling import CouplingMonitor
from solar_seed.monitoring.db import MonitoringDB


@pytest.fixture
def db(tmp_path):
    database = MonitoringDB(db_path=tmp_path / "test.db")
    yield database
    database.close()


def _seed_coupling(db, pair, values, day='2026-03-01', resolution='1k'):
    """Insert `values` as one measurement per minute."""
    for i, v in enumerate(values):
        db.insert_coupling(
            timestamp=f"{day}T{i // 60:02d}:{i % 60:02d}:00",
            pair=pair, delta_mi=v, resolution=resolution,
        )


class TestLoadBaselines:
    def test_missing_file_falls_back_to_provisional(self, tmp_path):
        loaded = load_baselines(tmp_path / "nope.json")
        assert loaded['_meta']['source'] == 'provisional'
        assert loaded['1k']['193-211'] == PROVISIONAL_BASELINES['1k']['193-211']

    def test_corrupt_file_falls_back_rather_than_raising(self, tmp_path):
        path = tmp_path / "baselines.json"
        path.write_text("{not json")
        assert load_baselines(path)['_meta']['source'] == 'provisional'

    def test_measured_file_wins(self, tmp_path):
        path = tmp_path / "baselines.json"
        save_baselines({
            '1k': {'193-211': {'mean': 0.79, 'std': 0.14}},
            '4k': {},
            '_meta': {'source': 'monitoring.db'},
        }, path)
        loaded = load_baselines(path)
        assert loaded['1k']['193-211']['mean'] == 0.79
        # Pairs absent from the measured file still resolve
        assert loaded['1k']['211-335'] == PROVISIONAL_BASELINES['1k']['211-335']

    def test_save_is_atomic(self, tmp_path):
        path = tmp_path / "baselines.json"
        save_baselines({'1k': {}, '4k': {}, '_meta': {}}, path)
        assert path.exists()
        assert not path.with_suffix('.json.tmp').exists()
        json.loads(path.read_text())  # valid JSON


class TestComputeBaselines:
    def test_uses_median_not_mean(self, db):
        # A heavy low tail (flare collapses) must not drag the baseline down.
        values = [0.80] * 240 + [0.10] * 60
        _seed_coupling(db, '193-211', values)
        result = compute_baselines_from_db(db, min_samples=100)
        assert result['1k']['193-211']['mean'] == pytest.approx(0.80, abs=0.01)

    def test_below_min_samples_keeps_provisional(self, db):
        _seed_coupling(db, '193-211', [0.42] * 10)
        result = compute_baselines_from_db(db, min_samples=100)
        assert result['1k']['193-211'] == PROVISIONAL_BASELINES['1k']['193-211']
        assert result['_meta']['pairs']['1k/193-211']['used'] == 'provisional'

    def test_std_floored_to_fraction_of_mean(self, db):
        # A constant series has MAD 0; a zero sigma turns everything into a
        # multi-sigma anomaly.
        _seed_coupling(db, '193-211', [0.80] * 300)
        result = compute_baselines_from_db(db, min_samples=100)
        assert result['1k']['193-211']['std'] == pytest.approx(0.08, abs=0.001)

    def test_flare_windows_excluded(self, db):
        _seed_coupling(db, '193-211', [0.80] * 120, day='2026-03-01')
        # Depressed values during a flare
        _seed_coupling(db, '193-211', [0.20] * 120, day='2026-03-02')
        db.insert_flare_event(
            start_time='2026-03-02T00:30:00', flare_class='M', magnitude=1.0,
            peak_time='2026-03-02T01:00:00',
        )
        result = compute_baselines_from_db(
            db, exclude_flare_hours=3.0, min_samples=50)
        assert result['_meta']['excluded_near_flare'] == 120
        assert result['1k']['193-211']['mean'] == pytest.approx(0.80, abs=0.01)

    def test_data_error_rows_excluded(self, db):
        _seed_coupling(db, '193-211', [0.80] * 200)
        for i in range(50):
            db.insert_coupling(timestamp=f"2026-03-05T00:{i:02d}:00", pair='193-211',
                               delta_mi=0.0, resolution='1k', status='DATA_ERROR')
        result = compute_baselines_from_db(db, min_samples=100)
        assert result['_meta']['pairs']['1k/193-211']['n'] == 200


class TestMonitorUsesLoadedBaselines:
    def test_monitor_reads_baseline_file(self, tmp_path):
        path = tmp_path / "baselines.json"
        save_baselines({
            '1k': {'193-211': {'mean': 0.80, 'std': 0.10}},
            '4k': {'193-211': {'mean': 0.80, 'std': 0.10}},
            '_meta': {'source': 'monitoring.db'},
        }, path)
        monitor = CouplingMonitor(
            history_file=tmp_path / "history.json", baseline_file=path)

        assert monitor.get_baselines('1k')['193-211']['mean'] == 0.80
        assert monitor.baseline_source == 'monitoring.db'

        # 0.80 is now exactly nominal, not the +1.75 sigma the old 0.59
        # baseline would have reported.
        result = monitor.compute_residual('193-211', 0.80)
        assert result['residual'] == pytest.approx(0.0)
        assert result['status'] == 'NORMAL'

    def test_resolution_selects_table(self, tmp_path):
        path = tmp_path / "baselines.json"
        save_baselines({
            '1k': {'193-304': {'mean': 0.18, 'std': 0.06}},
            '4k': {'193-304': {'mean': 0.11, 'std': 0.05}},
            '_meta': {'source': 'monitoring.db'},
        }, path)
        monitor = CouplingMonitor(
            history_file=tmp_path / "history.json", baseline_file=path)

        assert monitor.get_baselines('1k')['193-304']['mean'] == 0.18
        assert monitor.get_baselines('4k')['193-304']['mean'] == 0.11


class TestBackfillConsistency:
    def test_derived_fields_recomputed_against_4k_baseline(self, db):
        db.insert_coupling(
            timestamp='2026-03-01T00:00:00', pair='193-304', delta_mi=0.18,
            mi_original=1.5, residual=5.5, deviation_pct=1.57,
            status='NORMAL', resolution='1k',
        )
        db.update_measurement_backfill(
            timestamp='2026-03-01T00:00:00', pair='193-304',
            new_delta_mi=0.02, original_delta_mi=0.18,
            new_mi_original=1.1,
            baselines={'193-304': {'mean': 0.11, 'std': 0.025}},
        )
        row = db.conn.execute(
            "SELECT * FROM coupling_measurements WHERE pair='193-304'").fetchone()

        assert row['resolution'] == '4k'
        assert row['delta_mi'] == pytest.approx(0.02)
        assert row['original_delta_mi'] == pytest.approx(0.18)
        assert row['mi_original'] == pytest.approx(1.1)
        # Recomputed, not the stale 1k values. Status comes from the z-score:
        # (0.02 - 0.11) / 0.025 = -3.6 sigma -> ALERT
        assert row['residual'] == pytest.approx((0.02 - 0.11) / 0.025)
        assert row['status'] == 'ALERT'

    def test_no_baseline_clears_derived_fields(self, db):
        db.insert_coupling(
            timestamp='2026-03-01T00:00:00', pair='193-304', delta_mi=0.18,
            residual=5.5, deviation_pct=1.57, status='NORMAL', resolution='1k',
        )
        db.update_measurement_backfill(
            timestamp='2026-03-01T00:00:00', pair='193-304',
            new_delta_mi=0.055, baselines={},
        )
        row = db.conn.execute(
            "SELECT * FROM coupling_measurements WHERE pair='193-304'").fetchone()
        # Better a NULL than a 1k label next to a 4k value
        assert row['residual'] is None
        assert row['status'] is None

    def test_repair_fixes_legacy_rows(self, db):
        db.insert_coupling(
            timestamp='2026-03-01T00:00:00', pair='193-304', delta_mi=0.10,
            residual=5.5, deviation_pct=1.57, status='ALERT', resolution='1k',
        )
        # Simulate the old backfill: ΔMI and resolution updated, labels stale
        db.conn.execute("""
            UPDATE coupling_measurements
            SET resolution='4k', backfilled_at='2026-03-05T00:00:00'
        """)
        db.conn.commit()

        result = db.repair_backfilled_rows(
            baselines={'193-304': {'mean': 0.11, 'std': 0.025}})
        assert result == {'examined': 1, 'updated': 1, 'skipped_no_baseline': 0}

        row = db.conn.execute("SELECT * FROM coupling_measurements").fetchone()
        assert row['status'] == 'NORMAL'
        assert row['deviation_pct'] == pytest.approx((0.10 - 0.11) / 0.11)
        assert row['mi_original'] is None
