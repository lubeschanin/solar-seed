"""
Tests for the shared measurement-quality verdict and its backfill path.

Background: quality_ok used to be decided by four separate copies of the same
`spread < 60` check, and only one of the failing branches ever wrote a reason.
12513 of 12665 flagged rows in the production database carry no reason at all.
"""

import pytest

from solar_seed.monitoring.constants import SYNC_SPREAD_MAX_SEC
from solar_seed.monitoring.validation import assess_measurement_quality
from solar_seed.monitoring.db import MonitoringDB


class TestAssessMeasurementQuality:

    def test_clean_measurement_passes_without_reason(self):
        q = assess_measurement_quality(
            time_spread_sec=5.0,
            robustness_check={'is_robust': True, 'change_pct': 2.0},
        )
        assert q['quality_ok'] is True
        assert q['veto_reason'] is None

    def test_unknown_spread_does_not_flag(self):
        """None means 'not measured', which is not the same as 'failed'."""
        q = assess_measurement_quality(time_spread_sec=None)
        assert q['quality_ok'] is True
        assert q['veto_reason'] is None

    def test_every_failure_carries_a_reason(self):
        """The invariant: quality_ok False implies a non-empty veto_reason."""
        failures = [
            dict(time_spread_sec=SYNC_SPREAD_MAX_SEC + 1),
            dict(robustness_check={'is_robust': False, 'change_pct': 35.0}),
            dict(is_data_error=True),
            dict(break_vetoed='spike'),
            dict(break_vetoed=True),
        ]
        for kwargs in failures:
            q = assess_measurement_quality(**kwargs)
            assert q['quality_ok'] is False, kwargs
            assert q['veto_reason'], f"flagged with no reason: {kwargs}"

    def test_threshold_is_inclusive_at_the_limit(self):
        assert assess_measurement_quality(
            time_spread_sec=SYNC_SPREAD_MAX_SEC)['quality_ok'] is True
        assert assess_measurement_quality(
            time_spread_sec=SYNC_SPREAD_MAX_SEC + 0.1)['quality_ok'] is False

    def test_reasons_are_joined_not_overwritten(self):
        q = assess_measurement_quality(
            time_spread_sec=360.0,
            robustness_check={'is_robust': False, 'change_pct': 130.6},
            is_data_error=True,
        )
        assert q['veto_reason'].count('+') == 2
        for token in ('data_error', 'time_sync', 'robustness'):
            assert token in q['veto_reason']

    def test_robustness_none_is_not_a_failure(self):
        """is_robust None means the check did not run, not that it failed."""
        q = assess_measurement_quality(
            robustness_check={'is_robust': None, 'error': 'not computed'})
        assert q['quality_ok'] is True

    def test_same_reason_in_two_details_collapses_to_the_detailed_one(self):
        """
        Break detection reports a bare 'robustness'; the robustness check
        reports 'robustness(36%)'. De-duplicating on the exact string let both
        through, and the first live row written after the fix read
        'robustness+robustness(36%)'. Kind, not string, is the identity.
        """
        q = assess_measurement_quality(
            break_vetoed='robustness',
            robustness_check={'is_robust': False, 'change_pct': 36.0},
        )
        assert q['veto_reason'] == 'robustness(36%)'
        assert q['veto_reason'].count('robustness') == 1

    def test_distinct_reasons_are_all_kept(self):
        """De-duplicating by kind must not swallow genuinely different kinds."""
        q = assess_measurement_quality(
            break_vetoed='spike',
            time_spread_sec=180.0,
            robustness_check={'is_robust': False, 'change_pct': 36.0},
        )
        kinds = {r.split('(')[0] for r in q['veto_reason'].split('+')}
        assert kinds == {'spike', 'time_sync', 'robustness'}

    def test_inherited_reasons_are_not_duplicated(self):
        q = assess_measurement_quality(
            time_spread_sec=180.0,
            inherited_reasons='time_sync(180s)+robustness(130%)',
        )
        assert q['veto_reason'].count('time_sync') == 1
        assert 'robustness(130%)' in q['veto_reason']


class TestBackfillQualityFields:

    @pytest.fixture
    def db(self, tmp_path):
        d = MonitoringDB(db_path=tmp_path / "test.db")
        yield d
        d.close()

    def _insert_1k(self, db, **kwargs):
        defaults = dict(
            timestamp="2026-01-15T12:00:00", pair="193-211", delta_mi=0.59,
            resolution="1k", sync_delta_s=184.5, quality_ok=False,
            veto_reason="time_sync(184s)",
        )
        defaults.update(kwargs)
        return db.insert_coupling(**defaults)

    def _row(self, db, pair="193-211"):
        cur = db.conn.cursor()
        cur.execute(
            "SELECT resolution, sync_delta_s, quality_ok, veto_reason "
            "FROM coupling_measurements WHERE pair = ?", (pair,))
        return cur.fetchone()

    def test_backfill_replaces_stale_1k_sync_value(self, db):
        """
        The 4k row must carry the 4k spread.

        Backfill used to leave sync_delta_s untouched, so every 4k row echoed
        the spread of the 1k frame it replaced - which is why the stored 1k and
        4k spread distributions looked identical (p50 4.8s, p90 184.5s in both).
        That is an artefact of the copy, not a property of the resolution.
        """
        self._insert_1k(db)
        db.update_measurement_backfill(
            timestamp="2026-01-15T12:00:00", pair="193-211",
            new_delta_mi=0.80, sync_delta_s=4.8,
        )
        row = self._row(db)
        assert row['resolution'] == '4k'
        assert row['sync_delta_s'] == pytest.approx(4.8)
        assert row['quality_ok'] == 1
        assert row['veto_reason'] is None

    def test_backfill_clears_sync_when_unmeasured(self, db):
        """An unknown 4k spread becomes NULL, never the inherited 1k value."""
        self._insert_1k(db)
        db.update_measurement_backfill(
            timestamp="2026-01-15T12:00:00", pair="193-211",
            new_delta_mi=0.80, sync_delta_s=None,
        )
        row = self._row(db)
        assert row['sync_delta_s'] is None

    def test_backfill_flags_a_desynced_4k_frame(self, db):
        self._insert_1k(db, sync_delta_s=5.0, quality_ok=True, veto_reason=None)
        db.update_measurement_backfill(
            timestamp="2026-01-15T12:00:00", pair="193-211",
            new_delta_mi=0.80, sync_delta_s=200.0,
        )
        row = self._row(db)
        assert row['quality_ok'] == 0
        assert 'time_sync' in row['veto_reason']

    def test_backfill_keeps_reasons_it_did_not_recompute(self, db):
        """
        Robustness was measured on the 1k image pair and is not recomputed at
        4k, so dropping it would quietly promote a bad row to clean.
        """
        self._insert_1k(db, veto_reason="time_sync(184s)+robustness(130%)")
        db.update_measurement_backfill(
            timestamp="2026-01-15T12:00:00", pair="193-211",
            new_delta_mi=0.80, sync_delta_s=4.8,
        )
        row = self._row(db)
        assert row['quality_ok'] == 0
        assert 'robustness(130%)' in row['veto_reason']
        # the recomputed test passed at 4k, so its old reason must be gone
        assert 'time_sync' not in row['veto_reason']
