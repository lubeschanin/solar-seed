"""
Tests for episode-based predictions, the break window anchor and the
measurement quality gate.

These three used to interact badly:
  - one prediction per 10-min reading inflated the count to 14708 in six
    months, so a hit rate said nothing about detection skill
  - the rolling break window was anchored on wallclock while measurements are
    stored under the (lagging) AIA observation time
  - the quality gate discarded deep collapses as "data errors" - the very
    events the system exists to catch
"""

from datetime import datetime, timedelta, timezone

import pytest

from solar_seed.monitoring.coupling import CouplingMonitor
from solar_seed.monitoring.db import MonitoringDB, classify_trigger_kind
from solar_seed.monitoring.detection import (
    AnomalyStatus,
    BreakType,
    classify_anomaly_status,
    detect_coupling_break,
)
from solar_seed.monitoring.validation import validate_mi_measurement


@pytest.fixture
def db(tmp_path):
    database = MonitoringDB(db_path=tmp_path / "test.db")
    yield database
    database.close()


@pytest.fixture
def monitor(tmp_path):
    return CouplingMonitor(history_file=tmp_path / "history.json")


class TestPredictionEpisodes:
    def test_repeated_triggers_collapse_into_one_prediction(self, db):
        """Six consecutive ELEVATED readings = one prediction, not six."""
        for i in range(6):
            db.insert_or_extend_prediction(
                prediction_time=f"2026-03-01T10:{i * 10:02d}:00",
                trigger_pair='193-211', trigger_status='ELEVATED',
                trigger_kind='THRESHOLD',
            )
        count = db.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        assert count == 1

    def test_gap_opens_new_episode(self, db):
        db.insert_or_extend_prediction(
            prediction_time="2026-03-01T10:00:00",
            trigger_pair='193-211', trigger_status='ELEVATED')
        # Two hours later - well past the 30 min episode gap
        db.insert_or_extend_prediction(
            prediction_time="2026-03-01T12:00:00",
            trigger_pair='193-211', trigger_status='ELEVATED')
        assert db.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 2

    def test_escalation_upgrades_in_place(self, db):
        pred_id, action = db.insert_or_extend_prediction(
            prediction_time="2026-03-01T10:00:00",
            trigger_pair='193-211', trigger_status='ELEVATED',
            trigger_kind='THRESHOLD', predicted_class='C')
        assert action == 'created'

        same_id, action = db.insert_or_extend_prediction(
            prediction_time="2026-03-01T10:10:00",
            trigger_pair='193-211', trigger_status='ALERT',
            trigger_kind='SUDDEN_DROP', predicted_class='M')
        assert action == 'escalated'
        assert same_id == pred_id
        assert db.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1

        row = db.conn.execute("SELECT * FROM predictions").fetchone()
        assert row['trigger_status'] == 'ALERT'
        assert row['trigger_kind'] == 'SUDDEN_DROP'
        assert row['predicted_class'] == 'M'
        # Window extended from the escalating trigger
        assert row['valid_to'].startswith('2026-03-01T11:40')

    def test_de_escalation_is_absorbed(self, db):
        db.insert_or_extend_prediction(
            prediction_time="2026-03-01T10:00:00",
            trigger_pair='193-211', trigger_status='ALERT')
        _, action = db.insert_or_extend_prediction(
            prediction_time="2026-03-01T10:10:00",
            trigger_pair='193-211', trigger_status='ELEVATED')
        assert action == 'absorbed'
        row = db.conn.execute("SELECT * FROM predictions").fetchone()
        assert row['trigger_status'] == 'ALERT'  # not downgraded

    def test_pairs_are_independent_episodes(self, db):
        db.insert_or_extend_prediction(
            prediction_time="2026-03-01T10:00:00",
            trigger_pair='193-211', trigger_status='ELEVATED')
        db.insert_or_extend_prediction(
            prediction_time="2026-03-01T10:00:00",
            trigger_pair='193-304', trigger_status='ELEVATED')
        assert db.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 2


class TestBatchExtractionUsesEpisodes:
    """extract_predictions_from_coupling must agree with the live path.

    It used to be a raw INSERT that bypassed episode folding and trigger_kind,
    and deduplicated on an exact prediction_time match. Because the live path
    anchors an episode at its START, every later measurement in that episode
    looked unextracted and got a duplicate row with NULL trigger fields.
    """

    @staticmethod
    def _seed(db, n=6, status='ELEVATED', start_minute=0):
        for i in range(n):
            db.insert_coupling(
                timestamp=f"2026-03-01T10:{start_minute + i * 10:02d}:00",
                pair='193-211', delta_mi=0.70, residual=-1.7,
                deviation_pct=-0.12, status=status, trend='DECLINING',
                resolution='1k',
            )

    def test_consecutive_measurements_become_one_episode(self, db):
        self._seed(db)
        created = db.extract_predictions_from_coupling()
        assert created == 1
        assert db.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1
        row = db.conn.execute("SELECT * FROM predictions").fetchone()
        assert row['n_triggers'] == 6

    def test_trigger_fields_are_populated(self, db):
        self._seed(db)
        db.extract_predictions_from_coupling()
        row = db.conn.execute("SELECT * FROM predictions").fetchone()
        # Previously NULL on every batch-created row
        assert row['trigger_kind'] == 'THRESHOLD'
        assert row['trigger_value'] == pytest.approx(-1.7)
        assert row['last_trigger_time'] is not None
        assert row['valid_to'] is not None

    def test_does_not_duplicate_live_path_episodes(self, db):
        """Re-running extraction over live-recorded episodes adds nothing."""
        for i in range(6):
            db.insert_or_extend_prediction(
                prediction_time=f"2026-03-01T10:{i * 10:02d}:00",
                trigger_pair='193-211', trigger_status='ELEVATED',
                trigger_kind='THRESHOLD')
        self._seed(db)

        before = db.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        created = db.extract_predictions_from_coupling()
        after = db.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

        assert before == 1
        assert created == 0
        assert after == 1

    def test_escalation_is_carried_over(self, db):
        self._seed(db, n=3, status='ELEVATED')
        db.insert_coupling(
            timestamp="2026-03-01T10:30:00", pair='193-211', delta_mi=0.50,
            residual=-3.0, deviation_pct=-0.37, status='ALERT',
            trend='DECLINING', resolution='1k')

        db.extract_predictions_from_coupling()
        rows = db.conn.execute("SELECT * FROM predictions").fetchall()
        assert len(rows) == 1
        assert rows[0]['trigger_status'] == 'ALERT'
        assert rows[0]['predicted_class'] == 'M'


class TestMergeDuplicateEpisodes:
    """Cleanup for rows the pre-fix batch path wrote inside existing episodes."""

    @staticmethod
    def _episode(db, start, last, status='ELEVATED', n=3, pair='193-211'):
        db.insert_prediction(prediction_time=start, trigger_pair=pair,
                             trigger_status=status, trigger_kind='THRESHOLD')
        db.conn.execute(
            "UPDATE predictions SET last_trigger_time=?, n_triggers=? "
            "WHERE prediction_time=? AND trigger_pair=?",
            (last, n, start, pair))
        db.conn.commit()

    @staticmethod
    def _loose_row(db, ts, status='ELEVATED', pair='193-211'):
        """A row as the old batch path wrote it: no episode fields."""
        db.conn.execute(
            "INSERT INTO predictions (prediction_time, trigger_pair, trigger_status) "
            "VALUES (?, ?, ?)", (ts, pair, status))
        db.conn.commit()

    def test_contained_row_is_folded_in(self, db):
        self._episode(db, '2026-03-01T10:00:00', '2026-03-01T10:50:00', n=6)
        self._loose_row(db, '2026-03-01T10:20:00')

        result = db.merge_duplicate_prediction_episodes()
        assert result['merged'] == 1
        assert result['remaining'] == 0

        rows = db.conn.execute("SELECT * FROM predictions").fetchall()
        assert len(rows) == 1
        assert rows[0]['prediction_time'] == '2026-03-01T10:00:00'
        assert rows[0]['n_triggers'] == 7  # 6 + the folded row

    def test_more_severe_status_survives(self, db):
        self._episode(db, '2026-03-01T10:00:00', '2026-03-01T10:50:00', 'ELEVATED')
        self._loose_row(db, '2026-03-01T10:20:00', 'ALERT')

        db.merge_duplicate_prediction_episodes()
        row = db.conn.execute("SELECT * FROM predictions").fetchone()
        assert row['trigger_status'] == 'ALERT'
        assert row['predicted_class'] == 'M'

    def test_row_outside_the_span_is_kept(self, db):
        self._episode(db, '2026-03-01T10:00:00', '2026-03-01T10:50:00')
        self._loose_row(db, '2026-03-01T14:00:00')  # separate alarm

        assert db.merge_duplicate_prediction_episodes()['merged'] == 0
        assert db.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 2

    def test_other_pair_is_not_touched(self, db):
        self._episode(db, '2026-03-01T10:00:00', '2026-03-01T10:50:00', pair='193-211')
        self._loose_row(db, '2026-03-01T10:20:00', pair='193-304')

        assert db.merge_duplicate_prediction_episodes()['merged'] == 0
        assert db.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 2

    def test_verified_row_is_left_alone(self, db):
        """Merging an evaluated prediction would rewrite a recorded result."""
        self._episode(db, '2026-03-01T10:00:00', '2026-03-01T10:50:00')
        self._loose_row(db, '2026-03-01T10:20:00')
        db.conn.execute(
            "UPDATE predictions SET verified=1 WHERE prediction_time='2026-03-01T10:20:00'")
        db.conn.commit()

        result = db.merge_duplicate_prediction_episodes()
        assert result['merged'] == 0
        assert result['remaining'] == 1  # reported, not silently dropped
        assert db.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 2

    def test_row_with_flare_match_is_left_alone(self, db):
        self._episode(db, '2026-03-01T10:00:00', '2026-03-01T10:50:00')
        self._loose_row(db, '2026-03-01T10:20:00')
        flare_id = db.insert_flare_event(
            start_time='2026-03-01T11:00:00', flare_class='M', magnitude=1.0)
        dup_id = db.conn.execute(
            "SELECT id FROM predictions WHERE prediction_time='2026-03-01T10:20:00'"
        ).fetchone()[0]
        db.insert_prediction_match(dup_id, flare_id, 'hit')

        assert db.merge_duplicate_prediction_episodes()['merged'] == 0
        assert db.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 2

    def test_is_idempotent(self, db):
        self._episode(db, '2026-03-01T10:00:00', '2026-03-01T10:50:00')
        self._loose_row(db, '2026-03-01T10:20:00')

        db.merge_duplicate_prediction_episodes()
        assert db.merge_duplicate_prediction_episodes()['merged'] == 0

    def test_dry_run_writes_nothing(self, db):
        self._episode(db, '2026-03-01T10:00:00', '2026-03-01T10:50:00')
        self._loose_row(db, '2026-03-01T10:20:00')

        self._loose_row(db, '2026-03-01T10:30:00')  # second duplicate

        before = db.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        result = db.merge_duplicate_prediction_episodes(dry_run=True)
        after = db.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        assert before == after == 3
        # Reports the true candidate count, not just the first one found
        assert result['merged'] == 2

    def test_since_bounds_the_cleanup(self, db):
        self._episode(db, '2026-01-01T10:00:00', '2026-01-01T10:50:00')
        self._loose_row(db, '2026-01-01T10:20:00')

        assert db.merge_duplicate_prediction_episodes(
            since='2026-03-01T00:00:00')['merged'] == 0
        assert db.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 2


class TestTriggerKindClassification:
    """Both paths share one classifier, so they cannot drift apart."""

    def test_sudden_drop_wins(self):
        kind, value, threshold = classify_trigger_kind(
            {'sudden_drop_severity': 'SEVERE', 'sudden_drop_pct': -0.3,
             'residual': -4.0, 'is_break': True})
        assert (kind, value, threshold) == ('SUDDEN_DROP', -0.3, 2.5)

    def test_break_before_threshold(self):
        kind, _, _ = classify_trigger_kind(
            {'is_break': True, 'z_mad': 3.0, 'residual': -4.0})
        assert kind == 'BREAK'

    def test_threshold_picks_the_steepest_band(self):
        """Bands are in sigma, matching classify_status()."""
        assert classify_trigger_kind({'residual': -3.5})[2] == -3.0
        assert classify_trigger_kind({'residual': -2.5})[2] == -2.0
        assert classify_trigger_kind({'residual': -1.7})[2] == -1.5

    def test_never_returns_none(self):
        """A NULL trigger_kind left 2376 stored rows unattributable."""
        kind, _, _ = classify_trigger_kind({})
        assert kind == 'STATUS_ONLY'
        kind, _, _ = classify_trigger_kind({'residual': 0.5, 'trend': 'STABLE'})
        assert kind == 'STATUS_ONLY'


class TestBreakWindowAnchor:
    @staticmethod
    def _history(monitor, values, base):
        monitor.history = [
            {
                'timestamp': (base + timedelta(minutes=10 * i)).strftime("%Y-%m-%dT%H:%M:%S"),
                'coupling': {'193-211': {'delta_mi': v}},
            }
            for i, v in enumerate(values)
        ]

    def test_stale_data_still_yields_a_window(self, monitor):
        """Data six hours behind wallclock must not empty the window.

        Anchoring on wallclock made every gap-backfilled or lagging batch
        report "insufficient data", which is indistinguishable from a quiet Sun.
        """
        base = datetime.now(timezone.utc) - timedelta(hours=6)
        self._history(monitor, [0.80, 0.81, 0.79, 0.80, 0.80], base)

        result = detect_coupling_break('193-211', 0.50, monitor)
        assert result['n_points'] == 5
        assert result['is_break'] is True

    def test_explicit_anchor_is_respected(self, monitor):
        base = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
        self._history(monitor, [0.80, 0.81, 0.79, 0.80, 0.80], base)

        # Anchor two hours after the data: the 60 min window is empty
        result = detect_coupling_break(
            '193-211', 0.50, monitor,
            now=base + timedelta(hours=2))
        assert 'Insufficient data' in result['reason']

    def test_data_error_frames_excluded_from_statistics(self, monitor):
        base = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
        monitor.history = [
            {'timestamp': (base + timedelta(minutes=10 * i)).strftime("%Y-%m-%dT%H:%M:%S"),
             'coupling': {'193-211': entry}}
            for i, entry in enumerate([
                {'delta_mi': 0.80},
                {'delta_mi': 0.0, 'status': 'DATA_ERROR'},
                {'delta_mi': 0.81},
                {'delta_mi': 0.79},
                {'delta_mi': 0.80},
            ])
        ]
        result = detect_coupling_break(
            '193-211', 0.79, monitor, now=base + timedelta(minutes=50))
        assert result['n_points'] == 4
        assert result['median'] == pytest.approx(0.80, abs=0.01)


class TestTrendIgnoresDataErrors:
    def test_zero_from_data_error_does_not_drive_the_slope(self, monitor):
        monitor.history = [
            {'timestamp': f'2026-03-01T10:{i * 10:02d}:00',
             'coupling': {'193-211': entry}}
            for i, entry in enumerate([
                {'delta_mi': 0.80}, {'delta_mi': 0.80}, {'delta_mi': 0.80},
                {'delta_mi': 0.0, 'status': 'DATA_ERROR'},
                {'delta_mi': 0.80}, {'delta_mi': 0.80},
            ])
        ]
        trend = monitor.analyze_trend('193-211')
        assert trend['n_points'] == 5
        assert trend['trend'] == 'STABLE'
        assert trend['slope_pct_per_hour'] == pytest.approx(0.0, abs=1e-6)


class TestQualityGate:
    def test_deep_collapse_stays_valid(self):
        """A -75% collapse is the signal, not a data error."""
        result = validate_mi_measurement(0.20, '193-211', baseline_mean=0.79)
        assert result['is_valid'] is True
        assert result['is_extreme_low'] is True

    def test_documented_flare_signature_stays_valid(self):
        # -47%, the deep end of the documented flare range
        result = validate_mi_measurement(0.42, '193-211', baseline_mean=0.79)
        assert result['is_valid'] is True
        assert result['is_extreme_low'] is False

    def test_noise_floor_still_rejected(self):
        assert validate_mi_measurement(0.001, '193-211')['is_valid'] is False
        assert validate_mi_measurement(-0.06, '193-211')['is_valid'] is False

    def test_non_finite_rejected(self):
        assert validate_mi_measurement(float('nan'), '193-211')['is_valid'] is False
        assert validate_mi_measurement(float('inf'), '193-211')['is_valid'] is False


class TestAnomalyStatusIsUnambiguous:
    @staticmethod
    def _break():
        return {'is_break': True, 'z_mad': 3.5, 'k': 2.0}

    def test_precursor_is_actionable_and_unvetoed(self):
        status = classify_anomaly_status(
            self._break(),
            robustness_check={'is_robust': True, 'change_pct': 2.0},
            time_spread_sec=10,
            goes_context={'rising': True},
        )
        assert status['status'] == AnomalyStatus.VALIDATED_BREAK
        assert status['is_actionable'] is True
        assert status['veto_reasons'] == []

    def test_phase_gated_break_gets_its_own_status(self):
        """A decay-phase break is validated but not actionable.

        It used to be reported as VALIDATED_BREAK with a veto_reason attached -
        contradicting the rule that veto_reasons implies ANOMALY_VETOED, so a
        caller reading only `status` saw an actionable break.
        """
        status = classify_anomaly_status(
            self._break(),
            robustness_check={'is_robust': True, 'change_pct': 2.0},
            time_spread_sec=10,
            trend_info={'slope_pct_per_hour': -1.0, 'acceleration': 3.0},
            goes_context={'rising': False, 'phase': 'decay'},
        )
        assert status['status'] == AnomalyStatus.PHASE_GATED
        assert status['is_actionable'] is False
        assert status['is_validated'] is True
        assert status['break_type'] == BreakType.POSTCURSOR
        # veto_reasons stays reserved for validation failures
        assert status['veto_reasons'] == []

    def test_validation_failure_vetoes(self):
        status = classify_anomaly_status(
            self._break(),
            robustness_check={'is_robust': False, 'change_pct': 130.0},
            time_spread_sec=10,
            goes_context={'rising': True},
        )
        assert status['status'] == AnomalyStatus.ANOMALY_VETOED
        assert status['is_actionable'] is False
        assert status['is_validated'] is False
        assert any('robustness' in r for r in status['veto_reasons'])


class TestDiagnosticRendering:
    """The diagnostics panel must never print an empty reason.

    PHASE_GATED breaks carry no veto_reasons - the reason they are not
    actionable is the phase, not a failed test - so the old
    `[VETOED: {', '.join(veto)}]` rendered as a bare "[VETOED: ]".
    """

    @staticmethod
    def _render(status, width=140):
        import io

        from rich.console import Console

        from solar_seed.monitoring.formatting import StatusFormatter

        fmt = StatusFormatter()
        fmt.console = Console(file=io.StringIO(), width=width, no_color=True)
        bd = {'is_break': True, 'z_mad': 2.6, 'k': 2.0}
        fmt._print_alerts(
            {'_validation': {'anomaly_statuses': {'193-211': status},
                             'break_detections': {'193-211': bd}}},
            AnomalyStatus, BreakType)
        return fmt.console.file.getvalue()

    @staticmethod
    def _break():
        return {'is_break': True, 'z_mad': 2.6, 'k': 2.0}

    def test_ambiguous_shows_phase_reason(self):
        status = classify_anomaly_status(
            self._break(),
            robustness_check={'is_robust': True, 'change_pct': 2.0},
            time_spread_sec=10,
            trend_info={'slope_pct_per_hour': 0.1, 'acceleration': 0.0},
            goes_context={'rising': False, 'phase': 'active'},
        )
        out = self._render(status)
        assert 'VETOED: ]' not in out
        assert 'phase-gated' in out
        assert BreakType.AMBIGUOUS in out

    def test_postcursor_keeps_its_own_line(self):
        status = classify_anomaly_status(
            self._break(),
            robustness_check={'is_robust': True, 'change_pct': 2.0},
            time_spread_sec=10,
            trend_info={'slope_pct_per_hour': -1.0, 'acceleration': 3.0},
            goes_context={'rising': False, 'phase': 'decay'},
        )
        assert 'POSTCURSOR' in self._render(status)

    def test_real_veto_still_names_the_test(self):
        status = classify_anomaly_status(
            self._break(),
            robustness_check={'is_robust': False, 'change_pct': 130.0},
            time_spread_sec=10,
            goes_context={'rising': True},
        )
        out = self._render(status)
        assert 'VETOED' in out
        assert 'robustness' in out
