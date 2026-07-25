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
from solar_seed.monitoring.db import MonitoringDB
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
