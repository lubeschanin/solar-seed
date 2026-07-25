"""
Coupling Baselines
==================

Baselines (mean/std of ΔMI per channel pair and resolution) drive every
residual, deviation_pct and status in the early warning system. They used to
be hardcoded class constants in CouplingMonitor, derived from an early
short-run measurement. Once six months of monitoring accumulated, the coded
values no longer matched the data they describe:

    193-304 @ 1k:  coded 0.07 ± 0.02, measured 0.181 ± 0.057  (+5.6σ offset)
    193-304 @ 4k:  coded 0.32 ± 0.12, measured 0.103          (65% of all
                                                               4k readings
                                                               fired ALERT)

Baselines are therefore computed from the monitoring database and stored in
a versioned JSON file. The hardcoded table below remains only as a cold-start
fallback and is explicitly marked provisional.

Estimator
---------
Median and MAD-scaled sigma (1.4826 × MAD), not mean/std: the ΔMI
distribution has a heavy low tail (flare collapses, data errors) that would
drag a plain mean downward and inflate std. Robust statistics keep the
baseline anchored to the quiet-Sun level, which is what a deviation is
supposed to be measured against.

Quiet-window definition
-----------------------
Measurements within `exclude_flare_hours` of an M/X flare (start_time to
peak_time + window) are excluded, so the baseline describes the undisturbed
state rather than an average over flaring and quiet periods alike.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASELINE_FILE = Path("results/early_warning/baselines.json")

# Cold-start fallback only. These are the historical hardcoded values; they are
# known to disagree with the measured distributions (see module docstring).
# Run `early_warning.py baselines --recompute` to replace them with data.
PROVISIONAL_BASELINES = {
    '1k': {
        '193-211': {'mean': 0.59, 'std': 0.12},
        '193-304': {'mean': 0.07, 'std': 0.02},
        '171-193': {'mean': 0.17, 'std': 0.04},
        '211-335': {'mean': 0.28, 'std': 0.06},
    },
    '4k': {
        '193-211': {'mean': 1.03, 'std': 0.31},
        '193-304': {'mean': 0.32, 'std': 0.12},
        '171-193': {'mean': 0.29, 'std': 0.07},
        '211-335': {'mean': 0.48, 'std': 0.10},
    },
}

# A baseline computed from fewer samples than this is not trusted; the
# provisional value is kept instead (and flagged in the metadata).
MIN_SAMPLES = 200

# Guard against a degenerate std: a near-zero sigma turns every reading into a
# multi-sigma anomaly. Floor at 10% of the mean.
MIN_STD_FRACTION = 0.10


def _median(values: list[float]) -> float:
    """Median of a non-empty list."""
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _mad_sigma(values: list[float], median: float) -> float:
    """MAD scaled to a normal-equivalent standard deviation."""
    return _median([abs(v - median) for v in values]) * 1.4826


def load_baselines(path: Path | None = None) -> dict:
    """
    Load baselines, preferring the measured file over the provisional table.

    Returns a dict {'1k': {pair: {mean, std}}, '4k': {...}, '_meta': {...}}.
    A missing, unreadable or malformed file falls back to PROVISIONAL_BASELINES
    rather than raising: the monitor must keep running.
    """
    path = path or BASELINE_FILE

    fallback = {
        '1k': dict(PROVISIONAL_BASELINES['1k']),
        '4k': dict(PROVISIONAL_BASELINES['4k']),
        '_meta': {'source': 'provisional', 'warning': 'hardcoded values, not measured'},
    }

    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return fallback

    if not isinstance(data, dict) or '1k' not in data or '4k' not in data:
        return fallback

    # Merge over the provisional table so pairs absent from the measured file
    # (e.g. 211-335, which the monitor does not currently sample) still resolve.
    merged = {
        '1k': {**PROVISIONAL_BASELINES['1k'], **data.get('1k', {})},
        '4k': {**PROVISIONAL_BASELINES['4k'], **data.get('4k', {})},
        '_meta': data.get('_meta', {'source': str(path)}),
    }
    return merged


def save_baselines(baselines: dict, path: Path | None = None) -> Path:
    """Write baselines to JSON atomically (temp file + replace)."""
    path = Path(path or BASELINE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(baselines, f, indent=2, sort_keys=True)
    tmp.replace(path)
    return path


def _flare_windows(db, days: int | None, exclude_flare_hours: float) -> list[tuple[str, str]]:
    """
    Time ranges to exclude, as (start_iso, end_iso) pairs.

    One window per M/X flare, spanning from `exclude_flare_hours` before the
    flare start to `exclude_flare_hours` after its peak (or start, if no peak
    time is recorded).
    """
    cursor = db.conn.cursor()
    sql = """
        SELECT start_time, peak_time FROM flare_events
        WHERE (class LIKE 'M%' OR class LIKE 'X%')
    """
    params: list = []
    if days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
        sql += " AND start_time >= ?"
        params.append(cutoff)

    delta = timedelta(hours=exclude_flare_hours)
    windows = []
    for start_time, peak_time in cursor.execute(sql, params):
        try:
            start = datetime.fromisoformat(str(start_time).replace('Z', '+00:00'))
            end = start
            if peak_time:
                end = datetime.fromisoformat(str(peak_time).replace('Z', '+00:00'))
        except (ValueError, TypeError):
            continue
        if start.tzinfo:
            start = start.astimezone(timezone.utc).replace(tzinfo=None)
        if end.tzinfo:
            end = end.astimezone(timezone.utc).replace(tzinfo=None)
        windows.append((
            (start - delta).strftime("%Y-%m-%dT%H:%M:%S"),
            (end + delta).strftime("%Y-%m-%dT%H:%M:%S"),
        ))
    return windows


def compute_baselines_from_db(
    db,
    days: int | None = None,
    exclude_flare_hours: float = 2.0,
    min_samples: int = MIN_SAMPLES,
) -> dict:
    """
    Compute per-(resolution, pair) baselines from stored coupling measurements.

    Args:
        db: MonitoringDB instance
        days: Restrict to the last N days (None = all history)
        exclude_flare_hours: Drop measurements within this many hours of an
            M/X flare, so the baseline describes quiet Sun
        min_samples: Below this count the provisional value is kept

    Returns:
        Baseline dict ready for save_baselines(), including a '_meta' block
        recording how each value was derived.
    """
    cursor = db.conn.cursor()
    windows = _flare_windows(db, days, exclude_flare_hours)

    sql = """
        SELECT resolution, pair, timestamp, delta_mi
        FROM coupling_measurements
        WHERE delta_mi IS NOT NULL
          AND resolution IS NOT NULL
          AND (status IS NULL OR status != 'DATA_ERROR')
    """
    params: list = []
    if days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
        sql += " AND timestamp >= ?"
        params.append(cutoff)

    buckets: dict[str, dict[str, list[float]]] = {'1k': {}, '4k': {}}
    n_excluded = 0

    for resolution, pair, timestamp, delta_mi in cursor.execute(sql, params):
        if resolution not in buckets:
            continue
        ts = str(timestamp)
        if any(lo <= ts <= hi for lo, hi in windows):
            n_excluded += 1
            continue
        buckets[resolution].setdefault(pair, []).append(float(delta_mi))

    result: dict = {
        '1k': dict(PROVISIONAL_BASELINES['1k']),
        '4k': dict(PROVISIONAL_BASELINES['4k']),
    }
    details: dict = {}

    for resolution, pairs in buckets.items():
        for pair, values in pairs.items():
            n = len(values)
            if n < min_samples:
                details[f'{resolution}/{pair}'] = {
                    'n': n,
                    'used': 'provisional',
                    'reason': f'only {n} samples (need {min_samples})',
                }
                continue
            median = _median(values)
            sigma = _mad_sigma(values, median)
            floor = abs(median) * MIN_STD_FRACTION
            if sigma < floor:
                sigma = floor
            result[resolution][pair] = {'mean': round(median, 4), 'std': round(sigma, 4)}
            details[f'{resolution}/{pair}'] = {
                'n': n,
                'used': 'measured',
                'median': round(median, 4),
                'mad_sigma': round(sigma, 4),
            }

    result['_meta'] = {
        'computed_at': datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        'source': 'monitoring.db',
        'estimator': 'median / 1.4826*MAD',
        'days': days,
        'exclude_flare_hours': exclude_flare_hours,
        'min_samples': min_samples,
        'excluded_near_flare': n_excluded,
        'pairs': details,
    }
    return result
