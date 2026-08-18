"""
Channel observation timestamps
==============================

Shared parsing of per-channel FITS observation times and the resulting
time spread (artifact test A).

This lives in its own module because three loaders and the 4k backfill all
need it. When only the synoptic loader had it, the JSOC backfill had no way to
measure its own spread - so backfilled rows silently kept the 1k value of the
row they replaced, and the stored 4k distribution was an echo of the 1k one
rather than a measurement.
"""

from datetime import datetime, timezone
from typing import Optional


def parse_obs_time(value) -> Optional[datetime]:
    """Parse a FITS T_OBS/DATE-OBS header value (e.g. '2026-01-11T11:46:04.84Z')."""
    try:
        s = str(value).strip().replace('Z', '+00:00')
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


def compute_time_spread_sec(timestamps: dict) -> Optional[float]:
    """
    Compute max-min spread (seconds) of per-channel observation times.

    Returns None if any timestamp cannot be parsed - an unknown spread must not
    be reported as a reassuring 0.
    """
    if not timestamps:
        return None
    if len(timestamps) < 2:
        return 0.0
    parsed = [parse_obs_time(v) for v in timestamps.values()]
    if any(p is None for p in parsed):
        return None
    return (max(parsed) - min(parsed)).total_seconds()
