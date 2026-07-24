#!/usr/bin/env python3
"""Gap backfill: recompute coupling measurements for a historical period
from the dated JSOC synoptic archive (1k, 2-min cadence).

Fills monitoring outages (e.g. Jul 1-24, 2026) with the same MI pipeline
as the live monitor:

    subtract_radial_geometry -> sector_ring_shuffle_test(n_rings=10, n_sectors=12)
    delta_mi = mi_original - mi_sector_shuffled

Rows are stored with pipeline_version='gap-backfill-1.0' and without
status/trend (those depend on live monitor state and are not
reconstructed retroactively). Existing measurements are never touched:
timestamps that already have a row for a pair are skipped, which also
makes reruns resumable.

Usage:
  uv run python scripts/gap_backfill.py --start 2026-07-01T00:00 --end 2026-07-24T15:00
  uv run python scripts/gap_backfill.py ... --cadence 12 --limit 5 --dry-run
"""

import argparse
import sys
from datetime import datetime, timedelta

from solar_seed.data_sources.synoptic import load_aia_synoptic_archive
from solar_seed.radial_profile import subtract_radial_geometry
from solar_seed.control_tests import sector_ring_shuffle_test
from solar_seed.monitoring.coupling import CouplingMonitor
from solar_seed.monitoring.db import MonitoringDB

PIPELINE_VERSION = "gap-backfill-1.0"
WAVELENGTHS = [193, 211, 304]
PAIRS = [(193, 211), (193, 304)]
MAX_CONSECUTIVE_FAILURES = 20


def existing_timestamps(db, start_iso, end_iso):
    """Set of (timestamp, pair) already present in the target window."""
    rows = db.conn.execute(
        "SELECT timestamp, pair FROM coupling_measurements "
        "WHERE timestamp >= ? AND timestamp <= ?",
        (start_iso, end_iso),
    ).fetchall()
    return {(r[0], r[1]) for r in rows}


def nearest_slot_iso(t: datetime) -> str:
    """DB timestamp for the 2-min archive slot of t (naive UTC, T-separator)."""
    slot = t.replace(second=0, microsecond=0)
    slot -= timedelta(minutes=slot.minute % 2)
    return slot.strftime("%Y-%m-%dT%H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description="Backfill coupling gap from synoptic archive")
    parser.add_argument("--start", required=True, help="Start (UTC ISO, e.g. 2026-07-01T00:00)")
    parser.add_argument("--end", required=True, help="End (UTC ISO)")
    parser.add_argument("--cadence", type=int, default=12, help="Cadence in minutes (default 12)")
    parser.add_argument("--limit", type=int, default=None, help="Max timepoints to process")
    parser.add_argument("--dry-run", action="store_true", help="Compute nothing, list planned timepoints")
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    baselines = CouplingMonitor.BASELINES_1K

    timepoints = []
    t = start
    while t <= end:
        timepoints.append(t)
        t += timedelta(minutes=args.cadence)

    db = MonitoringDB()
    have = existing_timestamps(db, nearest_slot_iso(start), nearest_slot_iso(end))
    print(f"Gap backfill {args.start} -> {args.end} ({len(timepoints)} timepoints, "
          f"cadence {args.cadence} min, {len(have)} existing rows in window)")

    todo = [t for t in timepoints
            if any((nearest_slot_iso(t), f"{a}-{b}") not in have for a, b in PAIRS)]
    print(f"{len(todo)} timepoints to process, {len(timepoints) - len(todo)} already complete")

    if args.limit:
        todo = todo[:args.limit]
    if args.dry_run:
        for t in todo[:10]:
            print(f"  would process {t}")
        if len(todo) > 10:
            print(f"  ... and {len(todo) - 10} more")
        return 0

    inserted = failed = skipped = 0
    consecutive_failures = 0

    for i, t in enumerate(todo, 1):
        try:
            channels, iso_ts, quality = load_aia_synoptic_archive(t, WAVELENGTHS)
        except Exception as e:
            channels, iso_ts, quality = None, None, None
            print(f"  ✗ {t:%Y-%m-%dT%H:%M}: loader error - {e}")

        if not channels:
            failed += 1
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"Aborting: {MAX_CONSECUTIVE_FAILURES} consecutive load failures")
                break
            continue
        consecutive_failures = 0

        ts_db = iso_ts.rstrip("Z")
        spread = quality.get("time_spread_sec")
        quality_ok = (spread is not None and spread < 60
                      and len(channels) == len(WAVELENGTHS))

        residuals = {}
        for wl, data in channels.items():
            res, _, _ = subtract_radial_geometry(data)
            residuals[wl] = res

        for wl1, wl2 in PAIRS:
            pair = f"{wl1}-{wl2}"
            if (ts_db, pair) in have:
                skipped += 1
                continue
            if wl1 not in residuals or wl2 not in residuals:
                failed += 1
                continue
            try:
                shuffle = sector_ring_shuffle_test(
                    residuals[wl1], residuals[wl2], n_rings=10, n_sectors=12
                )
                delta_mi = shuffle.mi_original - shuffle.mi_sector_shuffled
                base = baselines.get(pair)
                residual_z = (delta_mi - base["mean"]) / base["std"] if base else None
                deviation = (delta_mi - base["mean"]) / base["mean"] if base else None

                row_id = db.insert_coupling(
                    timestamp=ts_db, pair=pair, delta_mi=delta_mi,
                    mi_original=shuffle.mi_original,
                    residual=residual_z, deviation_pct=deviation,
                    pipeline_version=PIPELINE_VERSION,
                    quality_ok=quality_ok, sync_delta_s=spread,
                    resolution="1k",
                )
                if row_id == -1:
                    failed += 1
                    print(f"  ✗ {ts_db} {pair}: DB insert failed")
                else:
                    inserted += 1
                    have.add((ts_db, pair))
            except Exception as e:
                failed += 1
                print(f"  ✗ {ts_db} {pair}: {e}")

        if i % 25 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] {t:%Y-%m-%d %H:%M} | inserted={inserted} "
                  f"skipped={skipped} failed={failed}", flush=True)

    print(f"\nDone: inserted={inserted}, skipped={skipped}, failed={failed}")
    return 0 if inserted or not todo else 1


if __name__ == "__main__":
    sys.exit(main())
