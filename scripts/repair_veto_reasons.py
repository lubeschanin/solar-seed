#!/usr/bin/env python3
"""
Reconstruct veto_reason for historical rows flagged without one.

Rows written before the shared quality verdict (assess_measurement_quality)
carry quality_ok=0 with veto_reason NULL: the store path evaluated three tests
but only passed the break_vetoed one through to the database. A row excluded
for an unrecorded reason cannot be reviewed, and cannot be un-excluded by an
analysis that decides the reason does not matter for its question.

The stored fields still hold the evidence, so most reasons are recoverable:
sync_delta_s and robustness_score were written even when the reason was not.
What cannot be reconstructed is labelled 'unknown(legacy)' rather than left
NULL - "flagged before reasons were recorded" is itself information, and it
keeps the invariant "quality_ok=0 implies a reason" true from here on.

Usage:
    uv run python scripts/repair_veto_reasons.py --dry-run
    uv run python scripts/repair_veto_reasons.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from solar_seed.monitoring.constants import (  # noqa: E402
    ROBUSTNESS_MAX_CHANGE_PCT,
    SYNC_SPREAD_MAX_SEC,
)
from solar_seed.monitoring.db import MonitoringDB  # noqa: E402

DB_PATH = Path(__file__).parent.parent / "results/early_warning/monitoring.db"


def build_reason(status, sync_delta_s, robustness_score) -> str:
    """Rebuild the reason string the store path would write today."""
    reasons = []
    if status == "DATA_ERROR":
        reasons.append("data_error")
    if sync_delta_s is not None and sync_delta_s > SYNC_SPREAD_MAX_SEC:
        reasons.append(f"time_sync({sync_delta_s:.0f}s)")
    if robustness_score is not None and robustness_score > ROBUSTNESS_MAX_CHANGE_PCT:
        reasons.append(f"robustness({robustness_score:.0f}%)")
    return "+".join(reasons) if reasons else "unknown(legacy)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args()

    db = MonitoringDB(db_path=Path(args.db))
    cur = db.conn.cursor()
    rows = cur.execute("""
        SELECT id, status, sync_delta_s, robustness_score
        FROM coupling_measurements
        WHERE quality_ok = 0 AND veto_reason IS NULL
    """).fetchall()

    if not rows:
        print("Nothing to repair.")
        db.close()
        return 0

    updates, tally = [], {}
    for row in rows:
        reason = build_reason(row["status"], row["sync_delta_s"], row["robustness_score"])
        updates.append((reason, row["id"]))
        key = "+".join(sorted(p.split("(")[0] for p in reason.split("+")))
        tally[key] = tally.get(key, 0) + 1

    print(f"Rows with quality_ok=0 and no reason: {len(rows)}")
    for key, n in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {key:<28} {n:>6}  ({n / len(rows) * 100:.1f}%)")
    recovered = len(rows) - tally.get("unknown", 0)
    print(f"\nReconstructed: {recovered} ({recovered / len(rows) * 100:.1f}%)")
    print(f"Unrecoverable: {tally.get('unknown', 0)}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        db.close()
        return 0

    cur.executemany(
        "UPDATE coupling_measurements SET veto_reason = ? WHERE id = ?", updates
    )
    db.conn.commit()
    print(f"\nUpdated {cur.rowcount if cur.rowcount != -1 else len(updates)} rows.")

    remaining = cur.execute(
        "SELECT COUNT(*) FROM coupling_measurements "
        "WHERE quality_ok = 0 AND veto_reason IS NULL"
    ).fetchone()[0]
    print(f"Remaining unexplained: {remaining}")
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
