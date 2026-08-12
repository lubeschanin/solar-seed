#!/bin/bash
# Backfill 1k synoptic measurements with 4k JSOC data
#
# Usage:
#   ./scripts/backfill.sh              # Backfill last 14 days
#   ./scripts/backfill.sh --dry-run    # Preview only
#   ./scripts/backfill.sh --status     # Show backfill stats
#   ./scripts/backfill.sh --days 30    # Custom range
#
# Cron (täglich 07:00, nach Flare-Import):
#   0 7 * * * /Users/vl/git/4free/solar-seed-project/scripts/backfill.sh --days 14 >> /Users/vl/git/4free/solar-seed-project/results/early_warning/backfill.log 2>&1

set -euo pipefail

# Lock against concurrent runs (macOS has no flock; mkdir is atomic).
#
# The lock records its owner's PID. A bare mkdir lock is only correct while
# every holder exits: on 2026-08-09 a run hung in interpreter shutdown for
# three days, and the three following nightly runs each reported "already
# running" and exited 0 — a dead job that looked like a healthy one. If the
# recorded PID is gone, the lock is stale and gets reclaimed.
LOCKDIR="/tmp/solar-backfill.lock"
PIDFILE="$LOCKDIR/pid"

if ! mkdir "$LOCKDIR" 2>/dev/null; then
    OWNER="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [ -n "$OWNER" ] && kill -0 "$OWNER" 2>/dev/null; then
        echo "backfill.sh: another instance is already running (pid $OWNER, lock: $LOCKDIR) — exiting"
        exit 0
    fi
    echo "backfill.sh: reclaiming stale lock $LOCKDIR (owner ${OWNER:-unknown} is gone)" >&2
    rm -rf "$LOCKDIR"
    if ! mkdir "$LOCKDIR" 2>/dev/null; then
        echo "backfill.sh: lost the race for $LOCKDIR — exiting"
        exit 0
    fi
fi
echo $$ > "$PIDFILE"
trap 'rm -rf "$LOCKDIR" 2>/dev/null || true' EXIT

cd "$(dirname "$0")/.." || { echo "backfill.sh: cd to project root failed" >&2; exit 1; }

# Default --days 14 only when no args were given (avoid duplicate/conflicting flags)
if [ $# -eq 0 ]; then
    set -- --days 14
fi

# cron runs with a minimal PATH (/usr/bin:/bin:/usr/sbin:/sbin) that does not
# include Homebrew, so a bare `uv` fails with "command not found" every night
# without the script ever reporting anything useful. Resolve it explicitly.
UV="${UV:-}"
if [ -z "$UV" ]; then
    for candidate in /opt/homebrew/bin/uv /usr/local/bin/uv "$HOME/.local/bin/uv"; do
        [ -x "$candidate" ] && { UV="$candidate"; break; }
    done
fi
[ -z "$UV" ] && UV="$(command -v uv || true)"
if [ -z "$UV" ]; then
    echo "backfill.sh: uv not found (looked in /opt/homebrew/bin, /usr/local/bin, ~/.local/bin, PATH)" >&2
    exit 1
fi

"$UV" run python scripts/early_warning.py backfill "$@"
