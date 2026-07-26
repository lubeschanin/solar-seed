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

# Lock against concurrent runs (macOS has no flock; mkdir is atomic)
LOCKDIR="/tmp/solar-backfill.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
    echo "backfill.sh: another instance is already running (lock: $LOCKDIR) — exiting"
    exit 0
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

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
