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

cd "$(dirname "$0")/.."
uv run python scripts/early_warning.py backfill --days 14 "$@"
