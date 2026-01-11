"""
Time Utilities
==============

Common timestamp parsing and formatting functions.
Handles ISO 8601 timestamps with timezone awareness.
"""

from datetime import datetime, timezone
from typing import Optional


def parse_iso_timestamp(ts: str) -> Optional[datetime]:
    """
    Parse ISO 8601 timestamp to timezone-aware datetime.

    Handles common formats:
    - '2026-01-11T12:00:00Z'
    - '2026-01-11T12:00:00+00:00'
    - '2026-01-11T12:00:00' (assumes UTC)

    Args:
        ts: ISO timestamp string

    Returns:
        Timezone-aware datetime (UTC) or None if parsing fails
    """
    if not ts:
        return None

    # Handle 'Z' suffix (common in APIs)
    ts = ts.replace('Z', '+00:00')

    try:
        dt = datetime.fromisoformat(ts)
        # Ensure timezone awareness (default to UTC)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def format_timestamp(dt: datetime, fmt: str = 'iso') -> str:
    """
    Format datetime for display or storage.

    Args:
        dt: Datetime object
        fmt: Format type ('iso', 'display', 'compact')

    Returns:
        Formatted timestamp string
    """
    if dt is None:
        return ''

    if fmt == 'iso':
        return dt.isoformat()
    elif fmt == 'display':
        return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    elif fmt == 'compact':
        return dt.strftime('%Y%m%d_%H%M%S')
    else:
        return dt.isoformat()


def now_utc() -> datetime:
    """Get current time as timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)
