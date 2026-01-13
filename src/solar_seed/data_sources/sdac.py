"""
SDAC AIA Data Loader
====================

Load 4096x4096 full-resolution AIA data from NASA SDAC (Solar Data Analysis Center).

SDAC mirrors SDO/AIA data with ~3 day latency, but provides full resolution
without the JSOC export queue limitations.

Key characteristics:
- Resolution: 4096x4096 (full)
- Latency: ~3 days
- Access: Via SunPy/VSO with Provider='SDAC'
- Rate limits: None (direct file access)

Use case: Backfilling 1k synoptic measurements with accurate 4k MI values.
"""

from datetime import datetime, timedelta
from typing import Optional
import tempfile
import os


def load_aia_sdac(
    timestamp: str,
    wavelengths: list[int] = None,
    time_window_minutes: int = 3
) -> tuple[dict, dict] | tuple[None, None]:
    """
    Load 4k AIA data from SDAC for a specific historical timestamp.

    SDAC has ~3 day latency but provides full 4096x4096 resolution.
    Used for backfilling 1k synoptic measurements.

    Args:
        timestamp: ISO format timestamp (e.g., '2026-01-10T12:00:00')
        wavelengths: List of wavelengths to load (default: [193, 211, 304])
        time_window_minutes: Search window around timestamp (default: ±3 min)

    Returns:
        (channels_dict, metadata) where:
            channels_dict: {wavelength: numpy.ndarray} with 4096x4096 data
            metadata: {'source': 'SDAC', 'resolution': '4k', 'timestamps': {...}}
        Or (None, None) if data not available
    """
    if wavelengths is None:
        wavelengths = [193, 211, 304]

    try:
        from sunpy.net import Fido, attrs as a
        from sunpy.map import Map
        import astropy.units as u
        import numpy as np
    except ImportError as e:
        print(f"    SDAC loader requires sunpy: {e}")
        return None, None

    # Parse timestamp
    if isinstance(timestamp, str):
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    else:
        dt = timestamp

    start = dt - timedelta(minutes=time_window_minutes)
    end = dt + timedelta(minutes=time_window_minutes)

    print(f"    Loading from SDAC: {dt.isoformat()}")

    channels = {}
    timestamps = {}
    quality_flags = {}

    for wl in wavelengths:
        try:
            # Search SDAC specifically
            result = Fido.search(
                a.Time(start, end),
                a.Instrument.aia,
                a.Wavelength(wl * u.Angstrom),
                a.Provider('SDAC')
            )

            if len(result) == 0 or len(result[0]) == 0:
                print(f"      ✗ {wl}Å: No SDAC data available")
                continue

            # Get the closest result to target time
            n_results = len(result[0])

            with tempfile.TemporaryDirectory() as tmpdir:
                # Fetch the first (closest) result
                files = Fido.fetch(result[0, 0], path=tmpdir, progress=False)

                if not files:
                    print(f"      ✗ {wl}Å: Download failed")
                    continue

                # Load with SunPy Map
                smap = Map(files[0])
                data = smap.data.astype(np.float64)

                # Validate it's actually 4k
                if data.shape[0] < 4000 or data.shape[1] < 4000:
                    print(f"      ⚠️ {wl}Å: Unexpected resolution {data.shape}")

                channels[wl] = data
                timestamps[wl] = smap.date.isot

                # Get quality info
                quality_flags[wl] = smap.meta.get('QUALITY', 0)

                print(f"      ✓ {wl}Å: {data.shape} (mean={data.mean():.1f})")

        except Exception as e:
            print(f"      ✗ {wl}Å: Error - {e}")
            continue

    if not channels:
        return None, None

    metadata = {
        'source': 'SDAC',
        'resolution': '4k',
        'timestamps': timestamps,
        'quality_flags': quality_flags,
        'requested_time': dt.isoformat(),
    }

    return channels, metadata


def check_sdac_availability(timestamp: str, wavelength: int = 193) -> bool:
    """
    Quick check if SDAC has data for a given timestamp.

    Args:
        timestamp: ISO format timestamp
        wavelength: Wavelength to check (default: 193Å)

    Returns:
        True if data available, False otherwise
    """
    try:
        from sunpy.net import Fido, attrs as a
        import astropy.units as u

        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        start = dt - timedelta(minutes=2)
        end = dt + timedelta(minutes=2)

        result = Fido.search(
            a.Time(start, end),
            a.Instrument.aia,
            a.Wavelength(wavelength * u.Angstrom),
            a.Provider('SDAC')
        )

        return len(result) > 0 and len(result[0]) > 0

    except Exception:
        return False
