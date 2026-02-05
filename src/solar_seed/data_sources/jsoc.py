"""
JSOC AIA Data Loader (4k Backfill)
==================================

Load 4096x4096 full-resolution AIA data from JSOC for backfilling.

JSOC is the only source for true 4k AIA data. Other providers (SDAC, etc.)
only serve reduced 1k resolution.

Key characteristics:
- Resolution: 4096x4096 (true full-res)
- Latency: Variable (currently offline since Jan 8, 2026)
- Access: Via SunPy/Fido with Provider='JSOC'
- File size: ~65 MB per channel (vs ~4 MB for 1k)

Use case: Backfilling 1k synoptic measurements with accurate 4k MI values
once JSOC processing resumes.
"""

from datetime import datetime, timedelta
from typing import Optional
import tempfile
import os


def load_aia_jsoc(
    timestamp: str,
    wavelengths: list[int] = None,
    time_window_minutes: int = 3
) -> tuple[dict, dict] | tuple[None, None]:
    """
    Load 4k AIA data from JSOC for a specific historical timestamp.

    Only accepts true 4096x4096 data. Returns None if data is not 4k.

    Args:
        timestamp: ISO format timestamp (e.g., '2026-01-07T12:00:00')
        wavelengths: List of wavelengths to load (default: [193, 211, 304])
        time_window_minutes: Search window around timestamp (default: ±3 min)

    Returns:
        (channels_dict, metadata) where:
            channels_dict: {wavelength: numpy.ndarray} with 4096x4096 data
            metadata: {'source': 'JSOC', 'resolution': '4k', 'timestamps': {...}}
        Or (None, None) if 4k data not available
    """
    if wavelengths is None:
        wavelengths = [193, 211, 304]

    try:
        from sunpy.net import Fido, attrs as a
        from sunpy.map import Map
        from aiapy.calibrate import register
        import astropy.units as u
        import numpy as np
    except ImportError as e:
        print(f"    JSOC loader requires sunpy + aiapy: {e}")
        return None, None

    # Parse timestamp
    if isinstance(timestamp, str):
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    else:
        dt = timestamp

    start = dt - timedelta(minutes=time_window_minutes)
    end = dt + timedelta(minutes=time_window_minutes)

    print(f"    Loading from JSOC: {dt.isoformat()}")

    channels = {}
    timestamps = {}
    quality_flags = {}

    for wl in wavelengths:
        try:
            # Search JSOC specifically for 4k data
            result = Fido.search(
                a.Time(start, end),
                a.Instrument.aia,
                a.Wavelength(wl * u.Angstrom),
                a.Provider('JSOC')
            )

            if len(result) == 0 or len(result[0]) == 0:
                print(f"      ✗ {wl}Å: No JSOC data available")
                continue

            # Verify it's 4k data by checking extent
            # Note: JSOC returns astropy Quantities, must use float() for comparison
            first_result = result[0][0]
            try:
                extent_w = float(first_result['Extent Width'])
                extent_h = float(first_result['Extent Length'])
                if extent_w < 4000 or extent_h < 4000:
                    print(f"      ✗ {wl}Å: Not 4k ({extent_w:.0f}x{extent_h:.0f})")
                    continue
            except (KeyError, TypeError, ValueError):
                # If extent not in metadata, check file size (~65MB for 4k)
                try:
                    size_mb = float(first_result['Size'])
                    if size_mb < 50:  # 4k is ~65MB, 1k is ~4MB
                        print(f"      ✗ {wl}Å: File too small ({size_mb:.1f}MB, need ~65MB for 4k)")
                        continue
                except (KeyError, TypeError, ValueError):
                    pass

            with tempfile.TemporaryDirectory() as tmpdir:
                # Fetch the first (closest) result
                files = Fido.fetch(result[0, 0], path=tmpdir, progress=False)

                if not files:
                    print(f"      ✗ {wl}Å: Download failed")
                    continue

                # Load with SunPy Map
                smap = Map(files[0])

                # Final validation: must be 4k
                if smap.data.shape[0] < 4000 or smap.data.shape[1] < 4000:
                    print(f"      ✗ {wl}Å: Data is {smap.data.shape}, not 4k - skipping")
                    continue

                # Calibrate: Level 1 → 1.5 (register + exposure normalize)
                # register() fixes per-channel pointing, rotation, plate scale
                # Without this, channels are misaligned by 30-40px → MI ≈ 0
                smap_reg = register(smap)
                exptime = smap.meta.get('EXPTIME', 1.0)
                data = smap_reg.data.astype(np.float64) / exptime

                channels[wl] = data
                timestamps[wl] = smap.date.isot

                # Get quality info
                quality_flags[wl] = smap.meta.get('QUALITY', 0)

                print(f"      ✓ {wl}Å: {data.shape} lev1.5 (mean={data.mean():.1f} DN/s)")

        except Exception as e:
            print(f"      ✗ {wl}Å: Error - {e}")
            continue

    if not channels:
        return None, None

    # Verify all channels are 4k
    for wl, data in channels.items():
        if data.shape[0] < 4000:
            print(f"    ⚠️ Channel {wl}Å is not 4k - aborting")
            return None, None

    metadata = {
        'source': 'JSOC',
        'resolution': '4k',
        'timestamps': timestamps,
        'quality_flags': quality_flags,
        'requested_time': dt.isoformat(),
    }

    return channels, metadata


def check_jsoc_4k_availability(timestamp: str, wavelength: int = 193) -> bool:
    """
    Quick check if JSOC has 4k data for a given timestamp.

    Args:
        timestamp: ISO format timestamp
        wavelength: Wavelength to check (default: 193Å)

    Returns:
        True if 4k data available, False otherwise
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
            a.Provider('JSOC')
        )

        if len(result) == 0 or len(result[0]) == 0:
            return False

        # Check if it's 4k (by extent or file size)
        # Note: JSOC returns astropy Quantities, must use .value for comparison
        try:
            extent = float(result[0][0]['Extent Width'])
            return extent >= 4000
        except (KeyError, TypeError, ValueError):
            try:
                size = float(result[0][0]['Size'])
                return size > 50  # 4k is ~65 MiB
            except (KeyError, TypeError, ValueError):
                return True  # Assume 4k if we can't check

    except Exception:
        return False


def get_jsoc_latest_date() -> Optional[str]:
    """
    Find the most recent date with 4k JSOC data available.

    Returns:
        ISO date string (YYYY-MM-DD) or None if JSOC unavailable
    """
    from datetime import date, timedelta

    # Binary search for most recent available date
    today = date.today()

    for days_back in [0, 1, 2, 3, 5, 7, 10, 14, 21, 30]:
        check_date = today - timedelta(days=days_back)
        timestamp = f"{check_date.isoformat()}T12:00:00"

        if check_jsoc_4k_availability(timestamp):
            return check_date.isoformat()

    return None
