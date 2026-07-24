"""
AIA Synoptic Data Loader
========================

Load 1024x1024 resolution AIA synoptic data from JSOC.
This is the preferred loader for real-time monitoring:
- Direct HTTP access (no export queue)
- Updated every 2 minutes
- Reliable and stable

Scale considerations:
- 193-211 Å pair: Scale-invariant (~5% difference from full-res)
- 193-304 Å pair: Scale-dependent (+33% difference)
"""

import tempfile
import os
from datetime import datetime, timezone
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

# JSOC synoptic data endpoint
from solar_seed.endpoints import endpoint

SYNOPTIC_BASE_URL = endpoint('synoptic_base')


def _parse_obs_time(value) -> Optional[datetime]:
    """Parse a FITS T_OBS/DATE-OBS header value (e.g. '2026-01-11T11:46:04.84Z')."""
    try:
        s = str(value).strip().replace('Z', '+00:00')
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


def _compute_time_spread_sec(timestamps: dict) -> Optional[float]:
    """
    Compute max-min spread (seconds) of per-channel observation times.

    Returns None if any timestamp cannot be parsed (no false '0' assurance).
    """
    if len(timestamps) < 2:
        return 0.0
    parsed = [_parse_obs_time(v) for v in timestamps.values()]
    if any(p is None for p in parsed):
        return None
    return (max(parsed) - min(parsed)).total_seconds()


def load_aia_synoptic(
    wavelengths: Optional[list[int]] = None
) -> tuple[dict, str, dict] | tuple[None, None, None]:
    """
    Load most recent AIA synoptic data (1024x1024 resolution).

    The synoptic archive is directly accessible without JSOC export queue.
    Updated every 2 minutes. Stable and reliable for real-time monitoring.

    Args:
        wavelengths: List of wavelengths to load (default: [193, 211, 304])

    Returns:
        (channels_dict, timestamp_str, quality_info) or (None, None, None)
    """
    if wavelengths is None:
        wavelengths = [193, 211, 304]

    print("  Loading AIA synoptic data (1k resolution)...")

    try:
        # Check mostrecent timestamp
        times_url = f"{SYNOPTIC_BASE_URL}/mostrecent/image_times"
        req = Request(times_url, headers={'User-Agent': 'SolarSeed/1.0'})
        with urlopen(req, timeout=10) as response:
            times_content = response.read().decode()

        # Parse timestamp: "Time     20260111_114600"
        timestamp_str = None
        for line in times_content.split('\n'):
            if line.startswith('Time'):
                parts = line.split()
                if len(parts) >= 2:
                    timestamp_str = parts[1]  # "20260111_114600"
                    break

        if not timestamp_str:
            print("    Could not parse synoptic timestamp")
            return None, None, None

        # Convert to ISO format
        iso_timestamp = (
            f"{timestamp_str[:4]}-{timestamp_str[4:6]}-{timestamp_str[6:8]}T"
            f"{timestamp_str[9:11]}:{timestamp_str[11:13]}:{timestamp_str[13:15]}Z"
        )
        print(f"    Synoptic timestamp: {iso_timestamp}")

        # Load FITS files
        try:
            from astropy.io import fits
            import numpy as np
        except ImportError:
            print("    Error: astropy required for FITS loading")
            return None, None, None

        channels = {}
        timestamps = {}

        for wl in wavelengths:
            fits_url = f"{SYNOPTIC_BASE_URL}/mostrecent/AIAsynoptic{wl:04d}.fits"
            print(f"    Fetching {wl} Å from synoptic...")

            try:
                req = Request(fits_url, headers={'User-Agent': 'SolarSeed/1.0'})
                with urlopen(req, timeout=30) as response:
                    fits_data = response.read()

                # Save to temp file and load with astropy
                with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
                    tmp.write(fits_data)
                    tmp_path = tmp.name

                try:
                    with fits.open(tmp_path) as hdul:
                        # Synoptic FITS uses compressed images in HDU[1]
                        data = None
                        header = None
                        if len(hdul) > 1 and hdul[1].data is not None:
                            data = hdul[1].data
                            header = hdul[1].header
                        elif hdul[0].data is not None:
                            data = hdul[0].data
                            header = hdul[0].header

                        if data is not None:
                            channels[wl] = data.astype(np.float64)
                            # Get timestamp from header if available
                            obs_time = header.get('T_OBS', header.get('DATE-OBS', iso_timestamp))
                            timestamps[wl] = obs_time
                            print(f"      ✓ {wl} Å: {data.shape} loaded")
                        else:
                            print(f"      ✗ {wl} Å: No data in FITS")
                finally:
                    # Always remove the temp file, even if FITS parsing fails
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            except URLError as e:
                print(f"      ✗ {wl} Å: Network error - {e}")
            except Exception as e:
                print(f"      ✗ {wl} Å: Load error - {e}")

        if not channels:
            print("    No synoptic data loaded")
            return None, None, None

        # Quality info
        # Resolution: always derived from actual array shape, never from labels
        first_shape = next(iter(channels.values())).shape
        resolution = f"{first_shape[1]}x{first_shape[0]}"

        # Real time spread between channels (None if headers unparseable)
        time_spread_sec = _compute_time_spread_sec(timestamps)

        quality_info = {
            'source': 'synoptic',
            'resolution': resolution,
            'is_good_quality': len(channels) == len(wavelengths),
            'time_spread_sec': time_spread_sec,
            'timestamps': timestamps,
            'warnings': [],
        }

        if time_spread_sec is None:
            # Unknown sync is a quality issue, not a silent pass
            quality_info['warnings'].append("Time spread unknown (T_OBS unparseable)")
            quality_info['is_good_quality'] = False

        if len(channels) < len(wavelengths):
            missing = [wl for wl in wavelengths if wl not in channels]
            quality_info['warnings'].append(f"Missing wavelengths: {missing}")
            quality_info['is_good_quality'] = False

        return channels, iso_timestamp, quality_info

    except Exception as e:
        print(f"    Synoptic load error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
