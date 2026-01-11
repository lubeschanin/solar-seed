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
SYNOPTIC_BASE_URL = "https://jsoc1.stanford.edu/data/aia/synoptic"


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

                os.remove(tmp_path)

            except URLError as e:
                print(f"      ✗ {wl} Å: Network error - {e}")
            except Exception as e:
                print(f"      ✗ {wl} Å: Load error - {e}")

        if not channels:
            print("    No synoptic data loaded")
            return None, None, None

        # Quality info
        quality_info = {
            'source': 'synoptic',
            'resolution': '1024x1024',
            'is_good_quality': len(channels) == len(wavelengths),
            'time_spread_sec': 0,  # All same timestamp in synoptic
            'timestamps': timestamps,
            'warnings': [],
        }

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
