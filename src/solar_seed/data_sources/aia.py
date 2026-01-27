"""
SDO/AIA Full-Resolution Data Loaders
====================================

Load full-resolution (4096x4096) AIA data via VSO/JSOC.
Note: JSOC has rate limits (49GB/request post-Nov 2024).

For real-time monitoring, prefer synoptic.load_aia_synoptic() which
provides 1024x1024 resolution with direct access (no queue).
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError


def load_aia_latest(
    wavelengths: list[int],
    max_age_minutes: int = 60,
    search_timeout: int = 60,
    download_timeout: int = 240
) -> tuple[dict, str, dict] | tuple[None, None, None]:
    """
    Load the most recent available AIA data with quality metadata.

    Searches for available images in the last max_age_minutes and picks the newest.
    Returns (channels_dict, actual_timestamp, quality_info) or (None, None, None) if not found.

    Quality info includes:
    - timestamps: dict of wavelength -> timestamp
    - time_spread_sec: max time difference between channels
    - quality_flags: dict of wavelength -> QUALITY header value
    - exposure_times: dict of wavelength -> EXPTIME
    - warnings: list of quality warnings

    Args:
        wavelengths: List of AIA wavelengths to load (Angstroms)
        max_age_minutes: Search window (minutes before now)
        search_timeout: Timeout in seconds for VSO search (default: 60s)
        download_timeout: Timeout in seconds for FITS download (default: 240s = 4min)
    """
    try:
        from sunpy.net import Fido, attrs as a
        import astropy.units as u
        from sunpy.map import Map
        import tempfile
        import os

        now = datetime.now(timezone.utc)
        start = now - timedelta(minutes=max_age_minutes)

        channels = {}
        actual_timestamp = None

        # Quality tracking
        timestamps = {}
        quality_flags = {}
        exposure_times = {}
        warnings = []

        def _load_single_channel(wl):
            """Load a single wavelength channel with VSO (called with timeout)."""
            result = Fido.search(
                a.Time(start, now),
                a.Instrument('AIA'),
                a.Wavelength(wl * u.Angstrom)
            )
            return result

        for wl in wavelengths:
            print(f"    Searching {wl} Å (last {max_age_minutes} min)...")

            # Search with timeout
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_load_single_channel, wl)
                    result = future.result(timeout=search_timeout)
            except FutureTimeoutError:
                print(f"    ⚠ Timeout searching {wl} Å (>{search_timeout}s)")
                warnings.append(f"{wl}Å: VSO search timeout (>{search_timeout}s)")
                continue
            except Exception as e:
                print(f"    ⚠ Error searching {wl} Å: {e}")
                warnings.append(f"{wl}Å: VSO search error: {e}")
                continue

            if len(result) > 0 and len(result[0]) > 0:
                # Get the LAST (most recent) result
                n_results = len(result[0])
                latest_idx = n_results - 1

                # Extract timestamp from result table if available
                try:
                    result_time = result[0][latest_idx]['Start Time']
                    print(f"    Found {n_results} images, using latest: {result_time}")
                except:
                    print(f"    Found {n_results} images, using latest")

                def _fetch_file():
                    """Fetch FITS file (called with timeout)."""
                    with tempfile.TemporaryDirectory() as tmpdir:
                        files = Fido.fetch(result[0, latest_idx], path=tmpdir, progress=False)
                        if files:
                            smap = Map(files[0])
                            data = smap.data.copy()
                            meta = smap.meta.copy()
                            date = smap.date
                            os.remove(files[0])
                            return data, meta, date
                    return None, None, None

                # Fetch with timeout
                print(f"    Downloading {wl} Å (timeout={download_timeout}s)...")
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(_fetch_file)
                        fetch_result = future.result(timeout=download_timeout)
                        data, meta, date = fetch_result
                except FutureTimeoutError:
                    print(f"    ⚠ Timeout downloading {wl} Å (>{download_timeout}s)")
                    warnings.append(f"{wl}Å: VSO download timeout (>{download_timeout}s)")
                    continue
                except Exception as e:
                    print(f"    ⚠ Error downloading {wl} Å: {e}")
                    warnings.append(f"{wl}Å: VSO download error: {e}")
                    continue

                if data is not None:
                        channels[wl] = data

                        # Get actual timestamp from the FITS header
                        timestamps[wl] = date.datetime
                        if actual_timestamp is None:
                            actual_timestamp = date.isot
                            print(f"    Actual image time: {actual_timestamp}")

                        # Extract quality metadata from FITS header
                        quality_flags[wl] = meta.get('QUALITY', 0)
                        exposure_times[wl] = meta.get('EXPTIME', 0)

                        # Check quality flag (only warn on critical bits)
                        # Bit 30 (2^30 = 1073741824) = AEC flag (normal operation)
                        # Critical bits: 0-15 indicate actual data issues
                        critical_bits = quality_flags[wl] & 0x0000FFFF  # Lower 16 bits
                        if critical_bits != 0:
                            warnings.append(f"{wl}Å: QUALITY={quality_flags[wl]} (critical bits set)")
                            print(f"    ⚠ Quality flag: {quality_flags[wl]} (critical)")

                        # Check exposure time (typical: 1-2s for most channels)
                        expected_exp = {171: 2.0, 193: 2.0, 211: 2.0, 304: 2.0, 335: 2.9, 94: 2.9, 131: 2.9}
                        exp_expected = expected_exp.get(wl, 2.0)
                        if exposure_times[wl] < exp_expected * 0.5:
                            warnings.append(f"{wl}Å: Short exposure {exposure_times[wl]:.2f}s (expected ~{exp_expected}s)")
                            print(f"    ⚠ Short exposure: {exposure_times[wl]:.2f}s")
            else:
                print(f"    No {wl} Å images found in last {max_age_minutes} min")

        if not channels:
            print(f"    ✗ No channels loaded successfully")
            return None, None, None

        # Report partial success if some channels failed
        failed_channels = set(wavelengths) - set(channels.keys())
        if failed_channels:
            print(f"    ⚠ Loaded {len(channels)}/{len(wavelengths)} channels (failed: {sorted(failed_channels)})")
            warnings.append(f"Partial load: {len(channels)}/{len(wavelengths)} channels (failed: {sorted(failed_channels)})")
        else:
            print(f"    ✓ All {len(channels)} channels loaded successfully")

        # Check time synchronization between channels
        time_spread_sec = 0
        if len(timestamps) >= 2:
            ts_list = list(timestamps.values())
            time_spread_sec = (max(ts_list) - min(ts_list)).total_seconds()
            if time_spread_sec > 60:
                warnings.append(f"ASYNC: Channels spread over {time_spread_sec:.0f}s (>60s)")
                print(f"    ⚠ Time spread: {time_spread_sec:.0f}s between channels")
            elif time_spread_sec > 30:
                print(f"    Time spread: {time_spread_sec:.0f}s (acceptable)")

        quality_info = {
            'timestamps': {wl: ts.isoformat() for wl, ts in timestamps.items()},
            'time_spread_sec': time_spread_sec,
            'quality_flags': quality_flags,
            'exposure_times': exposure_times,
            'warnings': warnings,
            'is_good_quality': len(warnings) == 0,
        }

        return channels, actual_timestamp, quality_info

    except Exception as e:
        print(f"    VSO load error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def load_aia_direct(timestamp: str, wavelengths: list[int], timeout_per_channel: int = 240) -> Optional[dict]:
    """Load AIA data for a specific timestamp (legacy, fallback).

    Args:
        timestamp: ISO format timestamp
        wavelengths: List of AIA wavelengths to load
        timeout_per_channel: Timeout in seconds for each channel (default: 240s = 4min)
    """
    try:
        from sunpy.net import Fido, attrs as a
        import astropy.units as u
        from sunpy.map import Map
        import tempfile
        import os

        dt = datetime.fromisoformat(timestamp)
        start = dt - timedelta(minutes=3)
        end = dt + timedelta(minutes=3)

        def _load_channel(wl):
            """Load a single channel with timeout."""
            result = Fido.search(
                a.Time(start, end),
                a.Instrument('AIA'),
                a.Wavelength(wl * u.Angstrom)
            )
            if len(result) > 0 and len(result[0]) > 0:
                with tempfile.TemporaryDirectory() as tmpdir:
                    files = Fido.fetch(result[0, 0], path=tmpdir, progress=False)
                    if files:
                        smap = Map(files[0])
                        data = smap.data.copy()
                        os.remove(files[0])
                        return data
            return None

        channels = {}
        for wl in wavelengths:
            print(f"    Fetching {wl} Å...")
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_load_channel, wl)
                    data = future.result(timeout=timeout_per_channel)
                    if data is not None:
                        channels[wl] = data
                        print(f"    ✓ {wl} Å loaded successfully")
            except FutureTimeoutError:
                print(f"    ⚠ Timeout loading {wl} Å (>{timeout_per_channel}s)")
            except Exception as e:
                print(f"    ⚠ Error loading {wl} Å: {e}")

        return channels if channels else None

    except Exception as e:
        print(f"    VSO load error: {e}")
        return None
