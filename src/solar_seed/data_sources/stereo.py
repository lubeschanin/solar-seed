"""
STEREO-A EUVI Data Loader
=========================

Load data from STEREO-A's EUVI instrument.
STEREO-A is ~51° ahead of Earth, providing ~3.9 days advance warning.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional


# STEREO-A position info (updated periodically)
STEREO_A_INFO = {
    'separation_deg': 51.0,  # Degrees ahead of Earth
    'light_travel_min': 7.0,  # Light travel time to Earth
    'advance_warning_days': 3.9,  # How many days ahead it sees (51° / 13.2°/day)
}

# EUVI to AIA wavelength mapping
EUVI_TO_AIA = {
    171: 171,  # Fe IX - same
    195: 193,  # Fe XII - similar to AIA 193
    284: 211,  # Fe XV - similar to AIA 211
    304: 304,  # He II - same
}


def load_stereo_a_latest(
    wavelengths: Optional[list[int]] = None,
    max_age_minutes: int = 120
) -> tuple[dict, str, dict] | tuple[None, None, None]:
    """
    Load most recent STEREO-A EUVI data.

    STEREO-A is ~51° ahead of Earth, providing ~3.9 days advance warning.

    Args:
        wavelengths: EUVI wavelengths to load [171, 195, 284, 304]
        max_age_minutes: How far back to search (STEREO data may be delayed)

    Returns:
        (channels_dict, timestamp, metadata) or (None, None, None)
    """
    if wavelengths is None:
        wavelengths = [195, 284, 304]  # Similar to AIA 193, 211, 304

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

        print(f"\n  STEREO-A EUVI ({STEREO_A_INFO['separation_deg']:.0f}° ahead, ~{STEREO_A_INFO['advance_warning_days']:.1f} days warning)")

        for wl in wavelengths:
            print(f"    Searching EUVI {wl} Å (last {max_age_minutes} min)...")

            # Search STEREO-A EUVI data
            result = Fido.search(
                a.Time(start, now),
                a.Source('STEREO_A'),
                a.Instrument('EUVI'),
                a.Wavelength(wl * u.Angstrom)
            )

            if len(result) > 0 and len(result[0]) > 0:
                n_results = len(result[0])
                latest_idx = n_results - 1

                try:
                    result_time = result[0][latest_idx]['Start Time']
                    print(f"    Found {n_results} images, using latest: {result_time}")
                except (KeyError, IndexError):
                    print(f"    Found {n_results} images, using latest")

                with tempfile.TemporaryDirectory() as tmpdir:
                    files = Fido.fetch(result[0, latest_idx], path=tmpdir, progress=False)
                    if files:
                        smap = Map(files[0])
                        channels[wl] = smap.data

                        if actual_timestamp is None:
                            actual_timestamp = smap.date.isot
                            print(f"    Actual image time: {actual_timestamp}")

                        os.remove(files[0])
            else:
                print(f"    No EUVI {wl} Å images found")

        metadata = {
            'source': 'STEREO-A',
            'instrument': 'EUVI',
            'separation_deg': STEREO_A_INFO['separation_deg'],
            'advance_warning_days': STEREO_A_INFO['advance_warning_days'],
        }

        return (channels, actual_timestamp, metadata) if channels else (None, None, None)

    except Exception as e:
        print(f"    STEREO-A load error: {e}")
        return None, None, None
