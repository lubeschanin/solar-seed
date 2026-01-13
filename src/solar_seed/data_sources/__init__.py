"""
Solar Seed Data Sources
=======================

Data loading functions for various solar observation sources.

- AIA: SDO/AIA EUV imagery (synoptic and full-resolution)
- SDAC: NASA SDAC mirror for 4k backfill (~3 day latency)
- STEREO: STEREO-A EUVI data (51° ahead, ~3.9 days warning)

Resolution Strategy:
- Real-time: Synoptic 1k (fast, but 304Å MI inflated by +350%)
- Backfill: SDAC 4k after 3 days (accurate MI values)
"""

from .synoptic import load_aia_synoptic, SYNOPTIC_BASE_URL
from .aia import load_aia_latest, load_aia_direct
from .sdac import load_aia_sdac, check_sdac_availability
from .stereo import load_stereo_a_latest, STEREO_A_INFO, EUVI_TO_AIA

__all__ = [
    # Synoptic (real-time monitoring, 1k)
    'load_aia_synoptic',
    'SYNOPTIC_BASE_URL',
    # SDAC (backfill, 4k)
    'load_aia_sdac',
    'check_sdac_availability',
    # Full-resolution AIA via VSO (fallback)
    'load_aia_latest',
    'load_aia_direct',
    # STEREO-A
    'load_stereo_a_latest',
    'STEREO_A_INFO',
    'EUVI_TO_AIA',
]
