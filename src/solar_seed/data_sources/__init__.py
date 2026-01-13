"""
Solar Seed Data Sources
=======================

Data loading functions for various solar observation sources.

- AIA: SDO/AIA EUV imagery (synoptic and full-resolution)
- JSOC: True 4k data for backfill (only source with 4096² resolution)
- STEREO: STEREO-A EUVI data (51° ahead, ~3.9 days warning)

Resolution Strategy:
- Real-time: Synoptic 1k (fast, but 304Å MI inflated by +350%)
- Backfill: JSOC 4k when available (accurate MI values)

Note: SDAC and other mirrors only provide 1k data despite claiming "FULLDISK".
Only JSOC provides true 4096x4096 resolution (~65MB per file vs ~4MB for 1k).
"""

from .synoptic import load_aia_synoptic, SYNOPTIC_BASE_URL
from .aia import load_aia_latest, load_aia_direct
from .jsoc import load_aia_jsoc, check_jsoc_4k_availability, get_jsoc_latest_date
from .stereo import load_stereo_a_latest, STEREO_A_INFO, EUVI_TO_AIA

__all__ = [
    # Synoptic (real-time monitoring, 1k)
    'load_aia_synoptic',
    'SYNOPTIC_BASE_URL',
    # JSOC (backfill, true 4k)
    'load_aia_jsoc',
    'check_jsoc_4k_availability',
    'get_jsoc_latest_date',
    # Full-resolution AIA via VSO (fallback)
    'load_aia_latest',
    'load_aia_direct',
    # STEREO-A
    'load_stereo_a_latest',
    'STEREO_A_INFO',
    'EUVI_TO_AIA',
]
