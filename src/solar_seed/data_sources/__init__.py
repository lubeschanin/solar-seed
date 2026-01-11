"""
Solar Seed Data Sources
=======================

Data loading functions for various solar observation sources.

- AIA: SDO/AIA EUV imagery (synoptic and full-resolution)
- STEREO: STEREO-A EUVI data (51° ahead, ~3.9 days warning)
"""

from .synoptic import load_aia_synoptic, SYNOPTIC_BASE_URL
from .aia import load_aia_latest, load_aia_direct
from .stereo import load_stereo_a_latest, STEREO_A_INFO, EUVI_TO_AIA

__all__ = [
    # Synoptic (primary for monitoring)
    'load_aia_synoptic',
    'SYNOPTIC_BASE_URL',
    # Full-resolution AIA (fallback)
    'load_aia_latest',
    'load_aia_direct',
    # STEREO-A
    'load_stereo_a_latest',
    'STEREO_A_INFO',
    'EUVI_TO_AIA',
]
