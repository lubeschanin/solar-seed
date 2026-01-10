"""
Solar Seed - Searching for Information Patterns in Sunlight
===========================================================

One hypothesis. One test. One answer.

H1: Certain AIA wavelength pairs show higher mutual information
    than explainable by independent thermal processes.

4free - The Search for the Seed.
"""

__version__ = "0.1.0"

from solar_seed.mutual_info import mutual_information, normalized_mutual_information
from solar_seed.null_model import compute_null_distribution, compute_z_score, compute_p_value

__all__ = [
    "mutual_information",
    "normalized_mutual_information", 
    "compute_null_distribution",
    "compute_z_score",
    "compute_p_value",
]
