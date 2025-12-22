"""
Solar Seed - Suche nach Informationsmustern in Sonnenlicht
==========================================================

Eine Hypothese. Ein Test. Eine Antwort.

H1: Bestimmte AIA-Wellenlängenpaare zeigen höhere Mutual Information
    als durch unabhängige thermische Prozesse erklärbar.

4free - Die Suche nach dem Seed.
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
