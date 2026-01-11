"""
Monitoring Constants
====================

Physical thresholds and constants for anomaly detection.
"""

# Data Quality Gate
# =================
# Physical minimum thresholds to detect data errors BEFORE break detection.
# ΔMI = 0.0 or very low values indicate data pipeline failures, not real breaks.

MIN_MI_THRESHOLD = 0.05  # bits - below this is DATA_ERROR (not a real measurement)
MIN_ROI_STD = 0.5        # DN - minimum std dev in residual ROI (after geometry subtraction)
