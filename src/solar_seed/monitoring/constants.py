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


# =============================================================================
# ANOMALY LEVEL (Statistical)
# =============================================================================
# Pure statistical classification based on |z| = |residual in sigma|
# This is independent of physical interpretation.

class AnomalyLevel:
    """Statistical anomaly level based on |z-score|."""
    NORMAL = 'NORMAL'       # |z| < 2
    ELEVATED = 'ELEVATED'   # 2 <= |z| < 4
    STRONG = 'STRONG'       # 4 <= |z| < 7
    EXTREME = 'EXTREME'     # |z| >= 7


def get_anomaly_level(z_score: float) -> str:
    """
    Classify anomaly level based on absolute z-score.

    Args:
        z_score: Residual in standard deviations (can be positive or negative)

    Returns:
        AnomalyLevel constant
    """
    z = abs(z_score)
    if z < 2:
        return AnomalyLevel.NORMAL
    elif z < 4:
        return AnomalyLevel.ELEVATED
    elif z < 7:
        return AnomalyLevel.STRONG
    else:
        return AnomalyLevel.EXTREME


# =============================================================================
# PHASE (Interpretive)
# =============================================================================
# Rule-based phase classification combining multiple indicators.
# This provides physical context for the statistical anomaly.

class Phase:
    """Interpretive phase based on multi-indicator rules."""
    BASELINE = 'BASELINE'                     # Quiet conditions
    PRE_FLARE = 'PRE-FLARE'                   # Destabilization (neg z + rising GOES)
    FLARE = 'FLARE'                           # Active flare (high GOES flux)
    RECOVERY = 'RECOVERY'                     # Post-peak decay
    POST_FLARE_REORG = 'POST-FLARE REORG'     # Chromosphere coupling high + corona stable


def classify_phase(
    pairs_data: dict,
    goes_flux: float = None,
    goes_rising: bool = None,
    goes_class: str = None,
) -> tuple[str, str]:
    """
    Classify current phase based on multiple indicators.

    Rules:
    - FLARE: GOES >= M-class (1e-5) or C5+ during rise
    - PRE-FLARE: Multiple pairs with negative z AND GOES rising
    - POST-FLARE REORG: 193-304 elevated/strong positive z AND 193-211 stable
    - RECOVERY: GOES falling after elevated activity
    - BASELINE: Otherwise

    Args:
        pairs_data: Dict of pair -> {residual, delta_mi, trend, ...}
        goes_flux: Current GOES X-ray flux (W/m²)
        goes_rising: Whether GOES is trending upward
        goes_class: Flare class string (e.g., "M2.5", "C3.0")

    Returns:
        (phase, reason) tuple
    """
    # Extract key metrics
    z_211 = pairs_data.get('193-211', {}).get('residual', 0)
    z_304 = pairs_data.get('193-304', {}).get('residual', 0)
    trend_211 = pairs_data.get('193-211', {}).get('slope_pct_per_hour', 0)
    trend_304 = pairs_data.get('193-304', {}).get('slope_pct_per_hour', 0)

    # Count negative anomalies (potential destabilization)
    neg_anomalies = sum(1 for p, d in pairs_data.items()
                        if not p.startswith('_') and d.get('residual', 0) < -2)

    # Rule 1: FLARE - high GOES activity
    if goes_flux and goes_flux >= 1e-5:  # M-class
        return Phase.FLARE, f"M/X-class active ({goes_class})"
    if goes_flux and goes_flux >= 5e-6 and goes_rising:  # C5+ and rising
        return Phase.FLARE, f"C-class flare in progress ({goes_class})"

    # Rule 2: PRE-FLARE - destabilization signature
    if neg_anomalies >= 1 and goes_rising:
        return Phase.PRE_FLARE, f"{neg_anomalies} pair(s) destabilizing + GOES rising"
    if z_211 < -2 and trend_211 < -3:  # Strong negative trend in coronal core
        return Phase.PRE_FLARE, f"Coronal decoupling (z={z_211:.1f}σ, {trend_211:+.1f}%/h)"

    # Rule 3: POST-FLARE REORGANIZATION
    # Chromosphere-corona link strengthening while coronal core stable
    if z_304 > 4 and abs(z_211) < 3 and trend_304 > 0:
        return Phase.POST_FLARE_REORG, f"Chromosphere coupling elevated (+{z_304:.1f}σ)"

    # Rule 4: RECOVERY - decaying from elevated
    if goes_flux and 1e-6 < goes_flux < 5e-6 and not goes_rising:
        if goes_class and goes_class[0] in ['B', 'C']:
            return Phase.RECOVERY, f"Post-flare decay ({goes_class})"

    # Rule 5: BASELINE - normal conditions
    return Phase.BASELINE, "Quiet conditions"
