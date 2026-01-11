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
    ELEVATED = 'ELEVATED'                     # ΔMI anomaly (experimental only)
    POST_EVENT = 'POST-EVENT'                 # GOES quiet but ΔMI elevated (experimental)


# =============================================================================
# PHASE CLASSIFICATION: GOES-ONLY (Current Standard)
# =============================================================================

def classify_phase_goes_only(
    goes_flux: float = None,
    goes_rising: bool = None,
    goes_class: str = None,
) -> tuple[str, str]:
    """
    Classify phase using ONLY GOES X-ray data (traditional approach).

    This is the current operational standard - simple flux thresholds.

    Returns:
        (phase, reason) tuple
    """
    if goes_flux is None:
        return Phase.BASELINE, "No GOES data"

    # M/X-class = definitely active
    if goes_flux >= 1e-5:
        return Phase.FLARE, f"M/X-class active ({goes_class})"

    # C-class + rising = flare in progress
    if goes_flux >= 5e-6 and goes_rising:
        return Phase.FLARE, f"C-class flare ({goes_class})"

    # C-class falling = recovery
    if goes_flux >= 1e-6 and not goes_rising:
        return Phase.RECOVERY, f"Post-flare decay ({goes_class})"

    # B-class or below = quiet
    return Phase.BASELINE, f"Quiet ({goes_class or 'B-class'})"


# =============================================================================
# PHASE CLASSIFICATION: EXPERIMENTAL (ΔMI-integrated)
# =============================================================================

def classify_phase_experimental(
    pairs_data: dict,
    goes_flux: float = None,
    goes_rising: bool = None,
    goes_class: str = None,
) -> tuple[str, str]:
    """
    Experimental phase classification integrating ΔMI coupling residuals.

    This approach hypothesizes that:
    - Coupling breaks precede flares by 0.5-2h
    - Post-event reorganization is visible in ΔMI even when GOES is quiet
    - Coupling anomalies may be early warning indicators

    Returns:
        (phase, reason) tuple
    """
    # Extract key metrics
    z_211 = pairs_data.get('193-211', {}).get('residual', 0)
    z_304 = pairs_data.get('193-304', {}).get('residual', 0)
    trend_211 = pairs_data.get('193-211', {}).get('slope_pct_per_hour', 0)
    trend_304 = pairs_data.get('193-304', {}).get('slope_pct_per_hour', 0)

    # Maximum absolute z-score across pairs
    max_z = max(abs(pairs_data.get(p, {}).get('residual', 0))
                for p in pairs_data if not p.startswith('_'))

    # Count negative anomalies (potential destabilization)
    neg_anomalies = sum(1 for p, d in pairs_data.items()
                        if not p.startswith('_') and d.get('residual', 0) < -2)

    # Rule 1: FLARE - high GOES activity (same as GOES-only)
    if goes_flux and goes_flux >= 1e-5:
        return Phase.FLARE, f"M/X-class active ({goes_class})"
    if goes_flux and goes_flux >= 5e-6 and goes_rising:
        return Phase.FLARE, f"C-class flare ({goes_class})"

    # Rule 2: PRE-FLARE - destabilization signature (ΔMI specific)
    if neg_anomalies >= 1 and goes_rising:
        return Phase.PRE_FLARE, f"{neg_anomalies} pair(s) destabilizing + GOES ↑"
    if z_211 < -2 and trend_211 < -3:
        return Phase.PRE_FLARE, f"Coronal decoupling ({z_211:.1f}σ, {trend_211:+.1f}%/h)"

    # Rule 3: POST-EVENT - GOES quiet but ΔMI still anomalous
    # This is a KEY experimental hypothesis: we see something GOES doesn't
    if goes_flux and goes_flux < 1e-6:  # GOES says quiet (B-class)
        if max_z > 5:
            return Phase.POST_EVENT, f"GOES quiet, coupling {max_z:.1f}σ"
        if z_304 > 4 and trend_304 > 0:
            return Phase.POST_FLARE_REORG, f"Chromosphere restructuring (+{z_304:.1f}σ)"

    # Rule 4: ELEVATED - significant ΔMI anomaly even without GOES confirmation
    if max_z > 4:
        direction = "+" if z_211 > 0 else ""
        return Phase.ELEVATED, f"max(|r|) = {max_z:.1f}σ ({direction}{z_211:.1f}σ @ 193-211)"

    # Rule 5: RECOVERY
    if goes_flux and 1e-6 < goes_flux < 5e-6 and not goes_rising:
        return Phase.RECOVERY, f"Post-flare decay ({goes_class})"

    # Rule 6: BASELINE
    return Phase.BASELINE, "Quiet conditions"


def classify_phase_parallel(
    pairs_data: dict,
    goes_flux: float = None,
    goes_rising: bool = None,
    goes_class: str = None,
) -> dict:
    """
    Run both phase classifiers in parallel and report divergence.

    Returns dict with:
        - current: (phase, reason) from GOES-only
        - experimental: (phase, reason) from ΔMI-integrated
        - is_divergent: bool
        - divergence_note: str explaining the divergence
    """
    current = classify_phase_goes_only(goes_flux, goes_rising, goes_class)
    experimental = classify_phase_experimental(pairs_data, goes_flux, goes_rising, goes_class)

    is_divergent = current[0] != experimental[0]

    if is_divergent:
        divergence_note = f"GOES says {current[0]}, ΔMI says {experimental[0]}"
    else:
        divergence_note = "Both classifiers agree"

    return {
        'current': current,
        'experimental': experimental,
        'is_divergent': is_divergent,
        'divergence_note': divergence_note,
    }


# Legacy alias for backward compatibility
def classify_phase(
    pairs_data: dict,
    goes_flux: float = None,
    goes_rising: bool = None,
    goes_class: str = None,
) -> tuple[str, str]:
    """Legacy function - returns experimental classification."""
    return classify_phase_experimental(pairs_data, goes_flux, goes_rising, goes_class)
