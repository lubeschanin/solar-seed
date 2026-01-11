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
#
# Semantic palette:
#   🟢 BASELINE       → thermal & structural quiet
#   🟢 ELEVATED-QUIET → structurally active but stable (ΔMI elevated, no destabilization)
#   🟣 POST-EVENT     → non-flaring but reorganizing
#   🟡 RECOVERY       → decaying activity
#   ⚠️ PRE-FLARE      → destabilization detected
#   🔴 ACTIVE         → ongoing energy release

class Phase:
    """Interpretive phase based on multi-indicator rules."""
    # Quiet states (green)
    BASELINE = 'BASELINE'                     # Thermal & structural quiet
    ELEVATED_QUIET = 'ELEVATED-QUIET'         # Structurally active but stable

    # Transitional states (yellow/purple)
    POST_EVENT = 'POST-EVENT'                 # Non-flaring but reorganizing
    RECOVERY = 'RECOVERY'                     # Decaying activity

    # Alert states (orange/red)
    PRE_FLARE = 'PRE-FLARE'                   # Destabilization detected
    ACTIVE = 'ACTIVE'                         # Ongoing energy release (flare)

    # Legacy aliases for backward compatibility
    FLARE = ACTIVE                            # Alias: FLARE → ACTIVE
    ELEVATED = ELEVATED_QUIET                 # Alias: ELEVATED → ELEVATED-QUIET
    POST_FLARE_REORG = POST_EVENT             # Alias: POST-FLARE REORG → POST-EVENT


# =============================================================================
# DIVERGENCE TYPOLOGY
# =============================================================================
# When GOES-only and ΔMI-integrated classifiers disagree, we categorize the
# divergence type for later validation against actual outcomes.
#
# Purpose: Empirically determine which divergences are predictive vs artifacts.

class DivergenceType:
    """Classification of phase divergence events for validation."""

    # ΔMI sees anomaly BEFORE GOES rises → potential early warning
    PRECURSOR = 'PRECURSOR'

    # ΔMI sees anomaly AFTER GOES returns to quiet → structural relaxation
    POST_EVENT = 'POST_EVENT'

    # ΔMI anomaly with no GOES activity within validation window → needs review
    UNCONFIRMED = 'UNCONFIRMED'

    # Validated outcomes (set retrospectively)
    TRUE_POSITIVE = 'TRUE_POSITIVE'   # PRECURSOR followed by flare
    TRUE_NEGATIVE = 'TRUE_NEGATIVE'   # No divergence, no flare
    FALSE_POSITIVE = 'FALSE_POSITIVE' # PRECURSOR not followed by flare
    FALSE_NEGATIVE = 'FALSE_NEGATIVE' # Flare without prior PRECURSOR


def classify_divergence_type(
    phase_goes: str,
    phase_experimental: str,
    goes_trend_rising: bool = False,
    recent_flare_hours: float = None,
) -> str:
    """
    Classify a divergence event for later validation.

    Args:
        phase_goes: Phase from GOES-only classifier
        phase_experimental: Phase from ΔMI-integrated classifier
        goes_trend_rising: Whether GOES flux is trending upward
        recent_flare_hours: Hours since last significant flare (None if unknown)

    Returns:
        DivergenceType classification
    """
    # No divergence
    if phase_goes == phase_experimental:
        return None

    # GOES quiet, ΔMI sees something
    if phase_goes == Phase.BASELINE:
        # If GOES is rising, this could be a precursor
        if goes_trend_rising:
            return DivergenceType.PRECURSOR

        # If recent flare, this is post-event relaxation
        if recent_flare_hours is not None and recent_flare_hours < 24:
            return DivergenceType.POST_EVENT

        # Otherwise, we don't know yet - needs validation
        return DivergenceType.UNCONFIRMED

    # GOES active, ΔMI sees quiet (unusual - GOES leading)
    if phase_goes in [Phase.ACTIVE, Phase.RECOVERY]:
        return DivergenceType.POST_EVENT

    return DivergenceType.UNCONFIRMED


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
        return Phase.ACTIVE, f"M/X-class active ({goes_class})"

    # C-class + rising = flare in progress
    if goes_flux >= 5e-6 and goes_rising:
        return Phase.ACTIVE, f"C-class flare ({goes_class})"

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

    Phase palette:
    - BASELINE: thermal & structural quiet (GOES quiet, |z| < 3)
    - ELEVATED-QUIET: structurally active but stable (|z| > 3, stable trend)
    - POST-EVENT: non-flaring but reorganizing (GOES quiet, |z| > 5)
    - RECOVERY: decaying activity (GOES falling)
    - PRE-FLARE: destabilization detected (negative z + rising GOES)
    - ACTIVE: ongoing energy release (high GOES flux)

    Returns:
        (phase, reason) tuple
    """
    # Extract key metrics
    z_211 = pairs_data.get('193-211', {}).get('residual', 0)
    z_304 = pairs_data.get('193-304', {}).get('residual', 0)
    trend_211 = pairs_data.get('193-211', {}).get('slope_pct_per_hour', 0)
    trend_304 = pairs_data.get('193-304', {}).get('slope_pct_per_hour', 0)

    # Maximum absolute z-score across pairs
    z_values = [abs(pairs_data.get(p, {}).get('residual', 0))
                for p in pairs_data if not p.startswith('_')]
    max_z = max(z_values) if z_values else 0

    # Count negative anomalies (potential destabilization)
    neg_anomalies = sum(1 for p, d in pairs_data.items()
                        if not p.startswith('_') and d.get('residual', 0) < -2)

    # Rule 1: ACTIVE - high GOES activity (ongoing energy release)
    if goes_flux and goes_flux >= 1e-5:
        return Phase.ACTIVE, f"M/X-class active ({goes_class})"
    if goes_flux and goes_flux >= 5e-6 and goes_rising:
        return Phase.ACTIVE, f"C-class flare ({goes_class})"

    # Rule 2: PRE-FLARE - destabilization signature
    if neg_anomalies >= 1 and goes_rising:
        return Phase.PRE_FLARE, f"{neg_anomalies} pair(s) destabilizing + GOES ↑"
    if z_211 < -2 and trend_211 < -3:
        return Phase.PRE_FLARE, f"Coronal decoupling ({z_211:.1f}σ, {trend_211:+.1f}%/h)"

    # Rule 3: POST-EVENT - GOES quiet but ΔMI still elevated (reorganizing)
    # Key hypothesis: we see magnetic restructuring that GOES doesn't
    if goes_flux and goes_flux < 1e-6:  # GOES says quiet (B-class)
        if max_z > 5:
            # Identify trigger pair (which channel drives the anomaly)
            trigger_pair = "193-211" if abs(z_211) >= abs(z_304) else "193-304"
            dominant_trend = trend_211 if abs(z_211) >= abs(z_304) else trend_304

            # Check if recovering (dominant trend is falling)
            if dominant_trend < -3:  # Falling >3%/h = recovering toward baseline
                return Phase.POST_EVENT, f"Relaxing ({trigger_pair} at {max_z:.1f}σ, {dominant_trend:+.1f}%/h)"
            else:
                return Phase.POST_EVENT, f"Reorganizing ({trigger_pair} at {max_z:.1f}σ)"
        if z_304 > 4 and trend_304 > 0:
            return Phase.POST_EVENT, f"Chromosphere restructuring (193-304 at +{z_304:.1f}σ)"

    # Rule 4: ELEVATED-QUIET - structurally active but stable
    # ΔMI elevated but not destabilizing (no negative trend, no GOES rise)
    if max_z > 3:
        if abs(trend_211) < 3 and abs(trend_304) < 3:  # Stable trends
            return Phase.ELEVATED_QUIET, f"Structurally active, stable ({max_z:.1f}σ)"
        else:
            # Elevated with significant trend - still elevated-quiet but note trend
            direction = "↑" if trend_211 > 0 else "↓"
            return Phase.ELEVATED_QUIET, f"Active, {direction} trend ({max_z:.1f}σ)"

    # Rule 5: RECOVERY - decaying from elevated
    if goes_flux and 1e-6 < goes_flux < 5e-6 and not goes_rising:
        return Phase.RECOVERY, f"Post-flare decay ({goes_class})"

    # Rule 6: BASELINE - thermal & structural quiet
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
