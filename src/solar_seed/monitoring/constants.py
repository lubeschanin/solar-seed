"""
Monitoring Constants
====================

Physical thresholds and constants for anomaly detection.
"""

# Data Quality Gate
# =================
# Detects data errors BEFORE break detection. The gate must catch pipeline
# failures WITHOUT censoring the signal it exists to protect.
#
# History: this used to be a baseline-relative threshold, max(0.02, 0.3 *
# baseline_mean). The documented flare signature is a -25% to -47% collapse,
# and a -54% collapse was logged on 24 Apr 2026 - so a deeper collapse was
# classified as a data error AND excluded from the rolling MAD window, keeping
# the median high while the real event went unrecorded. Cutting the tail off
# the distribution you are trying to measure is not a quality gate.
#
# The gate is now a pure noise floor. ΔMI = MI_original - MI_sector_shuffled is
# a difference of two estimates of the same quantity under the null, so pure
# permutation noise scatters it around zero (the database holds values down to
# -0.062). At or below the floor there is no measurable coupling to speak of,
# which is indistinguishable from a pipeline failure. Everything above it is a
# real measurement, however low.
#
# Image-level failures (blank/constant frames) are caught separately and more
# reliably by validate_roi_variance().
MIN_MI_THRESHOLD = 0.01  # bits - noise floor; at or below this there is no measurement

# A reading this far below its baseline is flagged (is_extreme_low) but stays
# VALID: deep collapses are the signal, not an error.
EXTREME_LOW_BASELINE_FRACTION = 0.3

MIN_ROI_STD = 0.5        # DN - minimum std dev in residual ROI (after geometry subtraction)


# Artifact Test A: channel time alignment
# =======================================
# The reviewer-proof artifact test requires the channels of a pair to be
# observed within this many seconds of each other. It is a published pass
# criterion, so the number does not move without the paper moving with it.
#
# This is deliberately NOT resolution-dependent. The stored spread is bimodal:
# ~5 s when every channel is served from the same synoptic slot, and discrete
# multiples of ~180 s when one channel's slot file is missing and the loader
# falls back to a neighbouring slot. That is a real desync of the observation
# times, not a property of 1k vs 4k - the 4k rows only appear to share the 1k
# distribution because backfill used to leave sync_delta_s untouched.
#
# Failing this test marks the measurement, it does not delete it: the reason is
# recorded in veto_reason so an analysis can decide for itself whether a 3-minute
# channel offset matters for the question it is asking.
SYNC_SPREAD_MAX_SEC = 60.0

# Artifact Test C: 2x2 binning sensitivity (see validation/detection)
ROBUSTNESS_MAX_CHANGE_PCT = 20.0


# Status Thresholds (in sigma)
# ============================
# Status is decided on z = (ΔMI - baseline_mean) / baseline_sigma, NOT on a
# percentage deviation. A fixed percentage means something different for every
# pair, because the relative spread differs by a factor of ~3:
#
#   pair            sigma/mu    -25% is...
#   193-211 @1k       0.17       -1.44 sigma
#   193-304 @4k       0.48       -0.52 sigma
#
# So the old -25% ALERT fired at half a sigma for 193-304 - which is why 33%
# of that pair's readings alarmed while 193-211 alarmed at a third of the rate
# for the same nominal threshold. In sigma the criterion means the same thing
# everywhere.
Z_ELEVATED = -1.5
Z_WARNING = -2.0
Z_ALERT = -3.0

# Sudden drop, also in sigma: (recent_median - current) / baseline_sigma.
# Positive values mean "this far below the recent level".
#
# Calibrated against the stored history (n ~ 20k per pair at 1k). In sigma the
# rates finally agree across pairs, which the percentage version never did:
#
#   threshold   193-211 1k   193-304 1k   193-211 4k   193-304 4k
#     1.25          5.9%         5.3%         7.9%         6.2%
#     2.50          1.5%         0.5%         3.1%         0.5%
#
# MODERATE sits at 1.25 rather than 1.5 so the documented M3 precursor still
# registers: its window median was 0.917 against a reading of 0.714, a drop of
# 0.203 bits = 1.48 sigma, which a 1.5 threshold would have missed by a hair.
Z_SUDDEN_DROP_MODERATE = 1.25
Z_SUDDEN_DROP_SEVERE = 2.5


def classify_status(residual: float | None) -> str:
    """
    Map a z-score to a coupling status.

    Single definition shared by the live monitor, the batch extraction and the
    backfill, so the three cannot drift apart.

    Note on reachable range: ΔMI is floored at MIN_MI_THRESHOLD, so the deepest
    z a pair can express is (MIN_MI_THRESHOLD - mean) / sigma. For a weakly
    coupled pair that bound can sit above the ALERT threshold - 193-304 at 4k
    bottoms out near -1.9 sigma - and no ALERT is possible for it. That is a
    statement about the pair's dynamic range, not a threshold to be lowered
    until every pair can alarm.
    """
    if residual is None:
        return 'NORMAL'
    if residual < Z_ALERT:
        return 'ALERT'
    if residual < Z_WARNING:
        return 'WARNING'
    if residual < Z_ELEVATED:
        return 'ELEVATED'
    return 'NORMAL'


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

    # ΔMI anomaly with no GOES activity within validation window
    # Renamed from UNCONFIRMED: these are real structural events, not "unconfirmed"
    STRUCTURAL_EVENT = 'STRUCTURAL_EVENT'
    UNCONFIRMED = STRUCTURAL_EVENT  # Legacy alias

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
        return DivergenceType.STRUCTURAL_EVENT

    # GOES active, ΔMI sees quiet (unusual - GOES leading)
    if phase_goes in [Phase.ACTIVE, Phase.RECOVERY]:
        return DivergenceType.POST_EVENT

    return DivergenceType.STRUCTURAL_EVENT


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
# PHASE CLASSIFICATION: ΔMI-ONLY (Pure Coupling)
# =============================================================================

class MIPhase:
    """Pure ΔMI phase states - independent of GOES."""
    BASELINE = 'BASELINE'           # Normal coupling
    ELEVATED = 'ELEVATED'           # Above-normal coupling (structural activity)
    ANOMALY = 'ANOMALY'             # Coupling break detected (single reading)
    DESTABILIZING = 'DESTABILIZING' # Sustained decoupling trend (pre-flare candidate)
    DECOUPLED = 'DECOUPLED'         # Severe decoupling (strong break)


def classify_phase_mi_only(pairs_data: dict) -> tuple[str, str]:
    """
    Classify phase using ONLY ΔMI coupling data (no GOES input).

    This is the pure early-warning classifier that can detect precursors
    BEFORE GOES flux rises. No contamination from lagging indicators.

    Phase palette:
    - BASELINE: Normal coupling (|z| < 2)
    - ELEVATED: Structural activity (z > 2, positive)
    - ANOMALY: Coupling break detected (z < -2)
    - DESTABILIZING: Sustained decoupling trend (z < -2 AND declining)
    - DECOUPLED: Severe coupling break (z < -3)

    Returns:
        (phase, reason) tuple
    """
    # Extract key metrics from primary pair (193-211 = coronal coupling)
    z_211 = pairs_data.get('193-211', {}).get('residual', 0)
    z_304 = pairs_data.get('193-304', {}).get('residual', 0)
    trend_211 = pairs_data.get('193-211', {}).get('slope_pct_per_hour', 0)
    trend_304 = pairs_data.get('193-304', {}).get('slope_pct_per_hour', 0)

    # Count pairs with significant negative anomalies
    neg_pairs = []
    for pair, data in pairs_data.items():
        if pair.startswith('_'):
            continue
        z = data.get('residual', 0)
        if z < -2:
            neg_pairs.append((pair, z))

    # Maximum positive and negative z-scores
    z_values = [pairs_data.get(p, {}).get('residual', 0)
                for p in pairs_data if not p.startswith('_')]
    max_z = max(z_values) if z_values else 0
    min_z = min(z_values) if z_values else 0

    # Rule 1: DECOUPLED - severe coupling break (strongest signal)
    if min_z < -3:
        worst_pair = min(neg_pairs, key=lambda x: x[1]) if neg_pairs else ('193-211', min_z)
        return MIPhase.DECOUPLED, f"Severe break: {worst_pair[0]} at {worst_pair[1]:.1f}σ"

    # Rule 2: DESTABILIZING - coupling break with declining trend
    if z_211 < -2 and trend_211 < -3:
        return MIPhase.DESTABILIZING, f"Coronal decoupling: {z_211:.1f}σ, {trend_211:+.1f}%/h"
    if z_304 < -2 and trend_304 < -3:
        return MIPhase.DESTABILIZING, f"Chromospheric decoupling: {z_304:.1f}σ, {trend_304:+.1f}%/h"

    # Rule 3: ANOMALY - coupling break detected (but not yet trending)
    if neg_pairs:
        worst_pair = min(neg_pairs, key=lambda x: x[1])
        return MIPhase.ANOMALY, f"Break: {worst_pair[0]} at {worst_pair[1]:.1f}σ"

    # Rule 4: ELEVATED - above-normal coupling (structural activity, not alarm)
    if max_z > 2:
        return MIPhase.ELEVATED, f"Structural activity: {max_z:.1f}σ"

    # Rule 5: BASELINE - normal coupling
    return MIPhase.BASELINE, "Normal coupling"


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
    # NOTE: 'is not None' instead of truthiness - flux 0.0 is a valid
    # (quiet) measurement, not missing data.
    if goes_flux is not None and goes_flux >= 1e-5:
        return Phase.ACTIVE, f"M/X-class active ({goes_class})"
    if goes_flux is not None and goes_flux >= 5e-6 and goes_rising:
        return Phase.ACTIVE, f"C-class flare ({goes_class})"

    # Rule 2: PRE-FLARE - destabilization signature
    if neg_anomalies >= 1 and goes_rising:
        return Phase.PRE_FLARE, f"{neg_anomalies} pair(s) destabilizing + GOES ↑"
    if z_211 < -2 and trend_211 < -3:
        return Phase.PRE_FLARE, f"Coronal decoupling ({z_211:.1f}σ, {trend_211:+.1f}%/h)"

    # Rule 3: POST-EVENT - GOES quiet but ΔMI still elevated (reorganizing)
    # Key hypothesis: we see magnetic restructuring that GOES doesn't
    if goes_flux is not None and goes_flux < 1e-6:  # GOES says quiet (B-class or lower, incl. 0.0)
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
    if goes_flux is not None and 1e-6 < goes_flux < 5e-6 and not goes_rising:
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
    Run all three phase classifiers in parallel and report divergence.

    Returns dict with:
        - goes_only: (phase, reason) from GOES-only classifier
        - mi_only: (phase, reason) from pure ΔMI classifier (NO GOES input)
        - integrated: (phase, reason) from ΔMI+GOES hybrid
        - is_divergent: bool (GOES vs MI-only)
        - divergence_note: str explaining the divergence
        - mi_precursor: bool - True if MI sees anomaly while GOES is quiet

    The key comparison for early warning is GOES-only vs MI-only:
    - If MI-only sees ANOMALY/DESTABILIZING while GOES shows BASELINE → precursor candidate
    """
    goes_only = classify_phase_goes_only(goes_flux, goes_rising, goes_class)
    mi_only = classify_phase_mi_only(pairs_data)
    integrated = classify_phase_experimental(pairs_data, goes_flux, goes_rising, goes_class)

    # Key divergence: GOES quiet but MI sees anomaly
    mi_sees_anomaly = mi_only[0] in [MIPhase.ANOMALY, MIPhase.DESTABILIZING, MIPhase.DECOUPLED]
    goes_quiet = goes_only[0] == Phase.BASELINE

    mi_precursor = mi_sees_anomaly and goes_quiet
    is_divergent = goes_only[0] != integrated[0]

    if mi_precursor:
        divergence_note = f"⚠️ PRECURSOR: ΔMI={mi_only[0]} while GOES quiet"
    elif is_divergent:
        divergence_note = f"GOES={goes_only[0]}, integrated={integrated[0]}"
    else:
        divergence_note = "All classifiers agree"

    return {
        'goes_only': goes_only,
        'mi_only': mi_only,
        'integrated': integrated,
        'is_divergent': is_divergent,
        'mi_precursor': mi_precursor,
        'divergence_note': divergence_note,
        # Legacy keys for backward compatibility
        'current': goes_only,
        'experimental': integrated,
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
