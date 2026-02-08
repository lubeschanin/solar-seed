#!/usr/bin/env python3
"""
Solar Early Warning System Prototype
=====================================

Multi-layer early warning architecture:

  STEREO-A (51° ahead)          ← 2-4 days warning (active regions)
         ↓
  ┌─────────────────────────┐
  │  ΔMI COUPLING MONITOR   │   ← Hours before flare (this system)
  │  Residual r(t) tracking │      - Coupling drops as magnetic stress builds
  │  Trend analysis         │      - Based on Lubeschanin et al. (2026) findings
  └─────────────────────────┘
         ↓
  SDO/AIA + GOES X-ray          ← Minutes (flare detection)
         ↓
  DSCOVR L1 Solar Wind          ← 15-60 min (geomagnetic storm arrival)

Data Sources:
- GOES X-ray flux (NOAA SWPC) - real-time flare classification
- DSCOVR solar wind plasma & magnetic field (L1 point)
- SDO/AIA multichannel imagery - coupling analysis
- NOAA Space Weather Alerts

Usage:
    uv run python scripts/early_warning.py              # Single check
    uv run python scripts/early_warning.py --monitor    # Continuous monitoring (60s)
    uv run python scripts/early_warning.py --coupling   # Include ΔMI coupling analysis
    uv run python scripts/early_warning.py --monitor --coupling --interval 600  # Full monitoring
"""

import sys
import json
import time
import signal
from pathlib import Path
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Global flag for graceful shutdown
_shutdown_requested = False
_original_sigint_handler = None

def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\n  Shutdown requested... (press again to force)")


def _install_signal_handler():
    """Install signal handler for graceful shutdown (only in monitor mode)."""
    global _original_sigint_handler
    _original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _signal_handler)


def _restore_signal_handler():
    """Restore original signal handler."""
    global _original_sigint_handler
    if _original_sigint_handler is not None:
        signal.signal(signal.SIGINT, _original_sigint_handler)
        _original_sigint_handler = None

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from solar_seed.monitoring import (
    MonitoringDB,
    CouplingMonitor,
    MIN_MI_THRESHOLD,
    MIN_ROI_STD,
    validate_roi_variance,
    validate_mi_measurement,
    AnomalyStatus,
    BreakType,
    detect_artifact,
    detect_coupling_break,
    compute_registration_shift,
    compute_robustness_check,
    classify_break_type,
    classify_anomaly_status,
    classify_phase_parallel,
    classify_divergence_type,
    StatusFormatter,
)
from solar_seed.data_sources import (
    load_aia_synoptic,
    load_aia_latest,
    load_aia_direct,
    load_stereo_a_latest,
    STEREO_A_INFO,
    EUVI_TO_AIA,
    SYNOPTIC_BASE_URL,
)

# NOAA SWPC API endpoints
GOES_XRAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"
DSCOVR_PLASMA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json"
DSCOVR_MAG_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json"
ALERTS_URL = "https://services.swpc.noaa.gov/products/alerts.json"

# Flare classification thresholds (W/m²)
FLARE_THRESHOLDS = {
    'X': 1e-4,   # X-class: >= 10⁻⁴
    'M': 1e-5,   # M-class: >= 10⁻⁵
    'C': 1e-6,   # C-class: >= 10⁻⁶
    'B': 1e-7,   # B-class: >= 10⁻⁷
}


def fetch_json(url: str, timeout: int = 30) -> dict | list | None:
    """Fetch JSON from URL with error handling."""
    try:
        req = Request(url, headers={'User-Agent': 'SolarSeed/1.0'})
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except URLError as e:
        print(f"  Network error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return None


def classify_flare(flux: float) -> tuple[str, int]:
    """Classify X-ray flux into flare class."""
    if flux >= FLARE_THRESHOLDS['X']:
        level = flux / FLARE_THRESHOLDS['X']
        return f"X{level:.1f}", 4
    elif flux >= FLARE_THRESHOLDS['M']:
        level = flux / FLARE_THRESHOLDS['M']
        return f"M{level:.1f}", 3
    elif flux >= FLARE_THRESHOLDS['C']:
        level = flux / FLARE_THRESHOLDS['C']
        return f"C{level:.1f}", 2
    elif flux >= FLARE_THRESHOLDS['B']:
        level = flux / FLARE_THRESHOLDS['B']
        return f"B{level:.1f}", 1
    else:
        return "A", 0


def get_goes_xray() -> dict | None:
    """Fetch current GOES X-ray flux."""
    print("\n  Fetching GOES X-ray data...")
    data = fetch_json(GOES_XRAY_URL)

    if not data:
        return None

    # Get latest reading (last entry with valid flux)
    for entry in reversed(data):
        if entry.get('flux') is not None:
            flux = float(entry['flux'])
            time_tag = entry.get('time_tag', 'unknown')
            flare_class, severity = classify_flare(flux)

            return {
                'timestamp': time_tag,
                'flux': flux,
                'flare_class': flare_class,
                'severity': severity,
                'energy': entry.get('energy', 'unknown')
            }

    return None


def get_dscovr_solar_wind() -> dict | None:
    """Fetch current DSCOVR solar wind data."""
    print("  Fetching DSCOVR solar wind data...")

    plasma = fetch_json(DSCOVR_PLASMA_URL)
    mag = fetch_json(DSCOVR_MAG_URL)

    result = {}

    if plasma and len(plasma) > 1:
        # Skip header row, get latest
        for row in reversed(plasma[1:]):
            if row[1] is not None:  # density
                result['plasma'] = {
                    'timestamp': row[0],
                    'density': float(row[1]) if row[1] else None,  # p/cm³
                    'speed': float(row[2]) if row[2] else None,    # km/s
                    'temperature': float(row[3]) if row[3] else None  # K
                }
                break

    if mag and len(mag) > 1:
        for row in reversed(mag[1:]):
            if row[3] is not None:  # Bz
                result['mag'] = {
                    'timestamp': row[0],
                    'bx': float(row[1]) if row[1] else None,
                    'by': float(row[2]) if row[2] else None,
                    'bz': float(row[3]) if row[3] else None,  # nT (negative = geoeffective)
                    'bt': float(row[6]) if row[6] else None   # total field
                }
                break

    return result if result else None


def parse_noaa_alert(alert: dict) -> dict:
    """Parse NOAA alert and extract Kp/scale information."""
    import re
    import json

    message = alert.get('message', '')
    product_id = alert.get('product_id', '')

    # Extract NOAA message code (e.g., WARK05, ALTK06)
    code_match = re.search(r'Code:\s*(\w+)', message)
    message_code = code_match.group(1) if code_match else None

    # Determine alert type from product_id or code
    alert_type = None
    if product_id.startswith('WAR') or (message_code and message_code.startswith('WAR')):
        alert_type = 'WARNING'
    elif product_id.startswith('ALT') or (message_code and message_code.startswith('ALT')):
        alert_type = 'ALERT'
    elif product_id.startswith('WAT'):
        alert_type = 'WATCH'
    elif product_id.startswith('SUM'):
        alert_type = 'SUMMARY'

    # Extract Kp from code (e.g., WARK05 → Kp=5)
    kp_predicted = None
    if message_code and len(message_code) >= 6:
        try:
            kp_predicted = int(message_code[-1])
        except ValueError:
            pass

    # Extract Kp from message text
    kp_match = re.search(r'Kp\s*[=:]\s*(\d+)', message)
    kp_observed = int(kp_match.group(1)) if kp_match else None

    # Extract G/S/R scale (e.g., "G2 - Moderate", "S1 - Minor", "R1 - Minor")
    g_match = re.search(r'G(\d)', message)
    s_match = re.search(r'S(\d)', message)
    r_match = re.search(r'R(\d)', message)

    g_scale = int(g_match.group(1)) if g_match else None
    s_scale = int(s_match.group(1)) if s_match else None
    r_scale = int(r_match.group(1)) if r_match else None

    # Extract active region (e.g., AR3842, Region 3842)
    region_match = re.search(r'(?:AR|Region\s*)(\d{4})', message)
    source_region = f"AR{region_match.group(1)}" if region_match else None

    # Extract validity period
    valid_match = re.search(r'Valid\s+(\d{4}\s+\w+\s+\d+)\s+to\s+(\d{4}\s+\w+\s+\d+)', message)
    valid_from = valid_to = None  # Would need more parsing for actual dates

    return {
        'message_code': message_code or product_id,
        'alert_type': alert_type,
        'kp_observed': kp_observed,
        'kp_predicted': kp_predicted,
        'g_scale': g_scale,
        's_scale': s_scale,
        'r_scale': r_scale,
        'source_region': source_region,
        'raw_json': json.dumps(alert),
    }


def get_noaa_alerts() -> list:
    """Fetch active NOAA space weather alerts."""
    print("  Fetching NOAA alerts...")
    data = fetch_json(ALERTS_URL)

    if not data:
        return []

    # Filter recent alerts (last 24 hours)
    now = datetime.now(timezone.utc)
    recent = []

    for alert in data:
        try:
            issue_str = alert.get('issue_datetime', '')
            # Handle various datetime formats
            if issue_str:
                issue_str = issue_str.replace('Z', '+00:00')
                if '+' not in issue_str and issue_str[-1] != 'Z':
                    issue_str += '+00:00'
                issue_time = datetime.fromisoformat(issue_str)
                if issue_time.tzinfo is None:
                    issue_time = issue_time.replace(tzinfo=timezone.utc)
                if now - issue_time < timedelta(hours=24):
                    parsed = parse_noaa_alert(alert)
                    recent.append({
                        'type': alert.get('product_id', 'unknown'),
                        'message': alert.get('message', '')[:200],
                        'issued': alert['issue_datetime'],
                        **parsed,  # Include parsed Kp/scale info
                    })
        except (KeyError, ValueError, TypeError):
            continue

    return recent[:5]  # Top 5 recent


def store_noaa_alerts(alerts: list):
    """Store NOAA alerts in database."""
    if not alerts:
        return

    db = get_monitoring_db()
    for alert in alerts:
        db.insert_noaa_alert(
            alert_id=f"{alert['type']}_{alert['issued']}",
            issue_time=alert['issued'],
            message_code=alert.get('message_code'),
            alert_type=alert.get('alert_type'),
            kp_observed=alert.get('kp_observed'),
            kp_predicted=alert.get('kp_predicted'),
            g_scale=alert.get('g_scale'),
            s_scale=alert.get('s_scale'),
            r_scale=alert.get('r_scale'),
            source_region=alert.get('source_region'),
            message=alert.get('message'),
            raw_json=alert.get('raw_json'),
        )


def assess_geomagnetic_risk(solar_wind: dict) -> tuple[str, int]:
    """Assess geomagnetic storm risk from solar wind data."""
    if not solar_wind:
        return "Unknown", 0

    risk_level = 0
    factors = []

    # Check Bz (southward = negative = bad)
    if 'mag' in solar_wind and solar_wind['mag'].get('bz') is not None:
        bz = solar_wind['mag']['bz']
        if bz < -10:
            risk_level += 3
            factors.append(f"Bz={bz:.1f}nT (strong southward)")
        elif bz < -5:
            risk_level += 2
            factors.append(f"Bz={bz:.1f}nT (moderate southward)")
        elif bz < 0:
            risk_level += 1
            factors.append(f"Bz={bz:.1f}nT (weak southward)")

    # Check solar wind speed
    if 'plasma' in solar_wind and solar_wind['plasma'].get('speed') is not None:
        speed = solar_wind['plasma']['speed']
        if speed > 700:
            risk_level += 2
            factors.append(f"Speed={speed:.0f}km/s (high)")
        elif speed > 500:
            risk_level += 1
            factors.append(f"Speed={speed:.0f}km/s (elevated)")

    # Check density
    if 'plasma' in solar_wind and solar_wind['plasma'].get('density') is not None:
        density = solar_wind['plasma']['density']
        if density > 20:
            risk_level += 1
            factors.append(f"Density={density:.1f}/cm³ (high)")

    if risk_level >= 5:
        return f"HIGH - {', '.join(factors)}", 3
    elif risk_level >= 3:
        return f"MODERATE - {', '.join(factors)}", 2
    elif risk_level >= 1:
        return f"LOW - {', '.join(factors)}", 1
    else:
        return "QUIET", 0


# Global instances
_coupling_monitor = None
_monitoring_db = None

def get_coupling_monitor() -> CouplingMonitor:
    global _coupling_monitor
    if _coupling_monitor is None:
        _coupling_monitor = CouplingMonitor()
    return _coupling_monitor


def get_monitoring_db() -> MonitoringDB:
    """Get singleton database instance."""
    global _monitoring_db
    if _monitoring_db is None:
        _monitoring_db = MonitoringDB()
    return _monitoring_db


def store_goes_reading(xray: dict):
    """Store GOES X-ray reading in database."""
    if not xray:
        return
    db = get_monitoring_db()
    db.insert_goes_xray(
        timestamp=xray['timestamp'],
        flux=xray['flux'],
        flare_class=xray['flare_class'].split('.')[0] if '.' in xray['flare_class'] else xray['flare_class'][0],
        magnitude=float(xray['flare_class'][1:]) if len(xray['flare_class']) > 1 and xray['flare_class'][1:].replace('.','').isdigit() else None,
        energy=xray.get('energy')
    )


def store_solar_wind_reading(solar_wind: dict, risk: str, risk_level: int):
    """Store solar wind reading in database."""
    if not solar_wind:
        return
    db = get_monitoring_db()

    timestamp = None
    speed = density = temperature = bx = by = bz = bt = None

    if 'plasma' in solar_wind:
        p = solar_wind['plasma']
        timestamp = p.get('timestamp')
        speed = p.get('speed')
        density = p.get('density')
        temperature = p.get('temperature')

    if 'mag' in solar_wind:
        m = solar_wind['mag']
        if not timestamp:
            timestamp = m.get('timestamp')
        bx = m.get('bx')
        by = m.get('by')
        bz = m.get('bz')
        bt = m.get('bt')

    if timestamp:
        db.insert_solar_wind(
            timestamp=timestamp,
            speed=speed,
            density=density,
            temperature=temperature,
            bx=bx, by=by, bz=bz, bt=bt,
            risk_level=risk_level,
            risk_description=risk
        )


def store_coupling_reading(timestamp: str, coupling: dict, xray: dict = None):
    """Store coupling measurements in database."""
    if not coupling:
        return
    db = get_monitoring_db()

    # Extract resolution from quality metadata
    quality = coupling.get('_quality', {})
    resolution_str = quality.get('resolution', '1024x1024')
    resolution = '1k' if '1024' in resolution_str else '4k'
    time_spread_sec = quality.get('time_spread_sec')

    # Extract validation data for quality_ok
    validation = coupling.get('_validation', {})
    robustness_checks = validation.get('robustness_checks', {})

    for pair, data in coupling.items():
        # Skip internal metadata fields
        if pair.startswith('_'):
            continue

        status = data.get('status')

        # Determine quality_ok from validation checks
        quality_ok = None
        robustness_score = None
        sync_delta_s = time_spread_sec

        rob = robustness_checks.get(pair)
        if rob is not None:
            robustness_score = rob.get('change_pct')

        if status == 'DATA_ERROR':
            quality_ok = False
        elif data.get('break_vetoed'):
            quality_ok = False
        else:
            # Check individual tests
            tests_ok = True
            if rob is not None and rob.get('is_robust') is False:
                tests_ok = False
            if time_spread_sec is not None and time_spread_sec > 60:
                tests_ok = False
            quality_ok = tests_ok

        # Store coupling measurement with resolution and quality fields
        db.insert_coupling(
            timestamp=timestamp,
            pair=pair,
            delta_mi=data['delta_mi'],
            mi_original=data.get('mi_original'),
            residual=data.get('residual'),
            deviation_pct=data.get('deviation_pct'),
            status=status,
            trend=data.get('trend'),
            slope_pct_per_hour=data.get('slope_pct_per_hour'),
            acceleration=data.get('acceleration'),
            confidence=data.get('confidence'),
            n_points=data.get('n_points'),
            resolution=resolution,
            quality_ok=quality_ok,
            robustness_score=robustness_score,
            sync_delta_s=sync_delta_s,
        )

        # Auto-create prediction for ALERT/ELEVATED status
        if status in ('ALERT', 'ELEVATED'):
            predicted_class = 'M' if status == 'ALERT' else 'C'

            # Determine trigger_kind based on what caused the alert
            trigger_kind = None
            trigger_value = None
            trigger_threshold = None

            sudden_drop = data.get('sudden_drop_severity')
            is_break = data.get('is_break')
            deviation_pct = data.get('deviation_pct')
            z_mad = data.get('z_mad')

            if sudden_drop:
                # Sudden drop detector triggered
                trigger_kind = 'SUDDEN_DROP'
                trigger_value = data.get('sudden_drop_pct')
                trigger_threshold = -0.15 if sudden_drop == 'MODERATE' else -0.25
            elif is_break:
                # Coupling break detected
                trigger_kind = 'BREAK'
                trigger_value = z_mad
                trigger_threshold = 2.0
            elif deviation_pct is not None and deviation_pct < -0.25:
                # Absolute threshold exceeded (ALERT)
                trigger_kind = 'THRESHOLD'
                trigger_value = deviation_pct
                trigger_threshold = -0.25
            elif deviation_pct is not None and deviation_pct < -0.15:
                # Warning threshold (ELEVATED)
                trigger_kind = 'THRESHOLD'
                trigger_value = deviation_pct
                trigger_threshold = -0.15
            elif deviation_pct is not None and deviation_pct < -0.10:
                # Elevated threshold (matches coupling.py ELEVATED status)
                trigger_kind = 'THRESHOLD'
                trigger_value = deviation_pct
                trigger_threshold = -0.10
            else:
                # Check for z-score spike (STRONG/EXTREME anomaly level)
                residual = data.get('residual')
                if residual is not None and abs(residual) > 4.0:
                    trigger_kind = 'Z_SCORE_SPIKE'
                    trigger_value = residual
                    trigger_threshold = 4.0
                else:
                    # Fallback: check trend
                    trend = data.get('trend')
                    if trend in ('DECLINING', 'ACCELERATING_DOWN'):
                        trigger_kind = 'TREND'
                        trigger_value = data.get('slope_pct_per_hour')

            db.insert_prediction(
                prediction_time=timestamp,
                predicted_class=predicted_class,
                trigger_pair=pair,
                trigger_status=status,
                trigger_residual=data.get('residual'),
                trigger_trend=data.get('trend'),
                trigger_kind=trigger_kind,
                trigger_value=trigger_value,
                trigger_threshold=trigger_threshold,
                notes='Auto-created during monitoring'
            )

    # Check and log phase divergence
    check_and_log_divergence(timestamp, coupling, xray, db)


def check_and_log_divergence(timestamp: str, coupling: dict, xray: dict, db: MonitoringDB):
    """
    Check for phase classifier divergence and log to database.

    This enables empirical validation: do divergences predict flares?

    Divergence types:
    - PRECURSOR: ΔMI anomaly before GOES rises (potential early warning)
    - POST_EVENT: ΔMI anomaly after GOES quiet (structural relaxation)
    - UNCONFIRMED: Needs validation against subsequent events
    """
    # Extract GOES info
    goes_flux = xray.get('flux') if xray else None
    goes_rising = xray.get('rising', False) if xray else None
    goes_class = xray.get('flare_class') if xray else None

    # Build pairs data for phase classification
    pairs_data = {k: v for k, v in coupling.items() if not k.startswith('_')}

    # Skip if no pairs data
    if not pairs_data:
        return

    # Run parallel classification
    comparison = classify_phase_parallel(pairs_data, goes_flux, goes_rising, goes_class)

    # Only log if divergent
    if not comparison['is_divergent']:
        return

    # Find max z-score and trigger pair
    max_z = 0
    trigger_pair = None
    for pair, data in pairs_data.items():
        z = abs(data.get('residual', 0))
        if z > max_z:
            max_z = z
            trigger_pair = pair

    # Classify divergence type
    phase_goes, reason_goes = comparison['current']
    phase_exp, reason_exp = comparison['experimental']

    # Check for recent flare activity (query last 24h from DB)
    recent_flare_hours = None
    try:
        recent_flares = db.get_recent_flares(hours=24, min_class='C')
        if recent_flares:
            # Hours since most recent flare
            from datetime import datetime, timezone
            latest = recent_flares[0]
            flare_time = datetime.fromisoformat(latest['start_time'].replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            recent_flare_hours = (now - flare_time).total_seconds() / 3600
    except Exception:
        pass  # DB query failed, proceed without flare context

    div_type = classify_divergence_type(
        phase_goes=phase_goes,
        phase_experimental=phase_exp,
        goes_trend_rising=goes_rising,
        recent_flare_hours=recent_flare_hours,
    )

    db.insert_phase_divergence(
        timestamp=timestamp,
        phase_goes=phase_goes,
        phase_experimental=phase_exp,
        reason_goes=reason_goes,
        reason_experimental=reason_exp,
        divergence_type=div_type,
        goes_flux=goes_flux,
        goes_class=goes_class,
        goes_rising=goes_rising,
        max_z_score=max_z,
        trigger_pair=trigger_pair,
        notes=comparison['divergence_note'],
    )


def run_stereo_coupling_analysis() -> dict | None:
    """
    Run coupling analysis on STEREO-A EUVI data.

    This shows what's coming ~3.9 days before it faces Earth.
    """
    print("\n  Running STEREO-A coupling analysis...")

    try:
        from solar_seed.radial_profile import subtract_radial_geometry
        from solar_seed.control_tests import sector_ring_shuffle_test

        # Load STEREO-A data
        channels, timestamp, metadata = load_stereo_a_latest([195, 284, 304], max_age_minutes=180)

        if not channels or len(channels) < 2:
            print("  Could not load STEREO-A EUVI data")
            return None

        print(f"  Using STEREO-A data from: {timestamp}")

        results = {
            '_stereo_metadata': metadata,
            '_timestamp': timestamp,
        }

        # Map EUVI wavelengths to AIA-equivalent pairs
        # EUVI 195 ≈ AIA 193, EUVI 284 ≈ AIA 211, EUVI 304 = AIA 304
        pairs = [
            (195, 284, '193-211'),  # Corona pair (EUVI equiv)
            (195, 304, '193-304'),  # Corona-Chromosphere (EUVI equiv)
        ]

        for wl1, wl2, pair_name in pairs:
            if wl1 in channels and wl2 in channels:
                res1, _, _ = subtract_radial_geometry(channels[wl1])
                res2, _, _ = subtract_radial_geometry(channels[wl2])

                # Validate ROI variance before MI calculation
                roi_check = validate_roi_variance(res1, res2, pair_name)
                if not roi_check['is_valid']:
                    print(f"  ⚠ DATA_ERROR in {pair_name}: {roi_check['error_reason']}")
                    results[pair_name] = {
                        'delta_mi': 0.0,
                        'data_error': True,
                        'error_type': roi_check['error_type'],
                        'error_reason': roi_check['error_reason'],
                        'source': 'STEREO-A',
                    }
                    continue

                shuffle_result = sector_ring_shuffle_test(res1, res2, n_rings=10, n_sectors=12)
                delta_mi = shuffle_result.mi_original - shuffle_result.mi_sector_shuffled

                results[pair_name] = {
                    'delta_mi': delta_mi,
                    'mi_original': shuffle_result.mi_original,
                    'euvi_wavelengths': f"{wl1}-{wl2}",
                    'source': 'STEREO-A',
                }

        return results

    except Exception as e:
        print(f"  STEREO-A analysis error: {e}")
        import traceback
        traceback.print_exc()
        return None


def _load_channels(wavelengths: list[int], use_synoptic: bool = False) -> tuple[dict | None, str | None, dict | None, str | None]:
    """
    Load AIA channel data with fallback strategy.

    Default: 4k full-resolution (more accurate MI, especially for 304Å pairs).
    Synoptic 1k available as fallback but has +350% MI inflation for chromospheric pairs.

    Returns:
        (channels, timestamp, quality_info, data_source) or (None, None, None, None) on failure
    """
    channels = None
    timestamp = None
    quality_info = None
    data_source = None

    # Optional: Try synoptic data (fast but inaccurate for 304Å pairs)
    if use_synoptic:
        print("  Trying AIA synoptic data (1k, direct access)...")
        channels, timestamp, quality_info = load_aia_synoptic(wavelengths)
        if channels and len(channels) >= 2:
            data_source = 'synoptic'
            print(f"  ✓ Using synoptic data from: {timestamp}")

    # Primary: Full-res via VSO (4k, accurate MI)
    if not channels or len(channels) < 2:
        print("  Loading AIA full-res data (4k via VSO)...")
        channels, timestamp, quality_info = load_aia_latest(wavelengths, max_age_minutes=30)
        if channels and len(channels) >= 2:
            data_source = 'full-res'

    # Fallback 2: Synoptic (1k, less accurate for 304Å but reliable)
    if not channels or len(channels) < 2:
        print("  ⚠️ 4k unavailable, falling back to synoptic (1k)...")
        print("    Note: 304Å MI will be inflated by ~350%")
        channels, timestamp, quality_info = load_aia_synoptic(wavelengths)
        if channels and len(channels) >= 2:
            data_source = 'synoptic-fallback'
            if quality_info:
                quality_info['warnings'] = quality_info.get('warnings', [])
                quality_info['warnings'].append("1k resolution: 304Å MI inflated ~350%")

    if not channels or len(channels) < 2:
        print("  ✗ Could not load AIA data from any source")
        return None, None, None, None

    print(f"  Data source: {data_source}")
    return channels, timestamp, quality_info, data_source


def _analyze_pair(wl1: int, wl2: int, channels: dict, monitor, validate_breaks: bool,
                  subtract_radial_geometry, sector_ring_shuffle_test,
                  resolution: str = '1k') -> dict:
    """
    Analyze a single channel pair: compute MI, detect breaks, run validation.

    Args:
        resolution: Data resolution ('1k' or '4k') - affects baseline thresholds

    Returns dict with keys: result, break_detection, registration_check, robustness_check,
                            quality_warnings, artifact_warnings
    """
    pair_key = f"{wl1}-{wl2}"
    output = {
        'pair_key': pair_key,
        'result': None,
        'break_detection': None,
        'registration_check': None,
        'robustness_check': None,
        'quality_warnings': [],
        'artifact_warnings': [],
    }

    # Test B: Registration shift BEFORE geometry subtraction
    reg_check = compute_registration_shift(channels[wl1], channels[wl2])
    output['registration_check'] = reg_check

    if not reg_check['is_centered']:
        warn_msg = f"{pair_key}: Registration shift {reg_check['shift_pixels']:.1f}px (dx={reg_check['dx']}, dy={reg_check['dy']})"
        output['quality_warnings'].append(warn_msg)
        print(f"  ⚠ {warn_msg}")

    res1, _, _ = subtract_radial_geometry(channels[wl1])
    res2, _, _ = subtract_radial_geometry(channels[wl2])

    # Validate ROI variance before MI calculation
    roi_check = validate_roi_variance(res1, res2, pair_key)
    if not roi_check['is_valid']:
        print(f"  ⚠ DATA_ERROR in {pair_key}: {roi_check['error_reason']}")
        output['result'] = {
            'delta_mi': 0.0,
            'data_error': True,
            'error_type': roi_check['error_type'],
            'error_reason': roi_check['error_reason'],
            'std1': roi_check['std1'],
            'std2': roi_check['std2'],
            'status': 'DATA_ERROR',
        }
        output['break_detection'] = {
            'is_break': False,
            'data_error': True,
            'error_type': roi_check['error_type'],
            'error_reason': roi_check['error_reason'],
        }
        return output

    shuffle_result = sector_ring_shuffle_test(res1, res2, n_rings=10, n_sectors=12)
    delta_mi = shuffle_result.mi_original - shuffle_result.mi_sector_shuffled

    # Check for artifacts (3σ jump)
    artifact = detect_artifact(pair_key, delta_mi, monitor)
    if artifact:
        output['artifact_warnings'].append(f"{pair_key}: {artifact['message']}")
        print(f"  ⚠ Possible artifact in {pair_key}: {artifact['message']}")

    # Formal Coupling Break detection
    break_check = detect_coupling_break(pair_key, delta_mi, monitor)
    output['break_detection'] = break_check

    if break_check['is_break']:
        z_mad = break_check.get('z_mad', 0)
        print(f"  ⚠ COUPLING BREAK detected in {pair_key}:")
        print(f"     {break_check['criterion']}")
        print(f"     Current: {delta_mi:.4f}, Deviation: {z_mad:.1f} MAD below median")

        # Test C: Robustness check on detected breaks
        if validate_breaks:
            print(f"  → Running robustness check (2x2 binning)...")
            robust = compute_robustness_check(channels[wl1], channels[wl2], delta_mi)
            output['robustness_check'] = robust
            if robust.get('is_robust'):
                print(f"     ✓ Break is ROBUST under binning (Δ={robust['change_pct']:.1f}%)")

                # Persistence check: anti-spike filter (2+ consecutive frames)
                is_persistent = monitor.is_persistent_break(pair_key, True, min_frames=2)
                if is_persistent:
                    print(f"     ✓ Break is PERSISTENT (2+ frames)")
                else:
                    print(f"     ⚠ Break NOT persistent (single frame - possible spike)")
                    break_check['is_break'] = False
                    break_check['vetoed'] = 'spike'
                    output['break_detection'] = break_check
            else:
                change = robust.get('change_pct', 0)
                print(f"     ⚠ Break is UNRELIABLE (binning Δ={change:.1f}% > 20%)")
                break_check['is_break'] = False
                break_check['vetoed'] = 'robustness'
                output['break_detection'] = break_check

    residual_info = monitor.compute_residual(pair_key, delta_mi, resolution=resolution)
    trend_info = monitor.analyze_trend(pair_key)

    # Extract sudden drop info
    sudden_drop = residual_info.get('sudden_drop', {})
    sudden_drop_pct = sudden_drop.get('drop_pct', 0) if sudden_drop else 0
    sudden_drop_severity = sudden_drop.get('severity') if sudden_drop else None

    output['result'] = {
        'delta_mi': delta_mi,
        'mi_original': shuffle_result.mi_original,
        'residual': residual_info['residual'],
        'deviation_pct': residual_info['deviation_pct'],
        'status': residual_info['status'],
        'sudden_drop_pct': sudden_drop_pct,
        'sudden_drop_severity': sudden_drop_severity,
        'artifact_warning': artifact is not None,
        'is_break': break_check.get('is_break', False),
        'break_vetoed': break_check.get('vetoed'),
        'z_mad': break_check.get('z_mad', 0),
        'registration_shift': reg_check.get('shift_pixels', 0),
        **trend_info
    }
    return output


def _build_goes_context(xray: dict) -> dict | None:
    """Build GOES context for phase-gating from X-ray data."""
    if not xray:
        return None

    flux = xray.get('flux', 0)
    flare_class = xray.get('flare_class', 'A0')
    goes_rising = False

    if flare_class.startswith(('A', 'B')):
        goes_phase = 'quiet_or_decay'
    elif flare_class.startswith('C'):
        goes_phase = 'active'
    else:
        goes_phase = 'elevated'

    return {
        'flux': flux,
        'flare_class': flare_class,
        'rising': goes_rising,
        'phase': goes_phase,
    }


def run_coupling_analysis(validate_breaks: bool = True, xray: dict = None, use_synoptic: bool = False) -> dict | None:
    """
    Run coupling analysis on latest AIA data with quality checks.

    Default: 4k full-resolution (accurate MI, especially for 304Å pairs).
    1k synoptic inflates 304Å MI by +350% due to spatial aliasing.

    Includes validation: time alignment, registration shift, robustness check,
    coupling break detection, and phase-gating.
    """
    print("  Running coupling analysis...")

    try:
        from solar_seed.radial_profile import subtract_radial_geometry
        from solar_seed.control_tests import sector_ring_shuffle_test

        # Load data
        channels, timestamp, quality_info, data_source = _load_channels([193, 211, 304], use_synoptic)
        if not channels:
            return None

        # Report quality
        quality_warnings = []
        if quality_info:
            if quality_info['is_good_quality']:
                print(f"  ✓ Quality check: PASSED (sync={quality_info['time_spread_sec']:.0f}s)")
            else:
                print(f"  ⚠ Quality warnings:")
                for w in quality_info['warnings']:
                    print(f"    - {w}")
                quality_warnings = quality_info['warnings']

        # Analyze each pair
        results = {}
        monitor = get_coupling_monitor()
        artifact_warnings = []
        registration_checks = {}
        break_detections = {}
        robustness_checks = {}

        # Determine resolution from actual data shape (not data source label)
        first_channel = next(iter(channels.values()))
        actual_size = max(first_channel.shape) if hasattr(first_channel, 'shape') else 0
        resolution = '4k' if actual_size >= 4000 else '1k'

        pairs = [(193, 211), (193, 304), (171, 193)]
        for wl1, wl2 in pairs:
            if wl1 in channels and wl2 in channels:
                pair_output = _analyze_pair(
                    wl1, wl2, channels, monitor, validate_breaks,
                    subtract_radial_geometry, sector_ring_shuffle_test,
                    resolution=resolution
                )
                pair_key = pair_output['pair_key']
                results[pair_key] = pair_output['result']
                break_detections[pair_key] = pair_output['break_detection']
                registration_checks[pair_key] = pair_output['registration_check']
                if pair_output['robustness_check']:
                    robustness_checks[pair_key] = pair_output['robustness_check']
                quality_warnings.extend(pair_output['quality_warnings'])
                artifact_warnings.extend(pair_output['artifact_warnings'])

        # 4K Confirmation: If break detected with 1k, verify with full-res
        breaks_to_confirm = [
            (pair, bd) for pair, bd in break_detections.items()
            if bd.get('is_break') and not bd.get('vetoed') and not results.get(pair, {}).get('data_error')
        ]

        if breaks_to_confirm and data_source == 'synoptic':
            print(f"  🔬 {len(breaks_to_confirm)} break(s) detected - loading 4K for confirmation...")
            channels_4k, _, _ = load_aia_latest([193, 211, 304], max_age_minutes=30)

            if channels_4k and len(channels_4k) >= 2:
                print(f"    ✓ 4K data loaded ({list(channels_4k.keys())})")

                for pair_key, bd in breaks_to_confirm:
                    wl1, wl2 = map(int, pair_key.split('-'))
                    if wl1 in channels_4k and wl2 in channels_4k:
                        # Compute MI on 4K
                        res1_4k, _, _ = subtract_radial_geometry(channels_4k[wl1])
                        res2_4k, _, _ = subtract_radial_geometry(channels_4k[wl2])
                        shuffle_4k = sector_ring_shuffle_test(res1_4k, res2_4k, n_rings=10, n_sectors=12)
                        delta_mi_4k = shuffle_4k.mi_original - shuffle_4k.mi_sector_shuffled

                        delta_mi_1k = results[pair_key]['delta_mi']
                        diff_pct = abs(delta_mi_4k - delta_mi_1k) / delta_mi_1k * 100 if delta_mi_1k else 0

                        if diff_pct < 15:  # <15% difference = confirmed
                            print(f"    ✓ {pair_key}: 4K CONFIRMS (1k={delta_mi_1k:.3f}, 4k={delta_mi_4k:.3f}, Δ={diff_pct:.1f}%)")
                            results[pair_key]['confirmed_4k'] = True
                            results[pair_key]['delta_mi_4k'] = delta_mi_4k
                            break_detections[pair_key]['confirmed_4k'] = True
                        else:
                            print(f"    ⚠ {pair_key}: 4K DISAGREES (1k={delta_mi_1k:.3f}, 4k={delta_mi_4k:.3f}, Δ={diff_pct:.1f}%)")
                            results[pair_key]['confirmed_4k'] = False
                            results[pair_key]['delta_mi_4k'] = delta_mi_4k
                            break_detections[pair_key]['confirmed_4k'] = False
                            break_detections[pair_key]['vetoed'] = '4k_mismatch'
            else:
                print(f"    ⚠ Could not load 4K data for confirmation")

        # Save to history
        monitor.add_reading(timestamp, results)

        # Transfer state detection
        time_spread = quality_info['time_spread_sec'] if quality_info else None
        transfer = monitor.detect_transfer_state(
            robustness_checks=robustness_checks,
            time_spread_sec=time_spread
        )
        if transfer:
            results['_transfer_state'] = transfer

        # Quality metadata
        results['_quality'] = {
            'data_source': data_source,
            'resolution': f'{actual_size}x{actual_size}',
            'is_good': quality_info['is_good_quality'] if quality_info else None,
            'time_spread_sec': time_spread,
            'timestamps': quality_info['timestamps'] if quality_info else {},
            'warnings': quality_warnings + artifact_warnings,
            'n_warnings': len(quality_warnings) + len(artifact_warnings),
        }

        # Phase-gating and anomaly classification
        goes_context = _build_goes_context(xray)
        anomaly_statuses = {}
        for pair_key, bd in break_detections.items():
            if bd.get('is_break') or bd.get('vetoed'):
                trend_info = {
                    'slope_pct_per_hour': results.get(pair_key, {}).get('slope_pct_per_hour', 0),
                    'acceleration': results.get(pair_key, {}).get('acceleration', 0),
                    'trend': results.get(pair_key, {}).get('trend', 'stable'),
                }
                anomaly_statuses[pair_key] = classify_anomaly_status(
                    bd, robustness_checks.get(pair_key), registration_checks.get(pair_key),
                    time_spread, trend_info=trend_info, goes_context=goes_context
                )

        results['_validation'] = {
            'registration_checks': registration_checks,
            'break_detections': break_detections,
            'robustness_checks': robustness_checks,
            'anomaly_statuses': anomaly_statuses,
        }

        # Copy break flags to pair results for UI display
        for pair_key, bd in break_detections.items():
            if pair_key in results:
                results[pair_key]['is_break'] = bd.get('is_break', False)
                results[pair_key]['break_vetoed'] = bd.get('vetoed')
                if 'confirmed_4k' in bd:
                    results[pair_key]['confirmed_4k'] = bd['confirmed_4k']

        return results

    except Exception as e:
        print(f"  Coupling analysis error: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_status_report(xray: dict, solar_wind: dict, alerts: list, coupling: dict = None, stereo: dict = None):
    """Print formatted status report using Rich StatusFormatter."""
    fmt = StatusFormatter()

    fmt.print_header()
    fmt.print_xray_status(xray)

    # Solar wind with risk assessment
    risk_info = assess_geomagnetic_risk(solar_wind) if solar_wind else None
    fmt.print_solar_wind(solar_wind, risk_info)

    # Coupling analysis
    if coupling:
        fmt.print_coupling_analysis(coupling, AnomalyStatus, BreakType, xray=xray)
        fmt.print_event_narrative(xray, coupling)

    # STEREO-A section
    fmt.print_stereo_section(stereo, coupling)

    # NOAA alerts
    fmt.print_alerts_section(alerts)


def monitor_loop(interval: int = 60, with_coupling: bool = False, with_stereo: bool = False, store_db: bool = True):
    """Continuous monitoring loop."""
    global _shutdown_requested
    _shutdown_requested = False

    # Install signal handler for graceful shutdown
    _install_signal_handler()

    print(f"\n  Starting continuous monitoring (interval: {interval}s)")
    if store_db:
        print(f"  Database: {get_monitoring_db().db_path}")
    print(f"  Press Ctrl+C to stop\n")

    coupling_interval = 600  # Run coupling every 10 minutes
    stereo_interval = 1800   # Run STEREO every 30 minutes (data updates less frequently)
    last_coupling = 0
    last_stereo = 0

    while not _shutdown_requested:
        xray = get_goes_xray()
        if _shutdown_requested:
            break

        solar_wind = get_dscovr_solar_wind()
        if _shutdown_requested:
            break

        alerts = get_noaa_alerts()
        if _shutdown_requested:
            break

        # Store in database
        if store_db:
            store_goes_reading(xray)
            if solar_wind:
                risk, risk_level = assess_geomagnetic_risk(solar_wind)
                store_solar_wind_reading(solar_wind, risk, risk_level)
            if alerts:
                store_noaa_alerts(alerts)

        coupling = None
        if with_coupling and (time.time() - last_coupling) > coupling_interval:
            if not _shutdown_requested:
                coupling = run_coupling_analysis(xray=xray)
                last_coupling = time.time()
                # Store coupling and check for divergence
                if store_db and coupling:
                    now = datetime.now(timezone.utc)
                    store_coupling_reading(now.strftime("%Y-%m-%dT%H:%M:%S"), coupling, xray)

        if _shutdown_requested:
            break

        # STEREO-A analysis (less frequent due to data latency)
        stereo = None
        if with_stereo and (time.time() - last_stereo) > stereo_interval:
            if not _shutdown_requested:
                stereo = run_stereo_coupling_analysis()
                last_stereo = time.time()

        if _shutdown_requested:
            break

        print_status_report(xray, solar_wind, alerts, coupling, stereo)

        # Alert on significant events
        if xray and xray['severity'] >= 3:
            print(f"\a")  # Terminal bell

        # Interruptible sleep (check every second)
        for _ in range(interval):
            if _shutdown_requested:
                break
            time.sleep(1)

    # Cleanup
    _restore_signal_handler()
    print("\n  Monitoring stopped.")
    if store_db:
        stats = get_monitoring_db().get_database_stats()
        print(f"  Database stats: {stats['goes_xray']} X-ray, {stats['solar_wind']} wind, {stats['coupling_measurements']} coupling")


# Typer CLI app
app = typer.Typer(
    name="solar-warning",
    help="☀️ Solar Early Warning System - Multi-layer space weather monitoring",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def _select_location_interactive() -> str:
    """Interactive location selection with Rich."""
    from rich.prompt import Prompt
    from solar_seed.monitoring.relevance import LOCATIONS

    console.print("\n[bold cyan]📍 Select your location:[/]\n")

    # Show numbered list
    locations = list(LOCATIONS.keys())
    for i, loc in enumerate(locations, 1):
        lat, lon, tz = LOCATIONS[loc]
        console.print(f"  [{i}] {loc.title():12} ({lat:.1f}°, {lon:.1f}°) - {tz}")

    console.print(f"  [7] Custom coordinates")
    console.print()

    choice = Prompt.ask("Enter number", default="1")

    try:
        idx = int(choice)
        if 1 <= idx <= len(locations):
            return locations[idx - 1]
        elif idx == 7:
            lat = Prompt.ask("Latitude", default="52.5")
            lon = Prompt.ask("Longitude", default="13.4")
            return f"{lat},{lon}"
    except ValueError:
        pass

    # Try as location name or coords
    if choice.lower() in LOCATIONS:
        return choice.lower()
    return choice


def _get_saved_location() -> str | None:
    """Get saved location from config."""
    config_path = Path.home() / ".config" / "solar-seed" / "location.txt"
    if config_path.exists():
        return config_path.read_text().strip()
    return None


def _save_location(location: str):
    """Save location to config."""
    config_path = Path.home() / ".config" / "solar-seed" / "location.txt"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(location)


@app.command()
def check(
    coupling: bool = typer.Option(False, "--coupling", "-c", help="Include SDO/AIA coupling analysis"),
    stereo: bool = typer.Option(False, "--stereo", "-s", help="Include STEREO-A EUVI analysis (~3.9 days ahead)"),
    minimal: bool = typer.Option(False, "--minimal", "-m", help="Minimal alert view (only actionable info)"),
    location: str = typer.Option(None, "--location", "-l", help="Location (berlin, tokyo, etc. or 'select')"),
    no_db: bool = typer.Option(False, "--no-db", help="Disable database storage"),
):
    """
    🔍 Single status check of all data sources.

    Use --minimal for operator view (only 193-211 + GOES).
    Use -l select for interactive location picker.
    Use -l berlin (or tokyo, london, etc.) for direct selection.
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    store_db = not no_db

    # Handle location selection
    if location == "select":
        location = _select_location_interactive()
        # Ask to save
        from rich.prompt import Confirm
        if Confirm.ask("Save as default location?", default=True):
            _save_location(location)
            console.print(f"[dim]Saved to ~/.config/solar-seed/location.txt[/]\n")
    elif location is None:
        # Try to use saved location
        location = _get_saved_location()

    # Minimal mode always requires coupling
    if minimal:
        coupling = True

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Fetching GOES X-ray...", total=None)
        xray = get_goes_xray()

        if not minimal:
            progress.add_task("Fetching solar wind...", total=None)
            solar_wind = get_dscovr_solar_wind()

            progress.add_task("Fetching NOAA alerts...", total=None)
            alerts = get_noaa_alerts()
        else:
            solar_wind = None
            alerts = None

        if store_db:
            store_goes_reading(xray)
            if solar_wind:
                risk, risk_level = assess_geomagnetic_risk(solar_wind)
                store_solar_wind_reading(solar_wind, risk, risk_level)
            if alerts:
                store_noaa_alerts(alerts)

        coupling_data = None
        if coupling:
            progress.add_task("Running coupling analysis...", total=None)
            coupling_data = run_coupling_analysis(xray=xray)
            if store_db and coupling_data:
                now = datetime.now(timezone.utc)
                store_coupling_reading(now.strftime("%Y-%m-%dT%H:%M:%S"), coupling_data, xray)

        stereo_data = None
        if stereo and not minimal:
            progress.add_task("Fetching STEREO-A data...", total=None)
            stereo_data = run_stereo_coupling_analysis()

    # Output mode
    fmt = StatusFormatter()
    if minimal:
        fmt.print_minimal_alert(coupling_data, xray, next_check_min=10)
    else:
        print_status_report(xray, solar_wind, alerts, coupling_data, stereo_data)

    # Personal relevance panel
    if location:
        fmt.print_personal_relevance(location)

    if store_db:
        fmt.print_footer(str(get_monitoring_db().db_path))


@app.command()
def monitor(
    interval: int = typer.Option(60, "--interval", "-i", help="Monitoring interval in seconds"),
    coupling: bool = typer.Option(False, "--coupling", "-c", help="Include coupling analysis"),
    stereo: bool = typer.Option(False, "--stereo", "-s", help="Include STEREO-A analysis"),
    no_db: bool = typer.Option(False, "--no-db", help="Disable database storage"),
):
    """
    📡 Continuous monitoring mode with periodic updates.
    """
    console.print(f"[bold cyan]Starting continuous monitoring[/] (interval: {interval}s)")
    console.print(f"[dim]Press Ctrl+C to stop[/]\n")

    monitor_loop(
        interval=interval,
        with_coupling=coupling,
        with_stereo=stereo,
        store_db=not no_db,
    )


@app.command()
def stats():
    """
    📊 Show database statistics.
    """
    from rich.table import Table

    db = get_monitoring_db()
    db_stats = db.get_database_stats()

    table = Table(title="📊 Solar Monitoring Database", box=box.ROUNDED)
    table.add_column("Table", style="bold")
    table.add_column("Count", justify="right")

    for name, count in db_stats.items():
        if isinstance(count, dict):
            table.add_row(name, str(count))
        else:
            table.add_row(name, f"{count:,} rows")

    console.print(table)
    console.print(f"[dim]Path: {db.db_path}[/]")


@app.command()
def correlations():
    """
    📈 Show coupling-flare correlations from database.
    """
    from rich.table import Table

    db = get_monitoring_db()
    corrs = db.get_coupling_before_flares(hours_before=6)

    if not corrs:
        console.print("[yellow]No correlations found. Run monitoring to collect data.[/]")
        return

    table = Table(title="📈 Coupling-Flare Correlations", box=box.ROUNDED)
    table.add_column("Pair")
    table.add_column("ΔMI", justify="right")
    table.add_column("Status")
    table.add_column("Hours Before")
    table.add_column("Flare")

    for c in corrs[:20]:
        table.add_row(
            c['pair'],
            f"{c['delta_mi']:.3f}",
            c['status'],
            f"{c['hours_before_flare']:.1f}h",
            f"{c['flare_class']}{c['flare_magnitude']:.1f}",
        )

    console.print(table)

    accuracy = db.get_prediction_accuracy()
    console.print(f"\n[bold]Prediction Accuracy:[/] {accuracy['overall']}")


@app.command(name="extract-flares")
def extract_flares(
    min_class: str = typer.Option("C", "--min-class", "-c", help="Minimum flare class (C, M, or X)"),
    gap: int = typer.Option(30, "--gap", "-g", help="Max gap in minutes between readings for same event"),
):
    """
    🔥 Extract flare events from GOES X-ray data.

    Groups consecutive elevated flux readings into discrete flare events.
    Stores in flare_events table for correlation analysis.
    """
    from rich.table import Table

    db = get_monitoring_db()

    console.print(f"[bold]Extracting flare events (>= {min_class}-class, gap={gap}min)...[/]\n")

    count = db.extract_flare_events_from_goes(min_class=min_class, gap_minutes=gap)

    if count == 0:
        console.print("[yellow]No new flare events found.[/]")
    else:
        console.print(f"[green]✓ Extracted {count} new flare events[/]\n")

    # Show all flare events
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT start_time, peak_time, class, magnitude, peak_flux
        FROM flare_events
        ORDER BY peak_time DESC
        LIMIT 20
    """)
    events = cursor.fetchall()

    if events:
        table = Table(title="🔥 Flare Events", box=box.ROUNDED)
        table.add_column("Peak Time")
        table.add_column("Class", justify="center")
        table.add_column("Peak Flux", justify="right")
        table.add_column("Duration")

        for e in events:
            start = e['start_time']
            peak = e['peak_time']
            flare_class = f"{e['class']}{e['magnitude']:.1f}"
            flux = f"{e['peak_flux']:.2e}" if e['peak_flux'] else "?"

            # Calculate duration if we have timestamps
            try:
                from datetime import datetime
                start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                peak_dt = datetime.fromisoformat(peak.replace('Z', '+00:00'))
                duration = "—"  # Would need end_time for proper duration
            except (ValueError, AttributeError):
                duration = "—"

            table.add_row(peak[:16].replace('T', ' '), flare_class, flux, duration)

        console.print(table)

    # Show total count
    cursor.execute("SELECT COUNT(*) FROM flare_events")
    total = cursor.fetchone()[0]
    console.print(f"\n[dim]Total flare events in database: {total}[/]")


@app.command(name="import-flares")
def import_flares(
    start: str = typer.Option("2026-01-01", "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option("2026-12-31", "--end", "-e", help="End date (YYYY-MM-DD)"),
    min_class: str = typer.Option("M", "--min-class", "-c", help="Minimum flare class (C, M, or X)"),
):
    """
    📥 Import historical flares from NASA DONKI archive.

    Downloads M/X-class flare data for correlation analysis with coupling measurements.
    """
    from rich.table import Table

    db = get_monitoring_db()

    console.print(f"[bold]Importing flares from NASA DONKI ({start} to {end}, >= {min_class}-class)...[/]\n")

    count = db.import_flares_from_donki(start_date=start, end_date=end, min_class=min_class)

    if count == 0:
        console.print("[yellow]No new flares imported.[/]")
    else:
        console.print(f"[green]✓ Imported {count} flares[/]\n")

    # Show summary by class
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT class, COUNT(*) as count,
               MIN(start_time) as first,
               MAX(start_time) as last
        FROM flare_events
        GROUP BY class
        ORDER BY class DESC
    """)
    stats = cursor.fetchall()

    table = Table(title="📊 Flare Database Summary", box=box.ROUNDED)
    table.add_column("Class")
    table.add_column("Count", justify="right")
    table.add_column("Date Range")

    for row in stats:
        first = row['first'][:10] if row['first'] else '?'
        last = row['last'][:10] if row['last'] else '?'
        table.add_row(
            f"{row['class']}-class",
            str(row['count']),
            f"{first} → {last}"
        )

    console.print(table)

    # Total
    cursor.execute("SELECT COUNT(*) FROM flare_events")
    total = cursor.fetchone()[0]
    console.print(f"\n[dim]Total flare events: {total}[/]")


@app.command(name="validate-divergences")
def validate_divergences(
    window: int = typer.Option(24, "--window", "-w", help="Hours window for flare matching"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-validate all (not just unvalidated)"),
):
    """
    🔬 Validate divergences against GOES flare history.

    Retrospectively classifies divergences as:
    - POST_EVENT: Flare occurred BEFORE divergence (structural aftermath)
    - PRECURSOR: Flare occurred AFTER divergence (predictive signal!)
    - BETWEEN_EVENTS: Flares both before and after
    - ISOLATED: No flares in window (noise or missed event)
    """
    from rich.table import Table

    db = get_monitoring_db()

    if force:
        # Reset validation status
        db.conn.execute("UPDATE phase_divergence SET validated = 0")
        db.conn.commit()
        console.print("[yellow]Reset validation status for all divergences[/]\n")

    console.print(f"[bold]Validating divergences against GOES flares (±{window}h window)...[/]\n")

    result = db.validate_divergences_against_flares(window_hours=window)

    # Summary table
    table = Table(title="🔬 Divergence Validation Results", box=box.ROUNDED)
    table.add_column("Classification", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Meaning")

    table.add_row(
        "POST_EVENT",
        str(result['post_event']),
        "[dim]Flare before divergence (structural aftermath)[/]"
    )
    table.add_row(
        "[bold yellow]PRECURSOR[/]",
        f"[bold yellow]{result['precursor']}[/]",
        "[yellow]Flare AFTER divergence (predictive signal!)[/]"
    )
    table.add_row(
        "BETWEEN_EVENTS",
        str(result['between_events']),
        "[dim]Flares both before and after[/]"
    )
    table.add_row(
        "ISOLATED",
        str(result['isolated']),
        "[dim]No flares in window[/]"
    )

    console.print(table)
    console.print(f"\n[green]✓ Validated {result['total_checked']} divergences[/]")

    # Show updated stats
    stats = db.get_divergence_statistics()
    console.print(f"\n[bold]Updated Statistics:[/]")
    console.print(f"  Total divergences: {stats['overall']['total_divergences']}")

    if result['precursor'] > 0:
        console.print(f"\n[bold green]🎯 {result['precursor']} PRECURSOR events found![/]")
        console.print("[green]These divergences preceded flares - potential early warning signals.[/]")


@app.command(name="extract-predictions")
def extract_predictions(
    window: float = typer.Option(6.0, "--window", "-w", help="Hours window for flare matching"),
):
    """
    🎯 Extract predictions from coupling data and verify against NOAA flares.

    This separates our predictions from NOAA ground truth:
    - predictions table: Our system's alerts/warnings
    - flare_events table: NOAA/DONKI flares (ground truth)
    """
    db = MonitoringDB()

    console.print("\n[bold]Extracting predictions from coupling measurements...[/]")

    # Extract predictions
    extracted = db.extract_predictions_from_coupling()
    console.print(f"  Extracted: [cyan]{extracted}[/] new predictions")

    # Verify against flares
    console.print(f"\n[bold]Verifying against NOAA flares (±{window}h window)...[/]")
    verify_result = db.verify_predictions_against_flares(window_hours=window)

    console.print(f"  Checked:   [cyan]{verify_result['total_checked']}[/] predictions")
    console.print(f"  Matched:   [green]{verify_result['matched']}[/] (true positives)")
    console.print(f"  Unmatched: [yellow]{verify_result['unmatched']}[/] (false positives)")

    if verify_result['avg_lead_time']:
        console.print(f"  Avg lead:  [cyan]{verify_result['avg_lead_time']:.1f}[/] hours")

    # Show summary
    console.print("\n[bold]Summary:[/]")
    summary = db.get_prediction_summary()

    from rich.table import Table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")

    table.add_row("Our Predictions", str(summary['predictions']['total_predictions']))
    table.add_row("  - Pre-Flare (TP)", f"[green]{summary['metrics']['true_positives']}[/]")
    table.add_row("  - Post-Flare", f"[dim]{summary['metrics']['post_flare_alerts']}[/]")
    table.add_row("  - False Positives", f"[yellow]{summary['metrics']['false_positives']}[/]")
    table.add_row("", "")
    table.add_row("Flares (monitoring period)", str(summary['metrics']['true_positives'] + summary['metrics']['false_negatives']))
    table.add_row("  - Detected (TP)", f"[green]{summary['metrics']['true_positives']}[/]")
    table.add_row("  - Missed (FN)", f"[red]{summary['metrics']['false_negatives']}[/]")
    table.add_row("", "")
    table.add_row("[bold]Precision[/]", f"[bold]{summary['metrics']['precision']:.1%}[/]")
    table.add_row("[bold]Recall[/]", f"[bold]{summary['metrics']['recall']:.1%}[/]")
    table.add_row("[bold]F1 Score[/]", f"[bold]{summary['metrics']['f1_score']:.2f}[/]")
    if summary['metrics']['avg_lead_time_hours']:
        table.add_row("Avg Lead Time", f"{summary['metrics']['avg_lead_time_hours']:.1f}h")

    console.print(table)


@app.command(name="show-predictions")
def show_predictions(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of predictions to show"),
):
    """
    📊 Show predictions vs NOAA flares comparison.

    Displays our system's predictions alongside actual NOAA flare events.
    """
    db = MonitoringDB()

    from rich.table import Table

    # Show predictions
    console.print("\n[bold]Recent Predictions (our system):[/]")
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT prediction_time, predicted_class, trigger_pair, trigger_status,
               actual_flare_id, lead_time_hours, verified
        FROM predictions
        ORDER BY prediction_time DESC
        LIMIT ?
    """, (limit,))

    preds = cursor.fetchall()

    if preds:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Time", style="cyan")
        table.add_column("Predicted", style="yellow")
        table.add_column("Pair")
        table.add_column("Status")
        table.add_column("Flare?", justify="center")
        table.add_column("Lead", justify="right")

        for p in preds:
            ts = p[0][:16] if p[0] else ""
            pred_cls = p[1] or ""
            pair = p[2] or ""
            status = p[3] or ""
            flare = "[green]✓[/]" if p[4] else "[red]✗[/]"
            lead = f"{p[5]:.1f}h" if p[5] else ""

            table.add_row(ts, pred_cls, pair, status, flare, lead)

        console.print(table)
    else:
        console.print("[yellow]No predictions yet. Run 'extract-predictions' first.[/]")

    # Show recent NOAA flares
    console.print("\n[bold]Recent NOAA Flares (ground truth):[/]")
    cursor.execute("""
        SELECT start_time, class, magnitude, source, location
        FROM flare_events
        ORDER BY start_time DESC
        LIMIT ?
    """, (limit,))

    flares = cursor.fetchall()

    if flares:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Time", style="cyan")
        table.add_column("Class", style="yellow")
        table.add_column("Source")
        table.add_column("Location")

        for f in flares:
            ts = f[0][:16] if f[0] else ""
            cls = f"{f[1]}{f[2]:.1f}" if f[1] and f[2] else ""
            src = f[3] or ""
            loc = f[4] or ""

            table.add_row(ts, cls, src, loc)

        console.print(table)

    # Show summary
    summary = db.get_prediction_summary()
    console.print(f"\n[bold]Metrics:[/] Precision={summary['metrics']['precision']:.1%}, "
                  f"Recall={summary['metrics']['recall']:.1%}, "
                  f"F1={summary['metrics']['f1_score']:.3f}")


@app.command(name="export")
def export_csv(
    table: str = typer.Argument(None, help="Table to export (or 'all')"),
    output: str = typer.Option("results/exports", "--output", "-o", help="Output path"),
    days: int = typer.Option(None, "--days", "-d", help="Limit to last N days"),
):
    """
    📤 Export database tables to CSV.

    Examples:
      export all                    # Export all tables
      export predictions            # Export predictions only
      export flare_events -d 30     # Last 30 days of flares
      export coupling_measurements  # All coupling data
    """
    from pathlib import Path

    db = MonitoringDB()

    if table is None or table == 'all':
        # Export all tables
        output_dir = Path(output)
        console.print(f"\n[bold]Exporting all tables to {output_dir}/[/]\n")

        results = db.export_all_csv(output_dir, days)

        for tbl, count in results.items():
            if count > 0:
                console.print(f"  [green]✓[/] {tbl}.csv: {count} rows")
            else:
                console.print(f"  [dim]- {tbl}.csv: empty[/]")

        total = sum(results.values())
        console.print(f"\n[bold]Total: {total} rows exported[/]")

    else:
        # Export single table
        output_path = Path(output)
        if not output_path.suffix:
            output_path = output_path / f"{table}.csv"

        console.print(f"\n[bold]Exporting {table} to {output_path}[/]\n")

        try:
            count = db.export_to_csv(table, output_path, days)
            console.print(f"  [green]✓[/] {count} rows exported")
        except ValueError as e:
            console.print(f"  [red]Error: {e}[/]")


@app.command()
def location(
    set_location: str = typer.Argument(None, help="Set location (berlin, tokyo, or lat,lon)"),
    show: bool = typer.Option(False, "--show", "-s", help="Show current sun status"),
):
    """
    📍 Set or show your default location.

    Examples:
      location           # Interactive selection
      location berlin    # Set to Berlin
      location 52.5,13.4 # Set custom coordinates
      location -s        # Show sun status for saved location
    """
    from solar_seed.monitoring.relevance import LOCATIONS, get_sun_status, get_daylight_window_utc

    if set_location is None and not show:
        # Interactive selection
        set_location = _select_location_interactive()
        _save_location(set_location)
        console.print(f"\n[green]✓ Location saved:[/] {set_location}")

    elif set_location:
        # Direct set
        _save_location(set_location)
        console.print(f"[green]✓ Location saved:[/] {set_location}")

    # Show status
    saved = _get_saved_location()
    if saved:
        console.print(f"\n[bold]Current location:[/] {saved}")

        # Parse and show sun status
        if ',' in saved:
            lat, lon = map(float, saved.split(','))
            loc_name = f"{lat:.1f}°, {lon:.1f}°"
        elif saved.lower() in LOCATIONS:
            lat, lon, _ = LOCATIONS[saved.lower()]
            loc_name = saved.title()
        else:
            console.print("[yellow]Unknown location format[/]")
            return

        sun = get_sun_status(lat, lon)
        sunrise, sunset = get_daylight_window_utc(lat, lon)

        if sun.is_visible:
            console.print(f"[yellow]☀️ Sun is VISIBLE[/] (altitude: {sun.altitude_deg:.0f}°)")
            console.print(f"[dim]→ You are on the day side - radio/GPS effects would affect you[/]")
        else:
            console.print(f"[cyan]🌙 Sun is BELOW HORIZON[/] (altitude: {sun.altitude_deg:.0f}°)")
            console.print(f"[dim]→ You are on the night side - only geomagnetic effects apply[/]")

        console.print(f"\n[dim]Daylight window: {sunrise}-{sunset} UTC[/]")
    else:
        console.print("[yellow]No location saved. Use 'location <name>' to set one.[/]")


# Import box for tables
from rich import box


def _notify_jsoc_outage(failed_timestamps: list[str], console):
    """Send email to JSOC support about drms_export.cgi download failures."""
    import smtplib
    from email.mime.text import MIMEText
    from datetime import datetime as _dt
    import platform

    ts_list = "\n".join(f"  - {ts}" for ts in failed_timestamps[:20])
    if len(failed_timestamps) > 20:
        ts_list += f"\n  ... and {len(failed_timestamps) - 20} more"

    body = f"""\
Dear JSOC team,

The automated Solar Seed backfill pipeline is unable to download AIA Level 1
FITS files from the DRMS export service.

Symptoms:
  - Fido.search() with Provider='JSOC' returns results normally
  - Fido.fetch() times out with "Timeout on reading data from socket"
  - Affected server: sdo7.nascom.nasa.gov (drms_export.cgi)

Failed timestamps (AIA 193/211/304 Angstrom):
{ts_list}

Total failures: {len(failed_timestamps)} consecutive timestamps
Time of report: {_dt.utcnow().isoformat()}Z
Host: {platform.node()}
SunPy version: see pip show sunpy

The search API confirms 4k data (4096x4096, ~65 MiB) is available for these
timestamps, but the CGI export endpoint does not respond.

Is there a known outage or maintenance window for drms_export.cgi?

Best regards,
Solar Seed Backfill (automated report)
https://github.com/lubeschanin/solar-seed
"""

    msg = MIMEText(body)
    msg["Subject"] = f"JSOC drms_export.cgi timeout — {len(failed_timestamps)} failed downloads"
    msg["From"] = "solar-seed@localhost"
    msg["To"] = "jsoc@sun.stanford.edu"

    # Save report to file (always)
    report_path = Path("results/early_warning/jsoc_outage_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"To: {msg['To']}\nSubject: {msg['Subject']}\n\n{body}")
    console.print(f"  [dim]Outage report saved: {report_path}[/]")

    # Try to send via local SMTP
    try:
        with smtplib.SMTP("localhost", timeout=10) as smtp:
            smtp.send_message(msg)
        console.print(f"  [green]Outage report emailed to jsoc@sun.stanford.edu[/]")
    except Exception:
        console.print(f"  [yellow]Could not send email (no local SMTP). Report saved to {report_path}[/]")
        console.print(f"  [yellow]Send manually: mail jsoc@sun.stanford.edu < {report_path}[/]")


@app.command()
def backfill(
    days: int = typer.Option(7, "--days", "-d", help="Days to look back"),
    status: bool = typer.Option(False, "--status", "-s", help="Show backfill status only"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Check availability without updating"),
    limit: int = typer.Option(0, "--limit", "-l", help="Max measurements to process (0=unlimited)"),
    check_jsoc: bool = typer.Option(False, "--check-jsoc", help="Check JSOC 4k availability"),
):
    """
    🔄 Backfill 1k measurements with 4k JSOC data.

    JSOC is the ONLY source for true 4096x4096 resolution (~65MB/file).
    Other providers (SDAC, etc.) only serve 1k despite claiming "FULLDISK".

    This corrects the +350% MI inflation in 304Å pairs from 1k synoptic data.

    Note: JSOC has been offline since Jan 8, 2026. Backfill will work once
    JSOC processes the backlog.

    Examples:
      backfill --status        # Show backfill statistics
      backfill --check-jsoc    # Check when JSOC 4k data is available
      backfill --dry-run       # Check what would be backfilled
      backfill --days 14       # Backfill last 14 days
    """
    from rich.table import Table

    db = get_monitoring_db()

    # Check JSOC availability
    if check_jsoc:
        try:
            from solar_seed.data_sources import get_jsoc_latest_date
            latest = get_jsoc_latest_date()
            if latest:
                console.print(f"\n[bold]📡 JSOC Status[/]")
                console.print(f"  Latest 4k data: {latest}")
                console.print(f"  [dim]Measurements after this date cannot be backfilled yet[/]")
            else:
                console.print("[yellow]⚠️ JSOC 4k data currently unavailable[/]")
        except Exception as e:
            console.print(f"[red]Error checking JSOC: {e}[/]")
        return

    # Show status
    if status:
        stats = db.get_backfill_stats()
        console.print("\n[bold]📊 Backfill Status[/]")
        console.print(f"  Total measurements: {stats['total']}")
        console.print(f"  Backfilled (4k):    {stats['backfilled_4k']}")
        console.print(f"  Pending (1k):       {stats['pending_1k']}")
        console.print(f"  Eligible (>3 days): {stats['eligible_for_backfill']}")

        # Also check JSOC
        try:
            from solar_seed.data_sources import get_jsoc_latest_date
            latest = get_jsoc_latest_date()
            console.print(f"\n  JSOC 4k available until: {latest or 'unavailable'}")
        except Exception:
            pass
        return

    # Get measurements to backfill
    measurements = db.get_measurements_for_backfill(
        min_age_days=0,  # Try all, JSOC check will filter
        max_age_days=days + 30,
        limit=limit or 0
    )

    if not measurements:
        console.print("[green]✓ No measurements need backfilling[/]")
        return

    # Group by timestamp
    by_timestamp = {}
    for m in measurements:
        ts = m['timestamp']
        if ts not in by_timestamp:
            by_timestamp[ts] = []
        by_timestamp[ts].append(m)

    console.print(f"\n[bold]🔄 Backfill: {len(by_timestamp)} timestamps, {len(measurements)} measurements[/]")

    if dry_run:
        console.print("[dim](dry run - checking JSOC 4k availability)[/]\n")

    # Import JSOC loader
    try:
        from solar_seed.data_sources import load_aia_jsoc, check_jsoc_4k_availability, get_jsoc_latest_date
        from solar_seed.radial_profile import subtract_radial_geometry
        from solar_seed.control_tests import sector_ring_shuffle_test
    except ImportError as e:
        console.print(f"[red]Import error: {e}[/]")
        return

    # Pre-flight: check if JSOC has any 4k data at all
    console.print("  Checking JSOC availability...")
    latest_jsoc = get_jsoc_latest_date()
    if not latest_jsoc:
        console.print("[yellow]  JSOC has no 4k data available (offline?). Nothing to backfill.[/]")
        return

    console.print(f"  JSOC 4k available until: {latest_jsoc}")

    # Filter timestamps to only those within JSOC range
    earliest_ts = min(by_timestamp.keys())
    if earliest_ts > latest_jsoc + "T23:59:59":
        console.print(f"[yellow]  All measurements ({earliest_ts[:10]}) are newer than JSOC data ({latest_jsoc}). Nothing to backfill.[/]")
        return

    # Process each timestamp
    updated = 0
    skipped = 0
    failed = 0
    consecutive_failures = 0
    failed_timestamps = []
    MAX_CONSECUTIVE_FAILURES = 5

    for ts, pairs in sorted(by_timestamp.items()):
        # Skip timestamps beyond JSOC availability (no network call needed)
        if ts[:10] > latest_jsoc:
            skipped += len(pairs)
            continue

        # Quick availability check for 4k
        if not check_jsoc_4k_availability(ts):
            console.print(f"  [dim]{ts}: JSOC 4k not available[/]")
            skipped += len(pairs)
            continue

        if dry_run:
            console.print(f"  [green]{ts}: JSOC 4k available ({len(pairs)} pairs)[/]")
            continue

        # Load 4k data
        channels, meta = load_aia_jsoc(ts, [193, 211, 304])
        if not channels:
            consecutive_failures += 1
            failed_timestamps.append(ts)
            console.print(f"  [yellow]{ts}: 4k load failed (not true 4k?) [{consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}][/]")
            failed += len(pairs)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                console.print(f"\n[red bold]Aborting: {MAX_CONSECUTIVE_FAILURES} consecutive download failures — JSOC likely down[/]")
                _notify_jsoc_outage(failed_timestamps, console)
                break
            continue

        consecutive_failures = 0  # Reset on success

        # Calculate MI for each pair
        for m in pairs:
            pair = m['pair']
            wl1, wl2 = map(int, pair.split('-'))

            if wl1 not in channels or wl2 not in channels:
                continue

            try:
                # Subtract radial geometry
                res1, _, _ = subtract_radial_geometry(channels[wl1])
                res2, _, _ = subtract_radial_geometry(channels[wl2])

                # Calculate delta_mi via sector-ring shuffle (same as monitoring)
                shuffle_result = sector_ring_shuffle_test(res1, res2, n_rings=10, n_sectors=12)
                new_mi = shuffle_result.mi_original - shuffle_result.mi_sector_shuffled

                # Update in DB
                db.update_measurement_backfill(
                    timestamp=ts,
                    pair=pair,
                    new_delta_mi=new_mi,
                    original_delta_mi=m['delta_mi']
                )

                change_pct = ((new_mi / m['delta_mi']) - 1) * 100 if m['delta_mi'] else 0
                console.print(f"  [green]✓[/] {ts} {pair}: {m['delta_mi']:.3f} → {new_mi:.3f} ({change_pct:+.0f}%)")
                updated += 1

            except Exception as e:
                console.print(f"  [red]✗[/] {ts} {pair}: {e}")
                failed += 1

    # Summary
    console.print(f"\n[bold]Summary:[/]")
    console.print(f"  Updated:  {updated}")
    console.print(f"  Skipped:  {skipped} (JSOC 4k not available)")
    console.print(f"  Failed:   {failed}")

    if failed_timestamps and not dry_run:
        report_path = Path("results/early_warning/jsoc_failures.log")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "a") as f:
            from datetime import datetime as _dt
            f.write(f"\n--- {_dt.now().isoformat()} ---\n")
            f.write(f"Failed timestamps ({len(failed_timestamps)}):\n")
            for ts in failed_timestamps:
                f.write(f"  {ts}\n")
        console.print(f"  [dim]Failure log: {report_path}[/]")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
