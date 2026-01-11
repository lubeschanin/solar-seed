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
                    recent.append({
                        'type': alert.get('product_id', 'unknown'),
                        'message': alert.get('message', '')[:200],
                        'issued': alert['issue_datetime']
                    })
        except (KeyError, ValueError, TypeError):
            continue

    return recent[:5]  # Top 5 recent


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


def store_coupling_reading(timestamp: str, coupling: dict):
    """Store coupling measurements in database."""
    if not coupling:
        return
    db = get_monitoring_db()

    for pair, data in coupling.items():
        # Skip internal metadata fields
        if pair.startswith('_'):
            continue
        db.insert_coupling(
            timestamp=timestamp,
            pair=pair,
            delta_mi=data['delta_mi'],
            mi_original=data.get('mi_original'),
            residual=data.get('residual'),
            deviation_pct=data.get('deviation_pct'),
            status=data.get('status'),
            trend=data.get('trend'),
            slope_pct_per_hour=data.get('slope_pct_per_hour'),
            acceleration=data.get('acceleration'),
            confidence=data.get('confidence'),
            n_points=data.get('n_points')
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


def _load_channels(wavelengths: list[int], use_synoptic: bool = True) -> tuple[dict | None, str | None, dict | None, str | None]:
    """
    Load AIA channel data with fallback strategy.

    Returns:
        (channels, timestamp, quality_info, data_source) or (None, None, None, None) on failure
    """
    channels = None
    timestamp = None
    quality_info = None
    data_source = None

    # Primary: Try synoptic data (fast, reliable, no queue)
    if use_synoptic:
        print("  Trying AIA synoptic data (1k, direct access)...")
        channels, timestamp, quality_info = load_aia_synoptic(wavelengths)
        if channels and len(channels) >= 2:
            data_source = 'synoptic'
            print(f"  ✓ Using synoptic data from: {timestamp}")

    # Fallback 1: Full-res via VSO
    if not channels or len(channels) < 2:
        print("  Synoptic unavailable, trying full-res via VSO...")
        channels, timestamp, quality_info = load_aia_latest(wavelengths, max_age_minutes=30)
        if channels and len(channels) >= 2:
            data_source = 'full-res'

    # Fallback 2: Direct load (legacy)
    if not channels or len(channels) < 2:
        print("  VSO unavailable, trying direct fallback...")
        fallback_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        timestamp = fallback_time.strftime("%Y-%m-%dT%H:%M:00")
        channels = load_aia_direct(timestamp, wavelengths)
        quality_info = None
        if channels and len(channels) >= 2:
            data_source = 'direct-fallback'

    if not channels or len(channels) < 2:
        print("  ✗ Could not load AIA data from any source")
        return None, None, None, None

    print(f"  Data source: {data_source}")
    return channels, timestamp, quality_info, data_source


def _analyze_pair(wl1: int, wl2: int, channels: dict, monitor, validate_breaks: bool,
                  subtract_radial_geometry, sector_ring_shuffle_test) -> dict:
    """
    Analyze a single channel pair: compute MI, detect breaks, run validation.

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
            else:
                change = robust.get('change_pct', 0)
                print(f"     ⚠ Break is UNRELIABLE (binning Δ={change:.1f}% > 20%)")
                break_check['is_break'] = False
                break_check['vetoed'] = 'robustness'
                output['break_detection'] = break_check

    residual_info = monitor.compute_residual(pair_key, delta_mi)
    trend_info = monitor.analyze_trend(pair_key)

    output['result'] = {
        'delta_mi': delta_mi,
        'mi_original': shuffle_result.mi_original,
        'residual': residual_info['residual'],
        'deviation_pct': residual_info['deviation_pct'],
        'status': residual_info['status'],
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


def run_coupling_analysis(validate_breaks: bool = True, xray: dict = None, use_synoptic: bool = True) -> dict | None:
    """
    Run quick coupling analysis on latest AIA data with quality checks.

    Hybrid Approach:
    - Primary: AIA synoptic data (1024x1024, direct access, no queue)
    - Fallback: Full-res via VSO (4096x4096, may be rate-limited)

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

        pairs = [(193, 211), (193, 304), (171, 193)]
        for wl1, wl2 in pairs:
            if wl1 in channels and wl2 in channels:
                pair_output = _analyze_pair(
                    wl1, wl2, channels, monitor, validate_breaks,
                    subtract_radial_geometry, sector_ring_shuffle_test
                )
                pair_key = pair_output['pair_key']
                results[pair_key] = pair_output['result']
                break_detections[pair_key] = pair_output['break_detection']
                registration_checks[pair_key] = pair_output['registration_check']
                if pair_output['robustness_check']:
                    robustness_checks[pair_key] = pair_output['robustness_check']
                quality_warnings.extend(pair_output['quality_warnings'])
                artifact_warnings.extend(pair_output['artifact_warnings'])

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
            'resolution': '1024x1024' if data_source == 'synoptic' else '4096x4096',
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

        coupling = None
        if with_coupling and (time.time() - last_coupling) > coupling_interval:
            if not _shutdown_requested:
                coupling = run_coupling_analysis(xray=xray)
                last_coupling = time.time()
                # Store coupling
                if store_db and coupling:
                    now = datetime.now(timezone.utc)
                    store_coupling_reading(now.strftime("%Y-%m-%dT%H:%M:%S"), coupling)

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


@app.command()
def check(
    coupling: bool = typer.Option(False, "--coupling", "-c", help="Include SDO/AIA coupling analysis"),
    stereo: bool = typer.Option(False, "--stereo", "-s", help="Include STEREO-A EUVI analysis (~3.9 days ahead)"),
    minimal: bool = typer.Option(False, "--minimal", "-m", help="Minimal alert view (only actionable info)"),
    location: str = typer.Option(None, "--location", "-l", help="Your location (berlin, london, tokyo, or lat,lon)"),
    no_db: bool = typer.Option(False, "--no-db", help="Disable database storage"),
):
    """
    🔍 Single status check of all data sources.

    Use --minimal for operator view (only 193-211 + GOES).
    Use --location to see personal relevance (day/night, aurora).
    Use without --minimal for full scientific dashboard.
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    store_db = not no_db

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

        coupling_data = None
        if coupling:
            progress.add_task("Running coupling analysis...", total=None)
            coupling_data = run_coupling_analysis(xray=xray)
            if store_db and coupling_data:
                now = datetime.now(timezone.utc)
                store_coupling_reading(now.strftime("%Y-%m-%dT%H:%M:%S"), coupling_data)

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


# Import box for tables
from rich import box


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
