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
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError

# Global flag for graceful shutdown
_shutdown_requested = False

def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\n  Shutdown requested... (press again to force)")

# Register signal handler
signal.signal(signal.SIGINT, _signal_handler)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from monitoring_db import MonitoringDB

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


class CouplingMonitor:
    """Track coupling residuals over time for pre-flare detection."""

    # Baseline values from 8-day rotation analysis
    BASELINES = {
        '193-211': {'mean': 0.59, 'std': 0.12},
        '193-304': {'mean': 0.07, 'std': 0.02},
        '171-193': {'mean': 0.17, 'std': 0.04},
        '211-335': {'mean': 0.28, 'std': 0.06},
    }

    # Flare analysis showed -25% to -47% reduction during flares
    ALERT_THRESHOLD = -0.25  # 25% below baseline triggers warning

    def __init__(self, history_file: Path = None):
        self.history_file = history_file or Path("results/early_warning/coupling_history.json")
        self.history = self._load_history()

    def _load_history(self) -> list:
        """Load coupling history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    return json.load(f)
            except:
                pass
        return []

    def _save_history(self):
        """Save coupling history to file."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        # Keep last 24 hours (144 entries at 10min intervals)
        self.history = self.history[-144:]
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f)

    def compute_residual(self, pair: str, delta_mi: float) -> dict:
        """Compute residual r(t) = (ΔMI - baseline) / std."""
        if pair not in self.BASELINES:
            return {'residual': 0, 'deviation_pct': 0, 'status': 'unknown'}

        baseline = self.BASELINES[pair]
        residual = (delta_mi - baseline['mean']) / baseline['std']
        deviation_pct = (delta_mi - baseline['mean']) / baseline['mean']

        if deviation_pct < self.ALERT_THRESHOLD:
            status = 'ALERT'
        elif deviation_pct < -0.15:
            status = 'WARNING'
        elif deviation_pct < -0.10:
            status = 'ELEVATED'
        else:
            status = 'NORMAL'

        return {
            'residual': residual,
            'deviation_pct': deviation_pct,
            'status': status
        }

    def _theil_sen_slope(self, values: list) -> float:
        """Compute robust Theil-Sen median slope estimator."""
        n = len(values)
        if n < 2:
            return 0.0

        slopes = []
        for i in range(n):
            for j in range(i + 1, n):
                if j != i:
                    slopes.append((values[j] - values[i]) / (j - i))

        if not slopes:
            return 0.0

        slopes.sort()
        mid = len(slopes) // 2
        if len(slopes) % 2 == 0:
            return (slopes[mid - 1] + slopes[mid]) / 2
        return slopes[mid]

    def analyze_trend(self, pair: str) -> dict:
        """Analyze recent trend in coupling using robust Theil-Sen estimator."""
        pair_history = [h for h in self.history if pair in h.get('coupling', {})]
        n_available = len(pair_history)

        # Base result with metadata
        base_result = {
            'method': 'Theil-Sen',
            'interval_min': 10,  # Assumed interval between readings
            'window_max': 12,    # Max window size (2 hours)
        }

        # Minimum 3 points for any trend
        MIN_POINTS = 3
        if n_available < MIN_POINTS:
            if n_available == 0:
                return {
                    **base_result,
                    'trend': 'NO_DATA',
                    'slope_pct_per_hour': 0,
                    'n_points': 0,
                    'window_min': 0,
                    'confidence': 'none',
                    'reason': 'No readings available'
                }
            else:
                return {
                    **base_result,
                    'trend': 'COLLECTING',
                    'slope_pct_per_hour': 0,
                    'n_points': n_available,
                    'window_min': n_available * 10,
                    'confidence': 'insufficient',
                    'reason': f'Need {MIN_POINTS} points, have {n_available}'
                }

        # Rolling window: last 12 points (2 hours) or all available
        window_size = min(12, n_available)
        recent = pair_history[-window_size:]
        values = [h['coupling'][pair]['delta_mi'] for h in recent]
        n = len(values)

        # Calculate actual time span from timestamps
        try:
            from datetime import datetime
            t_first = datetime.fromisoformat(recent[0]['timestamp'].replace('Z', '+00:00'))
            t_last = datetime.fromisoformat(recent[-1]['timestamp'].replace('Z', '+00:00'))
            window_min = (t_last - t_first).total_seconds() / 60
        except:
            window_min = n * 10  # Fallback: assume 10min intervals

        # Robust Theil-Sen slope
        slope = self._theil_sen_slope(values)

        # Mean value for normalization
        y_mean = sum(values) / n if n > 0 else 1

        # Normalize slope to % per hour (assuming 10min intervals)
        slope_per_hour = slope * 6 / y_mean * 100 if y_mean else 0

        # Acceleration: compare first half vs second half slopes
        acceleration = 0
        if n >= 6:
            first_half = values[:n//2]
            second_half = values[n//2:]
            slope1 = self._theil_sen_slope(first_half)
            slope2 = self._theil_sen_slope(second_half)
            acceleration = (slope2 - slope1) / y_mean * 100 if y_mean else 0

        # Confidence based on sample size
        if n >= 9:
            confidence = 'high'
        elif n >= 6:
            confidence = 'medium'
        else:
            confidence = 'low'

        # Thresholds for trend classification
        EPSILON = 3.0      # %/hour for stable vs trending
        EPSILON_ACC = 2.0  # acceleration threshold

        # Determine trend label
        if abs(slope_per_hour) < EPSILON:
            trend = 'STABLE'
        elif slope_per_hour < -EPSILON:
            if acceleration < -EPSILON_ACC:
                trend = 'ACCELERATING_DOWN'  # Getting worse faster
            else:
                trend = 'DECLINING'
        else:  # slope_per_hour > EPSILON
            if acceleration > EPSILON_ACC:
                trend = 'ACCELERATING_UP'
            else:
                trend = 'RISING'

        return {
            **base_result,
            'trend': trend,
            'slope_pct_per_hour': slope_per_hour,
            'acceleration': acceleration,
            'n_points': n,
            'window_min': window_min,
            'confidence': confidence
        }

    def add_reading(self, timestamp: str, coupling_data: dict):
        """Add a new coupling reading to history."""
        self.history.append({
            'timestamp': timestamp,
            'coupling': coupling_data
        })
        self._save_history()

    def detect_transfer_state(self) -> dict | None:
        """
        Detect potential energy transfer between layers.

        TRANSFER_STATE: When chromospheric anchor (193-304) strengthens
        while coronal coupling (193-211) weakens - may indicate
        energy reorganization before flare.

        Returns dict with state info or None if not detected.
        """
        # Need trends for both pairs
        trend_304 = self.analyze_trend('193-304')
        trend_211 = self.analyze_trend('193-211')

        # Require at least medium confidence
        if trend_304.get('confidence') in ['none', 'low']:
            return None
        if trend_211.get('confidence') in ['none', 'low']:
            return None

        slope_304 = trend_304.get('slope_pct_per_hour', 0)
        slope_211 = trend_211.get('slope_pct_per_hour', 0)

        # Thresholds for transfer detection
        RISING_THRESHOLD = 3.0   # %/hour
        FALLING_THRESHOLD = -3.0  # %/hour

        # Transfer state: 304 rising while 211 falling
        if slope_304 > RISING_THRESHOLD and slope_211 < FALLING_THRESHOLD:
            return {
                'state': 'TRANSFER_STATE',
                'description': 'Chromospheric anchor strengthening, coronal coupling weakening',
                'slope_193_304': slope_304,
                'slope_193_211': slope_211,
                'confidence': min(trend_304['confidence'], trend_211['confidence']),
                'interpretation': 'Possible energy reorganization / magnetic stress buildup'
            }

        # Inverse: recovery after flare?
        if slope_304 < FALLING_THRESHOLD and slope_211 > RISING_THRESHOLD:
            return {
                'state': 'RECOVERY_STATE',
                'description': 'Coronal coupling recovering, chromospheric anchor releasing',
                'slope_193_304': slope_304,
                'slope_193_211': slope_211,
                'confidence': min(trend_304['confidence'], trend_211['confidence']),
                'interpretation': 'Possible post-flare recovery / relaxation'
            }

        return None


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


def load_aia_direct(timestamp: str, wavelengths: list[int]) -> dict | None:
    """Load AIA data directly via VSO (more reliable for recent data)."""
    try:
        from sunpy.net import Fido, attrs as a
        import astropy.units as u
        from sunpy.map import Map
        import tempfile
        import os

        dt = datetime.fromisoformat(timestamp)
        start = dt - timedelta(minutes=3)
        end = dt + timedelta(minutes=3)

        channels = {}
        for wl in wavelengths:
            print(f"    Fetching {wl} Å...")
            result = Fido.search(
                a.Time(start, end),
                a.Instrument('AIA'),
                a.Wavelength(wl * u.Angstrom)
            )

            if len(result) > 0 and len(result[0]) > 0:
                with tempfile.TemporaryDirectory() as tmpdir:
                    files = Fido.fetch(result[0, 0], path=tmpdir, progress=False)
                    if files:
                        smap = Map(files[0])
                        channels[wl] = smap.data
                        # Clean up
                        os.remove(files[0])

        return channels if channels else None

    except Exception as e:
        print(f"    VSO load error: {e}")
        return None


def run_coupling_analysis() -> dict | None:
    """Run quick coupling analysis on latest AIA data."""
    print("  Running coupling analysis (this may take a few minutes)...")

    try:
        from solar_seed.radial_profile import subtract_radial_geometry
        from solar_seed.control_tests import sector_ring_shuffle_test

        # Load data from 30 minutes ago (to ensure availability)
        now = datetime.now(timezone.utc) - timedelta(minutes=30)
        timestamp = now.strftime("%Y-%m-%dT%H:%M:00")

        print(f"  Loading AIA data for {timestamp}...")

        # Try direct VSO load first (more reliable for recent data)
        channels = load_aia_direct(timestamp, [193, 211, 304])

        if not channels or len(channels) < 2:
            print("  Could not load AIA data via VSO")
            # Fallback to multichannel loader
            try:
                from solar_seed.multichannel import load_aia_multichannel
                channels, _ = load_aia_multichannel(timestamp, wavelengths=[193, 211, 304])
            except:
                pass

        if not channels or len(channels) < 2:
            print("  Could not load AIA data")
            return None

        results = {}
        monitor = get_coupling_monitor()

        # Key pairs for monitoring
        pairs = [(193, 211), (193, 304), (171, 193)]

        for wl1, wl2 in pairs:
            if wl1 in channels and wl2 in channels:
                res1, _, _ = subtract_radial_geometry(channels[wl1])
                res2, _, _ = subtract_radial_geometry(channels[wl2])

                shuffle_result = sector_ring_shuffle_test(res1, res2, n_rings=10, n_sectors=12)
                delta_mi = shuffle_result.mi_original - shuffle_result.mi_sector_shuffled

                pair_key = f"{wl1}-{wl2}"
                residual_info = monitor.compute_residual(pair_key, delta_mi)
                trend_info = monitor.analyze_trend(pair_key)

                results[pair_key] = {
                    'delta_mi': delta_mi,
                    'mi_original': shuffle_result.mi_original,
                    'residual': residual_info['residual'],
                    'deviation_pct': residual_info['deviation_pct'],
                    'status': residual_info['status'],
                    'trend': trend_info['trend'],
                    'slope_pct_per_hour': trend_info.get('slope_pct_per_hour', 0)
                }

        # Save to history
        monitor.add_reading(timestamp, results)

        # Check for transfer state (debug label)
        transfer = monitor.detect_transfer_state()
        if transfer:
            results['_transfer_state'] = transfer

        return results

    except Exception as e:
        print(f"  Coupling analysis error: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_status_report(xray: dict, solar_wind: dict, alerts: list, coupling: dict = None):
    """Print formatted status report."""
    now = datetime.now(timezone.utc)

    print(f"\n{'='*70}")
    print(f"  SOLAR EARLY WARNING SYSTEM - {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*70}")

    # X-ray status
    print(f"\n  GOES X-RAY STATUS")
    print(f"  {'-'*40}")
    if xray:
        severity_icons = ['', '', '', '', '']
        icon = severity_icons[min(xray['severity'], 4)]
        print(f"  Flux:        {xray['flux']:.2e} W/m²")
        print(f"  Flare Class: {icon} {xray['flare_class']}")
        print(f"  Timestamp:   {xray['timestamp']}")

        if xray['severity'] >= 3:
            print(f"\n  *** FLARE ALERT: {xray['flare_class']} class flare detected! ***")
    else:
        print("  Data unavailable")

    # Solar wind status
    print(f"\n  SOLAR WIND (DSCOVR L1)")
    print(f"  {'-'*40}")
    if solar_wind:
        if 'plasma' in solar_wind:
            p = solar_wind['plasma']
            print(f"  Speed:       {p.get('speed', 'N/A')} km/s")
            print(f"  Density:     {p.get('density', 'N/A')} p/cm³")
        if 'mag' in solar_wind:
            m = solar_wind['mag']
            print(f"  Bz:          {m.get('bz', 'N/A')} nT")
            print(f"  Bt:          {m.get('bt', 'N/A')} nT")

        risk, risk_level = assess_geomagnetic_risk(solar_wind)
        risk_icons = ['', '', '', '']
        print(f"\n  Geomag Risk: {risk_icons[risk_level]} {risk}")
    else:
        print("  Data unavailable")

    # Coupling analysis with residual tracking
    if coupling:
        print(f"\n  ΔMI COUPLING MONITOR (Pre-Flare Detection)")
        print(f"  {'-'*40}")

        status_icons = {
            'NORMAL': '',
            'ELEVATED': '',
            'WARNING': '',
            'ALERT': ''
        }
        trend_icons = {
            'ACCELERATING_DOWN': '',
            'DECLINING': '',
            'STABLE': '',
            'RISING': '',
            'ACCELERATING_UP': '',
            'INITIALIZING': '',
            'NO_DATA': ''
        }
        confidence_markers = {
            'high': '',
            'medium': '',
            'low': '',
            'none': ''
        }

        any_alert = False
        for pair, data in coupling.items():
            if pair.startswith('_'):  # Skip internal fields like _transfer_state
                continue

            icon = status_icons.get(data.get('status', 'NORMAL'), '')
            trend = data.get('trend', 'NO_DATA')
            trend_icon = trend_icons.get(trend, '')
            conf = data.get('confidence', 'none')
            conf_marker = confidence_markers.get(conf, '')

            residual = data.get('residual', 0)
            slope = data.get('slope_pct_per_hour', 0)
            n_pts = data.get('n_points', 0)
            window_min = data.get('window_min', 0)
            method = data.get('method', 'Theil-Sen')

            print(f"  {pair} Å: {data['delta_mi']:.3f} bits  r={residual:+.1f}σ  {icon} {data.get('status', '?')}")

            # Show trend with full metadata
            if trend == 'NO_DATA':
                reason = data.get('reason', 'No data')
                print(f"           Trend: {trend_icon} {trend} — {reason}")
            elif trend == 'COLLECTING':
                reason = data.get('reason', '')
                print(f"           Trend: {trend_icon} {trend} — {reason}")
            else:
                acc = data.get('acceleration', 0)
                acc_str = f", acc={acc:+.1f}%/h²" if abs(acc) > 1 else ""
                # Format window time nicely
                if window_min >= 60:
                    window_str = f"{window_min/60:.1f}h"
                else:
                    window_str = f"{window_min:.0f}min"
                print(f"           Trend: {trend_icon} {trend} ({slope:+.1f}%/h{acc_str})")
                print(f"                  {conf_marker} {conf} confidence | n={n_pts} | {window_str} window | {method}")

            if data.get('status') in ['WARNING', 'ALERT']:
                any_alert = True

            # Special warning for accelerating decline
            if trend == 'ACCELERATING_DOWN' and data.get('status') in ['ELEVATED', 'WARNING', 'ALERT']:
                print(f"           ⚠ Coupling declining and accelerating!")

        if any_alert:
            print(f"\n  *** COUPLING ANOMALY DETECTED ***")
            print(f"  Reduced coupling may indicate magnetic stress buildup")
            print(f"  Monitor for potential flare activity in coming hours")

        # Transfer state detection (debug label)
        transfer = coupling.get('_transfer_state')
        if transfer:
            state_icons = {'TRANSFER_STATE': '', 'RECOVERY_STATE': ''}
            icon = state_icons.get(transfer['state'], '')
            print(f"\n  {icon} [{transfer['state']}] ({transfer['confidence']} confidence)")
            print(f"     {transfer['description']}")
            print(f"     193-304: {transfer['slope_193_304']:+.1f}%/h  193-211: {transfer['slope_193_211']:+.1f}%/h")
            print(f"     → {transfer['interpretation']}")

    # Active alerts
    if alerts:
        print(f"\n  NOAA ALERTS (last 24h)")
        print(f"  {'-'*40}")
        for alert in alerts[:3]:
            print(f"  [{alert['type']}] {alert['message'][:60]}...")

    print(f"\n{'='*70}\n")


def monitor_loop(interval: int = 60, with_coupling: bool = False, store_db: bool = True):
    """Continuous monitoring loop."""
    global _shutdown_requested
    _shutdown_requested = False

    print(f"\n  Starting continuous monitoring (interval: {interval}s)")
    if store_db:
        print(f"  Database: {get_monitoring_db().db_path}")
    print(f"  Press Ctrl+C to stop\n")

    coupling_interval = 600  # Run coupling every 10 minutes
    last_coupling = 0

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
                coupling = run_coupling_analysis()
                last_coupling = time.time()
                # Store coupling
                if store_db and coupling:
                    now = datetime.now(timezone.utc)
                    store_coupling_reading(now.strftime("%Y-%m-%dT%H:%M:%S"), coupling)

        if _shutdown_requested:
            break

        print_status_report(xray, solar_wind, alerts, coupling)

        # Alert on significant events
        if xray and xray['severity'] >= 3:
            print(f"\a")  # Terminal bell

        # Interruptible sleep (check every second)
        for _ in range(interval):
            if _shutdown_requested:
                break
            time.sleep(1)

    # Cleanup
    print("\n  Monitoring stopped.")
    if store_db:
        stats = get_monitoring_db().get_database_stats()
        print(f"  Database stats: {stats['goes_xray']} X-ray, {stats['solar_wind']} wind, {stats['coupling_measurements']} coupling")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--monitor', action='store_true',
                        help='Continuous monitoring mode')
    parser.add_argument('--interval', type=int, default=60,
                        help='Monitoring interval in seconds (default: 60)')
    parser.add_argument('--coupling', action='store_true',
                        help='Include AIA coupling analysis')
    parser.add_argument('--no-db', action='store_true',
                        help='Disable database storage')
    parser.add_argument('--db-stats', action='store_true',
                        help='Show database statistics and exit')
    parser.add_argument('--correlations', action='store_true',
                        help='Show coupling-flare correlations from database')
    args = parser.parse_args()

    # Database stats mode
    if args.db_stats:
        db = get_monitoring_db()
        stats = db.get_database_stats()
        print("\n  Solar Monitoring Database")
        print("  " + "=" * 50)
        print(f"  Path: {db.db_path}")
        print("\n  Table Statistics:")
        for table, count in stats.items():
            if isinstance(count, dict):
                print(f"    {table}: {count}")
            else:
                print(f"    {table}: {count} rows")
        return 0

    # Correlations mode
    if args.correlations:
        db = get_monitoring_db()
        print("\n  Coupling-Flare Correlations")
        print("  " + "=" * 50)
        correlations = db.get_coupling_before_flares(hours_before=6)
        if correlations:
            for c in correlations[:20]:
                print(f"  {c['pair']}: ΔMI={c['delta_mi']:.3f}, status={c['status']}")
                print(f"    → {c['hours_before_flare']:.1f}h before {c['flare_class']}{c['flare_magnitude']:.1f} flare")
        else:
            print("  No correlations found. Run monitoring to collect data.")

        accuracy = db.get_prediction_accuracy()
        print(f"\n  Prediction Accuracy: {accuracy['overall']}")
        return 0

    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║          SOLAR EARLY WARNING SYSTEM - Prototype v0.2                  ║
╚═══════════════════════════════════════════════════════════════════════╝

  Data Sources:
    - GOES X-ray flux (NOAA SWPC)
    - DSCOVR solar wind plasma & magnetic field (L1)
    - NOAA Space Weather Alerts
    - SDO/AIA coupling analysis (optional)
""")

    store_db = not args.no_db

    if args.monitor:
        monitor_loop(interval=args.interval, with_coupling=args.coupling, store_db=store_db)
    else:
        # Single check
        xray = get_goes_xray()
        solar_wind = get_dscovr_solar_wind()
        alerts = get_noaa_alerts()

        # Store in database
        if store_db:
            store_goes_reading(xray)
            if solar_wind:
                risk, risk_level = assess_geomagnetic_risk(solar_wind)
                store_solar_wind_reading(solar_wind, risk, risk_level)

        coupling = None
        if args.coupling:
            coupling = run_coupling_analysis()
            if store_db and coupling:
                now = datetime.now(timezone.utc)
                store_coupling_reading(now.strftime("%Y-%m-%dT%H:%M:%S"), coupling)

        print_status_report(xray, solar_wind, alerts, coupling)

        if store_db:
            print(f"  Data stored in: {get_monitoring_db().db_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
