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
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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

    def analyze_trend(self, pair: str) -> dict:
        """Analyze recent trend in coupling for a pair."""
        pair_history = [h for h in self.history if pair in h.get('coupling', {})]

        if len(pair_history) < 3:
            return {'trend': 'insufficient_data', 'slope': 0}

        # Last 6 readings (1 hour at 10min intervals)
        recent = pair_history[-6:]
        values = [h['coupling'][pair]['delta_mi'] for h in recent]

        # Simple linear trend
        n = len(values)
        if n < 2:
            return {'trend': 'insufficient_data', 'slope': 0}

        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        slope = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        slope /= sum((i - x_mean) ** 2 for i in range(n)) or 1

        # Normalize slope to % per hour
        slope_per_hour = slope * 6 / y_mean * 100 if y_mean else 0

        if slope_per_hour < -10:
            trend = 'DROPPING'
        elif slope_per_hour < -5:
            trend = 'DECLINING'
        elif slope_per_hour > 10:
            trend = 'RISING'
        elif slope_per_hour > 5:
            trend = 'INCREASING'
        else:
            trend = 'STABLE'

        return {'trend': trend, 'slope_pct_per_hour': slope_per_hour}

    def add_reading(self, timestamp: str, coupling_data: dict):
        """Add a new coupling reading to history."""
        self.history.append({
            'timestamp': timestamp,
            'coupling': coupling_data
        })
        self._save_history()


# Global monitor instance
_coupling_monitor = None

def get_coupling_monitor() -> CouplingMonitor:
    global _coupling_monitor
    if _coupling_monitor is None:
        _coupling_monitor = CouplingMonitor()
    return _coupling_monitor


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
            'DROPPING': '',
            'DECLINING': '',
            'STABLE': '',
            'INCREASING': '',
            'RISING': '',
            'insufficient_data': ''
        }

        any_alert = False
        for pair, data in coupling.items():
            icon = status_icons.get(data.get('status', 'NORMAL'), '')
            trend_icon = trend_icons.get(data.get('trend', 'STABLE'), '')

            residual = data.get('residual', 0)
            deviation = data.get('deviation_pct', 0) * 100
            trend = data.get('trend', 'unknown')

            print(f"  {pair} Å: {data['delta_mi']:.3f} bits  r={residual:+.1f}σ  {icon} {data.get('status', '?')}")
            print(f"           Trend: {trend_icon} {trend} ({data.get('slope_pct_per_hour', 0):+.1f}%/h)")

            if data.get('status') in ['WARNING', 'ALERT']:
                any_alert = True

        if any_alert:
            print(f"\n  *** COUPLING ANOMALY DETECTED ***")
            print(f"  Reduced coupling may indicate magnetic stress buildup")
            print(f"  Monitor for potential flare activity in coming hours")

    # Active alerts
    if alerts:
        print(f"\n  NOAA ALERTS (last 24h)")
        print(f"  {'-'*40}")
        for alert in alerts[:3]:
            print(f"  [{alert['type']}] {alert['message'][:60]}...")

    print(f"\n{'='*70}\n")


def monitor_loop(interval: int = 60, with_coupling: bool = False):
    """Continuous monitoring loop."""
    print(f"\n  Starting continuous monitoring (interval: {interval}s)")
    print(f"  Press Ctrl+C to stop\n")

    coupling_interval = 600  # Run coupling every 10 minutes
    last_coupling = 0

    while True:
        try:
            xray = get_goes_xray()
            solar_wind = get_dscovr_solar_wind()
            alerts = get_noaa_alerts()

            coupling = None
            if with_coupling and (time.time() - last_coupling) > coupling_interval:
                coupling = run_coupling_analysis()
                last_coupling = time.time()

            print_status_report(xray, solar_wind, alerts, coupling)

            # Alert on significant events
            if xray and xray['severity'] >= 3:
                print(f"\a")  # Terminal bell

            time.sleep(interval)

        except KeyboardInterrupt:
            print("\n  Monitoring stopped.")
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--monitor', action='store_true',
                        help='Continuous monitoring mode')
    parser.add_argument('--interval', type=int, default=60,
                        help='Monitoring interval in seconds (default: 60)')
    parser.add_argument('--coupling', action='store_true',
                        help='Include AIA coupling analysis')
    args = parser.parse_args()

    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║          SOLAR EARLY WARNING SYSTEM - Prototype v0.1                  ║
╚═══════════════════════════════════════════════════════════════════════╝

  Data Sources:
    - GOES X-ray flux (NOAA SWPC)
    - DSCOVR solar wind plasma & magnetic field (L1)
    - NOAA Space Weather Alerts
    - SDO/AIA coupling analysis (optional)
""")

    if args.monitor:
        monitor_loop(interval=args.interval, with_coupling=args.coupling)
    else:
        # Single check
        xray = get_goes_xray()
        solar_wind = get_dscovr_solar_wind()
        alerts = get_noaa_alerts()

        coupling = None
        if args.coupling:
            coupling = run_coupling_analysis()

        print_status_report(xray, solar_wind, alerts, coupling)

    return 0


if __name__ == "__main__":
    sys.exit(main())
