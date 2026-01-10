#!/usr/bin/env python3
"""
Flare Event Analysis for Solar Seed
====================================

Analyzes ΔMI_sector before, during and after flare events.

Hypothesis:
- BEFORE:  Baseline coupling
- DURING:  Strong increase (especially 94-131 Å)
- AFTER:   Decay back to baseline

Known flares for testing:
- X5.0: 2024-01-01 00:55 UTC (AR 3536)
- X2.8: 2023-12-14 17:02 UTC
- X1.0: 2024-01-10 15:40 UTC
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from pathlib import Path
from datetime import datetime, timedelta
import json

from solar_seed.multichannel import (
    AIA_CHANNELS, WAVELENGTHS, WAVELENGTH_TO_TEMP,
    analyze_pair, load_aia_multichannel,
    generate_multichannel_sun, AIA_DATA_SOURCE
)


# ============================================================================
# KNOWN FLARE EVENTS
# ============================================================================

KNOWN_FLARES = {
    "X5.0_2024-01-01": {
        "peak_time": "2024-01-01T00:55:00",
        "class": "X5.0",
        "location": "AR 3536",
        "description": "Major X-class flare, strong in 94/131 Å"
    },
    "X2.8_2023-12-14": {
        "peak_time": "2023-12-14T17:02:00",
        "class": "X2.8",
        "location": "AR 3514",
        "description": "Strong X-class flare"
    },
    "X1.0_2024-01-10": {
        "peak_time": "2024-01-10T15:40:00",
        "class": "X1.0",
        "location": "AR 3536",
        "description": "Moderate X-class flare"
    },
    "M5.0_2024-01-22": {
        "peak_time": "2024-01-22T23:30:00",
        "class": "M5.0",
        "location": "AR 3559",
        "description": "Strong M-class flare"
    }
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FlarePhase:
    """Results for a flare phase."""
    phase: str  # "before", "during", "after"
    n_samples: int
    timestamps: List[str]

    # Mean values per pair
    pair_values: Dict[Tuple[int, int], float]  # ΔMI_sector
    pair_stds: Dict[Tuple[int, int], float]

    # 94Å intensity as activity indicator
    mean_94A_intensity: float


@dataclass
class FlareAnalysisResult:
    """Overall result of flare analysis."""
    flare_id: str
    flare_class: str
    peak_time: str

    # Phases
    before: FlarePhase
    during: FlarePhase
    after: FlarePhase

    # Changes
    coupling_change: Dict[Tuple[int, int], Dict[str, float]]  # Pair -> {before_to_during, during_to_after}

    # Top pairs by change
    most_affected: List[Tuple[Tuple[int, int], float]]


# ============================================================================
# SYNTHETIC FLARE SIMULATION
# ============================================================================

def generate_flare_timeseries(
    n_before: int = 10,
    n_during: int = 5,
    n_after: int = 10,
    flare_intensity: float = 3.0,
    seed: int = 42
) -> Tuple[List[Tuple[Dict[int, NDArray], str]], List[str]]:
    """
    Generates synthetic flare timeseries.

    Args:
        n_before: Timepoints before flare
        n_during: Timepoints during flare
        n_after: Timepoints after flare
        flare_intensity: Amplification factor for flare channels
        seed: Random seed

    Returns:
        (timeseries, phases) - Timeseries and phase per timepoint
    """
    rng = np.random.default_rng(seed)
    results = []
    phases = []

    base_time = datetime.now()
    t_idx = 0

    # BEFORE: normal sun
    for i in range(n_before):
        timestamp = (base_time + timedelta(minutes=2 * t_idx)).isoformat()
        channels = generate_multichannel_sun(
            n_active_regions=3 + rng.integers(0, 3),
            seed=seed + t_idx
        )
        results.append((channels, timestamp))
        phases.append("before")
        t_idx += 1

    # DURING: amplified flare channels
    for i in range(n_during):
        timestamp = (base_time + timedelta(minutes=2 * t_idx)).isoformat()
        channels = generate_multichannel_sun(
            n_active_regions=5 + rng.integers(0, 3),
            seed=seed + t_idx
        )

        # Amplify 94 and 131 Å (flare channels)
        # Amplification increases towards the middle (Gaussian profile)
        peak_factor = np.exp(-((i - n_during/2)**2) / (n_during/2))
        intensity = 1.0 + (flare_intensity - 1.0) * peak_factor

        for wl in [94, 131]:
            mask = channels[wl] > 0
            channels[wl][mask] *= intensity
            # Additional flare structure
            channels[wl][mask] += rng.normal(0, 500 * intensity, mask.sum())

        results.append((channels, timestamp))
        phases.append("during")
        t_idx += 1

    # AFTER: decaying activity
    for i in range(n_after):
        timestamp = (base_time + timedelta(minutes=2 * t_idx)).isoformat()

        # Exponential decay
        decay = np.exp(-i / 5)
        n_regions = int(5 - 2 * (1 - decay))

        channels = generate_multichannel_sun(
            n_active_regions=max(3, n_regions),
            seed=seed + t_idx
        )

        # Slightly elevated residual activity
        for wl in [94, 131]:
            mask = channels[wl] > 0
            channels[wl][mask] *= (1.0 + 0.5 * decay)

        results.append((channels, timestamp))
        phases.append("after")
        t_idx += 1

    return results, phases


# ============================================================================
# REAL DATA LOADING
# ============================================================================

def load_flare_timeseries(
    peak_time: str,
    minutes_before: int = 30,
    minutes_after: int = 30,
    cadence_minutes: int = 2,
    data_dir: str = "data/aia",
    verbose: bool = True
) -> Tuple[List[Tuple[Dict[int, NDArray], str]], List[str]]:
    """
    Loads real AIA data around a flare event.

    Args:
        peak_time: Peak time of the flare (ISO format)
        minutes_before: Minutes before peak
        minutes_after: Minutes after peak
        cadence_minutes: Time interval
        data_dir: Download directory
        verbose: Verbose output

    Returns:
        (timeseries, phases)
    """
    peak = datetime.fromisoformat(peak_time.replace('Z', '+00:00'))

    results = []
    phases = []

    # Calculate timepoints
    n_before = minutes_before // cadence_minutes
    n_after = minutes_after // cadence_minutes

    # Window for "during" = Peak ± 5 minutes
    during_window = 5

    total = n_before + n_after + 1

    if verbose:
        print(f"  📥 Loading {total} timepoints around {peak_time[:19]}...")

    for i in range(-n_before, n_after + 1):
        t = peak + timedelta(minutes=cadence_minutes * i)
        timestamp = t.isoformat()

        # Determine phase
        minutes_from_peak = abs(i * cadence_minutes)
        if minutes_from_peak <= during_window:
            phase = "during"
        elif i < 0:
            phase = "before"
        else:
            phase = "after"

        if verbose:
            print(f"    {phase:>6}: {timestamp[:19]}")

        channels, metadata = load_aia_multichannel(
            timestamp,
            data_dir=data_dir
        )

        if channels is not None:
            results.append((channels, timestamp))
            phases.append(phase)
        elif verbose:
            print(f"    ⚠️  Skipped")

    return results, phases


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_flare_phase(
    timeseries: List[Tuple[Dict[int, NDArray], str]],
    phase_name: str,
    seed: int = 42
) -> FlarePhase:
    """
    Analyzes a single flare phase.
    """
    if len(timeseries) == 0:
        return FlarePhase(
            phase=phase_name,
            n_samples=0,
            timestamps=[],
            pair_values={},
            pair_stds={},
            mean_94A_intensity=0.0
        )

    # Collect values per pair
    pair_values: Dict[Tuple[int, int], List[float]] = {
        pair: [] for pair in combinations(WAVELENGTHS, 2)
    }
    intensities_94 = []
    timestamps = []

    for t_idx, (channels, timestamp) in enumerate(timeseries):
        timestamps.append(timestamp)

        # 94Å intensity
        intensities_94.append(float(np.mean(channels[94][channels[94] > 0])))

        # Analyze all pairs
        for wl1, wl2 in combinations(WAVELENGTHS, 2):
            result = analyze_pair(
                channels[wl1], channels[wl2],
                wl1, wl2,
                bins=64,
                seed=seed + t_idx
            )
            pair_values[(wl1, wl2)].append(result.delta_mi_sector)

    return FlarePhase(
        phase=phase_name,
        n_samples=len(timeseries),
        timestamps=timestamps,
        pair_values={p: float(np.mean(v)) for p, v in pair_values.items()},
        pair_stds={p: float(np.std(v)) for p, v in pair_values.items()},
        mean_94A_intensity=float(np.mean(intensities_94))
    )


def run_flare_analysis(
    flare_id: str = None,
    peak_time: str = None,
    minutes_before: int = 30,
    minutes_after: int = 30,
    cadence_minutes: int = 2,
    use_real_data: bool = False,
    output_dir: str = "results/flare",
    verbose: bool = True
) -> FlareAnalysisResult:
    """
    Runs complete flare analysis.

    Args:
        flare_id: ID from KNOWN_FLARES (e.g. "X5.0_2024-01-01")
        peak_time: Alternatively: direct peak time
        minutes_before/after: Analysis window
        cadence_minutes: Time interval
        use_real_data: Use real AIA data
        output_dir: Output directory
        verbose: Verbose output

    Returns:
        FlareAnalysisResult
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Determine flare info
    if flare_id and flare_id in KNOWN_FLARES:
        flare_info = KNOWN_FLARES[flare_id]
        peak_time = flare_info["peak_time"]
        flare_class = flare_info["class"]
    else:
        flare_id = "custom"
        flare_class = "unknown"
        if peak_time is None:
            peak_time = datetime.now().isoformat()

    if verbose:
        print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║              🔥 FLARE EVENT ANALYSIS 🌞                                ║
╚═══════════════════════════════════════════════════════════════════════╝

  Flare:    {flare_id} ({flare_class})
  Peak:     {peak_time[:19]}
  Window:   -{minutes_before}min ... +{minutes_after}min
  Cadence:  {cadence_minutes}min
  Data:     {'Real (AIA)' if use_real_data else 'Synthetic'}
""")

    # Load data
    if use_real_data:
        timeseries, phases = load_flare_timeseries(
            peak_time=peak_time,
            minutes_before=minutes_before,
            minutes_after=minutes_after,
            cadence_minutes=cadence_minutes,
            verbose=verbose
        )
    else:
        n_before = minutes_before // cadence_minutes
        n_after = minutes_after // cadence_minutes
        n_during = 5  # ~10 Minuten Peak

        timeseries, phases = generate_flare_timeseries(
            n_before=n_before,
            n_during=n_during,
            n_after=n_after,
            flare_intensity=3.0
        )

    if len(timeseries) == 0:
        raise RuntimeError("No data loaded")

    if verbose:
        print(f"\n  📊 Analyzing {len(timeseries)} timepoints...")

    # Group by phase
    before_data = [(ch, ts) for (ch, ts), p in zip(timeseries, phases) if p == "before"]
    during_data = [(ch, ts) for (ch, ts), p in zip(timeseries, phases) if p == "during"]
    after_data = [(ch, ts) for (ch, ts), p in zip(timeseries, phases) if p == "after"]

    if verbose:
        print(f"     Before: {len(before_data)}, During: {len(during_data)}, After: {len(after_data)}")

    # Analyze each phase
    before_result = analyze_flare_phase(before_data, "before")
    during_result = analyze_flare_phase(during_data, "during")
    after_result = analyze_flare_phase(after_data, "after")

    # Calculate changes
    coupling_change = {}
    for pair in combinations(WAVELENGTHS, 2):
        before_val = before_result.pair_values.get(pair, 0)
        during_val = during_result.pair_values.get(pair, 0)
        after_val = after_result.pair_values.get(pair, 0)

        # Percentage change
        before_to_during = ((during_val - before_val) / before_val * 100) if before_val > 0 else 0
        during_to_after = ((after_val - during_val) / during_val * 100) if during_val > 0 else 0

        coupling_change[pair] = {
            "before_to_during": before_to_during,
            "during_to_after": during_to_after,
            "before": before_val,
            "during": during_val,
            "after": after_val
        }

    # Top-5 by increase during flare
    sorted_by_change = sorted(
        coupling_change.items(),
        key=lambda x: x[1]["before_to_during"],
        reverse=True
    )
    most_affected = [(pair, data["before_to_during"]) for pair, data in sorted_by_change[:5]]

    result = FlareAnalysisResult(
        flare_id=flare_id,
        flare_class=flare_class,
        peak_time=peak_time,
        before=before_result,
        during=during_result,
        after=after_result,
        coupling_change=coupling_change,
        most_affected=most_affected
    )

    # Save and print
    save_flare_results(result, out_path)

    if verbose:
        print_flare_summary(result)

    return result


def save_flare_results(result: FlareAnalysisResult, output_dir: Path) -> None:
    """Saves flare results."""

    with open(output_dir / "flare_analysis.txt", "w") as f:
        f.write("FLARE EVENT ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        f.write("DATA SOURCE:\n")
        f.write(f"  Instrument:   {AIA_DATA_SOURCE['instrument']}\n")
        f.write(f"  Operator:     {AIA_DATA_SOURCE['operator']}\n")
        f.write(f"  Data:         {AIA_DATA_SOURCE['data_provider']}\n")
        f.write(f"  URL:          {AIA_DATA_SOURCE['data_url']}\n")
        f.write(f"  Reference:    {AIA_DATA_SOURCE['reference']}\n\n")

        f.write(f"Flare: {result.flare_id} ({result.flare_class})\n")
        f.write(f"Peak:  {result.peak_time}\n\n")

        f.write("PHASE OVERVIEW:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Phase':<10} {'n':<5} {'94Å Intensity':<15} {'ΔMI 94-131'}\n")
        f.write("-" * 50 + "\n")

        for phase in [result.before, result.during, result.after]:
            mi_94_131 = phase.pair_values.get((94, 131), 0)
            f.write(f"{phase.phase:<10} {phase.n_samples:<5} "
                    f"{phase.mean_94A_intensity:<15.1f} {mi_94_131:.4f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("\nCHANGES DURING FLARE (Before → During):\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Pair':<12} {'Before':<10} {'During':<10} {'After':<10} {'Change'}\n")
        f.write("-" * 60 + "\n")

        for pair, data in sorted(result.coupling_change.items(),
                                  key=lambda x: -x[1]["before_to_during"]):
            f.write(f"{pair[0]}-{pair[1]:<7} "
                    f"{data['before']:<10.4f} "
                    f"{data['during']:<10.4f} "
                    f"{data['after']:<10.4f} "
                    f"{data['before_to_during']:+.1f}%\n")

    # JSON
    data = {
        "flare_id": result.flare_id,
        "flare_class": result.flare_class,
        "peak_time": result.peak_time,
        "phases": {
            "before": {
                "n_samples": result.before.n_samples,
                "mean_94A": result.before.mean_94A_intensity,
                "pair_values": {f"{p[0]}-{p[1]}": v for p, v in result.before.pair_values.items()}
            },
            "during": {
                "n_samples": result.during.n_samples,
                "mean_94A": result.during.mean_94A_intensity,
                "pair_values": {f"{p[0]}-{p[1]}": v for p, v in result.during.pair_values.items()}
            },
            "after": {
                "n_samples": result.after.n_samples,
                "mean_94A": result.after.mean_94A_intensity,
                "pair_values": {f"{p[0]}-{p[1]}": v for p, v in result.after.pair_values.items()}
            }
        },
        "coupling_change": {
            f"{p[0]}-{p[1]}": data for p, data in result.coupling_change.items()
        },
        "most_affected": [
            {"pair": f"{p[0]}-{p[1]}", "change_percent": c}
            for p, c in result.most_affected
        ],
        "data_source": AIA_DATA_SOURCE
    }

    with open(output_dir / "flare_analysis.json", "w") as f:
        json.dump(data, f, indent=2)


def print_flare_summary(result: FlareAnalysisResult) -> None:
    """Prints summary."""

    # 94-131 Werte
    before_94_131 = result.before.pair_values.get((94, 131), 0)
    during_94_131 = result.during.pair_values.get((94, 131), 0)
    after_94_131 = result.after.pair_values.get((94, 131), 0)

    change_94_131 = result.coupling_change.get((94, 131), {}).get("before_to_during", 0)

    print(f"""
  ════════════════════════════════════════════════════════════════════════

  PHASE COMPARISON:

    Phase      n    94Å Intensity     ΔMI_sector (94-131)
    ─────────────────────────────────────────────────────
    BEFORE    {result.before.n_samples:>2}    {result.before.mean_94A_intensity:>10.1f}         {before_94_131:.4f} bits
    DURING    {result.during.n_samples:>2}    {result.during.mean_94A_intensity:>10.1f}         {during_94_131:.4f} bits
    AFTER     {result.after.n_samples:>2}    {result.after.mean_94A_intensity:>10.1f}         {after_94_131:.4f} bits

  ────────────────────────────────────────────────────────────────────────

  TOP 5 AFFECTED PAIRS (Before → During):
""")

    for i, (pair, change) in enumerate(result.most_affected, 1):
        print(f"    {i}. {pair[0]}-{pair[1]} Å: {change:+.1f}%")

    print(f"""
  ────────────────────────────────────────────────────────────────────────

  FLARE CHANNEL COUPLING (94-131 Å):

    Before:  {before_94_131:.4f} bits
    During:  {during_94_131:.4f} bits  ({change_94_131:+.1f}%)
    After:   {after_94_131:.4f} bits

  INTERPRETATION:
    {'✓ Strong flare effect: coupling increases during flare.' if change_94_131 > 50 else
     '~ Moderate flare effect.' if change_94_131 > 20 else
     '○ Weak flare effect.'}

═══════════════════════════════════════════════════════════════════════
""")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Flare Event Analysis for Solar Seed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Known flares:
  X5.0_2024-01-01  - Strong X-flare
  X2.8_2023-12-14  - X-flare
  X1.0_2024-01-10  - Moderate X-flare
  M5.0_2024-01-22  - M-flare

Examples:
  python -m solar_seed.flare_analysis
  python -m solar_seed.flare_analysis --flare X5.0_2024-01-01 --real
  python -m solar_seed.flare_analysis --peak "2024-01-01T00:55:00" --real
        """
    )
    parser.add_argument("--flare", type=str, default=None,
                        help="Flare ID from known flares")
    parser.add_argument("--peak", type=str, default=None,
                        help="Peak time (ISO format)")
    parser.add_argument("--before", type=int, default=30,
                        help="Minutes before peak (default: 30)")
    parser.add_argument("--after", type=int, default=30,
                        help="Minutes after peak (default: 30)")
    parser.add_argument("--cadence", type=int, default=2,
                        help="Cadence in minutes (default: 2)")
    parser.add_argument("--real", action="store_true",
                        help="Use real AIA data")
    parser.add_argument("--output", type=str, default="results/flare",
                        help="Output directory")
    parser.add_argument("--list", action="store_true",
                        help="List known flares")

    args = parser.parse_args()

    if args.list:
        print("\nKnown flares:")
        print("-" * 60)
        for fid, info in KNOWN_FLARES.items():
            print(f"  {fid:<20} {info['class']:<6} {info['peak_time'][:19]}")
        print()
        return

    run_flare_analysis(
        flare_id=args.flare,
        peak_time=args.peak,
        minutes_before=args.before,
        minutes_after=args.after,
        cadence_minutes=args.cadence,
        use_real_data=args.real,
        output_dir=args.output,
        verbose=True
    )


if __name__ == "__main__":
    main()
