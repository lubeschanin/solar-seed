#!/usr/bin/env python3
"""
Final Analyses for Solar Seed
=============================

Two concluding analyses:

1. TIMESCALE COMPARISON (24h vs 27d)
   - Does the temperature ordering persist across different timescales?
   - Spearman correlation of rankings
   - Stability as validation

2. ACTIVITY CONDITIONING
   - ΔMI_sector vs 94Å proxy
   - Conditioning on quiet (low 94Å) vs active (high 94Å)
   - Shows physical coupling between activity and structure
"""

# Fix for macOS: Avoid fork crash with async libraries (SunPy/aiohttp)
# Must be BEFORE all other imports!
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=False)
except RuntimeError:
    pass  # Already set

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from pathlib import Path
from datetime import datetime, timedelta
import json

from scipy import stats

from solar_seed.multichannel import (
    AIA_CHANNELS, WAVELENGTHS, WAVELENGTH_TO_TEMP,
    generate_multichannel_sun, analyze_pair, PairResult,
    load_aia_multichannel_timeseries, AIA_DATA_SOURCE
)


# ============================================================================
# PARALLEL PROCESSING
# ============================================================================

def _analyze_pair_worker(args: Tuple) -> Tuple[Tuple[int, int], float]:
    """
    Worker function for parallel pair analysis.
    Must be at module level for pickling.

    Args:
        args: Tuple of (image_1, image_2, wl1, wl2, bins, seed)

    Returns:
        Tuple of ((wl1, wl2), delta_mi_sector)
    """
    image_1, image_2, wl1, wl2, bins, seed = args
    result = analyze_pair(image_1, image_2, wl1, wl2, bins=bins, seed=seed)
    return ((wl1, wl2), result.delta_mi_sector)


def analyze_pairs_parallel(
    channels: Dict[int, NDArray[np.float64]],
    bins: int = 64,
    seed: int = 42,
    n_workers: Optional[int] = None
) -> Dict[Tuple[int, int], float]:
    """
    Analyze all 21 wavelength pairs in parallel (for small images) or
    sequentially (for large images to avoid memory issues).

    Args:
        channels: Dict mapping wavelength -> image data
        bins: Number of bins for MI calculation
        seed: Base random seed
        n_workers: Number of parallel workers (default: CPU count - 1)

    Returns:
        Dict mapping (wl1, wl2) -> delta_mi_sector
    """
    pairs = list(combinations(WAVELENGTHS, 2))

    # For large images (>1024x1024), use sequential to avoid memory issues
    # with process serialization of ~500MB per timepoint
    first_channel = next(iter(channels.values()))
    use_sequential = first_channel.shape[0] > 1024

    if use_sequential:
        # Sequential for large (real AIA) images
        results = {}
        for i, (wl1, wl2) in enumerate(pairs):
            result = analyze_pair(
                channels[wl1], channels[wl2], wl1, wl2,
                bins=bins, seed=seed + i
            )
            results[(wl1, wl2)] = result.delta_mi_sector
        return results

    # Parallel processing for smaller (synthetic) images
    import os
    from concurrent.futures import ProcessPoolExecutor

    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)

    args_list = [
        (channels[wl1], channels[wl2], wl1, wl2, bins, seed + i)
        for i, (wl1, wl2) in enumerate(pairs)
    ]

    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for pair, delta_mi in executor.map(_analyze_pair_worker, args_list):
            results[pair] = delta_mi

    return results


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TimescaleResult:
    """Result of the timescale comparison."""
    timescale_hours: float
    n_points: int
    pair_rankings: Dict[Tuple[int, int], int]  # Pair -> Rank
    pair_values: Dict[Tuple[int, int], float]  # Pair -> ΔMI_sector
    timestamp: str


@dataclass
class TimescaleComparison:
    """Comparison between different timescales."""
    short_scale: TimescaleResult
    long_scale: TimescaleResult

    # Correlations
    spearman_rho: float
    spearman_p: float
    kendall_tau: float
    kendall_p: float

    # Stability
    top5_overlap: int  # How many of the top-5 pairs match?
    rank_changes: Dict[Tuple[int, int], int]  # Rank differences


@dataclass
class ActivityBin:
    """Result for an activity range."""
    bin_label: str  # "quiet", "moderate", "active"
    mean_94A_intensity: float
    n_samples: int
    pair_values: Dict[Tuple[int, int], float]  # Pair -> mean ΔMI_sector
    pair_stds: Dict[Tuple[int, int], float]


@dataclass
class ActivityConditioningResult:
    """Result of activity conditioning."""
    bins: List[ActivityBin]

    # Correlation between activity and coupling
    activity_vs_coupling: Dict[Tuple[int, int], Tuple[float, float]]  # Pair -> (r, p)

    # Strongest signal
    most_activity_dependent: List[Tuple[Tuple[int, int], float]]  # Top-5


# ============================================================================
# TIMESCALE ANALYSIS
# ============================================================================

def analyze_timescale(
    n_hours: float,
    cadence_minutes: int = 12,
    seed: int = 42,
    use_real_data: bool = False,
    start_time_str: Optional[str] = None,
    verbose: bool = True
) -> TimescaleResult:
    """
    Analyzes a timescale and returns pair rankings.

    Args:
        n_hours: Time period
        cadence_minutes: Cadence
        seed: Random Seed
        use_real_data: Use real data
        start_time_str: Start time
        verbose: Verbose output

    Returns:
        TimescaleResult with rankings
    """
    n_points = max(1, int(n_hours * 60 / cadence_minutes))

    if verbose:
        print(f"  📊 Analyzing {n_hours}h ({n_points} timepoints)...")

    # Generate or load data
    if use_real_data:
        if start_time_str is None:
            start_time_str = (datetime.now() - timedelta(hours=n_hours + 24)).isoformat()

        timeseries = load_aia_multichannel_timeseries(
            start_time=start_time_str,
            n_points=n_points,
            cadence_minutes=cadence_minutes,
            verbose=verbose
        )
    else:
        from solar_seed.multichannel import generate_multichannel_timeseries
        timeseries = generate_multichannel_timeseries(n_points, seed=seed)

    # Collect results
    pair_values: Dict[Tuple[int, int], List[float]] = {
        pair: [] for pair in combinations(WAVELENGTHS, 2)
    }

    for t_idx, (channels, _) in enumerate(timeseries):
        for wl1, wl2 in combinations(WAVELENGTHS, 2):
            result = analyze_pair(
                channels[wl1], channels[wl2],
                wl1, wl2,
                bins=64,
                seed=seed + t_idx
            )
            pair_values[(wl1, wl2)].append(result.delta_mi_sector)

    # Mean values
    mean_values = {pair: float(np.mean(vals)) for pair, vals in pair_values.items()}

    # Rankings
    sorted_pairs = sorted(mean_values.items(), key=lambda x: -x[1])
    rankings = {pair: rank + 1 for rank, (pair, _) in enumerate(sorted_pairs)}

    return TimescaleResult(
        timescale_hours=n_hours,
        n_points=n_points,
        pair_rankings=rankings,
        pair_values=mean_values,
        timestamp=datetime.now().isoformat()
    )


def compare_timescales(
    short_result: TimescaleResult,
    long_result: TimescaleResult
) -> TimescaleComparison:
    """
    Compares two timescales and calculates correlations.
    """
    pairs = list(short_result.pair_rankings.keys())

    short_ranks = [short_result.pair_rankings[p] for p in pairs]
    long_ranks = [long_result.pair_rankings[p] for p in pairs]

    # Spearman correlation
    spearman_rho, spearman_p = stats.spearmanr(short_ranks, long_ranks)

    # Kendall Tau
    kendall_tau, kendall_p = stats.kendalltau(short_ranks, long_ranks)

    # Top-5 Overlap
    short_top5 = set(p for p, r in short_result.pair_rankings.items() if r <= 5)
    long_top5 = set(p for p, r in long_result.pair_rankings.items() if r <= 5)
    top5_overlap = len(short_top5 & long_top5)

    # Rank differences
    rank_changes = {
        pair: abs(short_result.pair_rankings[pair] - long_result.pair_rankings[pair])
        for pair in pairs
    }

    return TimescaleComparison(
        short_scale=short_result,
        long_scale=long_result,
        spearman_rho=spearman_rho,
        spearman_p=spearman_p,
        kendall_tau=kendall_tau,
        kendall_p=kendall_p,
        top5_overlap=top5_overlap,
        rank_changes=rank_changes
    )


def run_timescale_comparison(
    short_hours: float = 24.0,
    long_hours: float = 648.0,  # 27 days
    cadence_minutes: int = 12,
    seed: int = 42,
    output_dir: str = "results/final",
    use_real_data: bool = False,
    verbose: bool = True
) -> TimescaleComparison:
    """
    Runs the timescale comparison.

    Args:
        short_hours: Short timescale (default: 24h)
        long_hours: Long timescale (default: 27 days = 648h)
        cadence_minutes: Cadence
        seed: Random Seed
        output_dir: Output directory
        use_real_data: Real data
        verbose: Verbose output

    Returns:
        TimescaleComparison
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║            📊 TIMESCALE COMPARISON (FINAL ANALYSIS 1)                  ║
╚═══════════════════════════════════════════════════════════════════════╝

  Question: Does the temperature ordering remain stable across different timescales?

  Timescales:
    Short: {short_hours}h
    Long:  {long_hours}h ({long_hours/24:.0f} days)
""")

    # Analyze both timescales
    short_result = analyze_timescale(
        short_hours,
        cadence_minutes=cadence_minutes,
        seed=seed,
        use_real_data=use_real_data,
        verbose=verbose
    )

    long_result = analyze_timescale(
        long_hours,
        cadence_minutes=cadence_minutes * 10,  # Coarser cadence for long timescale
        seed=seed + 1000,
        use_real_data=use_real_data,
        verbose=verbose
    )

    # Compare
    comparison = compare_timescales(short_result, long_result)

    # Save and print
    save_timescale_results(comparison, out_path)

    if verbose:
        print_timescale_summary(comparison)

    return comparison


def save_timescale_results(result: TimescaleComparison, output_dir: Path) -> None:
    """Saves timescale results."""

    with open(output_dir / "timescale_comparison.txt", "w") as f:
        f.write("TIMESCALE COMPARISON\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Short timescale: {result.short_scale.timescale_hours}h "
                f"({result.short_scale.n_points} timepoints)\n")
        f.write(f"Long timescale: {result.long_scale.timescale_hours}h "
                f"({result.long_scale.n_points} timepoints)\n\n")

        f.write("CORRELATIONS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Spearman rho: {result.spearman_rho:.4f} (p = {result.spearman_p:.2e})\n")
        f.write(f"Kendall tau:  {result.kendall_tau:.4f} (p = {result.kendall_p:.2e})\n\n")

        f.write(f"Top-5 Overlap: {result.top5_overlap}/5\n\n")

        f.write("RANKING COMPARISON:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Pair':<12} {'Rank (short)':<12} {'Rank (long)':<12} {'Difference'}\n")
        f.write("-" * 70 + "\n")

        pairs = sorted(result.short_scale.pair_rankings.keys(),
                       key=lambda p: result.short_scale.pair_rankings[p])

        for pair in pairs:
            short_r = result.short_scale.pair_rankings[pair]
            long_r = result.long_scale.pair_rankings[pair]
            diff = result.rank_changes[pair]

            f.write(f"{pair[0]}-{pair[1]:<7} {short_r:<12} {long_r:<12} "
                    f"{'+' if long_r > short_r else '-' if long_r < short_r else '='}{abs(diff)}\n")

    # JSON
    data = {
        "short_scale": {
            "hours": result.short_scale.timescale_hours,
            "n_points": result.short_scale.n_points,
            "rankings": {f"{p[0]}-{p[1]}": r for p, r in result.short_scale.pair_rankings.items()},
            "values": {f"{p[0]}-{p[1]}": v for p, v in result.short_scale.pair_values.items()}
        },
        "long_scale": {
            "hours": result.long_scale.timescale_hours,
            "n_points": result.long_scale.n_points,
            "rankings": {f"{p[0]}-{p[1]}": r for p, r in result.long_scale.pair_rankings.items()},
            "values": {f"{p[0]}-{p[1]}": v for p, v in result.long_scale.pair_values.items()}
        },
        "correlations": {
            "spearman_rho": result.spearman_rho,
            "spearman_p": result.spearman_p,
            "kendall_tau": result.kendall_tau,
            "kendall_p": result.kendall_p
        },
        "top5_overlap": result.top5_overlap,
        "rank_changes": {f"{p[0]}-{p[1]}": c for p, c in result.rank_changes.items()}
    }

    with open(output_dir / "timescale_comparison.json", "w") as f:
        json.dump(data, f, indent=2)


def print_timescale_summary(result: TimescaleComparison) -> None:
    """Prints summary."""

    stability = "STABLE" if result.spearman_rho > 0.8 else "VARIABLE" if result.spearman_rho > 0.5 else "UNSTABLE"

    print(f"""
  ════════════════════════════════════════════════════════════════════════

  RESULT:

    Spearman rho = {result.spearman_rho:.4f}  (p = {result.spearman_p:.2e})
    Kendall tau  = {result.kendall_tau:.4f}  (p = {result.kendall_p:.2e})

    Top-5 Overlap: {result.top5_overlap}/5 pairs match

    -> Ordering is {stability}

  INTERPRETATION:
    {'+ Temperature coupling remains preserved across timescales.' if result.spearman_rho > 0.7 else
     '! Coupling varies with timescale - dynamic effects.' if result.spearman_rho > 0.4 else
     '- No stable ordering - coupling is time-dependent.'}
    {'  This supports the physical interpretation.' if result.spearman_rho > 0.7 else ''}

═══════════════════════════════════════════════════════════════════════
""")


# ============================================================================
# ACTIVITY CONDITIONING
# ============================================================================

def run_activity_conditioning(
    n_hours: float = 48.0,
    cadence_minutes: int = 12,
    n_bins: int = 3,
    seed: int = 42,
    output_dir: str = "results/final",
    use_real_data: bool = False,
    verbose: bool = True
) -> ActivityConditioningResult:
    """
    Runs activity conditioning.

    Uses 94A as proxy for solar activity and calculates
    delta_MI_sector for different activity levels.

    Args:
        n_hours: Time period
        cadence_minutes: Cadence
        n_bins: Number of activity bins
        seed: Random Seed
        output_dir: Output directory
        use_real_data: Real data
        verbose: Verbose output

    Returns:
        ActivityConditioningResult
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n_points = max(1, int(n_hours * 60 / cadence_minutes))

    if verbose:
        print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║          🔥 ACTIVITY CONDITIONING (FINAL ANALYSIS 2)                   ║
╚═══════════════════════════════════════════════════════════════════════╝

  Question: Does delta_MI_sector correlate with solar activity (94A proxy)?

  Method:
    1. Calculate mean 94A intensity per timepoint
    2. Divide timepoints into {n_bins} activity bins
    3. Calculate delta_MI_sector per bin for all pairs
    4. Correlate activity with coupling

  Time period: {n_hours}h ({n_points} timepoints)
""")

    # Generate or load data
    if use_real_data:
        start_time_str = (datetime.now() - timedelta(hours=n_hours + 24)).isoformat()
        timeseries = load_aia_multichannel_timeseries(
            start_time=start_time_str,
            n_points=n_points,
            cadence_minutes=cadence_minutes,
            verbose=verbose
        )
    else:
        from solar_seed.multichannel import generate_multichannel_timeseries
        timeseries = generate_multichannel_timeseries(n_points, seed=seed)

    if verbose:
        print(f"  📊 Analyzing {len(timeseries)} timepoints...")

    # Collect all data
    all_data: List[Dict] = []

    for t_idx, (channels, timestamp) in enumerate(timeseries):
        # 94Å intensity as activity proxy
        intensity_94 = float(np.mean(channels[94][channels[94] > 0]))

        # Analyze all pairs
        pair_results = {}
        for wl1, wl2 in combinations(WAVELENGTHS, 2):
            result = analyze_pair(
                channels[wl1], channels[wl2],
                wl1, wl2,
                bins=64,
                seed=seed + t_idx
            )
            pair_results[(wl1, wl2)] = result.delta_mi_sector

        all_data.append({
            "timestamp": timestamp,
            "intensity_94": intensity_94,
            "pair_results": pair_results
        })

    # Divide into activity bins
    intensities = np.array([d["intensity_94"] for d in all_data])
    percentiles = np.percentile(intensities, [100/n_bins * i for i in range(1, n_bins)])

    bin_labels = ["quiet", "moderate", "active"] if n_bins == 3 else [f"bin_{i}" for i in range(n_bins)]

    bins: List[ActivityBin] = []

    for bin_idx in range(n_bins):
        if bin_idx == 0:
            mask = intensities <= percentiles[0]
        elif bin_idx == n_bins - 1:
            mask = intensities > percentiles[-1]
        else:
            mask = (intensities > percentiles[bin_idx-1]) & (intensities <= percentiles[bin_idx])

        bin_data = [all_data[i] for i in range(len(all_data)) if mask[i]]

        if len(bin_data) == 0:
            continue

        # Mean values per pair
        pair_values: Dict[Tuple[int, int], float] = {}
        pair_stds: Dict[Tuple[int, int], float] = {}

        for pair in combinations(WAVELENGTHS, 2):
            values = [d["pair_results"][pair] for d in bin_data]
            pair_values[pair] = float(np.mean(values))
            pair_stds[pair] = float(np.std(values))

        bins.append(ActivityBin(
            bin_label=bin_labels[bin_idx],
            mean_94A_intensity=float(np.mean([d["intensity_94"] for d in bin_data])),
            n_samples=len(bin_data),
            pair_values=pair_values,
            pair_stds=pair_stds
        ))

    if verbose:
        print(f"  📈 Computing correlations...")

    # Correlation between activity and coupling per pair
    activity_vs_coupling: Dict[Tuple[int, int], Tuple[float, float]] = {}

    for pair in combinations(WAVELENGTHS, 2):
        activities = [d["intensity_94"] for d in all_data]
        couplings = [d["pair_results"][pair] for d in all_data]

        r, p = stats.pearsonr(activities, couplings)
        activity_vs_coupling[pair] = (r, p)

    # Top-5 activity-dependent pairs
    sorted_by_correlation = sorted(
        activity_vs_coupling.items(),
        key=lambda x: abs(x[1][0]),
        reverse=True
    )
    most_activity_dependent = [(pair, r) for pair, (r, p) in sorted_by_correlation[:5]]

    result = ActivityConditioningResult(
        bins=bins,
        activity_vs_coupling=activity_vs_coupling,
        most_activity_dependent=most_activity_dependent
    )

    # Save and print
    save_activity_results(result, out_path)

    if verbose:
        print_activity_summary(result)

    return result


def save_activity_results(result: ActivityConditioningResult, output_dir: Path) -> None:
    """Saves activity results."""

    with open(output_dir / "activity_conditioning.txt", "w") as f:
        f.write("ACTIVITY CONDITIONING\n")
        f.write("=" * 70 + "\n\n")

        f.write("94A as proxy for solar activity\n\n")

        f.write("ACTIVITY BINS:\n")
        f.write("-" * 70 + "\n")

        for bin in result.bins:
            f.write(f"\n{bin.bin_label.upper()} (n={bin.n_samples}, "
                    f"mean 94Å={bin.mean_94A_intensity:.1f}):\n")

            sorted_pairs = sorted(bin.pair_values.items(), key=lambda x: -x[1])
            for pair, value in sorted_pairs[:5]:
                f.write(f"  {pair[0]}-{pair[1]}: ΔMI_sector = {value:.4f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("\nCORRELATION ACTIVITY <-> COUPLING:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Pair':<12} {'Pearson r':<12} {'p-value':<15} {'Interpretation'}\n")
        f.write("-" * 50 + "\n")

        for pair, (r, p) in sorted(result.activity_vs_coupling.items(),
                                    key=lambda x: -abs(x[1][0])):
            if abs(r) > 0.5:
                interp = "strong" + (" positive" if r > 0 else " negative")
            elif abs(r) > 0.3:
                interp = "moderate"
            else:
                interp = "weak"

            f.write(f"{pair[0]}-{pair[1]:<7} {r:<12.4f} {p:<15.2e} {interp}\n")

    # JSON
    data = {
        "bins": [
            {
                "label": bin.bin_label,
                "mean_94A_intensity": bin.mean_94A_intensity,
                "n_samples": bin.n_samples,
                "pair_values": {f"{p[0]}-{p[1]}": v for p, v in bin.pair_values.items()},
                "pair_stds": {f"{p[0]}-{p[1]}": v for p, v in bin.pair_stds.items()}
            }
            for bin in result.bins
        ],
        "correlations": {
            f"{p[0]}-{p[1]}": {"r": r, "p": p_val}
            for p, (r, p_val) in result.activity_vs_coupling.items()
        },
        "most_activity_dependent": [
            {"pair": f"{p[0]}-{p[1]}", "r": r}
            for p, r in result.most_activity_dependent
        ]
    }

    with open(output_dir / "activity_conditioning.json", "w") as f:
        json.dump(data, f, indent=2)


def print_activity_summary(result: ActivityConditioningResult) -> None:
    """Prints summary."""

    print(f"""
  ════════════════════════════════════════════════════════════════════════

  ACTIVITY BINS:
""")

    for bin in result.bins:
        print(f"    {bin.bin_label.upper():>10}: n={bin.n_samples:>3}, 94Å={bin.mean_94A_intensity:>8.1f}")

    print(f"""
  ────────────────────────────────────────────────────────────────────────

  TOP 5 ACTIVITY-DEPENDENT PAIRS:
""")

    for i, (pair, r) in enumerate(result.most_activity_dependent, 1):
        direction = "+" if r > 0 else "-"
        print(f"    {i}. {pair[0]}-{pair[1]} Å: r = {direction}{abs(r):.3f}")

    # Compare quiet vs active for top pair
    if len(result.bins) >= 2:
        top_pair = result.most_activity_dependent[0][0]
        quiet_val = result.bins[0].pair_values.get(top_pair, 0)
        active_val = result.bins[-1].pair_values.get(top_pair, 0)
        change = (active_val - quiet_val) / quiet_val * 100 if quiet_val > 0 else 0

        print(f"""
  ────────────────────────────────────────────────────────────────────────

  EXAMPLE: {top_pair[0]}-{top_pair[1]} A

    Quiet:  delta_MI_sector = {quiet_val:.4f} bits
    Active: delta_MI_sector = {active_val:.4f} bits
    Change: {change:+.1f}%

  INTERPRETATION:
    {'+ Strong activity dependence: Coupling varies with solar activity.' if abs(change) > 20 else
     '~ Moderate activity dependence.' if abs(change) > 10 else
     'o Weak activity dependence: Coupling is relatively stable.'}

═══════════════════════════════════════════════════════════════════════
""")


# ============================================================================
# 27-DAY ROTATION ANALYSIS
# ============================================================================

@dataclass
class RotationAnalysisResult:
    """Result of the 27-day rotation analysis."""
    hours: float
    n_points: int
    cadence_minutes: int
    start_time: str
    end_time: str

    # Coupling values over time
    pair_timeseries: Dict[Tuple[int, int], List[float]]
    pair_means: Dict[Tuple[int, int], float]
    pair_stds: Dict[Tuple[int, int], float]

    # Temporal stability
    temporal_correlations: Dict[Tuple[int, int], float]  # Autocorrelation

    # Rankings
    pair_rankings: Dict[Tuple[int, int], int]


def _compute_interim_result(
    pair_timeseries: Dict[Tuple[int, int], List[float]],
    timestamps: List[str],
    hours: float,
    cadence_minutes: int,
    start_time: str,
    end_time: str
) -> "RotationAnalysisResult":
    """Computes interim result from current data."""
    # Means and standard deviations
    pair_means = {pair: float(np.mean(vals)) if vals else 0.0
                  for pair, vals in pair_timeseries.items()}
    pair_stds = {pair: float(np.std(vals)) if vals else 0.0
                 for pair, vals in pair_timeseries.items()}

    # Autocorrelation
    temporal_correlations = {}
    for pair, values in pair_timeseries.items():
        if len(values) > 2:
            vals = np.array(values)
            corr = np.corrcoef(vals[:-1], vals[1:])[0, 1]
            temporal_correlations[pair] = float(corr) if not np.isnan(corr) else 0.0
        else:
            temporal_correlations[pair] = 0.0

    # Rankings
    sorted_pairs = sorted(pair_means.items(), key=lambda x: -x[1])
    pair_rankings = {pair: rank + 1 for rank, (pair, _) in enumerate(sorted_pairs)}

    return RotationAnalysisResult(
        hours=hours,
        n_points=len(timestamps),
        cadence_minutes=cadence_minutes,
        start_time=start_time,
        end_time=end_time,
        pair_timeseries=pair_timeseries,
        pair_means=pair_means,
        pair_stds=pair_stds,
        temporal_correlations=temporal_correlations,
        pair_rankings=pair_rankings
    )


def load_checkpoint(checkpoint_path: Path) -> Tuple[Dict, List[str], int]:
    """Loads checkpoint if present."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            data = json.load(f)
        # Convert string keys back to tuples
        pair_timeseries = {}
        for key, values in data.get("pair_timeseries", {}).items():
            wl1, wl2 = map(int, key.split("-"))
            pair_timeseries[(wl1, wl2)] = values
        return pair_timeseries, data.get("timestamps", []), data.get("last_index", 0)
    return {}, [], 0


def save_checkpoint(
    checkpoint_path: Path,
    pair_timeseries: Dict[Tuple[int, int], List[float]],
    timestamps: List[str],
    last_index: int,
    auto_push: bool = False
) -> None:
    """Saves checkpoint for resume.

    Args:
        checkpoint_path: Path to checkpoint file
        pair_timeseries: Timeseries data
        timestamps: List of timestamps
        last_index: Last processed index
        auto_push: Automatically git commit & push after saving
    """
    data = {
        "pair_timeseries": {f"{p[0]}-{p[1]}": v for p, v in pair_timeseries.items()},
        "timestamps": timestamps,
        "last_index": last_index
    }
    with open(checkpoint_path, "w") as f:
        json.dump(data, f)

    if auto_push:
        git_push_checkpoint(checkpoint_path, last_index, len(timestamps))


def git_push_checkpoint(checkpoint_path: Path, current: int, total: int) -> None:
    """Git commit & push of checkpoint for cross-system resume."""
    import subprocess

    try:
        # Find git root - from checkpoint directory
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=checkpoint_path.parent,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("    ⚠️  Auto-push: Not a git repository")
            return

        project_root = Path(result.stdout.strip())

        # Add all rotation files
        rotation_dir = checkpoint_path.resolve().parent
        files_to_add = [
            rotation_dir / "checkpoint.json",
            rotation_dir / "coupling_evolution.csv",
            rotation_dir / "rotation_analysis.json",
            rotation_dir / "rotation_analysis.txt"
        ]

        for f in files_to_add:
            if f.exists():
                rel_path = f.relative_to(project_root)
                subprocess.run(
                    ["git", "add", str(rel_path)],
                    cwd=project_root,
                    capture_output=True
                )

        # Commit
        commit_msg = f"Auto-checkpoint: {current}/{total} timepoints ({current*100//total}%)"
        commit_result = subprocess.run(
            ["git", "commit", "-m", commit_msg, "--no-verify"],
            cwd=project_root,
            capture_output=True,
            text=True
        )

        if commit_result.returncode != 0:
            if "nothing to commit" in commit_result.stdout or "nothing to commit" in commit_result.stderr:
                return  # No changes, no push needed
            print(f"    ⚠️  Auto-push: Commit failed - {commit_result.stderr.strip()}")
            return

        # Push
        push_result = subprocess.run(
            ["git", "push"],
            cwd=project_root,
            capture_output=True,
            text=True
        )

        if push_result.returncode != 0:
            print(f"    ⚠️  Auto-push: Push failed - {push_result.stderr.strip()}")
            return

        print(f"    📤 Checkpoint pushed ({current}/{total})")

    except Exception as e:
        print(f"    ⚠️  Auto-push error: {e}")


def run_rotation_analysis(
    hours: float = 648.0,  # 27 days
    cadence_minutes: int = 60,  # Hourly cadence
    seed: int = 42,
    output_dir: str = "results/rotation",
    use_real_data: bool = True,
    start_time_str: Optional[str] = None,
    verbose: bool = True,
    resume: bool = True,  # Automatically resume if checkpoint exists
    auto_push: bool = False  # Git push after each checkpoint
) -> RotationAnalysisResult:
    """
    Runs 27-day rotation analysis with real AIA data.

    Analyzes coupling stability over a complete solar rotation.

    Args:
        hours: Time period (default: 648h = 27 days)
        cadence_minutes: Cadence (default: 60 min for efficiency)
        seed: Random Seed
        output_dir: Output directory
        use_real_data: Use real AIA data
        start_time_str: Start time (ISO format)
        verbose: Verbose output

    Returns:
        RotationAnalysisResult
    """
    import os
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n_points = max(1, int(hours * 60 / cadence_minutes))
    days = hours / 24
    n_workers = max(1, os.cpu_count() - 1)

    # Parallel only for synthetic (256x256), sequential for real AIA (4096x4096)
    parallel_info = f"{n_workers} workers" if not use_real_data else "sequential (4096² images)"

    if verbose:
        print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║            🌞 ROTATION ANALYSIS 🌱                                      ║
╚═══════════════════════════════════════════════════════════════════════╝

  Solar rotation period: ~27.3 days (Carrington rotation)

  Configuration:
    Period:       {hours}h ({days:.1f} days)
    Cadence:      {cadence_minutes} min
    Datapoints:   {n_points}
    Data source:  {'Real AIA data' if use_real_data else 'Synthetic data'}
    Processing:   {parallel_info}
""")

    # Determine start and end time
    if start_time_str is None:
        # Default: 27 days before now
        start_time = datetime.now() - timedelta(hours=hours)
        start_time_str = start_time.isoformat()
    else:
        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))

    end_time = start_time + timedelta(hours=hours)

    if verbose:
        print(f"  Period:       {start_time_str[:10]} to {end_time.isoformat()[:10]}")
        print()

    # Checkpoint path
    checkpoint_path = out_path / "checkpoint.json"

    # Check if resume is possible
    pair_timeseries: Dict[Tuple[int, int], List[float]] = {
        pair: [] for pair in combinations(WAVELENGTHS, 2)
    }
    timestamps: List[str] = []
    start_index = 0

    if resume and checkpoint_path.exists():
        pair_timeseries, timestamps, start_index = load_checkpoint(checkpoint_path)
        if start_index > 0 and verbose:
            print(f"  🔄 Resuming from checkpoint: {start_index}/{n_points} already processed")
            print(f"     Continuing from timepoint {start_index + 1}...")
            print()

    # Process timepoints one by one (streaming instead of batch)
    if use_real_data:
        import gc
        from solar_seed.multichannel import load_aia_multichannel

        if verbose and start_index == 0:
            print(f"  📡 Loading and analyzing {n_points} timepoints...")
        elif verbose:
            print(f"  📡 Loading and analyzing remaining {n_points - start_index} timepoints...")

        t = start_time + timedelta(minutes=cadence_minutes * start_index)
        failed_count = 0

        for i in range(start_index, n_points):
            timestamp = t.isoformat()

            if verbose:
                print(f"    📥 [{i+1}/{n_points}] {timestamp[:19]}...", end=" ", flush=True)

            channels, metadata = load_aia_multichannel(
                timestamp,
                data_dir="data/aia",
                cleanup=True
            )

            if channels is not None:
                # Analyze all pairs (parallel if multiple cores available)
                pair_results = analyze_pairs_parallel(
                    channels, bins=64, seed=seed + i
                )
                for pair, delta_mi in pair_results.items():
                    pair_timeseries[pair].append(delta_mi)

                timestamps.append(timestamp)
                failed_count = 0

                if verbose:
                    print("✓")

                # Save results + checkpoint at each timepoint
                interim_result = _compute_interim_result(
                    pair_timeseries, timestamps, hours, cadence_minutes,
                    start_time_str, end_time.isoformat()
                )
                save_rotation_results(interim_result, out_path, timestamps)
                save_checkpoint(checkpoint_path, pair_timeseries, timestamps, i + 1, auto_push)

                # Garbage collection every 10 timepoints
                if (i + 1) % 10 == 0:
                    gc.collect()
            else:
                failed_count += 1
                if verbose:
                    print("⚠️ skipped")

                if failed_count >= 10:
                    if verbose:
                        print(f"    ✗ Abort: 10 consecutive failures")
                    break

            t += timedelta(minutes=cadence_minutes)

        # Final checkpoint only if data available
        if len(timestamps) > 0:
            save_checkpoint(checkpoint_path, pair_timeseries, timestamps, len(timestamps), auto_push)

        if len(timestamps) == 0:
            print("  ✗ No data loaded.")
            raise RuntimeError("No AIA data available")

        if verbose:
            print(f"\n  ✓ {len(timestamps)} timepoints successfully processed")

    else:
        if verbose:
            print(f"  📊 Generating synthetic data...")

        from solar_seed.multichannel import generate_multichannel_timeseries
        timeseries = generate_multichannel_timeseries(
            n_points=n_points,
            seed=seed,
            cadence_minutes=cadence_minutes
        )

        for t_idx, (channels, timestamp) in enumerate(timeseries):
            if verbose and (t_idx + 1) % 50 == 0:
                print(f"     Timepoint {t_idx + 1}/{len(timeseries)}...")

            timestamps.append(timestamp)

            # Analyze all pairs (parallel)
            pair_results = analyze_pairs_parallel(
                channels, bins=64, seed=seed + t_idx
            )
            for pair, delta_mi in pair_results.items():
                pair_timeseries[pair].append(delta_mi)

    if verbose:
        print("\n  📈 Computing statistics...")

    # Calculate means, standard deviations
    pair_means = {pair: float(np.mean(vals)) for pair, vals in pair_timeseries.items()}
    pair_stds = {pair: float(np.std(vals)) for pair, vals in pair_timeseries.items()}

    # Temporal autocorrelation (lag-1)
    temporal_correlations = {}
    for pair, values in pair_timeseries.items():
        if len(values) > 2:
            vals = np.array(values)
            corr = np.corrcoef(vals[:-1], vals[1:])[0, 1]
            temporal_correlations[pair] = float(corr) if not np.isnan(corr) else 0.0
        else:
            temporal_correlations[pair] = 0.0

    # Rankings by mean ΔMI_sector
    sorted_pairs = sorted(pair_means.items(), key=lambda x: -x[1])
    pair_rankings = {pair: rank + 1 for rank, (pair, _) in enumerate(sorted_pairs)}

    result = RotationAnalysisResult(
        hours=hours,
        n_points=len(timestamps),
        cadence_minutes=cadence_minutes,
        start_time=start_time_str,
        end_time=end_time.isoformat(),
        pair_timeseries=pair_timeseries,
        pair_means=pair_means,
        pair_stds=pair_stds,
        temporal_correlations=temporal_correlations,
        pair_rankings=pair_rankings
    )

    # Save results
    if verbose:
        print("\n  💾 Saving results...")

    save_rotation_results(result, out_path, timestamps)

    if verbose:
        print_rotation_summary(result)

    return result


def save_rotation_results(
    result: RotationAnalysisResult,
    output_dir: Path,
    timestamps: List[str]
) -> None:
    """Saves rotation results."""

    # 1. Main result as text
    with open(output_dir / "rotation_analysis.txt", "w") as f:
        f.write("ROTATION ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        f.write("DATA SOURCE:\n")
        f.write(f"  Instrument:   {AIA_DATA_SOURCE['instrument']}\n")
        f.write(f"  Operator:     {AIA_DATA_SOURCE['operator']}\n")
        f.write(f"  Data:         {AIA_DATA_SOURCE['data_provider']}\n")
        f.write(f"  URL:          {AIA_DATA_SOURCE['data_url']}\n")
        f.write(f"  Reference:    {AIA_DATA_SOURCE['reference']}\n\n")

        f.write("ANALYSIS PARAMETERS:\n")
        f.write(f"  Period:       {result.start_time[:10]} to {result.end_time[:10]}\n")
        f.write(f"  Duration:     {result.hours}h ({result.hours/24:.1f} days)\n")
        f.write(f"  Cadence:      {result.cadence_minutes} min\n")
        f.write(f"  Datapoints:   {result.n_points}\n\n")

        f.write("COUPLING RANKING (ΔMI_sector):\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Rank':<6} {'Pair':<12} {'Mean':<12} {'Std':<12} {'Autocorr'}\n")
        f.write("-" * 70 + "\n")

        sorted_pairs = sorted(result.pair_rankings.items(), key=lambda x: x[1])
        for pair, rank in sorted_pairs:
            mean = result.pair_means[pair]
            std = result.pair_stds[pair]
            autocorr = result.temporal_correlations[pair]
            f.write(f"{rank:<6} {pair[0]}-{pair[1]:<7} {mean:<12.4f} {std:<12.4f} {autocorr:.3f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("\nTEMPORAL STABILITY (Autocorrelation lag-1):\n")
        f.write("-" * 50 + "\n")

        sorted_by_autocorr = sorted(
            result.temporal_correlations.items(),
            key=lambda x: -x[1]
        )

        f.write("\nHighest autocorrelation (most stable coupling):\n")
        for pair, corr in sorted_by_autocorr[:5]:
            f.write(f"  {pair[0]}-{pair[1]} Å: r = {corr:.3f}\n")

        f.write("\nLowest autocorrelation (most variable coupling):\n")
        for pair, corr in sorted_by_autocorr[-5:]:
            f.write(f"  {pair[0]}-{pair[1]} Å: r = {corr:.3f}\n")

    # 2. Timeseries as CSV
    import csv
    with open(output_dir / "coupling_evolution.csv", "w", newline="") as f:
        writer = csv.writer(f)

        pairs = list(result.pair_timeseries.keys())
        header = ["timestamp"] + [f"{p[0]}-{p[1]}" for p in pairs]
        writer.writerow(header)

        for i, ts in enumerate(timestamps):
            row = [ts[:19]]
            for pair in pairs:
                if i < len(result.pair_timeseries[pair]):
                    row.append(f"{result.pair_timeseries[pair][i]:.4f}")
                else:
                    row.append("")
            writer.writerow(row)

    # 3. JSON for further processing
    data = {
        "metadata": {
            "hours": result.hours,
            "days": result.hours / 24,
            "n_points": result.n_points,
            "cadence_minutes": result.cadence_minutes,
            "start_time": result.start_time,
            "end_time": result.end_time
        },
        "pair_means": {f"{p[0]}-{p[1]}": v for p, v in result.pair_means.items()},
        "pair_stds": {f"{p[0]}-{p[1]}": v for p, v in result.pair_stds.items()},
        "temporal_correlations": {f"{p[0]}-{p[1]}": v for p, v in result.temporal_correlations.items()},
        "pair_rankings": {f"{p[0]}-{p[1]}": v for p, v in result.pair_rankings.items()},
        "data_source": AIA_DATA_SOURCE
    }

    with open(output_dir / "rotation_analysis.json", "w") as f:
        json.dump(data, f, indent=2)


def print_rotation_summary(result: RotationAnalysisResult) -> None:
    """Prints rotation analysis summary."""

    # Top-5 pairs
    sorted_pairs = sorted(result.pair_means.items(), key=lambda x: -x[1])

    # Mean autocorrelation
    mean_autocorr = np.mean(list(result.temporal_correlations.values()))

    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║              🌞 ROTATION ANALYSIS RESULT 🌱                             ║
╚═══════════════════════════════════════════════════════════════════════╝

  Period:   {result.start_time[:10]} -> {result.end_time[:10]}
  Duration: {result.hours/24:.1f} days ({result.n_points} data points)

  ════════════════════════════════════════════════════════════════════════

  TOP 5 STRONGEST COUPLINGS (27-day mean):
""")

    for i, (pair, mean) in enumerate(sorted_pairs[:5], 1):
        std = result.pair_stds[pair]
        autocorr = result.temporal_correlations[pair]
        print(f"    {i}. {pair[0]}-{pair[1]} Å: ΔMI = {mean:.4f} ± {std:.4f} (r = {autocorr:.2f})")

    print(f"""
  ────────────────────────────────────────────────────────────────────────

  TEMPORAL STABILITY:

    Mean autocorrelation: {mean_autocorr:.3f}
    -> {'High temporal stability' if mean_autocorr > 0.7 else 'Moderate stability' if mean_autocorr > 0.4 else 'Variable coupling'}

  ────────────────────────────────────────────────────────────────────────

  OUTPUT FILES:
    results/rotation/rotation_analysis.txt
    results/rotation/rotation_analysis.json
    results/rotation/coupling_evolution.csv

═══════════════════════════════════════════════════════════════════════
""")


# ============================================================================
# SEGMENT-BASED ROTATION ANALYSIS
# ============================================================================

@dataclass
class SegmentResult:
    """Result of a single segment (e.g., one day)."""
    date: str  # YYYY-MM-DD
    start_time: str
    end_time: str
    n_points: int
    cadence_minutes: int

    # Raw data for this segment
    timestamps: List[str]
    pair_values: Dict[str, List[float]]  # "304-171" -> [values]

    # Segment statistics
    pair_means: Dict[str, float]
    pair_stds: Dict[str, float]


def _load_partial_checkpoint(partial_file: Path) -> Tuple[Dict[str, List[float]], List[str], int]:
    """Load partial day checkpoint if exists."""
    if partial_file.exists():
        try:
            with open(partial_file) as f:
                data = json.load(f)
            return (
                data.get("pair_values", {}),
                data.get("timestamps", []),
                data.get("last_index", 0)
            )
        except Exception:
            pass
    return {}, [], 0


def _save_partial_checkpoint(
    partial_file: Path,
    pair_values: Dict[str, List[float]],
    timestamps: List[str],
    last_index: int
) -> None:
    """Save partial day checkpoint."""
    data = {
        "pair_values": pair_values,
        "timestamps": timestamps,
        "last_index": last_index
    }
    with open(partial_file, "w") as f:
        json.dump(data, f)


def run_segment_analysis(
    date: str,  # "2025-12-01"
    cadence_minutes: int = 12,
    seed: int = 42,
    output_dir: str = "results/rotation/segments",
    verbose: bool = True
) -> Optional[SegmentResult]:
    """
    Analyzes a single segment (one day).

    Supports intra-day checkpointing for resume after abort.
    Partial progress is saved to {date}.partial.json.

    Args:
        date: Date in format YYYY-MM-DD
        cadence_minutes: Cadence in minutes
        seed: Random seed
        output_dir: Output directory
        verbose: Verbose output

    Returns:
        SegmentResult or None on error
    """
    import os
    import gc
    from solar_seed.multichannel import load_aia_multichannel

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Check segment file (already analyzed?)
    segment_file = out_path / f"{date}.json"
    if segment_file.exists():
        if verbose:
            print(f"  ✓ Segment {date} already exists, skipping")
        return load_segment(segment_file)

    # Time range for this day
    start_time = datetime.fromisoformat(f"{date}T00:00:00")
    end_time = start_time + timedelta(days=1)
    n_points = int(24 * 60 / cadence_minutes)  # 120 at 12-min cadence

    # Check for partial checkpoint (intra-day resume)
    partial_file = out_path / f"{date}.partial.json"
    pair_keys = [f"{a}-{b}" for a, b in combinations(WAVELENGTHS, 2)]

    pair_values, timestamps, start_index = _load_partial_checkpoint(partial_file)

    # Initialize missing pair keys
    if not pair_values:
        pair_values = {key: [] for key in pair_keys}

    if start_index > 0 and verbose:
        print(f"\n  📅 Segment {date}: resuming from {start_index}/{n_points}")
    elif verbose:
        print(f"\n  📅 Segment {date}: {n_points} timepoints")

    failed_count = 0

    # Analyze each timepoint (starting from checkpoint)
    t = start_time + timedelta(minutes=cadence_minutes * start_index)
    for i in range(start_index, n_points):
        timestamp = t.isoformat()

        if verbose:
            print(f"    [{i+1}/{n_points}] {timestamp[:19]}...", end=" ", flush=True)

        channels, metadata = load_aia_multichannel(
            timestamp,
            data_dir="data/aia",
            cleanup=True
        )

        if channels is not None:
            # Analyze all pairs
            pair_results = analyze_pairs_parallel(channels, bins=64, seed=seed + i)

            for pair, delta_mi in pair_results.items():
                key = f"{pair[0]}-{pair[1]}"
                pair_values[key].append(delta_mi)

            timestamps.append(timestamp)
            failed_count = 0

            if verbose:
                print("✓")

            # Save partial checkpoint every 5 successful timepoints
            if (len(timestamps) % 5) == 0:
                _save_partial_checkpoint(partial_file, pair_values, timestamps, i + 1)

            # Garbage Collection
            if (i + 1) % 10 == 0:
                gc.collect()
        else:
            failed_count += 1
            if verbose:
                print("⚠️")

            # Save checkpoint on failure too (to preserve progress)
            if len(timestamps) > 0:
                _save_partial_checkpoint(partial_file, pair_values, timestamps, i + 1)

            if failed_count >= 10:
                if verbose:
                    print(f"    ✗ Abort: 10 consecutive failures (progress saved)")
                return None  # Return None but progress is saved

        t += timedelta(minutes=cadence_minutes)

    # No data?
    if not timestamps:
        if verbose:
            print(f"    ✗ No data for {date}")
        # Clean up partial file if no data
        if partial_file.exists():
            partial_file.unlink()
        return None

    # Calculate statistics
    pair_means = {key: float(np.mean(vals)) if vals else 0.0
                  for key, vals in pair_values.items()}
    pair_stds = {key: float(np.std(vals)) if vals else 0.0
                 for key, vals in pair_values.items()}

    # Create result
    result = SegmentResult(
        date=date,
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        n_points=len(timestamps),
        cadence_minutes=cadence_minutes,
        timestamps=timestamps,
        pair_values=pair_values,
        pair_means=pair_means,
        pair_stds=pair_stds
    )

    # Save
    save_segment(result, segment_file)

    # Clean up partial checkpoint on success
    if partial_file.exists():
        partial_file.unlink()

    if verbose:
        print(f"    ✓ Segment {date}: {len(timestamps)} points saved")

    return result


def save_segment(result: SegmentResult, path: Path) -> None:
    """Saves a segment as JSON."""
    data = {
        "date": result.date,
        "start_time": result.start_time,
        "end_time": result.end_time,
        "n_points": result.n_points,
        "cadence_minutes": result.cadence_minutes,
        "timestamps": result.timestamps,
        "pair_values": result.pair_values,
        "pair_means": result.pair_means,
        "pair_stds": result.pair_stds
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_segment(path: Path) -> SegmentResult:
    """Loads a segment from JSON."""
    with open(path) as f:
        data = json.load(f)
    return SegmentResult(**data)


def aggregate_segments(
    segment_dir: str = "results/rotation/segments",
    output_dir: str = "results/rotation",
    verbose: bool = True
) -> Optional[RotationAnalysisResult]:
    """
    Aggregates all available segments into an overall result.

    Args:
        segment_dir: Directory with segment files
        output_dir: Output directory for aggregated result
        verbose: Verbose output

    Returns:
        RotationAnalysisResult or None
    """
    seg_path = Path(segment_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Find all segment files
    segment_files = sorted(seg_path.glob("*.json"))

    if not segment_files:
        if verbose:
            print("  ✗ No segments found")
        return None

    if verbose:
        print(f"\n  📊 Aggregating {len(segment_files)} segments...")

    # Load all segments
    segments: List[SegmentResult] = []
    for sf in segment_files:
        seg = load_segment(sf)
        segments.append(seg)
        if verbose:
            print(f"    ✓ {seg.date}: {seg.n_points} points")

    # Combine data
    all_timestamps: List[str] = []
    pair_timeseries: Dict[Tuple[int, int], List[float]] = {
        pair: [] for pair in combinations(WAVELENGTHS, 2)
    }

    for seg in segments:
        all_timestamps.extend(seg.timestamps)
        for key, values in seg.pair_values.items():
            a, b = map(int, key.split("-"))
            pair_timeseries[(a, b)].extend(values)

    # Calculate total statistics
    total_hours = len(segments) * 24
    cadence = segments[0].cadence_minutes if segments else 12

    # Create RotationAnalysisResult
    pair_means = {pair: float(np.mean(vals)) if vals else 0.0
                  for pair, vals in pair_timeseries.items()}
    pair_stds = {pair: float(np.std(vals)) if vals else 0.0
                 for pair, vals in pair_timeseries.items()}

    # Autocorrelation
    temporal_correlations = {}
    for pair, vals in pair_timeseries.items():
        if len(vals) > 1:
            arr = np.array(vals)
            if np.std(arr) > 0:
                corr = np.corrcoef(arr[:-1], arr[1:])[0, 1]
                temporal_correlations[pair] = float(corr) if not np.isnan(corr) else 0.0
            else:
                temporal_correlations[pair] = 0.0
        else:
            temporal_correlations[pair] = 0.0

    # Rankings
    sorted_pairs = sorted(pair_means.items(), key=lambda x: -x[1])
    pair_rankings = {pair: rank for rank, (pair, _) in enumerate(sorted_pairs, 1)}

    result = RotationAnalysisResult(
        hours=total_hours,
        n_points=len(all_timestamps),
        cadence_minutes=cadence,
        start_time=segments[0].start_time if segments else "",
        end_time=segments[-1].end_time if segments else "",
        pair_timeseries=pair_timeseries,
        pair_means=pair_means,
        pair_stds=pair_stds,
        temporal_correlations=temporal_correlations,
        pair_rankings=pair_rankings
    )

    # Save
    save_rotation_results(result, out_path, all_timestamps)

    if verbose:
        print(f"\n  ✓ Aggregated: {len(segments)} days, {len(all_timestamps)} points")
        print(f"    → {out_path / 'rotation_analysis.json'}")

    return result


def convert_checkpoint_to_segments(
    checkpoint_path: str = "results/rotation/checkpoint.json",
    output_dir: str = "results/rotation/segments",
    verbose: bool = True
) -> int:
    """
    Converts existing monolithic checkpoint to segments.

    Args:
        checkpoint_path: Path to checkpoint
        output_dir: Output directory for segments
        verbose: Verbose output

    Returns:
        Number of created segments
    """
    ckpt_path = Path(checkpoint_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        if verbose:
            print(f"  ✗ Checkpoint not found: {checkpoint_path}")
        return 0

    # Load checkpoint
    with open(ckpt_path) as f:
        data = json.load(f)

    timestamps = data.get("timestamps", [])
    pair_data = data.get("pair_timeseries", {})

    if not timestamps:
        if verbose:
            print("  ✗ No data in checkpoint")
        return 0

    if verbose:
        print(f"\n  📦 Converting {len(timestamps)} timepoints to segments...")

    # Group by date
    from collections import defaultdict
    daily_indices: Dict[str, List[int]] = defaultdict(list)

    for i, ts in enumerate(timestamps):
        date = ts[:10]  # YYYY-MM-DD
        daily_indices[date].append(i)

    # Create segments
    segments_created = 0

    for date in sorted(daily_indices.keys()):
        indices = daily_indices[date]
        segment_file = out_path / f"{date}.json"

        if segment_file.exists():
            if verbose:
                print(f"    ⏭️  {date}: already exists")
            continue

        # Extract data for this day
        day_timestamps = [timestamps[i] for i in indices]
        day_pair_values: Dict[str, List[float]] = {}

        for pair_key, values in pair_data.items():
            day_pair_values[pair_key] = [values[i] for i in indices if i < len(values)]

        # Calculate statistics
        pair_means = {key: float(np.mean(vals)) if vals else 0.0
                      for key, vals in day_pair_values.items()}
        pair_stds = {key: float(np.std(vals)) if vals else 0.0
                     for key, vals in day_pair_values.items()}

        # Determine cadence from timestamps
        if len(day_timestamps) >= 2:
            t1 = datetime.fromisoformat(day_timestamps[0].replace('Z', '+00:00'))
            t2 = datetime.fromisoformat(day_timestamps[1].replace('Z', '+00:00'))
            cadence = int((t2 - t1).total_seconds() / 60)
        else:
            cadence = 12

        # Create segment
        result = SegmentResult(
            date=date,
            start_time=f"{date}T00:00:00",
            end_time=f"{date}T23:59:59",
            n_points=len(day_timestamps),
            cadence_minutes=cadence,
            timestamps=day_timestamps,
            pair_values=day_pair_values,
            pair_means=pair_means,
            pair_stds=pair_stds
        )

        save_segment(result, segment_file)
        segments_created += 1

        if verbose:
            print(f"    ✓ {date}: {len(day_timestamps)} points")

    if verbose:
        print(f"\n  ✓ {segments_created} segments created in {output_dir}")

    return segments_created


def run_segmented_rotation(
    start_date: str,
    end_date: str,
    cadence_minutes: int = 12,
    output_dir: str = "results/rotation",
    verbose: bool = True,
    auto_push: bool = False
) -> Optional[RotationAnalysisResult]:
    """
    Runs segment-based rotation analysis.

    Analyzes each day individually and aggregates at the end.
    Already analyzed days are skipped.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        cadence_minutes: Cadence
        output_dir: Output directory
        verbose: Verbose output
        auto_push: Git push after each segment

    Returns:
        Aggregated RotationAnalysisResult
    """
    segment_dir = f"{output_dir}/segments"

    # Parse dates
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    n_days = (end - start).days + 1

    if verbose:
        print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║          🌞 SEGMENT-BASED ROTATION ANALYSIS 🌱                         ║
╚═══════════════════════════════════════════════════════════════════════╝

  Period:      {start_date} -> {end_date} ({n_days} days)
  Cadence:     {cadence_minutes} min
  Segments:    {segment_dir}
""")

    # Analyze each day
    current = start
    completed = 0

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")

        result = run_segment_analysis(
            date=date_str,
            cadence_minutes=cadence_minutes,
            output_dir=segment_dir,
            verbose=verbose
        )

        if result is not None:
            completed += 1

            # Auto-push after each segment
            if auto_push:
                _git_push_segment(segment_dir, date_str, completed, n_days)

        current += timedelta(days=1)

    if verbose:
        print(f"\n  ════════════════════════════════════════════════════════════")
        print(f"  ✓ {completed}/{n_days} segments analyzed")

    # Aggregate all segments
    return aggregate_segments(segment_dir, output_dir, verbose)


def _git_push_segment(segment_dir: str, date: str, current: int, total: int) -> None:
    """Git push after segment analysis."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=segment_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return

        project_root = Path(result.stdout.strip())
        seg_path = Path(segment_dir).resolve()

        # Add segment file
        seg_file = seg_path / f"{date}.json"
        if seg_file.exists():
            rel_path = seg_file.relative_to(project_root)
            subprocess.run(["git", "add", str(rel_path)], cwd=project_root, capture_output=True)

        # Commit
        commit_msg = f"Segment {date}: {current}/{total} days"
        commit_result = subprocess.run(
            ["git", "commit", "-m", commit_msg, "--no-verify"],
            cwd=project_root,
            capture_output=True,
            text=True
        )

        if commit_result.returncode != 0:
            if "nothing to commit" in commit_result.stdout + commit_result.stderr:
                return
            return

        # Push
        subprocess.run(["git", "push"], cwd=project_root, capture_output=True)
        print(f"    📤 Segment {date} pushed")

    except Exception:
        pass


# ============================================================================
# COMBINED FINAL ANALYSIS
# ============================================================================

def run_final_analysis(
    output_dir: str = "results/final",
    use_real_data: bool = False,
    verbose: bool = True
) -> Tuple[TimescaleComparison, ActivityConditioningResult]:
    """
    Runs both final analyses.

    Args:
        output_dir: Output directory
        use_real_data: Use real data
        verbose: Verbose output

    Returns:
        (TimescaleComparison, ActivityConditioningResult)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("""
╔═══════════════════════════════════════════════════════════════════════╗
║               🌞 FINAL SOLAR SEED ANALYSES 🌱                           ║
╚═══════════════════════════════════════════════════════════════════════╝

  Two concluding analyses for depth over breadth:

    1. Timescale Comparison (24h vs 27d)
       -> Is the temperature ordering stable?

    2. Activity Conditioning (94A proxy)
       -> Does coupling correlate with solar activity?

═══════════════════════════════════════════════════════════════════════
""")

    # Analysis 1: Timescales
    timescale_result = run_timescale_comparison(
        short_hours=24.0,
        long_hours=648.0,  # 27 days
        output_dir=output_dir,
        use_real_data=use_real_data,
        verbose=verbose
    )

    # Analysis 2: Activity
    activity_result = run_activity_conditioning(
        n_hours=48.0,
        output_dir=output_dir,
        use_real_data=use_real_data,
        verbose=verbose
    )

    # Combined summary
    if verbose:
        print_final_summary(timescale_result, activity_result, out_path)

    return timescale_result, activity_result


def print_final_summary(
    timescale: TimescaleComparison,
    activity: ActivityConditioningResult,
    output_dir: Path
) -> None:
    """Prints combined summary."""

    summary = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                  🌞 FINAL RESULTS 🌱                                    ║
╚═══════════════════════════════════════════════════════════════════════╝

  1. TIMESCALE STABILITY:

     Spearman rho = {timescale.spearman_rho:.3f}
     -> {'The coupling ordering is temporally STABLE' if timescale.spearman_rho > 0.7 else 'Dynamic variation across timescales'}

  2. ACTIVITY DEPENDENCE:

     Strongest signal: {activity.most_activity_dependent[0][0][0]}-{activity.most_activity_dependent[0][0][1]} A
     r = {activity.most_activity_dependent[0][1]:.3f}
     -> {'Coupling correlates with activity' if abs(activity.most_activity_dependent[0][1]) > 0.3 else 'Coupling is activity-independent'}

  ════════════════════════════════════════════════════════════════════════

  SCIENTIFIC CONCLUSION:

  {"+ The local structure coupling (delta_MI_sector) is a robust signal." if timescale.spearman_rho > 0.6 else ""}
  {"  It remains preserved across timescales." if timescale.spearman_rho > 0.6 else ""}
  {"  It shows physically meaningful activity dependence." if abs(activity.most_activity_dependent[0][1]) > 0.2 else ""}

  OUTPUT FILES:
    {output_dir}/timescale_comparison.txt
    {output_dir}/timescale_comparison.json
    {output_dir}/activity_conditioning.txt
    {output_dir}/activity_conditioning.json

═══════════════════════════════════════════════════════════════════════
"""
    print(summary)

    # Also save as file
    with open(output_dir / "final_summary.txt", "w") as f:
        f.write(summary)


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Final analyses for Solar Seed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard analyses
  python -m solar_seed.final_analysis
  python -m solar_seed.final_analysis --timescale-only
  python -m solar_seed.final_analysis --activity-only

  # Legacy: Monolithic rotation (old)
  python -m solar_seed.final_analysis --rotation --start "2024-01-01"

  # NEW: Segment-based rotation (recommended)
  python -m solar_seed.final_analysis --segments --start 2025-12-01 --end 2025-12-27
  python -m solar_seed.final_analysis --segment 2025-12-15
  python -m solar_seed.final_analysis --aggregate
  python -m solar_seed.final_analysis --convert-checkpoint
        """
    )
    parser.add_argument("--output", type=str, default="results/final",
                        help="Output directory")
    parser.add_argument("--real", action="store_true",
                        help="Use real AIA data")
    parser.add_argument("--timescale-only", action="store_true",
                        help="Only timescale comparison")
    parser.add_argument("--activity-only", action="store_true",
                        help="Only activity conditioning")

    # Legacy rotation
    parser.add_argument("--rotation", action="store_true",
                        help="27-day rotation analysis (legacy, monolithic)")
    parser.add_argument("--short-hours", type=float, default=24.0,
                        help="Short timescale in hours")
    parser.add_argument("--long-hours", type=float, default=648.0,
                        help="Long timescale in hours (27d = 648)")

    # Segment-based rotation (new)
    parser.add_argument("--segments", action="store_true",
                        help="Segment-based rotation analysis (recommended)")
    parser.add_argument("--segment", type=str, default=None,
                        help="Analyze single segment (YYYY-MM-DD)")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate all segments")
    parser.add_argument("--convert-checkpoint", action="store_true",
                        help="Convert existing checkpoint to segments")

    # Common options
    parser.add_argument("--start", type=str, default=None,
                        help="Start date (YYYY-MM-DD or ISO format)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date for segment analysis (YYYY-MM-DD)")
    parser.add_argument("--cadence", type=int, default=12,
                        help="Cadence in minutes (default: 12)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not resume from checkpoint, start fresh")
    parser.add_argument("--auto-push", action="store_true",
                        help="Git push after each segment/checkpoint")

    args = parser.parse_args()

    # Segment-based analysis (new, recommended)
    if args.segments:
        if not args.start or not args.end:
            print("Error: --segments requires --start and --end")
            print("Example: --segments --start 2025-12-01 --end 2025-12-27")
            return
        run_segmented_rotation(
            start_date=args.start[:10],  # Date only
            end_date=args.end[:10],
            cadence_minutes=args.cadence,
            output_dir="results/rotation",
            verbose=True,
            auto_push=args.auto_push
        )
    elif args.segment:
        run_segment_analysis(
            date=args.segment,
            cadence_minutes=args.cadence,
            output_dir="results/rotation/segments",
            verbose=True
        )
    elif args.aggregate:
        aggregate_segments(
            segment_dir="results/rotation/segments",
            output_dir="results/rotation",
            verbose=True
        )
    elif args.convert_checkpoint:
        convert_checkpoint_to_segments(
            checkpoint_path="results/rotation/checkpoint.json",
            output_dir="results/rotation/segments",
            verbose=True
        )
    # Legacy rotation
    elif args.rotation:
        run_rotation_analysis(
            hours=args.long_hours,
            cadence_minutes=args.cadence,
            output_dir="results/rotation",
            use_real_data=True,
            start_time_str=args.start,
            verbose=True,
            resume=not args.no_resume,
            auto_push=args.auto_push
        )
    elif args.timescale_only:
        run_timescale_comparison(
            short_hours=args.short_hours,
            long_hours=args.long_hours,
            output_dir=args.output,
            use_real_data=args.real,
            verbose=True
        )
    elif args.activity_only:
        run_activity_conditioning(
            output_dir=args.output,
            use_real_data=args.real,
            verbose=True
        )
    else:
        run_final_analysis(
            output_dir=args.output,
            use_real_data=args.real,
            verbose=True
        )


if __name__ == "__main__":
    main()
