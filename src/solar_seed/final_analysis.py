#!/usr/bin/env python3
"""
Finale Analysen für Solar Seed
==============================

Zwei Abschluss-Analysen:

1. TIMESCALE COMPARISON (24h vs 27d)
   - Bleibt die Temperatur-Ordnung über verschiedene Zeitskalen erhalten?
   - Spearman-Korrelation der Rankings
   - Stabilität als Validierung

2. ACTIVITY CONDITIONING
   - ΔMI_sector vs 94Å-Proxy
   - Konditionierung auf ruhig (low 94Å) vs aktiv (high 94Å)
   - Zeigt physikalische Kopplung zwischen Aktivität und Struktur
"""

# Fix für macOS: Fork-Crash vermeiden bei async-Bibliotheken (SunPy/aiohttp)
# Muss VOR allen anderen Imports stehen!
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=False)
except RuntimeError:
    pass  # Bereits gesetzt

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
    load_aia_multichannel_timeseries
)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TimescaleResult:
    """Ergebnis des Zeitskalen-Vergleichs."""
    timescale_hours: float
    n_points: int
    pair_rankings: Dict[Tuple[int, int], int]  # Paar -> Rang
    pair_values: Dict[Tuple[int, int], float]  # Paar -> ΔMI_sector
    timestamp: str


@dataclass
class TimescaleComparison:
    """Vergleich zwischen verschiedenen Zeitskalen."""
    short_scale: TimescaleResult
    long_scale: TimescaleResult

    # Korrelationen
    spearman_rho: float
    spearman_p: float
    kendall_tau: float
    kendall_p: float

    # Stabilität
    top5_overlap: int  # Wie viele der Top-5 Paare sind gleich?
    rank_changes: Dict[Tuple[int, int], int]  # Rang-Differenzen


@dataclass
class ActivityBin:
    """Ergebnis für einen Aktivitäts-Bereich."""
    bin_label: str  # "quiet", "moderate", "active"
    mean_94A_intensity: float
    n_samples: int
    pair_values: Dict[Tuple[int, int], float]  # Paar -> mittlerer ΔMI_sector
    pair_stds: Dict[Tuple[int, int], float]


@dataclass
class ActivityConditioningResult:
    """Ergebnis der Aktivitäts-Konditionierung."""
    bins: List[ActivityBin]

    # Korrelation zwischen Aktivität und Kopplung
    activity_vs_coupling: Dict[Tuple[int, int], Tuple[float, float]]  # Paar -> (r, p)

    # Stärkstes Signal
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
    Analysiert eine Zeitskala und gibt Ranking der Paare zurück.

    Args:
        n_hours: Zeitraum
        cadence_minutes: Kadenz
        seed: Random Seed
        use_real_data: Echte Daten verwenden
        start_time_str: Startzeit
        verbose: Ausführliche Ausgabe

    Returns:
        TimescaleResult mit Rankings
    """
    n_points = max(1, int(n_hours * 60 / cadence_minutes))

    if verbose:
        print(f"  📊 Analysiere {n_hours}h ({n_points} Zeitpunkte)...")

    # Generiere oder lade Daten
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

    # Sammle Ergebnisse
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

    # Mittelwerte
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
    Vergleicht zwei Zeitskalen und berechnet Korrelationen.
    """
    pairs = list(short_result.pair_rankings.keys())

    short_ranks = [short_result.pair_rankings[p] for p in pairs]
    long_ranks = [long_result.pair_rankings[p] for p in pairs]

    # Spearman Korrelation
    spearman_rho, spearman_p = stats.spearmanr(short_ranks, long_ranks)

    # Kendall Tau
    kendall_tau, kendall_p = stats.kendalltau(short_ranks, long_ranks)

    # Top-5 Overlap
    short_top5 = set(p for p, r in short_result.pair_rankings.items() if r <= 5)
    long_top5 = set(p for p, r in long_result.pair_rankings.items() if r <= 5)
    top5_overlap = len(short_top5 & long_top5)

    # Rang-Differenzen
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
    long_hours: float = 648.0,  # 27 Tage
    cadence_minutes: int = 12,
    seed: int = 42,
    output_dir: str = "results/final",
    use_real_data: bool = False,
    verbose: bool = True
) -> TimescaleComparison:
    """
    Führt den Zeitskalen-Vergleich durch.

    Args:
        short_hours: Kurze Zeitskala (default: 24h)
        long_hours: Lange Zeitskala (default: 27 Tage = 648h)
        cadence_minutes: Kadenz
        seed: Random Seed
        output_dir: Output-Verzeichnis
        use_real_data: Echte Daten
        verbose: Ausführliche Ausgabe

    Returns:
        TimescaleComparison
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║            📊 ZEITSKALEN-VERGLEICH (FINALE ANALYSE 1)                  ║
╚═══════════════════════════════════════════════════════════════════════╝

  Frage: Bleibt die Temperatur-Ordnung über verschiedene Zeitskalen stabil?

  Zeitskalen:
    Kurz:  {short_hours}h
    Lang:  {long_hours}h ({long_hours/24:.0f} Tage)
""")

    # Analysiere beide Zeitskalen
    short_result = analyze_timescale(
        short_hours,
        cadence_minutes=cadence_minutes,
        seed=seed,
        use_real_data=use_real_data,
        verbose=verbose
    )

    long_result = analyze_timescale(
        long_hours,
        cadence_minutes=cadence_minutes * 10,  # Gröbere Kadenz für lange Zeitskala
        seed=seed + 1000,
        use_real_data=use_real_data,
        verbose=verbose
    )

    # Vergleiche
    comparison = compare_timescales(short_result, long_result)

    # Speichere und drucke
    save_timescale_results(comparison, out_path)

    if verbose:
        print_timescale_summary(comparison)

    return comparison


def save_timescale_results(result: TimescaleComparison, output_dir: Path) -> None:
    """Speichert Zeitskalen-Ergebnisse."""

    with open(output_dir / "timescale_comparison.txt", "w") as f:
        f.write("ZEITSKALEN-VERGLEICH\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Kurze Zeitskala: {result.short_scale.timescale_hours}h "
                f"({result.short_scale.n_points} Zeitpunkte)\n")
        f.write(f"Lange Zeitskala: {result.long_scale.timescale_hours}h "
                f"({result.long_scale.n_points} Zeitpunkte)\n\n")

        f.write("KORRELATIONEN:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Spearman ρ: {result.spearman_rho:.4f} (p = {result.spearman_p:.2e})\n")
        f.write(f"Kendall τ:  {result.kendall_tau:.4f} (p = {result.kendall_p:.2e})\n\n")

        f.write(f"Top-5 Overlap: {result.top5_overlap}/5\n\n")

        f.write("RANKING-VERGLEICH:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Paar':<12} {'Rang (kurz)':<12} {'Rang (lang)':<12} {'Differenz'}\n")
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
    """Druckt Zusammenfassung."""

    stability = "STABIL" if result.spearman_rho > 0.8 else "VARIABEL" if result.spearman_rho > 0.5 else "INSTABIL"

    print(f"""
  ════════════════════════════════════════════════════════════════════════

  ERGEBNIS:

    Spearman ρ = {result.spearman_rho:.4f}  (p = {result.spearman_p:.2e})
    Kendall τ  = {result.kendall_tau:.4f}  (p = {result.kendall_p:.2e})

    Top-5 Overlap: {result.top5_overlap}/5 Paare stimmen überein

    → Ordnung ist {stability}

  INTERPRETATION:
    {'✓ Die Temperatur-Kopplung bleibt über Zeitskalen erhalten.' if result.spearman_rho > 0.7 else
     '⚠ Die Kopplung variiert mit Zeitskala - dynamische Effekte.' if result.spearman_rho > 0.4 else
     '✗ Keine stabile Ordnung - Kopplung ist zeitabhängig.'}
    {'  Dies unterstützt die physikalische Interpretation.' if result.spearman_rho > 0.7 else ''}

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
    Führt Aktivitäts-Konditionierung durch.

    Verwendet 94Å als Proxy für Sonnenaktivität und berechnet
    ΔMI_sector für verschiedene Aktivitätsniveaus.

    Args:
        n_hours: Zeitraum
        cadence_minutes: Kadenz
        n_bins: Anzahl Aktivitäts-Bins
        seed: Random Seed
        output_dir: Output-Verzeichnis
        use_real_data: Echte Daten
        verbose: Ausführliche Ausgabe

    Returns:
        ActivityConditioningResult
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n_points = max(1, int(n_hours * 60 / cadence_minutes))

    if verbose:
        print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║          🔥 AKTIVITÄTS-KONDITIONIERUNG (FINALE ANALYSE 2)              ║
╚═══════════════════════════════════════════════════════════════════════╝

  Frage: Korreliert ΔMI_sector mit Sonnenaktivität (94Å-Proxy)?

  Methode:
    1. Berechne mittlere 94Å-Intensität pro Zeitpunkt
    2. Teile Zeitpunkte in {n_bins} Aktivitäts-Bins
    3. Berechne ΔMI_sector pro Bin für alle Paare
    4. Korreliere Aktivität mit Kopplung

  Zeitraum: {n_hours}h ({n_points} Zeitpunkte)
""")

    # Generiere oder lade Daten
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
        print(f"  📊 Analysiere {len(timeseries)} Zeitpunkte...")

    # Sammle alle Daten
    all_data: List[Dict] = []

    for t_idx, (channels, timestamp) in enumerate(timeseries):
        # 94Å Intensität als Aktivitäts-Proxy
        intensity_94 = float(np.mean(channels[94][channels[94] > 0]))

        # Analysiere alle Paare
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

    # Teile in Aktivitäts-Bins
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

        # Mittlere Werte pro Paar
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
        print(f"  📈 Berechne Korrelationen...")

    # Korrelation zwischen Aktivität und Kopplung pro Paar
    activity_vs_coupling: Dict[Tuple[int, int], Tuple[float, float]] = {}

    for pair in combinations(WAVELENGTHS, 2):
        activities = [d["intensity_94"] for d in all_data]
        couplings = [d["pair_results"][pair] for d in all_data]

        r, p = stats.pearsonr(activities, couplings)
        activity_vs_coupling[pair] = (r, p)

    # Top-5 aktivitätsabhängige Paare
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

    # Speichere und drucke
    save_activity_results(result, out_path)

    if verbose:
        print_activity_summary(result)

    return result


def save_activity_results(result: ActivityConditioningResult, output_dir: Path) -> None:
    """Speichert Aktivitäts-Ergebnisse."""

    with open(output_dir / "activity_conditioning.txt", "w") as f:
        f.write("AKTIVITÄTS-KONDITIONIERUNG\n")
        f.write("=" * 70 + "\n\n")

        f.write("94Å als Proxy für Sonnenaktivität\n\n")

        f.write("AKTIVITÄTS-BINS:\n")
        f.write("-" * 70 + "\n")

        for bin in result.bins:
            f.write(f"\n{bin.bin_label.upper()} (n={bin.n_samples}, "
                    f"mean 94Å={bin.mean_94A_intensity:.1f}):\n")

            sorted_pairs = sorted(bin.pair_values.items(), key=lambda x: -x[1])
            for pair, value in sorted_pairs[:5]:
                f.write(f"  {pair[0]}-{pair[1]}: ΔMI_sector = {value:.4f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("\nKORRELATION AKTIVITÄT ↔ KOPPLUNG:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Paar':<12} {'Pearson r':<12} {'p-Wert':<15} {'Interpretation'}\n")
        f.write("-" * 50 + "\n")

        for pair, (r, p) in sorted(result.activity_vs_coupling.items(),
                                    key=lambda x: -abs(x[1][0])):
            if abs(r) > 0.5:
                interp = "stark" + (" positiv" if r > 0 else " negativ")
            elif abs(r) > 0.3:
                interp = "moderat"
            else:
                interp = "schwach"

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
    """Druckt Zusammenfassung."""

    print(f"""
  ════════════════════════════════════════════════════════════════════════

  AKTIVITÄTS-BINS:
""")

    for bin in result.bins:
        print(f"    {bin.bin_label.upper():>10}: n={bin.n_samples:>3}, 94Å={bin.mean_94A_intensity:>8.1f}")

    print(f"""
  ────────────────────────────────────────────────────────────────────────

  TOP 5 AKTIVITÄTSABHÄNGIGE PAARE:
""")

    for i, (pair, r) in enumerate(result.most_activity_dependent, 1):
        direction = "+" if r > 0 else "-"
        print(f"    {i}. {pair[0]}-{pair[1]} Å: r = {direction}{abs(r):.3f}")

    # Vergleiche quiet vs active für Top-Paar
    if len(result.bins) >= 2:
        top_pair = result.most_activity_dependent[0][0]
        quiet_val = result.bins[0].pair_values.get(top_pair, 0)
        active_val = result.bins[-1].pair_values.get(top_pair, 0)
        change = (active_val - quiet_val) / quiet_val * 100 if quiet_val > 0 else 0

        print(f"""
  ────────────────────────────────────────────────────────────────────────

  BEISPIEL: {top_pair[0]}-{top_pair[1]} Å

    Ruhig:  ΔMI_sector = {quiet_val:.4f} bits
    Aktiv:  ΔMI_sector = {active_val:.4f} bits
    Änderung: {change:+.1f}%

  INTERPRETATION:
    {'✓ Starke Aktivitätsabhängigkeit: Kopplung variiert mit Sonnenaktivität.' if abs(change) > 20 else
     '~ Moderate Aktivitätsabhängigkeit.' if abs(change) > 10 else
     '○ Schwache Aktivitätsabhängigkeit: Kopplung ist relativ stabil.'}

═══════════════════════════════════════════════════════════════════════
""")


# ============================================================================
# 27-DAY ROTATION ANALYSIS
# ============================================================================

@dataclass
class RotationAnalysisResult:
    """Ergebnis der 27-Tage-Rotationsanalyse."""
    hours: float
    n_points: int
    cadence_minutes: int
    start_time: str
    end_time: str

    # Kopplungswerte über Zeit
    pair_timeseries: Dict[Tuple[int, int], List[float]]
    pair_means: Dict[Tuple[int, int], float]
    pair_stds: Dict[Tuple[int, int], float]

    # Zeitliche Stabilität
    temporal_correlations: Dict[Tuple[int, int], float]  # Autokorrelation

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
    """Berechnet Zwischenergebnis aus aktuellen Daten."""
    # Mittelwerte und Standardabweichungen
    pair_means = {pair: float(np.mean(vals)) if vals else 0.0
                  for pair, vals in pair_timeseries.items()}
    pair_stds = {pair: float(np.std(vals)) if vals else 0.0
                 for pair, vals in pair_timeseries.items()}

    # Autokorrelation
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
    """Lädt Checkpoint falls vorhanden."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            data = json.load(f)
        # Konvertiere String-Keys zurück zu Tupeln
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
    last_index: int
) -> None:
    """Speichert Checkpoint für Resume."""
    data = {
        "pair_timeseries": {f"{p[0]}-{p[1]}": v for p, v in pair_timeseries.items()},
        "timestamps": timestamps,
        "last_index": last_index
    }
    with open(checkpoint_path, "w") as f:
        json.dump(data, f)


def run_rotation_analysis(
    hours: float = 648.0,  # 27 Tage
    cadence_minutes: int = 60,  # Stündliche Kadenz
    seed: int = 42,
    output_dir: str = "results/rotation",
    use_real_data: bool = True,
    start_time_str: Optional[str] = None,
    verbose: bool = True,
    resume: bool = True  # Automatisch fortsetzen falls Checkpoint existiert
) -> RotationAnalysisResult:
    """
    Führt 27-Tage-Rotationsanalyse mit echten AIA-Daten durch.

    Analysiert die Kopplungsstabilität über eine vollständige Sonnenrotation.

    Args:
        hours: Zeitraum (default: 648h = 27 Tage)
        cadence_minutes: Kadenz (default: 60 min für Effizienz)
        seed: Random Seed
        output_dir: Output-Verzeichnis
        use_real_data: Echte AIA-Daten verwenden
        start_time_str: Startzeit (ISO format)
        verbose: Ausführliche Ausgabe

    Returns:
        RotationAnalysisResult
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n_points = max(1, int(hours * 60 / cadence_minutes))
    days = hours / 24

    if verbose:
        print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║            🌞 27-TAGE-ROTATIONSANALYSE 🌱                               ║
╚═══════════════════════════════════════════════════════════════════════╝

  Sonnenrotationsperiode: ~27.3 Tage (Carrington-Rotation)

  Konfiguration:
    Zeitraum:     {hours}h ({days:.1f} Tage)
    Kadenz:       {cadence_minutes} min
    Datenpunkte:  {n_points}
    Datenquelle:  {'Echte AIA-Daten' if use_real_data else 'Synthetische Daten'}
""")

    # Bestimme Start- und Endzeit
    if start_time_str is None:
        # Default: 27 Tage vor jetzt
        start_time = datetime.now() - timedelta(hours=hours)
        start_time_str = start_time.isoformat()
    else:
        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))

    end_time = start_time + timedelta(hours=hours)

    if verbose:
        print(f"  Zeitraum:     {start_time_str[:10]} bis {end_time.isoformat()[:10]}")
        print()

    # Checkpoint-Pfad
    checkpoint_path = out_path / "checkpoint.json"

    # Prüfe ob Resume möglich
    pair_timeseries: Dict[Tuple[int, int], List[float]] = {
        pair: [] for pair in combinations(WAVELENGTHS, 2)
    }
    timestamps: List[str] = []
    start_index = 0

    if resume and checkpoint_path.exists():
        pair_timeseries, timestamps, start_index = load_checkpoint(checkpoint_path)
        if start_index > 0 and verbose:
            print(f"  🔄 Resume von Checkpoint: {start_index}/{n_points} bereits verarbeitet")
            print(f"     Fortfahren ab Zeitpunkt {start_index + 1}...")
            print()

    # Verarbeite Zeitpunkte einzeln (streaming statt batch)
    if use_real_data:
        import gc
        from solar_seed.multichannel import load_aia_multichannel

        if verbose and start_index == 0:
            print(f"  📡 Lade und analysiere {n_points} Zeitpunkte...")
        elif verbose:
            print(f"  📡 Lade und analysiere verbleibende {n_points - start_index} Zeitpunkte...")

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
                # Analysiere alle Paare
                for wl1, wl2 in combinations(WAVELENGTHS, 2):
                    result = analyze_pair(
                        channels[wl1], channels[wl2],
                        wl1, wl2,
                        bins=64,
                        seed=seed + i
                    )
                    pair_timeseries[(wl1, wl2)].append(result.delta_mi_sector)

                timestamps.append(timestamp)
                failed_count = 0

                if verbose:
                    print("✓")

                # Bei jedem Zeitpunkt: Ergebnisse + Checkpoint speichern
                interim_result = _compute_interim_result(
                    pair_timeseries, timestamps, hours, cadence_minutes,
                    start_time_str, end_time.isoformat()
                )
                save_rotation_results(interim_result, out_path, timestamps)
                save_checkpoint(checkpoint_path, pair_timeseries, timestamps, i + 1)

                # Garbage Collection alle 10 Zeitpunkte
                if (i + 1) % 10 == 0:
                    gc.collect()
            else:
                failed_count += 1
                if verbose:
                    print("⚠️ übersprungen")

                if failed_count >= 10:
                    if verbose:
                        print(f"    ✗ Abbruch: 10 aufeinanderfolgende Fehler")
                    break

            t += timedelta(minutes=cadence_minutes)

        # Finaler Checkpoint nur wenn Daten vorhanden
        if len(timestamps) > 0:
            save_checkpoint(checkpoint_path, pair_timeseries, timestamps, len(timestamps))

        if len(timestamps) == 0:
            print("  ✗ Keine Daten geladen.")
            raise RuntimeError("Keine AIA-Daten verfügbar")

        if verbose:
            print(f"\n  ✓ {len(timestamps)} Zeitpunkte erfolgreich verarbeitet")

    else:
        if verbose:
            print(f"  📊 Generiere synthetische Daten...")

        from solar_seed.multichannel import generate_multichannel_timeseries
        timeseries = generate_multichannel_timeseries(
            n_points=n_points,
            seed=seed,
            cadence_minutes=cadence_minutes
        )

        for t_idx, (channels, timestamp) in enumerate(timeseries):
            if verbose and (t_idx + 1) % 50 == 0:
                print(f"     Zeitpunkt {t_idx + 1}/{len(timeseries)}...")

            timestamps.append(timestamp)

            for wl1, wl2 in combinations(WAVELENGTHS, 2):
                result = analyze_pair(
                    channels[wl1], channels[wl2],
                    wl1, wl2,
                    bins=64,
                    seed=seed + t_idx
                )
                pair_timeseries[(wl1, wl2)].append(result.delta_mi_sector)

    if verbose:
        print("\n  📈 Berechne Statistiken...")

    # Berechne Mittelwerte, Standardabweichungen
    pair_means = {pair: float(np.mean(vals)) for pair, vals in pair_timeseries.items()}
    pair_stds = {pair: float(np.std(vals)) for pair, vals in pair_timeseries.items()}

    # Zeitliche Autokorrelation (lag-1)
    temporal_correlations = {}
    for pair, values in pair_timeseries.items():
        if len(values) > 2:
            vals = np.array(values)
            corr = np.corrcoef(vals[:-1], vals[1:])[0, 1]
            temporal_correlations[pair] = float(corr) if not np.isnan(corr) else 0.0
        else:
            temporal_correlations[pair] = 0.0

    # Rankings nach mittlerem ΔMI_sector
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

    # Speichere Ergebnisse
    if verbose:
        print("\n  💾 Speichere Ergebnisse...")

    save_rotation_results(result, out_path, timestamps)

    if verbose:
        print_rotation_summary(result)

    return result


def save_rotation_results(
    result: RotationAnalysisResult,
    output_dir: Path,
    timestamps: List[str]
) -> None:
    """Speichert Rotations-Ergebnisse."""

    # 1. Hauptergebnis als Text
    with open(output_dir / "rotation_analysis.txt", "w") as f:
        f.write("27-TAGE-ROTATIONSANALYSE\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Zeitraum:     {result.start_time[:10]} bis {result.end_time[:10]}\n")
        f.write(f"Dauer:        {result.hours}h ({result.hours/24:.1f} Tage)\n")
        f.write(f"Kadenz:       {result.cadence_minutes} min\n")
        f.write(f"Datenpunkte:  {result.n_points}\n\n")

        f.write("KOPPLUNGS-RANKING (ΔMI_sector):\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Rang':<6} {'Paar':<12} {'Mean':<12} {'Std':<12} {'Autokorr'}\n")
        f.write("-" * 70 + "\n")

        sorted_pairs = sorted(result.pair_rankings.items(), key=lambda x: x[1])
        for pair, rank in sorted_pairs:
            mean = result.pair_means[pair]
            std = result.pair_stds[pair]
            autocorr = result.temporal_correlations[pair]
            f.write(f"{rank:<6} {pair[0]}-{pair[1]:<7} {mean:<12.4f} {std:<12.4f} {autocorr:.3f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("\nZEITLICHE STABILITÄT (Autokorrelation lag-1):\n")
        f.write("-" * 50 + "\n")

        sorted_by_autocorr = sorted(
            result.temporal_correlations.items(),
            key=lambda x: -x[1]
        )

        f.write("\nHöchste Autokorrelation (stabilste Kopplung):\n")
        for pair, corr in sorted_by_autocorr[:5]:
            f.write(f"  {pair[0]}-{pair[1]} Å: r = {corr:.3f}\n")

        f.write("\nNiedrigste Autokorrelation (variabelste Kopplung):\n")
        for pair, corr in sorted_by_autocorr[-5:]:
            f.write(f"  {pair[0]}-{pair[1]} Å: r = {corr:.3f}\n")

    # 2. Zeitreihen als CSV
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

    # 3. JSON für weitere Verarbeitung
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
        "pair_rankings": {f"{p[0]}-{p[1]}": v for p, v in result.pair_rankings.items()}
    }

    with open(output_dir / "rotation_analysis.json", "w") as f:
        json.dump(data, f, indent=2)


def print_rotation_summary(result: RotationAnalysisResult) -> None:
    """Druckt Zusammenfassung der Rotationsanalyse."""

    # Top-5 Paare
    sorted_pairs = sorted(result.pair_means.items(), key=lambda x: -x[1])

    # Mittlere Autokorrelation
    mean_autocorr = np.mean(list(result.temporal_correlations.values()))

    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║              🌞 ROTATIONSANALYSE ERGEBNIS 🌱                            ║
╚═══════════════════════════════════════════════════════════════════════╝

  Zeitraum: {result.start_time[:10]} → {result.end_time[:10]}
  Dauer:    {result.hours/24:.1f} Tage ({result.n_points} Datenpunkte)

  ════════════════════════════════════════════════════════════════════════

  TOP 5 STÄRKSTE KOPPLUNGEN (27-Tage-Mittel):
""")

    for i, (pair, mean) in enumerate(sorted_pairs[:5], 1):
        std = result.pair_stds[pair]
        autocorr = result.temporal_correlations[pair]
        print(f"    {i}. {pair[0]}-{pair[1]} Å: ΔMI = {mean:.4f} ± {std:.4f} (r = {autocorr:.2f})")

    print(f"""
  ────────────────────────────────────────────────────────────────────────

  ZEITLICHE STABILITÄT:

    Mittlere Autokorrelation: {mean_autocorr:.3f}
    → {'Hohe zeitliche Stabilität' if mean_autocorr > 0.7 else 'Moderate Stabilität' if mean_autocorr > 0.4 else 'Variable Kopplung'}

  ────────────────────────────────────────────────────────────────────────

  OUTPUT-DATEIEN:
    results/rotation/rotation_analysis.txt
    results/rotation/rotation_analysis.json
    results/rotation/coupling_evolution.csv

═══════════════════════════════════════════════════════════════════════
""")


# ============================================================================
# COMBINED FINAL ANALYSIS
# ============================================================================

def run_final_analysis(
    output_dir: str = "results/final",
    use_real_data: bool = False,
    verbose: bool = True
) -> Tuple[TimescaleComparison, ActivityConditioningResult]:
    """
    Führt beide finalen Analysen durch.

    Args:
        output_dir: Output-Verzeichnis
        use_real_data: Echte Daten verwenden
        verbose: Ausführliche Ausgabe

    Returns:
        (TimescaleComparison, ActivityConditioningResult)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("""
╔═══════════════════════════════════════════════════════════════════════╗
║               🌞 FINALE SOLAR SEED ANALYSEN 🌱                          ║
╚═══════════════════════════════════════════════════════════════════════╝

  Zwei Abschluss-Analysen für Tiefe statt Breite:

    1. Zeitskalen-Vergleich (24h vs 27d)
       → Ist die Temperatur-Ordnung stabil?

    2. Aktivitäts-Konditionierung (94Å-Proxy)
       → Korreliert Kopplung mit Sonnenaktivität?

═══════════════════════════════════════════════════════════════════════
""")

    # Analyse 1: Zeitskalen
    timescale_result = run_timescale_comparison(
        short_hours=24.0,
        long_hours=648.0,  # 27 Tage
        output_dir=output_dir,
        use_real_data=use_real_data,
        verbose=verbose
    )

    # Analyse 2: Aktivität
    activity_result = run_activity_conditioning(
        n_hours=48.0,
        output_dir=output_dir,
        use_real_data=use_real_data,
        verbose=verbose
    )

    # Kombinierte Zusammenfassung
    if verbose:
        print_final_summary(timescale_result, activity_result, out_path)

    return timescale_result, activity_result


def print_final_summary(
    timescale: TimescaleComparison,
    activity: ActivityConditioningResult,
    output_dir: Path
) -> None:
    """Druckt kombinierte Zusammenfassung."""

    summary = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                  🌞 FINALE ERGEBNISSE 🌱                                ║
╚═══════════════════════════════════════════════════════════════════════╝

  1. ZEITSKALEN-STABILITÄT:

     Spearman ρ = {timescale.spearman_rho:.3f}
     → {'Die Kopplungs-Ordnung ist zeitlich STABIL' if timescale.spearman_rho > 0.7 else 'Dynamische Variation über Zeitskalen'}

  2. AKTIVITÄTS-ABHÄNGIGKEIT:

     Stärkstes Signal: {activity.most_activity_dependent[0][0][0]}-{activity.most_activity_dependent[0][0][1]} Å
     r = {activity.most_activity_dependent[0][1]:.3f}
     → {'Kopplung korreliert mit Aktivität' if abs(activity.most_activity_dependent[0][1]) > 0.3 else 'Kopplung ist aktivitätsunabhängig'}

  ════════════════════════════════════════════════════════════════════════

  WISSENSCHAFTLICHE SCHLUSSFOLGERUNG:

  {"✓ Die lokale Strukturkopplung (ΔMI_sector) ist ein robustes Signal." if timescale.spearman_rho > 0.6 else ""}
  {"  Sie bleibt über Zeitskalen erhalten." if timescale.spearman_rho > 0.6 else ""}
  {"  Sie zeigt physikalisch sinnvolle Aktivitätsabhängigkeit." if abs(activity.most_activity_dependent[0][1]) > 0.2 else ""}

  OUTPUT-DATEIEN:
    {output_dir}/timescale_comparison.txt
    {output_dir}/timescale_comparison.json
    {output_dir}/activity_conditioning.txt
    {output_dir}/activity_conditioning.json

═══════════════════════════════════════════════════════════════════════
"""
    print(summary)

    # Auch als Datei speichern
    with open(output_dir / "final_summary.txt", "w") as f:
        f.write(summary)


# ============================================================================
# CLI
# ============================================================================

def main():
    """Hauptfunktion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Finale Analysen für Solar Seed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python -m solar_seed.final_analysis
  python -m solar_seed.final_analysis --timescale-only
  python -m solar_seed.final_analysis --activity-only
  python -m solar_seed.final_analysis --real
  python -m solar_seed.final_analysis --rotation --start "2024-01-01T00:00:00"
        """
    )
    parser.add_argument("--output", type=str, default="results/final",
                        help="Output-Verzeichnis")
    parser.add_argument("--real", action="store_true",
                        help="Echte AIA-Daten verwenden")
    parser.add_argument("--timescale-only", action="store_true",
                        help="Nur Zeitskalen-Vergleich")
    parser.add_argument("--activity-only", action="store_true",
                        help="Nur Aktivitäts-Konditionierung")
    parser.add_argument("--rotation", action="store_true",
                        help="27-Tage-Rotationsanalyse mit echten AIA-Daten")
    parser.add_argument("--short-hours", type=float, default=24.0,
                        help="Kurze Zeitskala in Stunden")
    parser.add_argument("--long-hours", type=float, default=648.0,
                        help="Lange Zeitskala in Stunden (27d = 648)")
    parser.add_argument("--start", type=str, default=None,
                        help="Startzeit für Rotationsanalyse (ISO format)")
    parser.add_argument("--cadence", type=int, default=60,
                        help="Kadenz in Minuten für Rotationsanalyse (default: 60)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Nicht von Checkpoint fortsetzen, neu starten")

    args = parser.parse_args()

    if args.rotation:
        run_rotation_analysis(
            hours=648.0,  # 27 Tage
            cadence_minutes=args.cadence,
            output_dir="results/rotation",
            use_real_data=True,  # Immer echte Daten für Rotation
            start_time_str=args.start,
            verbose=True,
            resume=not args.no_resume
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
