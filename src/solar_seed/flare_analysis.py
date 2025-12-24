#!/usr/bin/env python3
"""
Flare-Ereignis-Analyse für Solar Seed
=====================================

Analysiert ΔMI_sector vor, während und nach Flare-Ereignissen.

Hypothese:
- VOR:     Baseline-Kopplung
- WÄHREND: Starker Anstieg (besonders 94-131 Å)
- NACH:    Abklingen zur Baseline

Bekannte Flares für Tests:
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
    """Ergebnisse für eine Flare-Phase."""
    phase: str  # "before", "during", "after"
    n_samples: int
    timestamps: List[str]

    # Mittlere Werte pro Paar
    pair_values: Dict[Tuple[int, int], float]  # ΔMI_sector
    pair_stds: Dict[Tuple[int, int], float]

    # 94Å Intensität als Aktivitäts-Indikator
    mean_94A_intensity: float


@dataclass
class FlareAnalysisResult:
    """Gesamtergebnis der Flare-Analyse."""
    flare_id: str
    flare_class: str
    peak_time: str

    # Phasen
    before: FlarePhase
    during: FlarePhase
    after: FlarePhase

    # Änderungen
    coupling_change: Dict[Tuple[int, int], Dict[str, float]]  # Paar -> {before_to_during, during_to_after}

    # Top-Paare nach Änderung
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
    Generiert synthetische Flare-Zeitreihe.

    Args:
        n_before: Zeitpunkte vor Flare
        n_during: Zeitpunkte während Flare
        n_after: Zeitpunkte nach Flare
        flare_intensity: Verstärkungsfaktor für Flare-Kanäle
        seed: Random Seed

    Returns:
        (timeseries, phases) - Zeitreihe und Phase pro Zeitpunkt
    """
    rng = np.random.default_rng(seed)
    results = []
    phases = []

    base_time = datetime.now()
    t_idx = 0

    # BEFORE: normale Sonne
    for i in range(n_before):
        timestamp = (base_time + timedelta(minutes=2 * t_idx)).isoformat()
        channels = generate_multichannel_sun(
            n_active_regions=3 + rng.integers(0, 3),
            seed=seed + t_idx
        )
        results.append((channels, timestamp))
        phases.append("before")
        t_idx += 1

    # DURING: verstärkte Flare-Kanäle
    for i in range(n_during):
        timestamp = (base_time + timedelta(minutes=2 * t_idx)).isoformat()
        channels = generate_multichannel_sun(
            n_active_regions=5 + rng.integers(0, 3),
            seed=seed + t_idx
        )

        # Verstärke 94 und 131 Å (Flare-Kanäle)
        # Die Verstärkung nimmt zur Mitte zu (Gauss-Profil)
        peak_factor = np.exp(-((i - n_during/2)**2) / (n_during/2))
        intensity = 1.0 + (flare_intensity - 1.0) * peak_factor

        for wl in [94, 131]:
            mask = channels[wl] > 0
            channels[wl][mask] *= intensity
            # Zusätzliche Flare-Struktur
            channels[wl][mask] += rng.normal(0, 500 * intensity, mask.sum())

        results.append((channels, timestamp))
        phases.append("during")
        t_idx += 1

    # AFTER: abklingende Aktivität
    for i in range(n_after):
        timestamp = (base_time + timedelta(minutes=2 * t_idx)).isoformat()

        # Exponentielles Abklingen
        decay = np.exp(-i / 5)
        n_regions = int(5 - 2 * (1 - decay))

        channels = generate_multichannel_sun(
            n_active_regions=max(3, n_regions),
            seed=seed + t_idx
        )

        # Leicht erhöhte Restaktivität
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
    Lädt echte AIA-Daten um ein Flare-Ereignis.

    Args:
        peak_time: Peak-Zeit des Flares (ISO format)
        minutes_before: Minuten vor Peak
        minutes_after: Minuten nach Peak
        cadence_minutes: Zeitabstand
        data_dir: Download-Verzeichnis
        verbose: Ausführliche Ausgabe

    Returns:
        (timeseries, phases)
    """
    peak = datetime.fromisoformat(peak_time.replace('Z', '+00:00'))

    results = []
    phases = []

    # Zeitpunkte berechnen
    n_before = minutes_before // cadence_minutes
    n_after = minutes_after // cadence_minutes

    # Fenster für "during" = Peak ± 5 Minuten
    during_window = 5

    total = n_before + n_after + 1

    if verbose:
        print(f"  📥 Lade {total} Zeitpunkte um {peak_time[:19]}...")

    for i in range(-n_before, n_after + 1):
        t = peak + timedelta(minutes=cadence_minutes * i)
        timestamp = t.isoformat()

        # Phase bestimmen
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
            print(f"    ⚠️  Übersprungen")

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
    Analysiert eine einzelne Flare-Phase.
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

    # Sammle Werte pro Paar
    pair_values: Dict[Tuple[int, int], List[float]] = {
        pair: [] for pair in combinations(WAVELENGTHS, 2)
    }
    intensities_94 = []
    timestamps = []

    for t_idx, (channels, timestamp) in enumerate(timeseries):
        timestamps.append(timestamp)

        # 94Å Intensität
        intensities_94.append(float(np.mean(channels[94][channels[94] > 0])))

        # Analysiere alle Paare
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
    Führt komplette Flare-Analyse durch.

    Args:
        flare_id: ID aus KNOWN_FLARES (z.B. "X5.0_2024-01-01")
        peak_time: Alternativ: direkte Peak-Zeit
        minutes_before/after: Analyse-Fenster
        cadence_minutes: Zeitabstand
        use_real_data: Echte AIA-Daten
        output_dir: Output-Verzeichnis
        verbose: Ausführliche Ausgabe

    Returns:
        FlareAnalysisResult
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Flare-Info bestimmen
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
║              🔥 FLARE-EREIGNIS-ANALYSE 🌞                              ║
╚═══════════════════════════════════════════════════════════════════════╝

  Flare:    {flare_id} ({flare_class})
  Peak:     {peak_time[:19]}
  Fenster:  -{minutes_before}min ... +{minutes_after}min
  Kadenz:   {cadence_minutes}min
  Daten:    {'Echt (AIA)' if use_real_data else 'Synthetisch'}
""")

    # Daten laden
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
        raise RuntimeError("Keine Daten geladen")

    if verbose:
        print(f"\n  📊 Analysiere {len(timeseries)} Zeitpunkte...")

    # Nach Phase gruppieren
    before_data = [(ch, ts) for (ch, ts), p in zip(timeseries, phases) if p == "before"]
    during_data = [(ch, ts) for (ch, ts), p in zip(timeseries, phases) if p == "during"]
    after_data = [(ch, ts) for (ch, ts), p in zip(timeseries, phases) if p == "after"]

    if verbose:
        print(f"     Before: {len(before_data)}, During: {len(during_data)}, After: {len(after_data)}")

    # Analysiere jede Phase
    before_result = analyze_flare_phase(before_data, "before")
    during_result = analyze_flare_phase(during_data, "during")
    after_result = analyze_flare_phase(after_data, "after")

    # Berechne Änderungen
    coupling_change = {}
    for pair in combinations(WAVELENGTHS, 2):
        before_val = before_result.pair_values.get(pair, 0)
        during_val = during_result.pair_values.get(pair, 0)
        after_val = after_result.pair_values.get(pair, 0)

        # Prozentuale Änderung
        before_to_during = ((during_val - before_val) / before_val * 100) if before_val > 0 else 0
        during_to_after = ((after_val - during_val) / during_val * 100) if during_val > 0 else 0

        coupling_change[pair] = {
            "before_to_during": before_to_during,
            "during_to_after": during_to_after,
            "before": before_val,
            "during": during_val,
            "after": after_val
        }

    # Top-5 nach Anstieg während Flare
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

    # Speichere und drucke
    save_flare_results(result, out_path)

    if verbose:
        print_flare_summary(result)

    return result


def save_flare_results(result: FlareAnalysisResult, output_dir: Path) -> None:
    """Speichert Flare-Ergebnisse."""

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
    """Druckt Zusammenfassung."""

    # 94-131 Werte
    before_94_131 = result.before.pair_values.get((94, 131), 0)
    during_94_131 = result.during.pair_values.get((94, 131), 0)
    after_94_131 = result.after.pair_values.get((94, 131), 0)

    change_94_131 = result.coupling_change.get((94, 131), {}).get("before_to_during", 0)

    print(f"""
  ════════════════════════════════════════════════════════════════════════

  PHASEN-VERGLEICH:

    Phase      n    94Å Intensität    ΔMI_sector (94-131)
    ─────────────────────────────────────────────────────
    BEFORE    {result.before.n_samples:>2}    {result.before.mean_94A_intensity:>10.1f}         {before_94_131:.4f} bits
    DURING    {result.during.n_samples:>2}    {result.during.mean_94A_intensity:>10.1f}         {during_94_131:.4f} bits
    AFTER     {result.after.n_samples:>2}    {result.after.mean_94A_intensity:>10.1f}         {after_94_131:.4f} bits

  ────────────────────────────────────────────────────────────────────────

  TOP 5 BETROFFENE PAARE (Before → During):
""")

    for i, (pair, change) in enumerate(result.most_affected, 1):
        print(f"    {i}. {pair[0]}-{pair[1]} Å: {change:+.1f}%")

    print(f"""
  ────────────────────────────────────────────────────────────────────────

  FLARE-KANAL-KOPPLUNG (94-131 Å):

    Before:  {before_94_131:.4f} bits
    During:  {during_94_131:.4f} bits  ({change_94_131:+.1f}%)
    After:   {after_94_131:.4f} bits

  INTERPRETATION:
    {'✓ Starker Flare-Effekt: Kopplung steigt während des Flares.' if change_94_131 > 50 else
     '~ Moderater Flare-Effekt.' if change_94_131 > 20 else
     '○ Schwacher Flare-Effekt.'}

═══════════════════════════════════════════════════════════════════════
""")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Hauptfunktion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Flare-Ereignis-Analyse für Solar Seed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Bekannte Flares:
  X5.0_2024-01-01  - Starker X-Flare
  X2.8_2023-12-14  - X-Flare
  X1.0_2024-01-10  - Moderater X-Flare
  M5.0_2024-01-22  - M-Flare

Beispiele:
  python -m solar_seed.flare_analysis
  python -m solar_seed.flare_analysis --flare X5.0_2024-01-01 --real
  python -m solar_seed.flare_analysis --peak "2024-01-01T00:55:00" --real
        """
    )
    parser.add_argument("--flare", type=str, default=None,
                        help="Flare-ID aus bekannten Flares")
    parser.add_argument("--peak", type=str, default=None,
                        help="Peak-Zeit (ISO format)")
    parser.add_argument("--before", type=int, default=30,
                        help="Minuten vor Peak (default: 30)")
    parser.add_argument("--after", type=int, default=30,
                        help="Minuten nach Peak (default: 30)")
    parser.add_argument("--cadence", type=int, default=2,
                        help="Kadenz in Minuten (default: 2)")
    parser.add_argument("--real", action="store_true",
                        help="Echte AIA-Daten verwenden")
    parser.add_argument("--output", type=str, default="results/flare",
                        help="Output-Verzeichnis")
    parser.add_argument("--list", action="store_true",
                        help="Bekannte Flares auflisten")

    args = parser.parse_args()

    if args.list:
        print("\nBekannte Flares:")
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
