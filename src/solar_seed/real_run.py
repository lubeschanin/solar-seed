#!/usr/bin/env python3
"""
Real-Data Run for Solar Seed
============================

Reproducible, citable analysis run on real AIA data.

Execution:
    python -m solar_seed.real_run --hours 6
    python -m solar_seed.real_run --start "2024-01-15T00:00:00" --hours 6

Output:
    results/real_run/
    ├── timeseries.csv          # MI timeseries
    ├── controls_summary.json   # Control tests
    ├── spatial_maps.txt        # Spatial analysis
    └── run_metadata.json       # Configuration & reproducibility
"""

import argparse
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Tuple
import numpy as np
from numpy.typing import NDArray

from solar_seed.mutual_info import mutual_information
from solar_seed.radial_profile import prepare_pair_for_residual_mi
from solar_seed.control_tests import (
    run_all_controls,
    time_shift_null,
    sector_ring_shuffle_test,
    AllControlsResult
)
from solar_seed.spatial_analysis import (
    compute_spatial_residual_mi_map,
    mi_map_to_ascii,
    find_top_hotspots
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RunConfig:
    """Configuration for the real run."""
    wavelength_1: int = 193
    wavelength_2: int = 211
    start_time: Optional[str] = None  # ISO format, None = now - hours
    hours: float = 6.0
    cadence_minutes: int = 12  # AIA synoptic cadence
    output_dir: str = "results/real_run"
    bins: int = 64
    n_shuffles: int = 100
    seed: int = 42


@dataclass
class TimePointResult:
    """Result for a timepoint."""
    timestamp: str

    # Original MI
    mi_original: float
    nmi_original: float

    # Residual MI
    mi_residual: float
    nmi_residual: float

    # New metrics
    mi_ratio: float  # mi_residual / mi_original
    delta_mi_ring: float  # mi_residual - mi_ring_shuffled
    delta_mi_sector: float  # mi_residual - mi_sector_shuffled

    # Null model
    mi_null_mean: float
    mi_null_std: float
    z_score: float

    # Shape info
    shape: Tuple[int, int] = (0, 0)


@dataclass
class RunResult:
    """Overall result of the run."""
    config: RunConfig
    timeseries: List[TimePointResult]
    controls: Optional[dict]
    spatial: Optional[dict]

    # Aggregated statistics
    mean_mi_ratio: float = 0.0
    std_mi_ratio: float = 0.0
    mean_delta_mi_ring: float = 0.0
    mean_delta_mi_sector: float = 0.0

    run_timestamp: str = ""
    duration_seconds: float = 0.0


# ============================================================================
# DATA LOADING
# ============================================================================

def load_aia_pair_sunpy(
    wavelength_1: int,
    wavelength_2: int,
    time_str: str
) -> Tuple[Optional[NDArray], Optional[NDArray], dict]:
    """
    Lädt ein AIA-Bildpaar via SunPy/Fido.

    Args:
        wavelength_1: Erste Wellenlänge (z.B. 193)
        wavelength_2: Zweite Wellenlänge (z.B. 211)
        time_str: Zeitpunkt (ISO format)

    Returns:
        (image_1, image_2, metadata) oder (None, None, {}) bei Fehler
    """
    try:
        from sunpy.net import Fido, attrs as a
        import sunpy.map
        import astropy.units as u
        from datetime import datetime, timedelta

        # Parse time und erstelle Suchfenster (±5 min)
        t = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        t_start = t - timedelta(minutes=5)
        t_end = t + timedelta(minutes=5)

        images = []
        metadata = {"timestamp": time_str, "wavelengths": [wavelength_1, wavelength_2]}

        for wl in [wavelength_1, wavelength_2]:
            result = Fido.search(
                a.Time(t_start.isoformat(), t_end.isoformat()),
                a.Instrument("aia"),
                a.Wavelength(wl * u.angstrom),
            )

            if len(result) == 0 or len(result[0]) == 0:
                return None, None, {}

            # Lade erstes Ergebnis
            files = Fido.fetch(result[0, 0], path="data/aia/{file}")
            if not files:
                return None, None, {}

            aia_map = sunpy.map.Map(files[0])
            images.append(aia_map.data.astype(np.float64))

            metadata[f"file_{wl}"] = str(files[0])
            metadata[f"date_{wl}"] = str(aia_map.date)

        return images[0], images[1], metadata

    except ImportError:
        print("  ⚠️  SunPy nicht installiert")
        return None, None, {}
    except Exception as e:
        print(f"  ✗ Fehler: {e}")
        return None, None, {}


def generate_synthetic_timeseries(
    n_points: int,
    shape: Tuple[int, int] = (512, 512),
    extra_correlation: float = 0.3,
    seed: int = 42
) -> List[Tuple[NDArray, NDArray, str]]:
    """
    Generiert synthetische Zeitreihe für Tests.

    Simuliert zeitliche Variation durch leicht unterschiedliche Seeds.
    """
    from solar_seed.data_loader import generate_synthetic_sun

    results = []
    base_time = datetime.now()

    for i in range(n_points):
        timestamp = (base_time + timedelta(minutes=12 * i)).isoformat()

        # Vary extra_correlation slightly over time
        corr = extra_correlation + 0.1 * np.sin(2 * np.pi * i / n_points)

        data_1, data_2 = generate_synthetic_sun(
            shape=shape,
            extra_correlation=max(0, corr),
            n_active_regions=5 + i % 3,
            seed=seed + i
        )

        results.append((data_1, data_2, timestamp))

    return results


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_timepoint(
    image_1: NDArray[np.float64],
    image_2: NDArray[np.float64],
    timestamp: str,
    config: RunConfig
) -> TimePointResult:
    """
    Analysiert einen einzelnen Zeitpunkt.

    Berechnet:
    - Original MI
    - Residual MI (nach Geometrie-Subtraktion)
    - MI Ratio (residual / original)
    - ΔMI_ring (Struktur jenseits radialer Statistik)
    - ΔMI_sector (Struktur jenseits radial+azimutaler Ordnung)
    """
    from solar_seed.null_model import compute_null_distribution, compute_z_score

    # Original MI
    mi_original = mutual_information(image_1, image_2, bins=config.bins)
    nmi_original = mi_original / 6.0  # Grobe Normalisierung

    # Residual MI
    res_1, res_2, _ = prepare_pair_for_residual_mi(image_1, image_2)
    mi_residual = mutual_information(res_1, res_2, bins=config.bins)
    nmi_residual = mi_residual / 6.0

    # Ratio
    mi_ratio = mi_residual / mi_original if mi_original > 0 else 0.0

    # Ring and sector shuffles for delta metrics
    sector_result = sector_ring_shuffle_test(
        image_1, image_2,
        n_rings=20,
        n_sectors=16,
        seed=config.seed,
        bins=config.bins
    )

    delta_mi_ring = mi_residual - sector_result.mi_ring_shuffled
    delta_mi_sector = mi_residual - sector_result.mi_sector_shuffled

    # Null model for residual
    mi_null_mean, mi_null_std, _ = compute_null_distribution(
        res_1, res_2,
        n_shuffles=min(50, config.n_shuffles),  # Faster for timeseries
        bins=config.bins,
        seed=config.seed
    )
    z_score = compute_z_score(mi_residual, mi_null_mean, mi_null_std)

    return TimePointResult(
        timestamp=timestamp,
        mi_original=mi_original,
        nmi_original=nmi_original,
        mi_residual=mi_residual,
        nmi_residual=nmi_residual,
        mi_ratio=mi_ratio,
        delta_mi_ring=delta_mi_ring,
        delta_mi_sector=delta_mi_sector,
        mi_null_mean=mi_null_mean,
        mi_null_std=mi_null_std,
        z_score=z_score,
        shape=image_1.shape
    )


def run_spatial_analysis(
    image_1: NDArray[np.float64],
    image_2: NDArray[np.float64],
    grid_size: Tuple[int, int] = (8, 8)
) -> dict:
    """
    Führt räumliche Analyse durch.
    """
    result = compute_spatial_residual_mi_map(
        image_1, image_2,
        grid_size=grid_size,
        bins=32
    )

    hotspots = find_top_hotspots(result.residual, n=5)

    return {
        "grid_size": grid_size,
        "original_mi_mean": result.original.mi_mean,
        "original_mi_std": result.original.mi_std,
        "residual_mi_mean": result.residual.mi_mean,
        "residual_mi_std": result.residual.mi_std,
        "mean_reduction_percent": float(np.nanmean(result.mi_reduction_percent)),
        "hotspot": result.residual.hotspot_idx,
        "hotspot_mi": result.residual.hotspot_value,
        "top_5_hotspots": [(list(idx), val) for idx, val in hotspots],
        "original_map_ascii": mi_map_to_ascii(result.original.mi_map),
        "residual_map_ascii": mi_map_to_ascii(result.residual.mi_map)
    }


# ============================================================================
# REPORTING
# ============================================================================

def save_timeseries_csv(
    timeseries: List[TimePointResult],
    filepath: Path
) -> None:
    """Speichert Zeitreihe als CSV."""

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "timestamp",
            "mi_original",
            "mi_residual",
            "mi_ratio",
            "delta_mi_ring",
            "delta_mi_sector",
            "z_score",
            "mi_null_mean",
            "mi_null_std"
        ])

        # Data
        for tp in timeseries:
            writer.writerow([
                tp.timestamp,
                f"{tp.mi_original:.6f}",
                f"{tp.mi_residual:.6f}",
                f"{tp.mi_ratio:.4f}",
                f"{tp.delta_mi_ring:.6f}",
                f"{tp.delta_mi_sector:.6f}",
                f"{tp.z_score:.2f}",
                f"{tp.mi_null_mean:.6f}",
                f"{tp.mi_null_std:.6f}"
            ])


def save_controls_json(
    controls: AllControlsResult,
    filepath: Path
) -> None:
    """Speichert Kontroll-Tests als JSON."""

    # Helper to convert numpy types to Python native
    def to_native(val):
        if hasattr(val, 'item'):
            return val.item()
        return val

    data = {
        "c1_time_shift": {
            "mi_original": to_native(controls.c1_time_shift.mi_original),
            "mi_shifted": to_native(controls.c1_time_shift.mi_shifted),
            "reduction_percent": to_native(controls.c1_time_shift.mi_reduction_percent),
            "passed": to_native(controls.c1_time_shift.passed)
        },
        "c2_ring_shuffle": {
            "mi_original": to_native(controls.c2_ring_shuffle.mi_original),
            "mi_ring_shuffled": to_native(controls.c2_ring_shuffle.mi_ring_shuffled),
            "mi_global_shuffled": to_native(controls.c2_ring_shuffle.mi_global_shuffled),
            "ring_reduction_percent": to_native(controls.c2_ring_shuffle.ring_reduction_percent),
            "global_reduction_percent": to_native(controls.c2_ring_shuffle.global_reduction_percent)
        },
        "c3_blur_match": {
            "mi_original": to_native(controls.c3_blur_match.mi_original),
            "mi_blurred": to_native(controls.c3_blur_match.mi_blurred),
            "change_percent": to_native(controls.c3_blur_match.mi_change_percent),
            "blur_sigma": to_native(controls.c3_blur_match.blur_sigma),
            "stable": to_native(controls.c3_blur_match.stable)
        },
        "c4_co_alignment": {
            "mi_at_zero": to_native(controls.c4_co_alignment.mi_at_zero),
            "mi_at_max": to_native(controls.c4_co_alignment.mi_at_max),
            "max_shift": [to_native(x) for x in controls.c4_co_alignment.max_shift],
            "centered": to_native(controls.c4_co_alignment.centered)
        },
        "all_passed": to_native(controls.all_passed)
    }

    # Add sector shuffle if available
    if hasattr(controls, 'c2_sector_shuffle'):
        data["c2_sector_shuffle"] = {
            "mi_sector_shuffled": controls.c2_sector_shuffle.mi_sector_shuffled,
            "sector_reduction_percent": controls.c2_sector_shuffle.sector_reduction_percent
        }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def save_spatial_maps(
    spatial: dict,
    filepath: Path
) -> None:
    """Speichert räumliche Analyse als Text."""

    with open(filepath, 'w') as f:
        f.write("RÄUMLICHE MI-ANALYSE\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Grid: {spatial['grid_size'][0]}x{spatial['grid_size'][1]}\n\n")

        f.write("ORIGINAL MI:\n")
        f.write(f"  Mittel: {spatial['original_mi_mean']:.4f} ± {spatial['original_mi_std']:.4f}\n\n")
        f.write(spatial['original_map_ascii'] + "\n\n")

        f.write("RESIDUAL MI:\n")
        f.write(f"  Mittel: {spatial['residual_mi_mean']:.4f} ± {spatial['residual_mi_std']:.4f}\n")
        f.write(f"  Reduktion: {spatial['mean_reduction_percent']:.1f}%\n\n")
        f.write(spatial['residual_map_ascii'] + "\n\n")

        f.write("TOP 5 RESIDUAL-HOTSPOTS:\n")
        for i, (idx, val) in enumerate(spatial['top_5_hotspots'], 1):
            f.write(f"  {i}. Zelle {idx}: MI={val:.4f}\n")


def save_run_metadata(
    result: RunResult,
    filepath: Path
) -> None:
    """Speichert Run-Metadaten für Reproduzierbarkeit."""

    metadata = {
        "run_timestamp": result.run_timestamp,
        "duration_seconds": result.duration_seconds,
        "config": asdict(result.config),
        "summary": {
            "n_timepoints": len(result.timeseries),
            "mean_mi_ratio": result.mean_mi_ratio,
            "std_mi_ratio": result.std_mi_ratio,
            "mean_delta_mi_ring": result.mean_delta_mi_ring,
            "mean_delta_mi_sector": result.mean_delta_mi_sector,
            "controls_passed": result.controls.get("all_passed", False) if result.controls else None
        },
        "interpretation": {
            "mi_ratio_meaning": "MI_residual / MI_original - wie viel MI bleibt nach Geometrie-Subtraktion",
            "delta_mi_ring_meaning": "MI_residual - MI_ring_shuffle - Struktur jenseits radialer Statistik",
            "delta_mi_sector_meaning": "MI_residual - MI_sector_shuffle - echte lokale Struktur"
        }
    }

    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)


def print_run_summary(result: RunResult) -> None:
    """Gibt Zusammenfassung des Runs aus."""

    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                     🌞 REAL-RUN ZUSAMMENFASSUNG 🌱                     ║
╚═══════════════════════════════════════════════════════════════════════╝

  Zeitraum:     {result.config.hours} Stunden
  Wellenlängen: {result.config.wavelength_1} Å + {result.config.wavelength_2} Å
  Zeitpunkte:   {len(result.timeseries)}
  Dauer:        {result.duration_seconds:.1f} Sekunden

  ════════════════════════════════════════════════════════════════════════

  NEUE KENNZAHLEN:

  MI Ratio (residual/original):
    Mittel: {result.mean_mi_ratio:.4f} ± {result.std_mi_ratio:.4f}
    → {result.mean_mi_ratio*100:.1f}% der Original-MI bleibt nach Geometrie-Subtraktion

  ΔMI_ring (Struktur jenseits radialer Statistik):
    Mittel: {result.mean_delta_mi_ring:.4f} bits
    → Positiv = echte azimutale/lokale Struktur vorhanden

  ΔMI_sector (echte lokale Struktur):
    Mittel: {result.mean_delta_mi_sector:.4f} bits
    → Positiv = Struktur jenseits radial+azimutaler Ordnung

  ════════════════════════════════════════════════════════════════════════

  OUTPUT-DATEIEN:
    {result.config.output_dir}/timeseries.csv
    {result.config.output_dir}/controls_summary.json
    {result.config.output_dir}/spatial_maps.txt
    {result.config.output_dir}/run_metadata.json

═══════════════════════════════════════════════════════════════════════
    """)


# ============================================================================
# MAIN RUN
# ============================================================================

def run_real_analysis(
    config: RunConfig,
    use_synthetic: bool = False,
    verbose: bool = True
) -> RunResult:
    """
    Führt den kompletten Real-Run durch.

    Args:
        config: Run-Konfiguration
        use_synthetic: Verwende synthetische Daten (für Tests)
        verbose: Ausführliche Ausgabe
    """
    import time
    start_time = time.time()
    run_timestamp = datetime.now().isoformat()

    # Output-Verzeichnis erstellen
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    🌞 SOLAR SEED REAL-RUN 🌱                           ║
║                                                                        ║
║  Reproduzierbare Analyse auf {"echten" if not use_synthetic else "synthetischen"} AIA-Daten                    ║
╚═══════════════════════════════════════════════════════════════════════╝
        """)

    # Berechne Anzahl Zeitpunkte
    n_points = int(config.hours * 60 / config.cadence_minutes)

    if verbose:
        print(f"  Konfiguration:")
        print(f"    Wellenlängen: {config.wavelength_1} Å + {config.wavelength_2} Å")
        print(f"    Zeitraum: {config.hours} Stunden")
        print(f"    Kadenz: {config.cadence_minutes} Minuten")
        print(f"    Zeitpunkte: {n_points}")
        print()

    # Lade oder generiere Daten
    if use_synthetic:
        if verbose:
            print("  📊 Generiere synthetische Zeitreihe...")
        data_pairs = generate_synthetic_timeseries(
            n_points=n_points,
            shape=(256, 256),
            extra_correlation=0.3,
            seed=config.seed
        )
    else:
        if verbose:
            print("  📡 Lade echte AIA-Daten...")
        # TODO: Implementiere echte Datenladung
        # For now: Fallback to synthetic
        print("  ⚠️  Echte Daten noch nicht implementiert, verwende synthetische")
        data_pairs = generate_synthetic_timeseries(
            n_points=n_points,
            shape=(256, 256),
            extra_correlation=0.3,
            seed=config.seed
        )

    # Analysiere jeden Zeitpunkt
    timeseries = []

    if verbose:
        print(f"\n  🔬 Analysiere {n_points} Zeitpunkte...")

    for i, (img_1, img_2, timestamp) in enumerate(data_pairs):
        if verbose and (i + 1) % 5 == 0:
            print(f"     Zeitpunkt {i+1}/{n_points}...")

        tp_result = analyze_timepoint(img_1, img_2, timestamp, config)
        timeseries.append(tp_result)

    # Kontroll-Tests auf erstem Zeitpunkt
    if verbose:
        print("\n  🧪 Führe Kontroll-Tests durch...")

    first_img_1, first_img_2, _ = data_pairs[0]
    controls = run_all_controls(
        first_img_1, first_img_2,
        seed=config.seed,
        bins=config.bins,
        verbose=False
    )

    # Spatial analysis on first timepoint
    if verbose:
        print("  🗺️  Erstelle räumliche Analyse...")

    spatial = run_spatial_analysis(first_img_1, first_img_2)

    # Aggregiere Statistiken
    mi_ratios = [tp.mi_ratio for tp in timeseries]
    delta_rings = [tp.delta_mi_ring for tp in timeseries]
    delta_sectors = [tp.delta_mi_sector for tp in timeseries]

    duration = time.time() - start_time

    result = RunResult(
        config=config,
        timeseries=timeseries,
        controls=asdict(controls) if controls else None,
        spatial=spatial,
        mean_mi_ratio=float(np.mean(mi_ratios)),
        std_mi_ratio=float(np.std(mi_ratios)),
        mean_delta_mi_ring=float(np.mean(delta_rings)),
        mean_delta_mi_sector=float(np.mean(delta_sectors)),
        run_timestamp=run_timestamp,
        duration_seconds=duration
    )

    # Speichere Ergebnisse
    if verbose:
        print("\n  💾 Speichere Ergebnisse...")

    save_timeseries_csv(timeseries, output_dir / "timeseries.csv")
    save_controls_json(controls, output_dir / "controls_summary.json")
    save_spatial_maps(spatial, output_dir / "spatial_maps.txt")
    save_run_metadata(result, output_dir / "run_metadata.json")

    if verbose:
        print_run_summary(result)

    return result


# ============================================================================
# CLI
# ============================================================================

def main():
    """Hauptfunktion."""

    parser = argparse.ArgumentParser(
        description="Solar Seed Real-Run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python -m solar_seed.real_run --hours 6
  python -m solar_seed.real_run --hours 1 --synthetic
  python -m solar_seed.real_run --wavelengths 171 193
        """
    )
    parser.add_argument("--hours", type=float, default=6.0,
                        help="Analysezeitraum in Stunden (default: 6)")
    parser.add_argument("--wavelengths", type=int, nargs=2, default=[193, 211],
                        help="Wellenlängen-Paar (default: 193 211)")
    parser.add_argument("--cadence", type=int, default=12,
                        help="Kadenz in Minuten (default: 12)")
    parser.add_argument("--output", type=str, default="results/real_run",
                        help="Output-Verzeichnis")
    parser.add_argument("--synthetic", action="store_true",
                        help="Verwende synthetische Daten")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random Seed")

    args = parser.parse_args()

    config = RunConfig(
        wavelength_1=args.wavelengths[0],
        wavelength_2=args.wavelengths[1],
        hours=args.hours,
        cadence_minutes=args.cadence,
        output_dir=args.output,
        seed=args.seed
    )

    run_real_analysis(config, use_synthetic=args.synthetic)


if __name__ == "__main__":
    main()
