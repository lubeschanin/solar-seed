#!/usr/bin/env python3
"""
Solar Seed Hypothesis Test
==========================

EINE Hypothese. EIN Test. EINE Antwort.

H1: Bestimmte AIA-Wellenlängenpaare zeigen höhere Mutual Information 
    als durch unabhängige thermische Prozesse erklärbar.

Ausführung:
    python -m solar_seed.hypothesis_test
    python -m solar_seed.hypothesis_test --real-data
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

from solar_seed.mutual_info import mutual_information, normalized_mutual_information
from solar_seed.null_model import (
    compute_null_distribution, 
    compute_z_score, 
    compute_p_value,
    interpret_result
)
from solar_seed.data_loader import (
    generate_pure_noise,
    generate_correlated_noise,
    generate_synthetic_sun,
    load_sunpy_sample,
    load_aia_fits
)
from solar_seed.radial_profile import prepare_pair_for_residual_mi
from solar_seed.spatial_analysis import (
    compute_spatial_residual_mi_map,
    print_spatial_comparison
)
from solar_seed.control_tests import (
    run_all_controls,
    print_control_results
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TestConfig:
    """Konfiguration für den Hypothesentest."""
    downsample_factor: int = 8
    n_bins: int = 64
    n_shuffles: int = 100
    output_dir: str = "results"
    seed: int = 42


@dataclass
class TestResult:
    """Ergebnis eines einzelnen Tests."""
    label: str
    mi_real: float
    nmi_real: float
    mi_null_mean: float
    mi_null_std: float
    z_score: float
    p_value: float
    status: str
    interpretation: str


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_single_test(
    data_1: np.ndarray,
    data_2: np.ndarray,
    label: str,
    config: TestConfig
) -> TestResult:
    """
    Führt einen einzelnen Hypothesentest durch.
    
    Args:
        data_1: Erster Kanal
        data_2: Zweiter Kanal
        label: Bezeichnung des Tests
        config: Test-Konfiguration
        
    Returns:
        TestResult mit allen Metriken
    """
    # Downsample
    ds = config.downsample_factor
    if ds > 1:
        data_1 = data_1[::ds, ::ds]
        data_2 = data_2[::ds, ::ds]
    
    print(f"\n  📐 Shape: {data_1.shape}")
    
    # MI berechnen
    print(f"  🔬 Berechne MI...")
    mi_real = mutual_information(data_1, data_2, config.n_bins)
    nmi_real = normalized_mutual_information(data_1, data_2, config.n_bins)
    print(f"     MI_real  = {mi_real:.6f} bits")
    print(f"     NMI_real = {nmi_real:.6f}")
    
    # Nullmodell
    print(f"  🎲 Nullmodell ({config.n_shuffles} Shuffles)...")
    mi_null_mean, mi_null_std, _ = compute_null_distribution(
        data_1, data_2, 
        n_shuffles=config.n_shuffles, 
        bins=config.n_bins,
        seed=config.seed,
        verbose=True
    )
    print(f"     MI_null  = {mi_null_mean:.6f} ± {mi_null_std:.6f}")
    
    # Statistik
    z_score = compute_z_score(mi_real, mi_null_mean, mi_null_std)
    p_value = compute_p_value(mi_real, [])  # Vereinfacht
    
    # Empirischer p-Wert aus Z-Score (Normalapproximation)
    from math import erfc, sqrt
    p_value = 0.5 * erfc(z_score / sqrt(2)) if z_score > 0 else 1.0
    
    print(f"  📈 Z-Score  = {z_score:.2f}")
    print(f"     p-Wert   = {p_value:.4f}")
    
    status, interpretation = interpret_result(z_score, p_value)
    print(f"     Status   = {status}")
    
    return TestResult(
        label=label,
        mi_real=mi_real,
        nmi_real=nmi_real,
        mi_null_mean=mi_null_mean,
        mi_null_std=mi_null_std,
        z_score=z_score,
        p_value=p_value,
        status=status,
        interpretation=interpretation
    )


def run_all_tests(config: TestConfig, use_real_data: bool = False) -> list[TestResult]:
    """
    Führt alle Tests durch.
    
    Args:
        config: Test-Konfiguration
        use_real_data: Versuche echte Sonnendaten zu laden
        
    Returns:
        Liste aller TestResults
    """
    results = []
    
    # TEST 1: Reines Rauschen (Nullhypothese wahr)
    print("\n" + "="*72)
    print("TEST 1: VALIDIERUNG - Reines Rauschen (unabhängig)")
    print("="*72)
    print("  Erwartung: MI_real ≈ MI_null, Z ≈ 0")
    
    data_1, data_2 = generate_pure_noise(shape=(512, 512), seed=config.seed)
    results.append(run_single_test(data_1, data_2, "Reines Rauschen", config))
    
    # TEST 2: Korreliertes Rauschen (Alternative wahr)
    print("\n" + "="*72)
    print("TEST 2: VALIDIERUNG - Korreliertes Rauschen (r=0.5)")
    print("="*72)
    print("  Erwartung: MI_real >> MI_null, Z >> 3")
    
    data_1, data_2 = generate_correlated_noise(shape=(512, 512), correlation=0.5, seed=config.seed)
    results.append(run_single_test(data_1, data_2, "Korreliert (r=0.5)", config))
    
    # TEST 3: Synthetische Sonne - Nur Geometrie
    print("\n" + "="*72)
    print("TEST 3: SYNTHETISCHE SONNE - Nur gemeinsame Geometrie")
    print("="*72)
    print("  Erwartung: MI_real > MI_null (wegen Geometrie)")
    
    data_1, data_2 = generate_synthetic_sun(shape=(512, 512), extra_correlation=0.0, seed=config.seed)
    results.append(run_single_test(data_1, data_2, "Sonne (Geometrie)", config))
    
    # TEST 4: Synthetische Sonne - Mit Extra-Korrelation
    print("\n" + "="*72)
    print("TEST 4: SYNTHETISCHE SONNE - Mit extra Korrelation")
    print("="*72)
    print("  Erwartung: MI_real > Test 3")
    
    data_1, data_2 = generate_synthetic_sun(shape=(512, 512), extra_correlation=0.5, seed=config.seed)
    results.append(run_single_test(data_1, data_2, "Sonne (extra r=0.5)", config))
    
    # TEST 5: Synthetische Sonne - Residuen (nur Geometrie)
    print("\n" + "="*72)
    print("TEST 5: RESIDUAL-ANALYSE - Sonne ohne Geometrie")
    print("="*72)
    print("  Radiale Geometrie wird subtrahiert.")
    print("  Erwartung: MI_residual << MI_original (Geometrie erklärt viel)")

    data_1, data_2 = generate_synthetic_sun(shape=(512, 512), extra_correlation=0.0, seed=config.seed)
    residual_1, residual_2, _ = prepare_pair_for_residual_mi(data_1, data_2)
    results.append(run_single_test(residual_1, residual_2, "Residual (Geometrie)", config))

    # TEST 6: Synthetische Sonne - Residuen mit Extra-Korrelation
    print("\n" + "="*72)
    print("TEST 6: RESIDUAL-ANALYSE - Mit extra Korrelation")
    print("="*72)
    print("  Erwartung: MI_residual > Test 5 (extra Korrelation überlebt)")

    data_1, data_2 = generate_synthetic_sun(shape=(512, 512), extra_correlation=0.5, seed=config.seed)
    residual_1, residual_2, _ = prepare_pair_for_residual_mi(data_1, data_2)
    results.append(run_single_test(residual_1, residual_2, "Residual (extra r=0.5)", config))

    # TEST 7: Echte Daten (optional)
    if use_real_data:
        print("\n" + "="*72)
        print("TEST 7: ECHTE SONNENDATEN")
        print("="*72)
        
        data_1, data_2 = load_sunpy_sample()
        
        if data_1 is not None:
            results.append(run_single_test(data_1, data_2, "AIA Sample", config))
        else:
            print("  ⚠️  Keine echten Daten verfügbar")
            print("  → Installiere: pip install sunpy")
    
    return results


def print_summary(results: list[TestResult]) -> None:
    """Gibt Zusammenfassung aus."""
    
    print("\n" + "="*72)
    print("ZUSAMMENFASSUNG")
    print("="*72)
    
    print(f"\n  {'Test':<22} {'MI_real':>10} {'MI_null':>12} {'Z':>8} {'Status':<20}")
    print("  " + "-"*72)
    
    for r in results:
        print(f"  {r.label:<22} {r.mi_real:>10.4f} {r.mi_null_mean:>10.4f}±{r.mi_null_std:.2f} {r.z_score:>8.2f} {r.status:<20}")


def run_spatial_analysis(config: TestConfig) -> None:
    """
    Führt räumliche MI-Analyse durch.

    Zeigt wo auf der (synthetischen) Sonne die höchste Residual-MI ist.
    """
    print("\n" + "="*72)
    print("RÄUMLICHE MI-ANALYSE")
    print("="*72)
    print("  Wo auf der Sonne ist die Residual-MI am höchsten?")

    # Generiere synthetische Sonne mit Extra-Korrelation
    data_1, data_2 = generate_synthetic_sun(
        shape=(512, 512),
        extra_correlation=0.5,
        n_active_regions=5,
        seed=config.seed
    )

    print(f"\n  📐 Bildgröße: {data_1.shape}")
    print(f"  🔲 Grid: 8x8")
    print(f"  🔬 Berechne räumliche MI-Karten...")

    result = compute_spatial_residual_mi_map(
        data_1, data_2,
        grid_size=(8, 8),
        bins=32
    )

    print_spatial_comparison(result, "Synthetische Sonne mit Extra-Korrelation")


def run_control_tests(config: TestConfig) -> None:
    """
    Führt alle Kontroll-Tests durch.

    Testet ob die gemessene Residual-MI durch Artefakte verursacht wird.
    """
    print("\n" + "="*72)
    print("KONTROLL-TESTS")
    print("="*72)
    print("  Validierung der Residual-MI Messung")

    # Generiere synthetische Sonne mit Extra-Korrelation
    data_1, data_2 = generate_synthetic_sun(
        shape=(256, 256),  # Kleiner für schnellere Kontrollen
        extra_correlation=0.5,
        n_active_regions=5,
        seed=config.seed
    )

    print(f"\n  📐 Bildgröße: {data_1.shape}")
    print(f"  🔬 Führe Kontroll-Tests durch...")

    result = run_all_controls(
        data_1, data_2,
        seed=config.seed,
        bins=32,
        verbose=True
    )

    print_control_results(result)


def print_interpretation() -> None:
    """Gibt Interpretationshilfe aus."""

    print("""
═══════════════════════════════════════════════════════════════════════

  INTERPRETATION:

  Test 1 (Rauschen):     Z ≈ 0 → Nullmodell funktioniert ✓
  Test 2 (Korreliert):   Z >> 3 → MI-Berechnung funktioniert ✓
  Test 3 (Geometrie):    Z > 0 → Gemeinsame Geometrie erzeugt MI
  Test 4 (Extra Korr.):  Z >> Test 3 → Extra-Korrelation detektierbar

  RESIDUAL-ANALYSE (NEU):

  Test 5 (Residual):     MI << Test 3 → Geometrie-Subtraktion funktioniert
  Test 6 (Res. + Korr.): MI > Test 5 → Extra-Korrelation überlebt Subtraktion

  SCHLÜSSEL-VERGLEICH:

  Vergleiche Test 3 vs Test 5:
  - MI_residual << MI_original → Geometrie erklärt (fast) alles
  - MI_residual ≈ MI_original → Geometrie erklärt wenig (unwahrscheinlich)

  Vergleiche Test 5 vs Test 6:
  - Wenn MI_6 >> MI_5 → Extra-Korrelation ist NICHT geometrisch
  - Das ist die "versteckte Information"

═══════════════════════════════════════════════════════════════════════

  "Die Frage ist nicht, ob Struktur existiert.
   Die Frage ist, ob sie erklärbar ist."

  ☀️ → 🔬 → ?

═══════════════════════════════════════════════════════════════════════
    """)


def save_results(results: list[TestResult], output_dir: str) -> None:
    """Speichert Ergebnisse als JSON."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / "hypothesis_test_results.json"
    
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print(f"\n  💾 Gespeichert: {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Hauptfunktion."""
    
    parser = argparse.ArgumentParser(
        description="Solar Seed Hypothesis Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python -m solar_seed.hypothesis_test
  python -m solar_seed.hypothesis_test --real-data
  python -m solar_seed.hypothesis_test --spatial
  python -m solar_seed.hypothesis_test --controls
  python -m solar_seed.hypothesis_test --shuffles 500
        """
    )
    parser.add_argument("--real-data", action="store_true",
                        help="Versuche echte Sonnendaten zu laden")
    parser.add_argument("--spatial", action="store_true",
                        help="Führe räumliche MI-Analyse durch (8x8 Grid)")
    parser.add_argument("--controls", action="store_true",
                        help="Führe Kontroll-Tests durch (C1-C4)")
    parser.add_argument("--shuffles", type=int, default=100,
                        help="Anzahl Shuffles für Nullmodell (default: 100)")
    parser.add_argument("--bins", type=int, default=64,
                        help="Anzahl Bins für MI-Berechnung (default: 64)")
    parser.add_argument("--downsample", type=int, default=8,
                        help="Downsampling-Faktor (default: 8)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output-Verzeichnis (default: results)")
    
    args = parser.parse_args()
    
    config = TestConfig(
        downsample_factor=args.downsample,
        n_bins=args.bins,
        n_shuffles=args.shuffles,
        output_dir=args.output
    )
    
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                    🌞 SOLAR SEED HYPOTHESIS TEST 🌱                    ║
║                                                                        ║
║  EINE Hypothese. EIN Test. EINE Antwort.                              ║
║                                                                        ║
║  H1: MI zwischen AIA-Kanälen > als durch Zufall erklärbar            ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    results = run_all_tests(config, use_real_data=args.real_data)

    print_summary(results)
    save_results(results, config.output_dir)

    if args.spatial:
        run_spatial_analysis(config)

    if args.controls:
        run_control_tests(config)

    print_interpretation()


if __name__ == "__main__":
    main()
