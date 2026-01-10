#!/usr/bin/env python3
"""
Solar Seed Hypothesis Test
==========================

ONE hypothesis. ONE test. ONE answer.

H1: Certain AIA wavelength pairs show higher mutual information
    than explainable by independent thermal processes.

Usage:
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
    """Configuration for the hypothesis test."""
    downsample_factor: int = 8
    n_bins: int = 64
    n_shuffles: int = 100
    output_dir: str = "results"
    seed: int = 42


@dataclass
class TestResult:
    """Result of a single test."""
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
    Runs a single hypothesis test.

    Args:
        data_1: First channel
        data_2: Second channel
        label: Test label
        config: Test configuration

    Returns:
        TestResult with all metrics
    """
    # Downsample
    ds = config.downsample_factor
    if ds > 1:
        data_1 = data_1[::ds, ::ds]
        data_2 = data_2[::ds, ::ds]
    
    print(f"\n  📐 Shape: {data_1.shape}")

    # Calculate MI
    print(f"  🔬 Computing MI...")
    mi_real = mutual_information(data_1, data_2, config.n_bins)
    nmi_real = normalized_mutual_information(data_1, data_2, config.n_bins)
    print(f"     MI_real  = {mi_real:.6f} bits")
    print(f"     NMI_real = {nmi_real:.6f}")
    
    # Null model
    print(f"  🎲 Null model ({config.n_shuffles} shuffles)...")
    mi_null_mean, mi_null_std, _ = compute_null_distribution(
        data_1, data_2, 
        n_shuffles=config.n_shuffles, 
        bins=config.n_bins,
        seed=config.seed,
        verbose=True
    )
    print(f"     MI_null  = {mi_null_mean:.6f} ± {mi_null_std:.6f}")
    
    # Statistics
    z_score = compute_z_score(mi_real, mi_null_mean, mi_null_std)
    p_value = compute_p_value(mi_real, [])  # Simplified
    
    # Empirical p-value from Z-score (normal approximation)
    from math import erfc, sqrt
    p_value = 0.5 * erfc(z_score / sqrt(2)) if z_score > 0 else 1.0

    print(f"  📈 Z-Score  = {z_score:.2f}")
    print(f"     p-value  = {p_value:.4f}")
    
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
    Runs all tests.

    Args:
        config: Test configuration
        use_real_data: Try to load real solar data

    Returns:
        List of all TestResults
    """
    results = []

    # TEST 1: Pure noise (null hypothesis true)
    print("\n" + "="*72)
    print("TEST 1: VALIDATION - Pure Noise (independent)")
    print("="*72)
    print("  Expectation: MI_real ≈ MI_null, Z ≈ 0")

    data_1, data_2 = generate_pure_noise(shape=(512, 512), seed=config.seed)
    results.append(run_single_test(data_1, data_2, "Pure Noise", config))
    
    # TEST 2: Correlated noise (alternative true)
    print("\n" + "="*72)
    print("TEST 2: VALIDATION - Correlated Noise (r=0.5)")
    print("="*72)
    print("  Expectation: MI_real >> MI_null, Z >> 3")

    data_1, data_2 = generate_correlated_noise(shape=(512, 512), correlation=0.5, seed=config.seed)
    results.append(run_single_test(data_1, data_2, "Correlated (r=0.5)", config))
    
    # TEST 3: Synthetic sun - Geometry only
    print("\n" + "="*72)
    print("TEST 3: SYNTHETIC SUN - Geometry Only")
    print("="*72)
    print("  Expectation: MI_real > MI_null (due to geometry)")

    data_1, data_2 = generate_synthetic_sun(shape=(512, 512), extra_correlation=0.0, seed=config.seed)
    results.append(run_single_test(data_1, data_2, "Sun (Geometry)", config))
    
    # TEST 4: Synthetic sun - With extra correlation
    print("\n" + "="*72)
    print("TEST 4: SYNTHETIC SUN - With Extra Correlation")
    print("="*72)
    print("  Expectation: MI_real > Test 3")

    data_1, data_2 = generate_synthetic_sun(shape=(512, 512), extra_correlation=0.5, seed=config.seed)
    results.append(run_single_test(data_1, data_2, "Sun (extra r=0.5)", config))
    
    # TEST 5: Synthetic sun - Residuals (geometry only)
    print("\n" + "="*72)
    print("TEST 5: RESIDUAL ANALYSIS - Sun Without Geometry")
    print("="*72)
    print("  Radial geometry is subtracted.")
    print("  Expectation: MI_residual << MI_original (geometry explains most)")

    data_1, data_2 = generate_synthetic_sun(shape=(512, 512), extra_correlation=0.0, seed=config.seed)
    residual_1, residual_2, _ = prepare_pair_for_residual_mi(data_1, data_2)
    results.append(run_single_test(residual_1, residual_2, "Residual (Geometry)", config))

    # TEST 6: Synthetic sun - Residuals with extra correlation
    print("\n" + "="*72)
    print("TEST 6: RESIDUAL ANALYSIS - With Extra Correlation")
    print("="*72)
    print("  Expectation: MI_residual > Test 5 (extra correlation survives)")

    data_1, data_2 = generate_synthetic_sun(shape=(512, 512), extra_correlation=0.5, seed=config.seed)
    residual_1, residual_2, _ = prepare_pair_for_residual_mi(data_1, data_2)
    results.append(run_single_test(residual_1, residual_2, "Residual (extra r=0.5)", config))

    # TEST 7: Real data (optional)
    if use_real_data:
        print("\n" + "="*72)
        print("TEST 7: REAL SOLAR DATA")
        print("="*72)
        
        data_1, data_2 = load_sunpy_sample()
        
        if data_1 is not None:
            results.append(run_single_test(data_1, data_2, "AIA Sample", config))
        else:
            print("  ⚠️  No real data available")
            print("  → Install with: pip install sunpy")
    
    return results


def print_summary(results: list[TestResult]) -> None:
    """Prints summary."""
    
    print("\n" + "="*72)
    print("SUMMARY")
    print("="*72)
    
    print(f"\n  {'Test':<22} {'MI_real':>10} {'MI_null':>12} {'Z':>8} {'Status':<20}")
    print("  " + "-"*72)
    
    for r in results:
        print(f"  {r.label:<22} {r.mi_real:>10.4f} {r.mi_null_mean:>10.4f}±{r.mi_null_std:.2f} {r.z_score:>8.2f} {r.status:<20}")


def run_spatial_analysis(config: TestConfig) -> None:
    """
    Runs spatial MI analysis.

    Shows where on the (synthetic) Sun the highest residual MI is.
    """
    print("\n" + "="*72)
    print("SPATIAL MI ANALYSIS")
    print("="*72)
    print("  Where on the Sun is the residual MI highest?")

    # Generate synthetic sun with extra correlation
    data_1, data_2 = generate_synthetic_sun(
        shape=(512, 512),
        extra_correlation=0.5,
        n_active_regions=5,
        seed=config.seed
    )

    print(f"\n  📐 Image size: {data_1.shape}")
    print(f"  🔲 Grid: 8x8")
    print(f"  🔬 Computing spatial MI maps...")

    result = compute_spatial_residual_mi_map(
        data_1, data_2,
        grid_size=(8, 8),
        bins=32
    )

    print_spatial_comparison(result, "Synthetic Sun with Extra Correlation")


def run_control_tests(config: TestConfig) -> None:
    """
    Runs all control tests.

    Tests whether the measured residual MI is caused by artifacts.
    """
    print("\n" + "="*72)
    print("CONTROL TESTS")
    print("="*72)
    print("  Validation of residual MI measurement")

    # Generate synthetic sun with extra correlation
    data_1, data_2 = generate_synthetic_sun(
        shape=(256, 256),  # Smaller for faster controls
        extra_correlation=0.5,
        n_active_regions=5,
        seed=config.seed
    )

    print(f"\n  📐 Image size: {data_1.shape}")
    print(f"  🔬 Running control tests...")

    result = run_all_controls(
        data_1, data_2,
        seed=config.seed,
        bins=32,
        verbose=True
    )

    print_control_results(result)


def print_interpretation() -> None:
    """Prints interpretation help."""

    print("""
═══════════════════════════════════════════════════════════════════════

  INTERPRETATION:

  Test 1 (Noise):        Z ≈ 0 → Null model works ✓
  Test 2 (Correlated):   Z >> 3 → MI calculation works ✓
  Test 3 (Geometry):     Z > 0 → Shared geometry creates MI
  Test 4 (Extra Corr.):  Z >> Test 3 → Extra correlation detectable

  RESIDUAL ANALYSIS (NEW):

  Test 5 (Residual):     MI << Test 3 → Geometry subtraction works
  Test 6 (Res. + Corr.): MI > Test 5 → Extra correlation survives subtraction

  KEY COMPARISON:

  Compare Test 3 vs Test 5:
  - MI_residual << MI_original → Geometry explains (almost) everything
  - MI_residual ≈ MI_original → Geometry explains little (unlikely)

  Compare Test 5 vs Test 6:
  - If MI_6 >> MI_5 → Extra correlation is NOT geometric
  - This is the "hidden information"

═══════════════════════════════════════════════════════════════════════

  "The question is not whether structure exists.
   The question is whether it is explainable."

  ☀️ → 🔬 → ?

═══════════════════════════════════════════════════════════════════════
    """)


def save_results(results: list[TestResult], output_dir: str) -> None:
    """Saves results as JSON."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / "hypothesis_test_results.json"
    
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print(f"\n  💾 Saved: {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function."""

    parser = argparse.ArgumentParser(
        description="Solar Seed Hypothesis Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m solar_seed.hypothesis_test
  python -m solar_seed.hypothesis_test --real-data
  python -m solar_seed.hypothesis_test --spatial
  python -m solar_seed.hypothesis_test --controls
  python -m solar_seed.hypothesis_test --shuffles 500
        """
    )
    parser.add_argument("--real-data", action="store_true",
                        help="Try to load real solar data")
    parser.add_argument("--spatial", action="store_true",
                        help="Run spatial MI analysis (8x8 grid)")
    parser.add_argument("--controls", action="store_true",
                        help="Run control tests (C1-C4)")
    parser.add_argument("--shuffles", type=int, default=100,
                        help="Number of shuffles for null model (default: 100)")
    parser.add_argument("--bins", type=int, default=64,
                        help="Number of bins for MI calculation (default: 64)")
    parser.add_argument("--downsample", type=int, default=8,
                        help="Downsampling factor (default: 8)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory (default: results)")
    
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
║  ONE hypothesis. ONE test. ONE answer.                                ║
║                                                                        ║
║  H1: MI between AIA channels > than explainable by chance            ║
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
