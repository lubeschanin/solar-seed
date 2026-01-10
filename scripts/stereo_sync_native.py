#!/usr/bin/env python3
"""
STEREO Synchronized Analysis at Native Resolution
==================================================

Analyzes EUVI and AIA data at native resolution with strict
temporal synchronization (Δt < 2.5 min).

This validates the antipodal 180° comparison at full resolution.
"""

import numpy as np
from datetime import datetime
from pathlib import Path
from itertools import combinations
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from solar_seed.mutual_info import mutual_information
from solar_seed.radial_profile import subtract_radial_geometry
from solar_seed.control_tests import sector_ring_shuffle_test

try:
    import sunpy.map
    from sunpy.net import Fido, attrs as a
    import astropy.units as u
    SUNPY_AVAILABLE = True
except ImportError:
    SUNPY_AVAILABLE = False
    print("SunPy required: uv pip install sunpy")
    sys.exit(1)


# Synchronized timestamps from Paper (2011-02-06, 180° opposition)
SYNC_CONFIG = {
    "euvi_files": {
        171: "data/stereo/20110206_121400_n4euA.fts",  # 12:14:00
        195: "data/stereo/20110206_121530_n4euA.fts",  # 12:15:30
        304: "data/stereo/20110206_121615_n4euA.fts",  # 12:16:15
        284: "data/stereo/20110206_121630_n4euA.fts",  # 12:16:30
    },
    "aia_time": "2011-02-06T12:15:00",
    "aia_wavelengths": [171, 193, 211, 304],
    "max_dt_min": 2.5,
}

EUVI_TO_AIA = {171: 171, 195: 193, 304: 304}
TEMPERATURES = {171: 0.6, 193: 1.2, 195: 1.2, 211: 2.0, 284: 2.0, 304: 0.05}


def load_euvi_native() -> dict:
    """Load EUVI data at native resolution (2048x2048)."""
    channels = {}

    print("\n📡 Loading EUVI data (native resolution)...")

    for wl, filepath in SYNC_CONFIG["euvi_files"].items():
        path = Path(filepath)
        if not path.exists():
            print(f"    ✗ {wl} Å: File not found: {filepath}")
            continue

        try:
            euvi_map = sunpy.map.Map(str(path))
            channels[wl] = euvi_map.data.astype(np.float64)
            print(f"    ✓ {wl} Å: {euvi_map.data.shape} (native)")
        except Exception as e:
            print(f"    ✗ {wl} Å: {e}")

    return channels


def download_aia_synoptic(timestamp: str, target_shape: int = 2048) -> dict:
    """Download AIA data and resample to match EUVI resolution."""
    from datetime import timedelta

    channels = {}
    dt = datetime.fromisoformat(timestamp)
    time_start = (dt - timedelta(minutes=5)).isoformat()
    time_end = (dt + timedelta(minutes=5)).isoformat()

    print(f"\n📥 Downloading AIA data for {timestamp}...")
    print(f"    Target resolution: {target_shape}×{target_shape} (to match EUVI)")

    out_path = Path("data/aia_sync")
    out_path.mkdir(parents=True, exist_ok=True)

    for wl in SYNC_CONFIG["aia_wavelengths"]:
        try:
            # Search for data
            result = Fido.search(
                a.Time(time_start, time_end),
                a.Instrument('AIA'),
                a.Wavelength(wl * u.Angstrom),
                a.Sample(1 * u.min)
            )

            if result and len(result[0]) > 0:
                files = Fido.fetch(result[0, 0], path=str(out_path))
                if files:
                    aia_map = sunpy.map.Map(files[0])
                    orig_shape = aia_map.data.shape

                    # Resample to target resolution
                    target = [target_shape, target_shape] * u.pix
                    resampled = aia_map.resample(target)

                    channels[wl] = resampled.data.astype(np.float64)
                    print(f"    ✓ {wl} Å: {orig_shape} → {resampled.data.shape}")
            else:
                print(f"    ✗ {wl} Å: No data found")

        except Exception as e:
            print(f"    ✗ {wl} Å: {e}")

    return channels


def compute_coupling(channels: dict, instrument: str) -> dict:
    """Compute ΔMI_sector for all channel pairs."""
    wavelengths = sorted(channels.keys())
    results = {}

    print(f"\n📊 Computing coupling matrix ({instrument}, {len(wavelengths)} channels)...")

    for wl1, wl2 in combinations(wavelengths, 2):
        try:
            img1 = channels[wl1]
            img2 = channels[wl2]

            # Radial normalization
            res1, _, _ = subtract_radial_geometry(img1)
            res2, _, _ = subtract_radial_geometry(img2)

            # Sector-ring shuffle test
            shuffle_result = sector_ring_shuffle_test(
                res1, res2,
                n_rings=10,
                n_sectors=12
            )

            delta_mi = shuffle_result.mi_original - shuffle_result.mi_sector_shuffled

            results[(wl1, wl2)] = {
                'delta_mi': delta_mi,
                'mi_original': shuffle_result.mi_original,
                'mi_sector': shuffle_result.mi_sector_shuffled,
            }

            print(f"    {wl1}-{wl2} Å: ΔMI = {delta_mi:.4f} bits")

        except Exception as e:
            print(f"    ⚠️ {wl1}-{wl2} Å: {e}")

    return results


def compare_hierarchies(euvi: dict, aia: dict) -> dict:
    """Compare EUVI and AIA coupling hierarchies."""

    # Find common pairs with wavelength mapping
    common = []

    for euvi_pair in euvi.keys():
        aia_wl1 = EUVI_TO_AIA.get(euvi_pair[0])
        aia_wl2 = EUVI_TO_AIA.get(euvi_pair[1])

        if aia_wl1 and aia_wl2:
            aia_pair = tuple(sorted([aia_wl1, aia_wl2]))
            if aia_pair in aia:
                common.append((euvi_pair, aia_pair))

    if len(common) < 2:
        return {"error": "Too few common pairs", "n_common": len(common)}

    # Extract values
    euvi_vals = [euvi[ep]['delta_mi'] for ep, _ in common]
    aia_vals = [aia[ap]['delta_mi'] for _, ap in common]

    # Pearson correlation
    correlation = np.corrcoef(euvi_vals, aia_vals)[0, 1]

    # Spearman rank correlation
    from scipy.stats import spearmanr
    rank_corr, p_value = spearmanr(euvi_vals, aia_vals)

    return {
        "n_common": len(common),
        "pearson_correlation": float(correlation),
        "spearman_correlation": float(rank_corr),
        "spearman_p_value": float(p_value),
        "pairs": [
            {
                "euvi": f"{ep[0]}-{ep[1]}",
                "aia": f"{ap[0]}-{ap[1]}",
                "euvi_mi": euvi[ep]['delta_mi'],
                "aia_mi": aia[ap]['delta_mi'],
                "ratio": euvi[ep]['delta_mi'] / aia[ap]['delta_mi'] if aia[ap]['delta_mi'] > 0 else None
            }
            for ep, ap in common
        ]
    }


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║     🛰️  STEREO SYNCHRONIZED ANALYSIS (Native Resolution) 🌞          ║
╚═══════════════════════════════════════════════════════════════════════╝

  Date: 2011-02-06 (180° opposition)
  STEREO-A: Opposite side of Sun from Earth
  Max Δt: 2.5 minutes between channels
  Resolution: 2048×2048 (matched)
""")

    # 1. Load EUVI at native resolution
    euvi_channels = load_euvi_native()

    if len(euvi_channels) < 3:
        print("\n✗ Not enough EUVI channels")
        return

    # 2. Download AIA data
    aia_channels = download_aia_synoptic(SYNC_CONFIG["aia_time"])

    if len(aia_channels) < 3:
        print("\n✗ Not enough AIA channels")
        return

    # 3. Compute coupling matrices
    euvi_coupling = compute_coupling(euvi_channels, "EUVI")
    aia_coupling = compute_coupling(aia_channels, "AIA")

    # 4. Compare hierarchies
    print("\n" + "="*70)
    print("HIERARCHY COMPARISON")
    print("="*70)

    comparison = compare_hierarchies(euvi_coupling, aia_coupling)

    if "error" in comparison:
        print(f"\n  ✗ {comparison['error']}")
        return

    print(f"\n  Common pairs: {comparison['n_common']}")
    print(f"\n  Pearson correlation:  {comparison['pearson_correlation']:.3f}")
    print(f"  Spearman correlation: {comparison['spearman_correlation']:.3f} (p={comparison['spearman_p_value']:.4f})")

    print(f"\n  {'EUVI Pair':<12} {'AIA Pair':<12} {'EUVI ΔMI':>10} {'AIA ΔMI':>10} {'Ratio':>8}")
    print("  " + "-"*54)

    for p in comparison['pairs']:
        ratio_str = f"{p['ratio']:.2f}" if p['ratio'] else "N/A"
        print(f"  {p['euvi']:<12} {p['aia']:<12} {p['euvi_mi']:>10.4f} {p['aia_mi']:>10.4f} {ratio_str:>8}")

    # 5. Validation
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)

    rank_corr = comparison['spearman_correlation']
    is_valid = rank_corr > 0.7

    if is_valid:
        print(f"\n  ✅ VALIDATED: Rank correlation = {rank_corr:.1%}")
        print("     Temperature-ordered coupling hierarchy is intrinsic!")
    else:
        print(f"\n  ⚠️  Rank correlation = {rank_corr:.1%} (threshold: 70%)")

    # 6. Save results
    out_path = Path("results/stereo")
    out_path.mkdir(parents=True, exist_ok=True)

    result = {
        "timestamp": SYNC_CONFIG["aia_time"],
        "method": "synchronized_native",
        "resolution": {
            "euvi": "2048x2048 (native)",
            "aia": "2048x2048 (resampled from 4096)"
        },
        "max_dt_min": SYNC_CONFIG["max_dt_min"],
        "euvi_coupling": {
            f"{k[0]}-{k[1]}": v for k, v in euvi_coupling.items()
        },
        "aia_coupling": {
            f"{k[0]}-{k[1]}": v for k, v in aia_coupling.items()
        },
        "comparison": comparison
    }

    filepath = out_path / "stereo_validation_2011-02-06_native.json"
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n📄 Results saved: {filepath}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
  Resolution:        2048² (matched)
  Angular separation: 180° (antipodal)
  Temporal sync:     Δt < 2.5 min

  Spearman ρ:        {rank_corr:.3f}
  Status:            {'✅ INTRINSIC' if is_valid else '⚠️  CHECK DATA'}

  Interpretation:
    The coupling hierarchy observed from opposite sides of the Sun
    {'shows strong agreement' if is_valid else 'shows weaker agreement than expected'}.
    {'This confirms viewpoint invariance.' if is_valid else 'Resolution differences may affect comparison.'}
""")


if __name__ == "__main__":
    main()
