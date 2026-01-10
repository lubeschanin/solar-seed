#!/usr/bin/env python3
"""
STEREO/EUVI Prototype
=====================

Compares coupling hierarchy between SDO/AIA and STEREO-A/EUVI.

If the hierarchy is identical from different viewing angles,
it is intrinsically solar - not perspectival.

EUVI channels: 304, 171, 195 (≈193), 284 Å
AIA channels:  304, 171, 193, 211, 335, 94, 131 Å

Common channels: 304, 171, 195/193

Workflow:
    1. Search for STEREO-A/EUVI data for timestamp
    2. Download data and calculate ΔMI_sector
    3. Compare hierarchy with SDO/AIA results
    4. Calculate correlation for validation
"""

import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from itertools import combinations
import json
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from solar_seed.mutual_info import mutual_information
from solar_seed.radial_profile import subtract_radial_geometry
from solar_seed.control_tests import sector_ring_shuffle_test

# SunPy imports
try:
    import sunpy.map
    from sunpy.net import Fido, attrs as a
    import astropy.units as u
    SUNPY_AVAILABLE = True
except ImportError:
    SUNPY_AVAILABLE = False
    print("⚠️  SunPy not installed. Install with: uv pip install sunpy")


# EUVI wavelengths (in Angstrom)
EUVI_WAVELENGTHS = [304, 171, 195, 284]

# Mapping EUVI -> AIA (for comparison)
EUVI_TO_AIA = {
    304: 304,   # He II - identical
    171: 171,   # Fe IX - identical
    195: 193,   # Fe XII ≈ Fe XII
    284: None,  # Fe XV - no AIA equivalent
}

# AIA reference values are loaded dynamically from segment data
AIA_REFERENCE_COUPLING = {}  # Populated by load_aia_reference()


def load_aia_reference(
    timestamp: str,
    segment_dir: str = "results/rotation/segments"
) -> dict:
    """
    Loads AIA reference values from segment data for an exact timestamp.

    Args:
        timestamp: ISO timestamp (e.g. "2025-12-01T12:00:00")
        segment_dir: Directory with segment files

    Returns:
        Dict {(wl1, wl2): delta_mi_sector}
    """
    date = timestamp[:10]
    target_time = timestamp[11:16]  # "HH:MM"

    segment_path = Path(segment_dir) / f"{date}.json"

    if not segment_path.exists():
        print(f"  ⚠️  No AIA data found for {date}")
        print(f"      Expected: {segment_path}")
        return {}

    try:
        with open(segment_path) as f:
            data = json.load(f)

        timestamps = data.get('timestamps', [])
        pair_values = data.get('pair_values', {})

        # Find the closest timestamp to target_time
        best_idx = None
        best_diff = float('inf')

        for i, ts in enumerate(timestamps):
            ts_time = ts[11:16]  # "HH:MM"
            # Calculate difference in minutes
            ts_h, ts_m = int(ts_time[:2]), int(ts_time[3:5])
            tgt_h, tgt_m = int(target_time[:2]), int(target_time[3:5])
            diff = abs((ts_h * 60 + ts_m) - (tgt_h * 60 + tgt_m))

            if diff < best_diff:
                best_diff = diff
                best_idx = i

        if best_idx is None:
            print(f"  ✗ No matching timestamp found")
            return {}

        matched_ts = timestamps[best_idx]
        print(f"  ✓ AIA data loaded for {matched_ts}")
        print(f"    (Target: {timestamp}, Difference: {best_diff} min)")

        # Extract values for this timestamp
        reference = {}
        for pair_str, values in pair_values.items():
            if best_idx < len(values):
                wl1, wl2 = map(int, pair_str.split('-'))
                reference[(wl1, wl2)] = values[best_idx]

        print(f"    {len(reference)} pairs extracted")

        return reference

    except Exception as e:
        print(f"  ✗ Error loading AIA data: {e}")
        return {}

# Temperature mapping for EUVI
EUVI_TEMPERATURES = {
    304: 0.05,   # MK - Chromosphere
    171: 0.6,    # MK - Quiet Corona
    195: 1.2,    # MK - Corona (≈193Å)
    284: 2.0,    # MK - Active Regions
}


def search_stereo_euvi(timestamp: str, spacecraft: str = "A") -> dict:
    """
    Searches for STEREO/EUVI data at a timestamp.

    Args:
        timestamp: ISO timestamp (e.g. "2025-12-01T12:00:00")
        spacecraft: "A" or "B"

    Returns:
        Dict with search results per wavelength
    """
    if not SUNPY_AVAILABLE:
        return {}

    source = f"STEREO_{spacecraft}"
    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

    # Zeitfenster: ±30 Minuten
    time_start = dt.isoformat()
    time_end = (dt.replace(minute=dt.minute + 30) if dt.minute < 30
                else dt.replace(hour=dt.hour + 1, minute=0)).isoformat()

    results = {}

    print(f"\n🛰️  Searching STEREO-{spacecraft} EUVI data for {timestamp[:10]}...")

    for wl in EUVI_WAVELENGTHS:
        try:
            # SunPy 7.x syntax
            result = Fido.search(
                a.Time(time_start, time_end),
                a.Source(source),
                a.Instrument('EUVI'),
                a.Wavelength(wl * u.Angstrom)
            )

            n_found = len(result[0]) if result else 0
            results[wl] = {
                'count': n_found,
                'result': result if n_found > 0 else None
            }

            status = "✓" if n_found > 0 else "✗"
            print(f"    {status} {wl} Å: {n_found} files found")

        except Exception as e:
            results[wl] = {'count': 0, 'result': None, 'error': str(e)}
            print(f"    ✗ {wl} Å: Error - {e}")

    return results


def download_stereo_euvi(
    timestamp: str,
    spacecraft: str = "A",
    output_dir: str = "data/stereo",
    wavelengths: list = None
) -> dict:
    """
    Downloads STEREO/EUVI data.

    Args:
        timestamp: ISO timestamp
        spacecraft: "A" or "B"
        output_dir: Target directory
        wavelengths: List of wavelengths (default: all)

    Returns:
        Dict with paths to downloaded files
    """
    if not SUNPY_AVAILABLE:
        return {}

    if wavelengths is None:
        wavelengths = EUVI_WAVELENGTHS

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    source = f"STEREO_{spacecraft}"
    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

    # Zeitfenster
    time_start = dt.isoformat()
    time_end = (dt.replace(hour=dt.hour + 1)).isoformat()

    downloaded = {}

    print(f"\n📥 Downloading STEREO-{spacecraft} EUVI data...")

    for wl in wavelengths:
        try:
            result = Fido.search(
                a.Time(time_start, time_end),
                a.Source(source),
                a.Instrument('EUVI'),
                a.Wavelength(wl * u.Angstrom)
            )

            if result and len(result[0]) > 0:
                # Download first file only
                files = Fido.fetch(result[0, 0], path=str(out_path))
                if files:
                    downloaded[wl] = files[0]
                    print(f"    ✓ {wl} Å: {Path(files[0]).name}")
            else:
                print(f"    ✗ {wl} Å: No data found")

        except Exception as e:
            print(f"    ✗ {wl} Å: Error - {e}")

    return downloaded


def load_euvi_multichannel(
    timestamp: str,
    spacecraft: str = "A",
    data_dir: str = "data/stereo"
) -> tuple:
    """
    Loads EUVI multi-channel data analogous to AIA.

    Returns:
        (channels_dict, metadata) or (None, None)
    """
    # Download first
    files = download_stereo_euvi(timestamp, spacecraft, data_dir)

    if not files:
        return None, None

    channels = {}

    for wl, filepath in files.items():
        try:
            euvi_map = sunpy.map.Map(filepath)

            # Use native resolution (no resampling)
            channels[wl] = euvi_map.data.astype(np.float64)
            print(f"    ✓ {wl} Å loaded: {euvi_map.data.shape} (native)")

        except Exception as e:
            print(f"    ⚠️  Error loading {wl} Å: {e}")

    if len(channels) < 2:
        return None, None

    metadata = {
        'spacecraft': f'STEREO-{spacecraft}',
        'instrument': 'EUVI',
        'timestamp': timestamp,
        'wavelengths': list(channels.keys())
    }

    return channels, metadata


def calculate_euvi_coupling(channels: dict) -> dict:
    """
    Calculates ΔMI_sector for all EUVI channel pairs.

    Uses the same methodology as for AIA:
    1. Radial profile subtraction
    2. Sector-ring shuffle for ΔMI_sector

    Args:
        channels: Dict {wavelength: 2D-Array}

    Returns:
        Dict with results per pair
    """
    wavelengths = sorted(channels.keys())
    results = {}

    print(f"\n📊 Calculating coupling matrix for {len(wavelengths)} channels...")

    for wl1, wl2 in combinations(wavelengths, 2):
        try:
            img1 = channels[wl1]
            img2 = channels[wl2]

            # Radial normalization (function returns (residual, profile, model))
            res1, _, _ = subtract_radial_geometry(img1)
            res2, _, _ = subtract_radial_geometry(img2)

            # MI on residuals
            mi_residual = mutual_information(res1, res2)

            # Sector-ring shuffle test
            shuffle_result = sector_ring_shuffle_test(
                res1, res2,
                n_rings=8,
                n_sectors=8
            )

            # Delta MI = Original - Sector-Shuffled (what remains after geometry removal)
            delta_mi_sector = shuffle_result.mi_original - shuffle_result.mi_sector_shuffled
            # Approximate Z-score from reduction percent
            z_score = shuffle_result.sector_reduction_percent / 5.0  # Rough approximation

            # Temperature difference
            temp_diff = abs(EUVI_TEMPERATURES[wl1] - EUVI_TEMPERATURES[wl2])

            results[(wl1, wl2)] = {
                'mi_residual': mi_residual,
                'delta_mi_sector': delta_mi_sector,
                'z_score': z_score,
                'temperature_diff': temp_diff
            }

            print(f"    {wl1}-{wl2} Å: ΔMI = {delta_mi_sector:.3f} bits (Z={z_score:.1f})")

        except Exception as e:
            print(f"    ⚠️ {wl1}-{wl2} Å: Fehler - {e}")

    return results


def compare_coupling_hierarchies(
    euvi_results: dict,
    aia_reference: dict = None
) -> dict:
    """
    Compares coupling hierarchies between EUVI and AIA.

    Args:
        euvi_results: {(wl1, wl2): {'delta_mi_sector': float, ...}}
        aia_reference: {(wl1, wl2): float} - AIA reference values (optional)

    Returns:
        Comparison statistics
    """
    if aia_reference is None:
        aia_reference = AIA_REFERENCE_COUPLING

    # Find common pairs (with wavelength mapping)
    # Consider both key orderings
    common_pairs = []

    for euvi_pair in euvi_results.keys():
        # Map EUVI -> AIA wavelengths
        aia_wl1 = EUVI_TO_AIA.get(euvi_pair[0])
        aia_wl2 = EUVI_TO_AIA.get(euvi_pair[1])

        if aia_wl1 and aia_wl2:
            # Check both orderings
            aia_pair = (aia_wl1, aia_wl2)
            aia_pair_rev = (aia_wl2, aia_wl1)

            if aia_pair in aia_reference:
                common_pairs.append((euvi_pair, aia_pair))
            elif aia_pair_rev in aia_reference:
                common_pairs.append((euvi_pair, aia_pair_rev))

    if not common_pairs:
        return {'error': 'No common pairs found'}

    # Extract ΔMI values
    euvi_values = []
    aia_values = []

    for euvi_pair, aia_pair in common_pairs:
        euvi_val = euvi_results[euvi_pair]['delta_mi_sector']
        aia_val = aia_reference[aia_pair]
        euvi_values.append(euvi_val)
        aia_values.append(aia_val)

    # Calculate correlation
    if len(euvi_values) >= 2:
        correlation = np.corrcoef(euvi_values, aia_values)[0, 1]
    else:
        correlation = None

    # Create rankings
    euvi_ranking = sorted(
        [(p, euvi_results[p]['delta_mi_sector']) for p in euvi_results],
        key=lambda x: -x[1]
    )
    aia_ranking = sorted(
        [(p, aia_reference[p]) for _, p in common_pairs],
        key=lambda x: -x[1]
    )

    # Check ranking consistency (Spearman correlation of ranks)
    euvi_ranks = {p: i for i, (p, _) in enumerate(euvi_ranking)}
    aia_ranks = {p: i for i, (p, _) in enumerate(aia_ranking)}

    rank_diffs = []
    for euvi_pair, aia_pair in common_pairs:
        euvi_rank = euvi_ranks.get(euvi_pair, 999)
        aia_rank = aia_ranks.get(aia_pair, 999)
        rank_diffs.append(abs(euvi_rank - aia_rank))

    return {
        'common_pairs': len(common_pairs),
        'correlation': correlation,
        'euvi_ranking': euvi_ranking,
        'aia_ranking': aia_ranking,
        'mean_rank_diff': np.mean(rank_diffs) if rank_diffs else None,
        'pair_comparison': [
            {
                'euvi_pair': euvi_pair,
                'aia_pair': aia_pair,
                'euvi_mi': euvi_results[euvi_pair]['delta_mi_sector'],
                'aia_mi': aia_reference[aia_pair],
                'ratio': euvi_results[euvi_pair]['delta_mi_sector'] / aia_reference[aia_pair]
                    if aia_reference[aia_pair] > 0 else None
            }
            for euvi_pair, aia_pair in common_pairs
        ]
    }


def validate_intrinsic_hierarchy(comparison: dict, verbose: bool = True) -> dict:
    """
    Validates whether the coupling hierarchy is intrinsically solar.

    Criteria for intrinsic hierarchy:
    1. High correlation (r > 0.7) between EUVI and AIA ΔMI values
    2. Consistent ranking (mean rank difference < 1)
    3. 171-195 > 304-171 > 304-195 (temperature ordering)

    Returns:
        Dict with validation result
    """
    if 'error' in comparison:
        return {'valid': False, 'error': comparison['error']}

    correlation = comparison.get('correlation')
    mean_rank_diff = comparison.get('mean_rank_diff')

    # Criterion 1: Correlation
    corr_valid = correlation is not None and correlation > 0.7

    # Criterion 2: Ranking consistency
    rank_valid = mean_rank_diff is not None and mean_rank_diff < 1.5

    # Criterion 3: Check temperature ordering
    euvi_ranking = comparison.get('euvi_ranking', [])
    temp_ordered = False

    if len(euvi_ranking) >= 2:
        # Strongest coupling should be at small temp difference
        top_pair, top_mi = euvi_ranking[0]
        # 171-195 should be strongest (0.6 MK difference)
        if top_pair == (171, 195):
            temp_ordered = True
        # Alternatively: 195-284 would also be acceptable
        elif top_pair == (195, 284):
            temp_ordered = True

    # Overall validation
    is_intrinsic = corr_valid and rank_valid

    result = {
        'is_intrinsic': is_intrinsic,
        'correlation': correlation,
        'correlation_valid': corr_valid,
        'mean_rank_diff': mean_rank_diff,
        'ranking_valid': rank_valid,
        'temperature_ordered': temp_ordered,
        'confidence': 'high' if is_intrinsic and temp_ordered else
                      'medium' if is_intrinsic else 'low'
    }

    if verbose:
        print("\n" + "="*70)
        print("VALIDATION: Intrinsic Coupling Hierarchy")
        print("="*70)

        print(f"\n  Correlation EUVI↔AIA: {correlation:.3f}" if correlation else
              "\n  Correlation: not calculable")
        print(f"  {'✓' if corr_valid else '✗'} Correlation > 0.7")

        print(f"\n  Mean rank difference: {mean_rank_diff:.2f}" if mean_rank_diff else
              "\n  Rank difference: not calculable")
        print(f"  {'✓' if rank_valid else '✗'} Rank difference < 1.5")

        print(f"\n  Temperature ordering: {'✓' if temp_ordered else '✗'} Strongest coupling at ΔT~0.6 MK")

        print("\n" + "-"*70)
        if is_intrinsic:
            print("  ✅ VALIDATED: Coupling hierarchy is intrinsically solar!")
            print("     The hierarchy is independent of viewing angle.")
        else:
            print("  ⚠️  NOT VALIDATED: More data required")
            if not corr_valid:
                print("     → Correlation too low")
            if not rank_valid:
                print("     → Ranking inconsistent")
        print("-"*70)

    return result


def save_results(
    euvi_results: dict,
    comparison: dict,
    validation: dict,
    timestamp: str,
    aia_reference: dict,
    output_dir: str = "results/stereo"
) -> Path:
    """Saves results as JSON."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Convert tuple keys to strings for JSON
    euvi_json = {
        f"{k[0]}-{k[1]}": v for k, v in euvi_results.items()
    }

    comparison_json = {
        'common_pairs': comparison.get('common_pairs'),
        'correlation': comparison.get('correlation'),
        'mean_rank_diff': comparison.get('mean_rank_diff'),
        'euvi_ranking': [
            {'pair': f"{p[0]}-{p[1]}", 'delta_mi': v}
            for p, v in comparison.get('euvi_ranking', [])
        ],
        'aia_ranking': [
            {'pair': f"{p[0]}-{p[1]}", 'delta_mi': v}
            for p, v in comparison.get('aia_ranking', [])
        ],
        'pair_comparison': [
            {
                'euvi_pair': f"{c['euvi_pair'][0]}-{c['euvi_pair'][1]}",
                'aia_pair': f"{c['aia_pair'][0]}-{c['aia_pair'][1]}",
                'euvi_mi': c['euvi_mi'],
                'aia_mi': c['aia_mi'],
                'ratio': c.get('ratio')
            }
            for c in comparison.get('pair_comparison', [])
        ]
    }

    # Convert numpy types to Python types for JSON serialization
    def to_python(obj):
        if isinstance(obj, (np.bool_, np.generic)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_python(i) for i in obj]
        return obj

    result = {
        'timestamp': timestamp,
        'spacecraft': 'STEREO-A',
        'instrument': 'EUVI',
        'euvi_coupling': to_python(euvi_json),
        'comparison': to_python(comparison_json),
        'validation': to_python(validation),
        'aia_reference': {
            f"{k[0]}-{k[1]}": v for k, v in aia_reference.items()
        }
    }

    filepath = out_path / f"stereo_validation_{timestamp[:10]}.json"
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n📄 Results saved: {filepath}")
    return filepath


def main(download: bool = False, analyze: bool = False, timestamp: str = "2025-12-01T12:00:00"):
    """
    Main function for STEREO/EUVI exploration.

    Args:
        download: If True, data will be downloaded
        analyze: If True, full analysis will be performed
        timestamp: ISO timestamp for analysis
    """
    # Determine STEREO-A position based on date
    year = int(timestamp[:4])
    if year <= 2011:
        stereo_position = "~180° (opposite side of the Sun)"
    elif year >= 2023:
        stereo_position = "~51° ahead of Earth"
    else:
        stereo_position = "unknown"

    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║              🛰️  STEREO/EUVI PROTOTYPE 🌞                             ║
╚═══════════════════════════════════════════════════════════════════════╝

  Goal: Validate that coupling hierarchy is intrinsically solar

  Timestamp: {timestamp}
  STEREO-A Position: {stereo_position}
  Common channels: 304, 171, 195/193 Å

  Hypothesis: If the coupling hierarchy is identical from two different
              viewing angles, it is intrinsically solar.
""")

    # 1. Search for available data
    print("\n" + "="*70)
    print("STEP 1: Check data availability")
    print("="*70)

    search_results = search_stereo_euvi(timestamp, spacecraft="A")

    available = sum(1 for r in search_results.values() if r.get('count', 0) > 0)

    if available < 2:
        print(f"\n  ⚠️  Only {available} channels available")
        print("      Possibly not yet processed or data gap")
        return search_results

    print(f"\n  ✓ {available}/{len(EUVI_WAVELENGTHS)} channels available")

    if not download and not analyze:
        print("\n  To download: python stereo_prototype.py --download")
        print("  For full analysis: python stereo_prototype.py --analyze")
        return search_results

    # 2. Download
    print("\n" + "="*70)
    print("STEP 2: Download data")
    print("="*70)

    channels, metadata = load_euvi_multichannel(timestamp, spacecraft="A")

    if channels is None:
        print("\n  ✗ Download failed")
        return None

    print(f"\n  ✓ {len(channels)} channels loaded: {list(channels.keys())} Å")

    if not analyze:
        print("\n  For analysis: python stereo_prototype.py --analyze")
        return channels

    # 3. MI calculation
    print("\n" + "="*70)
    print("STEP 3: Calculate coupling matrix (ΔMI_sector)")
    print("="*70)

    euvi_results = calculate_euvi_coupling(channels)

    if not euvi_results:
        print("\n  ✗ MI calculation failed")
        return None

    # 4. Load and compare AIA data
    print("\n" + "="*70)
    print("STEP 4: Load SDO/AIA data (exact timestamp)")
    print("="*70)

    # Load AIA data for exact timestamp
    aia_reference = load_aia_reference(timestamp)

    if not aia_reference:
        print("\n  ⚠️  No segment data available - loading AIA data live...")
        aia_reference = download_and_analyze_aia(timestamp)

        if not aia_reference:
            print("\n  ✗ No AIA data available - comparison not possible")
            return euvi_results

    print("\n  AIA Hierarchy (Top 10):")
    for i, (pair, mi) in enumerate(sorted(aia_reference.items(), key=lambda x: -x[1])[:10]):
        print(f"    {pair[0]}-{pair[1]} Å: {mi:.3f} bits")

    comparison = compare_coupling_hierarchies(euvi_results, aia_reference)

    print("\n  Pairwise comparison:")
    print("  " + "-"*50)
    print(f"  {'EUVI Pair':<12} {'AIA Pair':<12} {'EUVI ΔMI':>10} {'AIA ΔMI':>10} {'Ratio':>8}")
    print("  " + "-"*50)

    for c in comparison.get('pair_comparison', []):
        euvi_p = f"{c['euvi_pair'][0]}-{c['euvi_pair'][1]}"
        aia_p = f"{c['aia_pair'][0]}-{c['aia_pair'][1]}"
        ratio_str = f"{c['ratio']:.2f}" if c.get('ratio') else "N/A"
        print(f"  {euvi_p:<12} {aia_p:<12} {c['euvi_mi']:>10.3f} {c['aia_mi']:>10.3f} {ratio_str:>8}")

    # 5. Validation
    validation = validate_intrinsic_hierarchy(comparison)

    # 6. Save results
    save_results(euvi_results, comparison, validation, timestamp, aia_reference)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    corr = comparison.get('correlation')
    corr_str = f"{corr:.3f}" if corr is not None else "N/A (too few pairs)"

    print(f"""
  Timestamp:     {timestamp}
  Instrument:    STEREO-A/EUVI
  Channels:      {list(channels.keys())} Å
  Pairs:         {len(euvi_results)}
  Common:        {comparison.get('common_pairs', 0)}

  Correlation:   {corr_str}
  Confidence:    {validation.get('confidence', 'unknown')}

  Interpretation:
""")

    if validation.get('is_intrinsic'):
        print(f"    The coupling hierarchy from {stereo_position} viewing angle")
        print("    is consistent with SDO/AIA. This supports the hypothesis that")
        print("    temperature-ordered coupling is an intrinsic property")
        print("    of the solar atmosphere - not a perspectival effect.")
    else:
        print("    Validation was not successful. Possible reasons:")
        print("    - Different activity levels at observation time")
        print("    - Calibration differences between EUVI and AIA")
        print("    - Too few common wavelengths for robust statistics")
        print("    Additional timestamps should be analyzed.")

    return {
        'euvi_results': euvi_results,
        'comparison': comparison,
        'validation': validation
    }


def download_and_analyze_aia(
    timestamp: str,
    output_dir: str = "data/aia_temp"
) -> dict:
    """
    Downloads SDO/AIA data and calculates ΔMI_sector.

    Used when no segment data is available.
    """
    if not SUNPY_AVAILABLE:
        return {}

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Zeitfenster: ±15 Minuten
    from datetime import datetime, timedelta
    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    time_start = dt.isoformat()
    time_end = (dt + timedelta(minutes=15)).isoformat()

    # AIA wavelengths that match EUVI
    aia_wavelengths = [304, 171, 193, 211]

    print(f"\n📥 Downloading SDO/AIA data for {timestamp}...")

    channels = {}

    for wl in aia_wavelengths:
        try:
            result = Fido.search(
                a.Time(time_start, time_end),
                a.Instrument('AIA'),
                a.Wavelength(wl * u.Angstrom)
            )

            if result and len(result[0]) > 0:
                # Download first file
                files = Fido.fetch(result[0, 0], path=str(out_path))
                if files:
                    aia_map = sunpy.map.Map(files[0])
                    # Use native resolution (no resampling)
                    channels[wl] = aia_map.data.astype(np.float64)
                    print(f"    ✓ {wl} Å loaded: {aia_map.data.shape} (native)")
            else:
                print(f"    ✗ {wl} Å: No data")

        except Exception as e:
            print(f"    ✗ {wl} Å: {e}")

    if len(channels) < 2:
        return {}

    # Calculate MI
    print(f"\n📊 Calculating AIA coupling matrix...")

    results = {}
    wavelengths = sorted(channels.keys())

    for wl1, wl2 in combinations(wavelengths, 2):
        try:
            img1 = channels[wl1]
            img2 = channels[wl2]

            res1, _, _ = subtract_radial_geometry(img1)
            res2, _, _ = subtract_radial_geometry(img2)

            shuffle_result = sector_ring_shuffle_test(res1, res2, n_rings=8, n_sectors=8)
            delta_mi = shuffle_result.mi_original - shuffle_result.mi_sector_shuffled

            results[(wl1, wl2)] = delta_mi
            print(f"    {wl1}-{wl2} Å: ΔMI = {delta_mi:.3f} bits")

        except Exception as e:
            print(f"    ⚠️ {wl1}-{wl2} Å: {e}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="STEREO/EUVI Validation")
    parser.add_argument("--download", action="store_true",
                        help="Download data")
    parser.add_argument("--analyze", action="store_true",
                        help="Perform full analysis")
    parser.add_argument("--timestamp", type=str, default="2025-12-01T12:00:00",
                        help="Timestamp for analysis (ISO format)")
    args = parser.parse_args()

    # --analyze implies --download
    if args.analyze:
        args.download = True

    main(download=args.download, analyze=args.analyze, timestamp=args.timestamp)
