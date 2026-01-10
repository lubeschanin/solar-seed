#!/usr/bin/env python3
"""
Double Prominence Eruption Analysis - September 13-14, 2013
============================================================

Analyzes and compares the double prominence eruption captured by STEREO-A.
NASA SOHO "Pick of the Week" - September 20, 2013.

Events:
    Prominence 1: 2013-09-13 19:16:15 UTC
    Prominence 2: 2013-09-14 12:00:00 UTC (estimated)

Usage:
    uv run python scripts/prominence_2013-09-13.py           # Compare both prominences
    uv run python scripts/prominence_2013-09-13.py --single  # First prominence only
    uv run python scripts/prominence_2013-09-13.py --render-only  # Render sun images
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Event configuration - Double Prominence Eruption
EVENTS = [
    {
        "name": "Prominence 1",
        "timestamp": "2013-09-13T19:16:15",
        "description": "First prominence eruption",
    },
    {
        "name": "Prominence 2",
        "timestamp": "2013-09-14T12:00:00",
        "description": "Second prominence eruption (estimated time)",
    }
]

EVENT_META = {
    "name": "Double Prominence Eruption",
    "date_range": "September 13-14, 2013",
    "source": "STEREO-A EUVI / NASA SOHO Pick of the Week",
    "significance": "Prominences are cooler plasma (0.05 MK) in magnetic loops"
}

def render_sun_images():
    """Download and render AIA images for both events."""
    for event in EVENTS:
        print(f"\n{'='*70}")
        print(f"RENDERING: {event['name']}")
        print(f"{'='*70}")
        print(f"  Timestamp: {event['timestamp']}")
        print()

        try:
            from solar_seed.render_sun import main as render_main
            import sys
            from datetime import datetime

            dt = datetime.fromisoformat(event['timestamp'])
            sys.argv = [
                'render_sun',
                '--date', dt.strftime('%d.%m.%Y'),
                '--time', dt.strftime('%H:%M'),
                '--timezone', 'UTC'
            ]
            render_main()
        except Exception as e:
            print(f"  Error rendering: {e}")
            return False
    return True


def analyze_single_event(event):
    """Analyze coupling for a single event timestamp."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {event['name']} - {event['timestamp']}")
    print(f"{'='*70}")

    try:
        from solar_seed.multichannel import load_aia_multichannel
        from solar_seed.radial_profile import subtract_radial_geometry
        from solar_seed.control_tests import sector_ring_shuffle_test
        from itertools import combinations

        print(f"\n  Loading AIA data...")
        channels, metadata = load_aia_multichannel(event['timestamp'])

        if channels is None or not channels:
            print("  No channels loaded. Server may be unavailable.")
            return None

        print(f"  Loaded {len(channels)} channels: {list(channels.keys())}")

        TEMPS = {304: 0.05, 171: 0.6, 193: 1.2, 211: 2.0, 335: 2.5, 94: 6.3, 131: 10.0}
        wavelengths = sorted(channels.keys(), key=lambda w: TEMPS.get(w, 0))

        print(f"\n  Computing coupling matrix...")
        results = {}

        for wl1, wl2 in combinations(wavelengths, 2):
            try:
                img1 = channels[wl1]
                img2 = channels[wl2]

                res1, _, _ = subtract_radial_geometry(img1)
                res2, _, _ = subtract_radial_geometry(img2)

                shuffle_result = sector_ring_shuffle_test(res1, res2, n_rings=10, n_sectors=12)
                delta_mi = shuffle_result.mi_original - shuffle_result.mi_sector_shuffled

                results[(wl1, wl2)] = {
                    'delta_mi': delta_mi,
                    'mi_original': shuffle_result.mi_original,
                    'local_structure': shuffle_result.local_structure
                }

                print(f"    {wl1:>3}-{wl2:<3} Å: ΔMI = {delta_mi:.4f} bits")

            except Exception as e:
                print(f"    {wl1:>3}-{wl2:<3} Å: Error - {e}")

        return results

    except Exception as e:
        import traceback
        print(f"  Error: {e}")
        traceback.print_exc()
        return None


def compare_events():
    """Analyze both prominences and compare coupling patterns."""
    from datetime import datetime as dt_module
    import json

    all_results = {}

    # Analyze each event
    for event in EVENTS:
        results = analyze_single_event(event)
        if results:
            all_results[event['name']] = {
                'timestamp': event['timestamp'],
                'coupling': results
            }

    if len(all_results) < 2:
        print("\n  Could not analyze both events for comparison.")
        return False

    # Comparison
    print(f"\n{'='*70}")
    print("DOUBLE PROMINENCE COMPARISON")
    print(f"{'='*70}")

    event_names = list(all_results.keys())
    r1 = all_results[event_names[0]]['coupling']
    r2 = all_results[event_names[1]]['coupling']

    # Find common pairs
    common_pairs = set(r1.keys()) & set(r2.keys())

    print(f"\n  {'Pair':<12} {'Prom 1':>12} {'Prom 2':>12} {'Diff':>10} {'Match':>8}")
    print(f"  {'-'*56}")

    matches = 0
    total = 0

    # Sort by first event's coupling strength
    sorted_pairs = sorted(common_pairs, key=lambda p: r1[p]['delta_mi'], reverse=True)

    for pair in sorted_pairs:
        wl1, wl2 = pair
        d1 = r1[pair]['delta_mi']
        d2 = r2[pair]['delta_mi']
        diff = d2 - d1
        # Check if same direction (both positive or both negative coupling changes)
        match = "✓" if (d1 > 0.05 and d2 > 0.05) or (d1 < 0.02 and d2 < 0.02) else "~"
        if match == "✓":
            matches += 1
        total += 1

        print(f"  {wl1}-{wl2} Å{' ':>4} {d1:>12.4f} {d2:>12.4f} {diff:>+10.4f} {match:>8}")

    # 304 Å comparison (prominence channel)
    print(f"\n  304 Å (Prominence Channel) Comparison:")
    print(f"  {'-'*56}")

    prom_pairs = [p for p in common_pairs if 304 in p]
    for pair in sorted(prom_pairs, key=lambda p: r1[p]['delta_mi'], reverse=True):
        wl1, wl2 = pair
        other = wl2 if wl1 == 304 else wl1
        d1 = r1[pair]['delta_mi']
        d2 = r2[pair]['delta_mi']
        diff = d2 - d1
        print(f"    304-{other} Å:  {d1:.4f} → {d2:.4f}  ({diff:+.4f})")

    # Summary statistics
    print(f"\n  Consistency: {matches}/{total} pairs show similar coupling pattern")

    # Check if ranking is preserved
    top5_1 = sorted_pairs[:5]
    top5_2 = sorted(common_pairs, key=lambda p: r2[p]['delta_mi'], reverse=True)[:5]
    rank_overlap = len(set(top5_1) & set(top5_2))
    print(f"  Top-5 Ranking Overlap: {rank_overlap}/5 pairs")

    # Save comparison results
    out_path = Path("results/prominence")
    out_path.mkdir(parents=True, exist_ok=True)

    comparison_data = {
        "meta": EVENT_META,
        "analysis_time": dt_module.now().isoformat(),
        "events": {
            name: {
                "timestamp": data['timestamp'],
                "coupling": {f"{k[0]}-{k[1]}": v for k, v in data['coupling'].items()}
            }
            for name, data in all_results.items()
        },
        "comparison": {
            "consistency": f"{matches}/{total}",
            "top5_overlap": f"{rank_overlap}/5"
        }
    }

    filepath = out_path / "double_prominence_comparison.json"
    with open(filepath, 'w') as f:
        json.dump(comparison_data, f, indent=2)

    print(f"\n  Results saved: {filepath}")
    return True


def run_single_timepoint_analysis():
    """Run analysis for first event only (backward compatibility)."""
    results = analyze_single_event(EVENTS[0])
    if results:
        import json
        from datetime import datetime as dt_module

        out_path = Path("results/prominence")
        out_path.mkdir(parents=True, exist_ok=True)

        result_data = {
            "event": EVENTS[0],
            "analysis_time": dt_module.now().isoformat(),
            "coupling": {f"{k[0]}-{k[1]}": v for k, v in results.items()}
        }

        filepath = out_path / "prominence_2013-09-13.json"
        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"\n  Results saved: {filepath}")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only', action='store_true',
                        help='Only render sun images')
    parser.add_argument('--single', action='store_true',
                        help='Analyze first prominence only')
    parser.add_argument('--compare', action='store_true',
                        help='Compare both prominences (default)')
    args = parser.parse_args()

    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║   🌞  DOUBLE PROMINENCE ERUPTION - September 13-14, 2013  🌞          ║
╚═══════════════════════════════════════════════════════════════════════╝

  Event: Two substantial prominences broke away from the Sun

  Prominence 1: 2013-09-13 19:16:15 UTC
  Prominence 2: 2013-09-14 12:00:00 UTC (estimated)

  Captured by STEREO-A EUVI in extreme UV light.
  NASA SOHO "Pick of the Week" - September 20, 2013
  https://soho.nascom.nasa.gov/pickoftheweek/old/20sep2013/
""")

    success = True

    if args.render_only:
        success = render_sun_images()
    elif args.single:
        success = run_single_timepoint_analysis()
    else:
        # Default: compare both prominences
        success = compare_events()

    if success:
        print(f"\n  ✅ Analysis complete!")
    else:
        print(f"\n  ⚠️  Analysis incomplete - check server availability")
        print(f"     Try again later (servers often faster at night)")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
