#!/usr/bin/env python3
"""
One-Figure Summary: Timeline + ΔMI + GOES + Bz
==============================================

Creates a publication-ready 4-panel figure showing:
1. GOES X-ray flux with flare classifications
2. ΔMI coupling (193-211 and 193-304)
3. IMF Bz component (southward = geoeffective)
4. Event timeline with breaks and alerts

Usage:
    uv run python scripts/figure_summary.py
    uv run python scripts/figure_summary.py --hours 6
    uv run python scripts/figure_summary.py --output figures/summary.png
"""

import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy required")
    print("Install with: uv pip install matplotlib numpy")
    sys.exit(1)


def load_data_from_db(db_path: str, hours: int = 6) -> dict:
    """Load all data from monitoring database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cutoff = f"-{hours} hours"

    # GOES X-ray
    cursor.execute("""
        SELECT timestamp, flux, flare_class, magnitude
        FROM goes_xray
        WHERE timestamp >= datetime('now', ?)
        ORDER BY timestamp
    """, (cutoff,))
    goes = [dict(r) for r in cursor.fetchall()]

    # Coupling measurements
    cursor.execute("""
        SELECT timestamp, pair, delta_mi, status, deviation_pct, residual
        FROM coupling_measurements
        WHERE timestamp >= datetime('now', ?)
        ORDER BY timestamp
    """, (cutoff,))
    coupling = [dict(r) for r in cursor.fetchall()]

    # Solar wind
    cursor.execute("""
        SELECT timestamp, bz, speed, bt
        FROM solar_wind
        WHERE timestamp >= datetime('now', ?)
        ORDER BY timestamp
    """, (cutoff,))
    wind = [dict(r) for r in cursor.fetchall()]

    conn.close()

    return {
        'goes': goes,
        'coupling': coupling,
        'wind': wind
    }


def parse_timestamp(ts: str) -> datetime:
    """Parse timestamp string to datetime."""
    if not ts:
        return None
    ts = ts.replace('Z', '+00:00')
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except:
        return None


def create_summary_figure(data: dict, output_path: str = None, hours: int = 6):
    """Create the 4-panel summary figure."""

    # Set up figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={'height_ratios': [1.2, 1.2, 1, 0.6],
                                         'hspace': 0.12})
    fig.patch.set_facecolor('white')

    colors = {
        'goes': '#e74c3c',
        '193-211': '#3498db',
        '193-304': '#9b59b6',
        'bz_south': '#e74c3c',
        'bz_north': '#27ae60',
        'break': '#e67e22',
        'flare': '#c0392b',
        'alert': '#e74c3c',
        'diagnostic': '#f39c12'
    }

    now = datetime.now(timezone.utc)

    # =========================================================================
    # Panel 1: GOES X-ray Flux
    # =========================================================================
    ax1 = axes[0]

    if data['goes']:
        times = [parse_timestamp(g['timestamp']) for g in data['goes']]
        flux = [g['flux'] for g in data['goes']]

        ax1.semilogy(times, flux, color=colors['goes'], linewidth=1.5, label='GOES XRS')

        # Flare class thresholds
        ax1.axhline(y=1e-4, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
        ax1.axhline(y=1e-5, color='orange', linestyle='--', alpha=0.5, linewidth=0.8)
        ax1.axhline(y=1e-6, color='yellow', linestyle='--', alpha=0.5, linewidth=0.8)

        # Class labels
        ax1.text(0.02, 0.92, 'X', transform=ax1.transAxes, fontsize=9, color='red', alpha=0.7)
        ax1.text(0.02, 0.72, 'M', transform=ax1.transAxes, fontsize=9, color='orange', alpha=0.7)
        ax1.text(0.02, 0.52, 'C', transform=ax1.transAxes, fontsize=9, color='#b8860b', alpha=0.7)

        # Mark flares
        for g in data['goes']:
            if g['flare_class'] and g['flare_class'] in ['M', 'X']:
                t = parse_timestamp(g['timestamp'])
                ax1.axvline(x=t, color=colors['flare'], alpha=0.3, linewidth=2)

    ax1.set_ylabel('GOES Flux\n(W/m²)', fontsize=10)
    ax1.set_ylim(1e-8, 1e-3)
    ax1.tick_params(axis='y', labelsize=9)
    ax1.set_title('Solar Early Warning Summary', fontsize=14, fontweight='bold', pad=10)

    # =========================================================================
    # Panel 2: ΔMI Coupling
    # =========================================================================
    ax2 = axes[1]

    # Separate by pair
    coupling_193_211 = [(parse_timestamp(c['timestamp']), c['delta_mi'], c['status'], c.get('residual', 0))
                        for c in data['coupling'] if c['pair'] == '193-211']
    coupling_193_304 = [(parse_timestamp(c['timestamp']), c['delta_mi'], c['status'])
                        for c in data['coupling'] if c['pair'] == '193-304']

    if coupling_193_211:
        times_211 = [c[0] for c in coupling_193_211]
        mi_211 = [c[1] for c in coupling_193_211]
        ax2.plot(times_211, mi_211, color=colors['193-211'], linewidth=2,
                label='193-211 Å (corona)', marker='o', markersize=4)

        # Mark breaks/warnings
        for t, mi, status, r in coupling_193_211:
            if status == 'ALERT':
                ax2.scatter([t], [mi], color=colors['alert'], s=100, zorder=5,
                           marker='v', edgecolors='black', linewidths=1)
            elif status == 'WARNING':
                ax2.scatter([t], [mi], color=colors['diagnostic'], s=80, zorder=5,
                           marker='s', edgecolors='black', linewidths=0.5)

    if coupling_193_304:
        times_304 = [c[0] for c in coupling_193_304]
        mi_304 = [c[1] for c in coupling_193_304]
        ax2.plot(times_304, mi_304, color=colors['193-304'], linewidth=2,
                label='193-304 Å (chromosphere)', marker='s', markersize=3, alpha=0.8)

    ax2.set_ylabel('ΔMI\n(bits)', fontsize=10)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.tick_params(axis='y', labelsize=9)

    # Add baseline reference if we have data
    if coupling_193_211:
        mean_mi = np.mean([c[1] for c in coupling_193_211])
        ax2.axhline(y=mean_mi, color=colors['193-211'], linestyle=':', alpha=0.5)

    # =========================================================================
    # Panel 3: IMF Bz
    # =========================================================================
    ax3 = axes[2]

    if data['wind']:
        times_bz = [parse_timestamp(w['timestamp']) for w in data['wind']]
        bz = [w['bz'] for w in data['wind']]

        # Fill regions
        bz_array = np.array(bz)
        times_array = np.array(times_bz)

        ax3.fill_between(times_bz, bz, 0, where=np.array(bz) < 0,
                        color=colors['bz_south'], alpha=0.3, label='Bz < 0 (geoeffective)')
        ax3.fill_between(times_bz, bz, 0, where=np.array(bz) >= 0,
                        color=colors['bz_north'], alpha=0.3, label='Bz ≥ 0')
        ax3.plot(times_bz, bz, color='#2c3e50', linewidth=1.5)

        ax3.axhline(y=0, color='black', linewidth=0.5)
        ax3.axhline(y=-10, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
        ax3.text(0.02, 0.15, 'Storm threshold', transform=ax3.transAxes,
                fontsize=8, color='red', alpha=0.7)

    ax3.set_ylabel('IMF Bz\n(nT)', fontsize=10)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.tick_params(axis='y', labelsize=9)

    # =========================================================================
    # Panel 4: Event Timeline
    # =========================================================================
    ax4 = axes[3]
    ax4.set_ylim(0, 3)
    ax4.set_yticks([0.5, 1.5, 2.5])
    ax4.set_yticklabels(['Breaks', 'Flares', 'Wind'], fontsize=9)

    # Plot events as markers
    event_y = {'break': 0.5, 'flare': 1.5, 'wind': 2.5}

    # Coupling breaks
    for c in data['coupling']:
        if c['status'] in ['ALERT', 'WARNING']:
            t = parse_timestamp(c['timestamp'])
            color = colors['alert'] if c['status'] == 'ALERT' else colors['diagnostic']
            label = 'Actionable' if c['status'] == 'ALERT' else 'Diagnostic'
            ax4.scatter([t], [event_y['break']], color=color, s=60,
                       marker='v' if c['status'] == 'ALERT' else 's',
                       edgecolors='black', linewidths=0.5)

    # Flares
    for g in data['goes']:
        if g['flare_class'] and g['flux'] >= 1e-6:  # C-class and above
            t = parse_timestamp(g['timestamp'])
            size = 40 + (np.log10(g['flux']) + 6) * 30  # Scale by flux
            ax4.scatter([t], [event_y['flare']], color=colors['flare'], s=size,
                       marker='*', edgecolors='black', linewidths=0.5)

    # Strong southward Bz
    for w in data['wind']:
        if w['bz'] and w['bz'] < -5:
            t = parse_timestamp(w['timestamp'])
            ax4.scatter([t], [event_y['wind']], color=colors['bz_south'], s=40,
                       marker='d', alpha=0.7)

    ax4.set_xlabel('Time (UTC)', fontsize=10)
    ax4.tick_params(axis='x', labelsize=9)

    # =========================================================================
    # Format x-axis
    # =========================================================================
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax4.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0, ha='center')

    # Add date to xlabel
    if data['goes']:
        first_time = parse_timestamp(data['goes'][0]['timestamp'])
        date_str = first_time.strftime('%Y-%m-%d')
        ax4.set_xlabel(f'Time (UTC) — {date_str}', fontsize=10)

    # =========================================================================
    # Legend for event types
    # =========================================================================
    legend_elements = [
        Line2D([0], [0], marker='v', color='w', markerfacecolor=colors['alert'],
               markersize=10, label='Actionable Alert', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['diagnostic'],
               markersize=8, label='Diagnostic (vetoed)', markeredgecolor='black'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=colors['flare'],
               markersize=12, label='C+ Flare', markeredgecolor='black'),
        Line2D([0], [0], marker='d', color='w', markerfacecolor=colors['bz_south'],
               markersize=8, label='Bz < -5 nT', markeredgecolor='black'),
    ]
    # Place legend below the panel to avoid overlap
    ax4.legend(handles=legend_elements, loc='upper center', fontsize=7, ncol=4,
               bbox_to_anchor=(0.5, -0.25), frameon=True, fancybox=True)

    # =========================================================================
    # Add annotation box
    # =========================================================================
    # Count events
    n_alerts = sum(1 for c in data['coupling'] if c['status'] == 'ALERT')
    n_diagnostic = sum(1 for c in data['coupling'] if c['status'] == 'WARNING')
    n_flares = sum(1 for g in data['goes'] if g['flare_class'] and g['flux'] >= 1e-6)

    info_text = f"Period: {hours}h | Alerts: {n_alerts} | Diagnostic: {n_diagnostic} | Flares (C+): {n_flares}"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=9,
             style='italic', color='#555555')

    # Watermark
    fig.text(0.99, 0.01, 'Solar Seed Early Warning System', ha='right', fontsize=8,
             color='#aaaaaa', style='italic')

    plt.subplots_adjust(bottom=0.12, top=0.94, left=0.10, right=0.95)

    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="One-Figure Summary: Timeline + ΔMI + GOES + Bz"
    )
    parser.add_argument('--hours', type=int, default=6,
                       help='Hours of data to show (default: 6)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path (PNG/PDF)')
    parser.add_argument('--db', type=str,
                       default='results/early_warning/monitoring.db',
                       help='Path to monitoring database')

    args = parser.parse_args()

    # Load data
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        sys.exit(1)

    print(f"Loading data from {db_path}...")
    data = load_data_from_db(str(db_path), args.hours)

    print(f"  GOES measurements: {len(data['goes'])}")
    print(f"  Coupling measurements: {len(data['coupling'])}")
    print(f"  Solar wind measurements: {len(data['wind'])}")

    # Create figure
    output = args.output or 'figures/early_warning_summary.png'
    create_summary_figure(data, output, args.hours)


if __name__ == "__main__":
    main()
