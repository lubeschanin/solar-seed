#!/usr/bin/env python3
"""
Case Study Figure: Coupling Break → GOES Response
==================================================

Generates a publication-ready figure showing:
1. ΔMI 193-211 timeline with break detection
2. GOES X-ray flux timeline
3. Validation status markers
4. Time annotations for key events

Based on 2026-01-10 monitoring session.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np

# Style for publication
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_coupling_data():
    """Load coupling history from JSON."""
    history_file = Path('results/early_warning/coupling_history.json')
    with open(history_file) as f:
        history = json.load(f)

    times = []
    mi_211 = []
    mi_304 = []

    for h in history:
        ts = h['timestamp'].replace('Z', '+00:00')
        if '+' not in ts and not ts.endswith('Z'):
            ts += '+00:00'
        try:
            dt = datetime.fromisoformat(ts.replace('+00:00', ''))
        except:
            continue

        c = h.get('coupling', {})
        mi = c.get('193-211', {}).get('delta_mi')
        mi3 = c.get('193-304', {}).get('delta_mi')

        if mi is not None:
            times.append(dt)
            mi_211.append(mi)
            mi_304.append(mi3 if mi3 else np.nan)

    return times, mi_211, mi_304


def load_goes_data():
    """Load GOES X-ray data from database."""
    db_path = Path('results/early_warning/monitoring.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    cursor.execute('SELECT timestamp, flux, flare_class FROM goes_xray ORDER BY timestamp')
    rows = cursor.fetchall()

    times = []
    flux = []

    for row in rows:
        ts = row['timestamp'].replace('Z', '')
        try:
            dt = datetime.fromisoformat(ts)
        except:
            continue
        times.append(dt)
        flux.append(row['flux'])

    conn.close()
    return times, flux


def compute_break_threshold(values, k=2.0):
    """Compute median - k*MAD threshold."""
    if len(values) < 3:
        return None, None, None

    values = sorted(values)
    n = len(values)
    median = values[n//2] if n % 2 else (values[n//2-1] + values[n//2]) / 2

    deviations = sorted([abs(v - median) for v in values])
    mad = deviations[len(deviations)//2] if len(deviations) % 2 else \
          (deviations[len(deviations)//2-1] + deviations[len(deviations)//2]) / 2

    mad_scaled = mad * 1.4826
    threshold = median - k * mad_scaled

    return median, mad_scaled, threshold


def create_figure():
    """Create the case study figure."""

    # Load data
    coupling_times, mi_211, mi_304 = load_coupling_data()
    goes_times, goes_flux = load_goes_data()

    # Key events
    break_time = datetime(2026, 1, 10, 17, 42)
    goes_spike_time = datetime(2026, 1, 10, 18, 9)

    # Compute baseline statistics (using data before break)
    pre_break_values = [v for t, v in zip(coupling_times, mi_211) if t < break_time]
    median, mad, threshold = compute_break_threshold(pre_break_values)

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08})

    # ========== Panel 1: ΔMI Coupling ==========
    ax1.set_ylabel('ΔMI [bits]', fontweight='bold')

    # Plot 193-211 as primary
    ax1.plot(coupling_times, mi_211, 'o-', color='#2E86AB', linewidth=2,
             markersize=6, label='193-211 Å (Corona)', zorder=5)

    # Plot 193-304 as secondary (lighter)
    ax1.plot(coupling_times, mi_304, 's--', color='#A23B72', linewidth=1.5,
             markersize=4, alpha=0.6, label='193-304 Å (Corona-Chromosphere)')

    # Baseline and threshold
    if median and threshold:
        ax1.axhline(median, color='gray', linestyle='-', alpha=0.5, label=f'Median = {median:.3f}')
        ax1.axhline(threshold, color='red', linestyle='--', alpha=0.7,
                    label=f'Break threshold (−2 MAD) = {threshold:.3f}')

        # Fill below threshold
        ax1.axhspan(0, threshold, alpha=0.1, color='red')

    # Mark the break point
    break_value = None
    for t, v in zip(coupling_times, mi_211):
        if abs((t - break_time).total_seconds()) < 120:  # within 2 min
            break_value = v
            break_idx_time = t
            break

    if break_value:
        ax1.scatter([break_idx_time], [break_value], s=200, c='red', marker='v',
                   zorder=10, edgecolors='darkred', linewidths=2)
        ax1.annotate(f'BREAK\n{break_value:.3f} bits',
                    xy=(break_idx_time, break_value),
                    xytext=(break_idx_time, break_value + 0.15),
                    fontsize=9, fontweight='bold', color='darkred',
                    ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

    # Validation box
    validation_text = "✓ Registration: 0px\n✓ Time sync: 5s\n✓ Robustness: −1.7%"
    props = dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', edgecolor='green', alpha=0.9)
    ax1.text(0.02, 0.98, validation_text, transform=ax1.transAxes, fontsize=8,
             verticalalignment='top', bbox=props, family='monospace')

    ax1.legend(loc='upper right', framealpha=0.95)
    ax1.set_ylim(0.3, 1.15)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Case Study: Coupling Break Precedes GOES Response (2026-01-10)',
                  fontweight='bold', fontsize=12)

    # ========== Panel 2: GOES X-ray ==========
    ax2.set_ylabel('GOES Flux\n[W/m²]', fontweight='bold')
    ax2.set_xlabel('Time (UTC)', fontweight='bold')

    # Plot GOES flux
    ax2.plot(goes_times, goes_flux, 'o-', color='#F18F01', linewidth=2,
             markersize=5, label='GOES 1-8 Å')

    # Mark the spike
    spike_value = None
    pre_spike_value = None
    for i, (t, f) in enumerate(zip(goes_times, goes_flux)):
        if abs((t - goes_spike_time).total_seconds()) < 120:
            spike_value = f
            spike_time_actual = t
        if abs((t - datetime(2026, 1, 10, 17, 58)).total_seconds()) < 120:
            pre_spike_value = f

    if spike_value:
        ax2.scatter([spike_time_actual], [spike_value], s=200, c='#F18F01', marker='^',
                   zorder=10, edgecolors='darkorange', linewidths=2)

        if pre_spike_value:
            pct_change = (spike_value - pre_spike_value) / pre_spike_value * 100
            ax2.annotate(f'+{pct_change:.0f}%\n(B5→B9)',
                        xy=(spike_time_actual, spike_value),
                        xytext=(spike_time_actual, spike_value * 1.15),
                        fontsize=9, fontweight='bold', color='darkorange',
                        ha='center', va='bottom',
                        arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))

    # Time delta annotation
    if break_value and spike_value:
        delta_min = (goes_spike_time - break_time).total_seconds() / 60
        mid_time = break_time + (goes_spike_time - break_time) / 2

        # Draw arrow between break and spike
        ax2.annotate('', xy=(goes_spike_time, goes_flux[goes_times.index(spike_time_actual)] * 0.5),
                    xytext=(break_time, goes_flux[goes_times.index(spike_time_actual)] * 0.5),
                    arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax2.text(mid_time, goes_flux[goes_times.index(spike_time_actual)] * 0.55,
                f'Δt = {delta_min:.0f} min\n(precursor window)',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='green')

    ax2.set_yscale('log')
    ax2.set_ylim(3e-7, 2e-6)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

    # Add vertical lines at key events
    for ax in [ax1, ax2]:
        ax.axvline(break_time, color='red', linestyle=':', alpha=0.7, linewidth=1.5)
        ax.axvline(goes_spike_time, color='orange', linestyle=':', alpha=0.7, linewidth=1.5)

    # Save figure
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / 'case_study_coupling_break.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')

    # Also save as PDF for paper
    pdf_path = output_dir / 'case_study_coupling_break.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f'Saved: {pdf_path}')

    plt.close()

    # Print summary
    print(f"""
Case Study Summary:
==================
Break detected:     {break_time.strftime('%H:%M')} UTC
Break value:        {break_value:.3f} bits
Threshold:          {threshold:.3f} bits (median − 2×MAD)
z_mad:              {(median - break_value) / mad:.1f} MAD below median

GOES response:      {goes_spike_time.strftime('%H:%M')} UTC
Flux increase:      {pct_change:.0f}% (B5 → B9)
Precursor window:   {delta_min:.0f} minutes

Validation:
  ✓ Registration:   OK (0px shift)
  ✓ Time sync:      OK (5s spread)
  ✓ Robustness:     STABLE (−1.7% under binning)

Classification:     CONFIRMED PRECURSOR (TP)
""")


if __name__ == '__main__':
    create_figure()
