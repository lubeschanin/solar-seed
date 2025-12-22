"""Visualization module for Solar Seed results.

Generates publication-ready figures from analysis results.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# AIA channel information
AIA_CHANNELS = {
    304: {"temp": 0.05, "region": "Chromosphere", "color": "#e41a1c"},
    171: {"temp": 0.6, "region": "Quiet Corona", "color": "#377eb8"},
    193: {"temp": 1.2, "region": "Corona", "color": "#4daf4a"},
    211: {"temp": 2.0, "region": "Active Regions", "color": "#984ea3"},
    335: {"temp": 2.5, "region": "Hot AR", "color": "#ff7f00"},
    94: {"temp": 6.3, "region": "Flares", "color": "#ffff33"},
    131: {"temp": 10.0, "region": "Hot Flares", "color": "#a65628"},
}


def load_coupling_data(results_dir: Path) -> dict:
    """Load coupling matrices from JSON."""
    json_path = results_dir / "coupling_matrices.json"
    if not json_path.exists():
        raise FileNotFoundError(f"No coupling data found at {json_path}")

    with open(json_path) as f:
        return json.load(f)


def load_pair_results(results_dir: Path) -> list[dict]:
    """Load pair results from CSV."""
    import csv
    csv_path = results_dir / "pair_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No pair results found at {csv_path}")

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_geometric_normalization(
    results_dir: Path,
    output_path: Path,
) -> None:
    """Figure 1: Effect of geometric normalization on MI.

    Shows MI before and after radial profile normalization.
    """
    import matplotlib.pyplot as plt

    pairs = load_pair_results(results_dir)

    # Extract data
    pair_labels = []
    mi_original = []
    mi_residual = []

    for p in pairs:
        label = f"{p['wavelength_1']}-{p['wavelength_2']}"
        pair_labels.append(label)
        mi_original.append(float(p['mi_original']))
        mi_residual.append(float(p['mi_residual']))

    # Sort by original MI for better visualization
    sorted_idx = np.argsort(mi_original)[::-1]
    pair_labels = [pair_labels[i] for i in sorted_idx]
    mi_original = [mi_original[i] for i in sorted_idx]
    mi_residual = [mi_residual[i] for i in sorted_idx]

    # Calculate statistics
    mean_original = np.mean(mi_original)
    mean_residual = np.mean(mi_residual)
    reduction_pct = (1 - mean_residual / mean_original) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(pair_labels))
    width = 0.7

    # Left: Original MI
    bars1 = ax1.bar(x, mi_original, width, color='#3498db', edgecolor='black', alpha=0.8)
    ax1.set_ylabel('Mutual Information (bits)', fontsize=11)
    ax1.set_xlabel('Channel Pair', fontsize=11)
    ax1.set_title('Original Images', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{l} Å" for l in pair_labels], rotation=45, ha='right', fontsize=8)
    ax1.axhline(y=mean_original, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_original:.2f} bits')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, max(mi_original) * 1.15)
    ax1.grid(True, alpha=0.3, axis='y')

    # Right: Residual MI
    bars2 = ax2.bar(x, mi_residual, width, color='#e74c3c', edgecolor='black', alpha=0.8)
    ax2.set_ylabel('Mutual Information (bits)', fontsize=11)
    ax2.set_xlabel('Channel Pair', fontsize=11)
    ax2.set_title('After Radial Normalization', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{l} Å" for l in pair_labels], rotation=45, ha='right', fontsize=8)
    ax2.axhline(y=mean_residual, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_residual:.2f} bits')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, max(mi_original) * 1.15)  # Same scale as left
    ax2.grid(True, alpha=0.3, axis='y')

    # Main title
    fig.suptitle(
        f'Effect of Geometric Normalization on Multichannel MI\n'
        f'(~{reduction_pct:.0f}% reduction, stable residual remains)',
        fontsize=13, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_coupling_heatmap(
    data: dict,
    output_path: Path,
    title: str = "Local Coupling Matrix (ΔMI_sector)",
) -> None:
    """Generate 7x7 coupling matrix heatmap."""
    import matplotlib.pyplot as plt

    wavelengths = data["wavelengths"]
    matrix = np.array(data["delta_mi_sector"])

    fig, ax = plt.subplots(figsize=(8, 7))

    # Create heatmap
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="equal")

    # Labels
    labels = [f"{w} Å" for w in wavelengths]
    ax.set_xticks(range(len(wavelengths)))
    ax.set_yticks(range(len(wavelengths)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Add values in cells
    for i in range(len(wavelengths)):
        for j in range(len(wavelengths)):
            value = matrix[i, j]
            if value > 0:
                color = "white" if value > 0.4 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                       color=color, fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label="ΔMI_sector (bits)")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("AIA Channel")
    ax.set_ylabel("AIA Channel")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_temperature_coupling(
    data: dict,
    output_path: Path,
) -> None:
    """Plot coupling strength vs temperature difference."""
    import matplotlib.pyplot as plt

    wavelengths = data["wavelengths"]
    matrix = np.array(data["delta_mi_sector"])

    # Extract pairs with temperature differences
    pairs = []
    for i in range(len(wavelengths)):
        for j in range(i + 1, len(wavelengths)):
            w1, w2 = wavelengths[i], wavelengths[j]
            temp_diff = abs(AIA_CHANNELS[w1]["temp"] - AIA_CHANNELS[w2]["temp"])
            coupling = matrix[i, j]
            pairs.append({
                "pair": f"{w1}-{w2}",
                "temp_diff": temp_diff,
                "coupling": coupling,
                "w1": w1,
                "w2": w2,
            })

    # Sort by coupling strength
    pairs.sort(key=lambda x: x["coupling"], reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Scatter plot
    temp_diffs = [p["temp_diff"] for p in pairs]
    couplings = [p["coupling"] for p in pairs]

    # Color by whether adjacent in temperature
    colors = []
    for p in pairs:
        if p["temp_diff"] < 1.0:
            colors.append("#2ecc71")  # Green for adjacent
        elif p["temp_diff"] < 3.0:
            colors.append("#3498db")  # Blue for moderate
        else:
            colors.append("#e74c3c")  # Red for distant

    ax1.scatter(temp_diffs, couplings, c=colors, s=100, alpha=0.7, edgecolors="black")

    # Annotate top pairs
    for p in pairs[:5]:
        ax1.annotate(
            p["pair"],
            (p["temp_diff"], p["coupling"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax1.set_xlabel("Temperature Difference (MK)", fontsize=11)
    ax1.set_ylabel("ΔMI_sector (bits)", fontsize=11)
    ax1.set_title("Coupling vs Temperature Distance", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Adjacent (<1 MK)"),
        Patch(facecolor="#3498db", label="Moderate (1-3 MK)"),
        Patch(facecolor="#e74c3c", label="Distant (>3 MK)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right")

    # Right: Bar chart of top pairs
    top_n = 10
    top_pairs = pairs[:top_n]

    y_pos = np.arange(len(top_pairs))
    bars = ax2.barh(y_pos, [p["coupling"] for p in top_pairs], color="#3498db", edgecolor="black")

    # Highlight 193-211 (strongest)
    for i, p in enumerate(top_pairs):
        if p["pair"] == "193-211":
            bars[i].set_color("#e74c3c")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{p['pair']} Å" for p in top_pairs])
    ax2.set_xlabel("ΔMI_sector (bits)", fontsize=11)
    ax2.set_title("Top Channel Pairs by Coupling", fontsize=12, fontweight="bold")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_null_model_hierarchy(output_path: Path) -> None:
    """Generate schematic of null model hierarchy."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Define boxes
    boxes = [
        {"x": 1, "y": 4.5, "label": "MI_global\n(noise)", "color": "#bdc3c7", "mi": "~0"},
        {"x": 3.5, "y": 4.5, "label": "MI_ring\n(radial)", "color": "#95a5a6", "mi": "0.54"},
        {"x": 6, "y": 4.5, "label": "MI_sector\n(azimuthal)", "color": "#7f8c8d", "mi": "0.71"},
        {"x": 8.5, "y": 4.5, "label": "MI_original\n(local)", "color": "#2c3e50", "mi": "0.88"},
    ]

    # Draw boxes and arrows
    for i, box in enumerate(boxes):
        rect = mpatches.FancyBboxPatch(
            (box["x"] - 0.8, box["y"] - 0.6),
            1.6, 1.2,
            boxstyle="round,pad=0.05",
            facecolor=box["color"],
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(rect)

        # Label
        text_color = "white" if box["color"] in ["#7f8c8d", "#2c3e50"] else "black"
        ax.text(box["x"], box["y"], box["label"], ha="center", va="center",
                fontsize=10, fontweight="bold", color=text_color)

        # MI value below
        ax.text(box["x"], box["y"] - 1.0, f"MI = {box['mi']} bits", ha="center",
                fontsize=9, style="italic")

        # Arrow to next
        if i < len(boxes) - 1:
            ax.annotate(
                "", xy=(boxes[i + 1]["x"] - 0.9, box["y"]),
                xytext=(box["x"] + 0.9, box["y"]),
                arrowprops=dict(arrowstyle="->", color="black", lw=2),
            )
            # Delta label
            ax.text(
                (box["x"] + boxes[i + 1]["x"]) / 2,
                box["y"] + 0.8,
                f"Δ = {float(boxes[i+1]['mi']) - float(box['mi']):.2f}" if box["mi"] != "~0" else "",
                ha="center", fontsize=8, color="#e74c3c",
            )

    # Title and explanation
    ax.text(5, 5.7, "Null Model Hierarchy", ha="center", fontsize=14, fontweight="bold")
    ax.text(5, 1.5,
            "Each level removes geometric structure:\n"
            "• Global shuffle → destroys all structure\n"
            "• Ring shuffle → preserves radial statistics\n"
            "• Sector shuffle → preserves radial + coarse azimuthal\n"
            "• Original → full local structure",
            ha="center", va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Key result
    ax.text(5, 0.3,
            "ΔMI_sector = MI_original − MI_sector = 0.17 bits (genuine local coupling)",
            ha="center", fontsize=11, fontweight="bold", color="#c0392b")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_channel_overview(output_path: Path) -> None:
    """Generate AIA channel temperature overview."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    channels = list(AIA_CHANNELS.keys())
    temps = [AIA_CHANNELS[c]["temp"] for c in channels]
    colors = [AIA_CHANNELS[c]["color"] for c in channels]
    regions = [AIA_CHANNELS[c]["region"] for c in channels]

    # Sort by temperature
    sorted_idx = np.argsort(temps)
    channels = [channels[i] for i in sorted_idx]
    temps = [temps[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]
    regions = [regions[i] for i in sorted_idx]

    # Bar chart
    x = np.arange(len(channels))
    bars = ax.bar(x, temps, color=colors, edgecolor="black", linewidth=1.5)

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c} Å" for c in channels], fontsize=11)
    ax.set_ylabel("Peak Temperature (MK)", fontsize=11)
    ax.set_title("SDO/AIA EUV Channels by Temperature", fontsize=12, fontweight="bold")

    # Add region labels on bars
    for i, (bar, region, temp) in enumerate(zip(bars, regions, temps)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            region,
            ha="center", va="bottom", fontsize=8, rotation=45,
        )

    ax.set_ylim(0, 12)
    ax.grid(True, alpha=0.3, axis="y")

    # Add log scale note
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.text(6.5, 1.2, "1 MK (Corona)", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_all_figures(
    results_dir: str | Path = "results/multichannel_real",
    output_dir: str | Path = "figures",
) -> None:
    """Generate all figures from results."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating figures from {results_dir}...")
    print(f"Output directory: {output_dir}\n")

    # Load data
    try:
        data = load_coupling_data(results_dir)
        has_data = True
    except FileNotFoundError:
        print("  Warning: No coupling data found, using synthetic results...")
        results_dir = Path("results/multichannel")
        try:
            data = load_coupling_data(results_dir)
            has_data = True
        except FileNotFoundError:
            has_data = False

    # Generate figures
    if has_data:
        plot_geometric_normalization(results_dir, output_dir / "figure1_geometric_normalization.png")
        plot_coupling_heatmap(data, output_dir / "coupling_matrix.png")
        plot_temperature_coupling(data, output_dir / "temperature_coupling.png")

    # These don't need data
    plot_null_model_hierarchy(output_dir / "null_model_hierarchy.png")
    plot_channel_overview(output_dir / "aia_channels.png")

    print(f"\nDone! Generated figures in {output_dir}/")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Solar Seed visualizations")
    parser.add_argument(
        "--results", "-r",
        default="results/multichannel_real",
        help="Results directory (default: results/multichannel_real)",
    )
    parser.add_argument(
        "--output", "-o",
        default="figures",
        help="Output directory for figures (default: figures)",
    )

    args = parser.parse_args()
    generate_all_figures(args.results, args.output)


if __name__ == "__main__":
    main()
