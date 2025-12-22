"""
Räumliche MI-Analyse für Solar Seed
====================================

Berechnet MI-Karten über die Sonnenscheibe um zu identifizieren,
wo die höchste (Residual-)MI auftritt.

Fragestellung: Wo auf der Sonne ist die "Extra-Information"?
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Optional
from dataclasses import dataclass

from solar_seed.mutual_info import mutual_information
from solar_seed.radial_profile import prepare_pair_for_residual_mi


@dataclass
class SpatialMIResult:
    """Ergebnis der räumlichen MI-Analyse."""
    mi_map: NDArray[np.float64]  # MI pro Region
    grid_size: Tuple[int, int]  # (rows, cols)
    cell_size: Tuple[int, int]  # Pixelgröße pro Zelle
    image_shape: Tuple[int, int]  # Originalbild-Größe

    # Statistiken
    mi_mean: float
    mi_std: float
    mi_max: float
    mi_min: float

    # Hotspot-Info
    hotspot_idx: Tuple[int, int]  # (row, col) des Maximums
    hotspot_value: float


@dataclass
class SpatialComparisonResult:
    """Vergleich von Original-MI und Residual-MI Maps."""
    original: SpatialMIResult
    residual: SpatialMIResult

    # Differenz-Analyse
    mi_reduction_map: NDArray[np.float64]  # original - residual
    mi_reduction_percent: NDArray[np.float64]  # (orig - res) / orig * 100

    # Wo bleibt die meiste MI nach Subtraktion?
    residual_hotspot_idx: Tuple[int, int]
    residual_hotspot_value: float


def compute_spatial_mi_map(
    image_1: NDArray[np.float64],
    image_2: NDArray[np.float64],
    grid_size: Tuple[int, int] = (8, 8),
    bins: int = 32,
    min_valid_pixels: int = 100
) -> SpatialMIResult:
    """
    Berechnet eine räumliche MI-Karte.

    Teilt die Bilder in ein Grid und berechnet MI pro Zelle.

    Args:
        image_1: Erstes Bild (z.B. 193 Å)
        image_2: Zweites Bild (z.B. 211 Å)
        grid_size: (rows, cols) Anzahl der Grid-Zellen
        bins: Bins für MI-Berechnung (weniger für kleinere Regionen)
        min_valid_pixels: Mindestanzahl nicht-null Pixel für MI-Berechnung

    Returns:
        SpatialMIResult mit MI-Karte und Statistiken
    """
    rows, cols = grid_size
    h, w = image_1.shape

    cell_h = h // rows
    cell_w = w // cols

    mi_map = np.zeros((rows, cols), dtype=np.float64)

    for i in range(rows):
        for j in range(cols):
            # Extrahiere Region
            y_start = i * cell_h
            y_end = (i + 1) * cell_h if i < rows - 1 else h
            x_start = j * cell_w
            x_end = (j + 1) * cell_w if j < cols - 1 else w

            region_1 = image_1[y_start:y_end, x_start:x_end]
            region_2 = image_2[y_start:y_end, x_start:x_end]

            # Prüfe ob genug valide Pixel vorhanden
            valid_mask = (region_1 > 0) & (region_2 > 0)
            n_valid = valid_mask.sum()

            if n_valid >= min_valid_pixels:
                mi_map[i, j] = mutual_information(
                    region_1[valid_mask],
                    region_2[valid_mask],
                    bins=bins
                )
            else:
                mi_map[i, j] = np.nan

    # Statistiken (ignoriere NaN)
    valid_mi = mi_map[~np.isnan(mi_map)]
    if len(valid_mi) > 0:
        mi_mean = float(np.mean(valid_mi))
        mi_std = float(np.std(valid_mi))
        mi_max = float(np.max(valid_mi))
        mi_min = float(np.min(valid_mi))

        # Finde Hotspot
        max_idx = np.unravel_index(np.nanargmax(mi_map), mi_map.shape)
        hotspot_idx = (int(max_idx[0]), int(max_idx[1]))
        hotspot_value = float(mi_map[hotspot_idx])
    else:
        mi_mean = mi_std = mi_max = mi_min = 0.0
        hotspot_idx = (0, 0)
        hotspot_value = 0.0

    return SpatialMIResult(
        mi_map=mi_map,
        grid_size=grid_size,
        cell_size=(cell_h, cell_w),
        image_shape=image_1.shape,
        mi_mean=mi_mean,
        mi_std=mi_std,
        mi_max=mi_max,
        mi_min=mi_min,
        hotspot_idx=hotspot_idx,
        hotspot_value=hotspot_value
    )


def compute_spatial_residual_mi_map(
    image_1: NDArray[np.float64],
    image_2: NDArray[np.float64],
    grid_size: Tuple[int, int] = (8, 8),
    bins: int = 32,
    n_profile_bins: int = 100
) -> SpatialComparisonResult:
    """
    Berechnet und vergleicht Original- und Residual-MI-Karten.

    Args:
        image_1: Erstes Bild
        image_2: Zweites Bild
        grid_size: Grid-Größe
        bins: Bins für MI
        n_profile_bins: Bins für Radialprofil

    Returns:
        SpatialComparisonResult mit beiden Karten und Vergleich
    """
    # Original MI-Karte
    original = compute_spatial_mi_map(image_1, image_2, grid_size, bins)

    # Residuen berechnen
    residual_1, residual_2, _ = prepare_pair_for_residual_mi(
        image_1, image_2, n_bins=n_profile_bins
    )

    # Residual MI-Karte
    residual = compute_spatial_mi_map(residual_1, residual_2, grid_size, bins)

    # Reduktions-Analyse
    mi_reduction_map = original.mi_map - residual.mi_map

    # Prozentuale Reduktion (vermeide Division durch 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        mi_reduction_percent = np.where(
            original.mi_map > 0,
            (original.mi_map - residual.mi_map) / original.mi_map * 100,
            0
        )

    return SpatialComparisonResult(
        original=original,
        residual=residual,
        mi_reduction_map=mi_reduction_map,
        mi_reduction_percent=mi_reduction_percent,
        residual_hotspot_idx=residual.hotspot_idx,
        residual_hotspot_value=residual.hotspot_value
    )


def find_top_hotspots(
    result: SpatialMIResult,
    n: int = 5
) -> List[Tuple[Tuple[int, int], float]]:
    """
    Findet die Top-N Hotspots (höchste MI-Werte).

    Args:
        result: SpatialMIResult
        n: Anzahl Hotspots

    Returns:
        Liste von ((row, col), mi_value) Tupeln
    """
    mi_map = result.mi_map.copy()
    mi_map = np.nan_to_num(mi_map, nan=-np.inf)

    # Flache Indizes sortiert nach Wert
    flat_indices = np.argsort(mi_map.ravel())[::-1]

    hotspots = []
    for idx in flat_indices[:n]:
        row, col = np.unravel_index(idx, mi_map.shape)
        value = result.mi_map[row, col]
        if not np.isnan(value):
            hotspots.append(((int(row), int(col)), float(value)))

    return hotspots


def get_region_coordinates(
    result: SpatialMIResult,
    grid_idx: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Gibt die Pixelkoordinaten einer Grid-Region zurück.

    Args:
        result: SpatialMIResult
        grid_idx: (row, col) im Grid

    Returns:
        (y_start, y_end, x_start, x_end) in Pixeln
    """
    row, col = grid_idx
    cell_h, cell_w = result.cell_size
    h, w = result.image_shape
    rows, cols = result.grid_size

    y_start = row * cell_h
    y_end = (row + 1) * cell_h if row < rows - 1 else h
    x_start = col * cell_w
    x_end = (col + 1) * cell_w if col < cols - 1 else w

    return y_start, y_end, x_start, x_end


# ============================================================================
# ASCII VISUALISIERUNG
# ============================================================================

def mi_map_to_ascii(
    mi_map: NDArray[np.float64],
    width: int = 40,
    show_values: bool = False
) -> str:
    """
    Konvertiert MI-Karte zu ASCII-Art für Terminal-Ausgabe.

    Args:
        mi_map: 2D MI-Karte
        width: Breite der Ausgabe in Zeichen
        show_values: Zeige numerische Werte

    Returns:
        ASCII-String der MI-Karte
    """
    # Intensitäts-Zeichen (von niedrig zu hoch)
    chars = " ·:░▒▓█"

    rows, cols = mi_map.shape

    # Normalisiere auf [0, 1]
    valid_mask = ~np.isnan(mi_map)
    if not valid_mask.any():
        return "Keine gültigen Daten"

    mi_min = np.nanmin(mi_map)
    mi_max = np.nanmax(mi_map)
    mi_range = mi_max - mi_min if mi_max > mi_min else 1.0

    normalized = (mi_map - mi_min) / mi_range
    normalized = np.nan_to_num(normalized, nan=0)

    # Berechne Zeichenbreite pro Zelle
    char_per_cell = max(1, width // cols)

    lines = []

    # Oberer Rahmen
    lines.append("┌" + "─" * (cols * char_per_cell) + "┐")

    for i in range(rows):
        line = "│"
        for j in range(cols):
            if np.isnan(mi_map[i, j]):
                char = " "
            else:
                idx = int(normalized[i, j] * (len(chars) - 1))
                idx = min(idx, len(chars) - 1)
                char = chars[idx]
            line += char * char_per_cell
        line += "│"
        lines.append(line)

    # Unterer Rahmen
    lines.append("└" + "─" * (cols * char_per_cell) + "┘")

    # Legende
    lines.append(f"  MI: {mi_min:.3f} {'·' * 3} {mi_max:.3f}")

    return "\n".join(lines)


def print_spatial_comparison(
    result: SpatialComparisonResult,
    title: str = "Räumliche MI-Analyse"
) -> None:
    """
    Gibt einen formatierten Vergleich der räumlichen MI aus.

    Args:
        result: SpatialComparisonResult
        title: Titel für die Ausgabe
    """
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")

    print(f"\n  Grid: {result.original.grid_size[0]}x{result.original.grid_size[1]}")
    print(f"  Zellgröße: {result.original.cell_size[0]}x{result.original.cell_size[1]} Pixel")

    # Original MI
    print(f"\n  ORIGINAL MI:")
    print(f"    Mittel: {result.original.mi_mean:.4f} ± {result.original.mi_std:.4f}")
    print(f"    Bereich: [{result.original.mi_min:.4f}, {result.original.mi_max:.4f}]")
    print(f"    Hotspot: Zelle {result.original.hotspot_idx} = {result.original.hotspot_value:.4f}")

    print(f"\n  ORIGINAL MI-KARTE:")
    print(mi_map_to_ascii(result.original.mi_map))

    # Residual MI
    print(f"\n  RESIDUAL MI (nach Geometrie-Subtraktion):")
    print(f"    Mittel: {result.residual.mi_mean:.4f} ± {result.residual.mi_std:.4f}")
    print(f"    Bereich: [{result.residual.mi_min:.4f}, {result.residual.mi_max:.4f}]")
    print(f"    Hotspot: Zelle {result.residual.hotspot_idx} = {result.residual.hotspot_value:.4f}")

    print(f"\n  RESIDUAL MI-KARTE:")
    print(mi_map_to_ascii(result.residual.mi_map))

    # Reduktions-Analyse
    valid_reduction = result.mi_reduction_percent[~np.isnan(result.mi_reduction_percent)]
    if len(valid_reduction) > 0:
        mean_reduction = np.mean(valid_reduction)
        print(f"\n  REDUKTION DURCH GEOMETRIE-SUBTRAKTION:")
        print(f"    Mittlere Reduktion: {mean_reduction:.1f}%")

        # Wo bleibt am meisten MI?
        min_reduction_idx = np.unravel_index(
            np.nanargmin(result.mi_reduction_percent),
            result.mi_reduction_percent.shape
        )
        min_reduction = result.mi_reduction_percent[min_reduction_idx]
        print(f"    Geringste Reduktion: Zelle {min_reduction_idx} ({min_reduction:.1f}%)")
        print(f"    → Diese Region hat die meiste 'Extra-Information'")

    # Top Hotspots im Residual
    print(f"\n  TOP 5 RESIDUAL-HOTSPOTS:")
    hotspots = find_top_hotspots(result.residual, n=5)
    for rank, (idx, value) in enumerate(hotspots, 1):
        coords = get_region_coordinates(result.residual, idx)
        print(f"    {rank}. Zelle {idx}: MI={value:.4f} (Pixel {coords[0]}-{coords[1]}, {coords[2]}-{coords[3]})")


def create_disk_mask(
    shape: Tuple[int, int],
    center: Optional[Tuple[float, float]] = None,
    radius_fraction: float = 0.9
) -> NDArray[np.bool_]:
    """
    Erstellt eine kreisförmige Maske für die Sonnenscheibe.

    Args:
        shape: (height, width) des Bildes
        center: (y, x) Zentrum, oder None für Bildmitte
        radius_fraction: Radius als Anteil der halben Bildgröße

    Returns:
        Boolesche Maske (True = innerhalb der Scheibe)
    """
    h, w = shape
    if center is None:
        center = (h / 2, w / 2)

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    radius = min(h, w) / 2 * radius_fraction

    return r <= radius


def compute_disk_spatial_mi(
    image_1: NDArray[np.float64],
    image_2: NDArray[np.float64],
    grid_size: Tuple[int, int] = (8, 8),
    bins: int = 32
) -> SpatialComparisonResult:
    """
    Berechnet räumliche MI nur innerhalb der Sonnenscheibe.

    Pixel außerhalb der Scheibe werden auf 0 gesetzt.

    Args:
        image_1: Erstes Bild
        image_2: Zweites Bild
        grid_size: Grid-Größe
        bins: Bins für MI

    Returns:
        SpatialComparisonResult
    """
    # Erstelle Disk-Maske
    mask = create_disk_mask(image_1.shape)

    # Wende Maske an
    image_1_masked = image_1.copy()
    image_2_masked = image_2.copy()
    image_1_masked[~mask] = 0
    image_2_masked[~mask] = 0

    return compute_spatial_residual_mi_map(
        image_1_masked, image_2_masked, grid_size, bins
    )
