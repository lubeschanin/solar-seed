"""
Radialprofil-Analyse für Solar Seed
====================================

Subtrahiert die radiale Geometrie (Limb Darkening) um
"echte" Korrelationen von geometrischen zu trennen.

Methode:
1. Berechne mittlere Intensität als Funktion des Radius
2. Rekonstruiere Geometrie-Bild aus diesem Profil
3. Residuum = Original / Geometrie-Modell
4. MI auf Residuen → zeigt nicht-geometrische Korrelation
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class RadialProfile:
    """Ergebnis der Radialprofil-Berechnung."""
    radii: NDArray[np.float64]  # Mittelpunkte der Radius-Bins
    intensities: NDArray[np.float64]  # Mean intensity per bin
    counts: NDArray[np.int64]  # Anzahl Pixel pro Bin
    center: Tuple[float, float]  # Verwendetes Zentrum (y, x)
    n_bins: int


def find_disk_center(
    image: NDArray[np.float64],
    threshold_fraction: float = 0.1
) -> Tuple[float, float]:
    """
    Findet das Zentrum der Sonnenscheibe.

    Verwendet den Intensitäts-Schwerpunkt der Pixel über dem Schwellwert.

    Args:
        image: 2D Bild-Array
        threshold_fraction: Schwellwert als Anteil des Maximums

    Returns:
        (center_y, center_x) Koordinaten des Zentrums
    """
    threshold = image.max() * threshold_fraction
    mask = image > threshold

    if not mask.any():
        # Fallback: Bildmitte
        return image.shape[0] / 2, image.shape[1] / 2

    y_coords, x_coords = np.where(mask)
    weights = image[mask]

    center_y = np.average(y_coords, weights=weights)
    center_x = np.average(x_coords, weights=weights)

    return float(center_y), float(center_x)


def compute_radial_profile(
    image: NDArray[np.float64],
    n_bins: int = 100,
    center: Optional[Tuple[float, float]] = None,
    max_radius: Optional[float] = None
) -> RadialProfile:
    """
    Berechnet das radiale Intensitätsprofil eines Bildes.

    Args:
        image: 2D Bild-Array
        n_bins: Anzahl der Radius-Bins
        center: (y, x) Zentrum, oder None für automatische Erkennung
        max_radius: Maximaler Radius, oder None für Bilddiagonale/2

    Returns:
        RadialProfile mit Radius-Bins und mittleren Intensitäten
    """
    if center is None:
        center = find_disk_center(image)

    center_y, center_x = center

    # Koordinaten-Grid
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    if max_radius is None:
        max_radius = min(image.shape) / 2

    # Bin-Grenzen
    bin_edges = np.linspace(0, max_radius, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Berechne Mittelwert pro Bin
    intensities = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        counts[i] = mask.sum()
        if counts[i] > 0:
            intensities[i] = image[mask].mean()

    return RadialProfile(
        radii=bin_centers,
        intensities=intensities,
        counts=counts,
        center=center,
        n_bins=n_bins
    )


def reconstruct_from_profile(
    profile: RadialProfile,
    shape: Tuple[int, int]
) -> NDArray[np.float64]:
    """
    Rekonstruiert ein Bild aus dem Radialprofil.

    Jeder Pixel erhält die mittlere Intensität seines Radius-Bins.

    Args:
        profile: RadialProfile mit Radius-Intensitäts-Mapping
        shape: (height, width) des Output-Bildes

    Returns:
        Rekonstruiertes Geometrie-Bild
    """
    center_y, center_x = profile.center

    # Koordinaten-Grid
    y, x = np.ogrid[:shape[0], :shape[1]]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Bin-Grenzen
    max_radius = profile.radii[-1] + (profile.radii[1] - profile.radii[0]) / 2
    bin_width = max_radius / profile.n_bins

    # Weise jedem Pixel seinen Bin zu
    bin_indices = np.clip((r / bin_width).astype(int), 0, profile.n_bins - 1)

    # Rekonstruiere Bild
    reconstructed = profile.intensities[bin_indices]

    return reconstructed


def compute_residual(
    image: NDArray[np.float64],
    model: NDArray[np.float64],
    method: str = "ratio",
    epsilon: float = 1.0
) -> NDArray[np.float64]:
    """
    Berechnet das Residuum zwischen Bild und Modell.

    Args:
        image: Original-Bild
        model: Geometrie-Modell (aus Radialprofil)
        method: "ratio" (image/model) oder "difference" (image-model)
        epsilon: Kleine Konstante um Division durch 0 zu vermeiden

    Returns:
        Residuum-Bild
    """
    if method == "ratio":
        # Ratio is better for multiplicative effects (limb darkening)
        return image / (model + epsilon)
    elif method == "difference":
        return image - model
    else:
        raise ValueError(f"Unbekannte Methode: {method}")


def subtract_radial_geometry(
    image: NDArray[np.float64],
    n_bins: int = 100,
    method: str = "ratio"
) -> Tuple[NDArray[np.float64], RadialProfile, NDArray[np.float64]]:
    """
    Kompletter Workflow: Profil berechnen, Modell erstellen, Residuum berechnen.

    Args:
        image: Original-Bild
        n_bins: Anzahl Radius-Bins
        method: "ratio" oder "difference"

    Returns:
        residual: Bild ohne radiale Geometrie
        profile: Berechnetes Radialprofil
        model: Rekonstruiertes Geometrie-Bild
    """
    profile = compute_radial_profile(image, n_bins=n_bins)
    model = reconstruct_from_profile(profile, image.shape)
    residual = compute_residual(image, model, method=method)

    return residual, profile, model


def prepare_pair_for_residual_mi(
    image_1: NDArray[np.float64],
    image_2: NDArray[np.float64],
    n_bins: int = 100,
    method: str = "ratio",
    shared_center: bool = True
) -> Tuple[NDArray[np.float64], NDArray[np.float64], dict]:
    """
    Bereitet ein Bildpaar für Residual-MI-Analyse vor.

    Args:
        image_1: Erstes Bild (z.B. 193 Å)
        image_2: Zweites Bild (z.B. 211 Å)
        n_bins: Anzahl Radius-Bins
        method: Residuum-Methode
        shared_center: Verwende gleiches Zentrum für beide Bilder

    Returns:
        residual_1: Residuum von Bild 1
        residual_2: Residuum von Bild 2
        info: Dictionary mit Profilen und Modellen
    """
    if shared_center:
        # Finde gemeinsames Zentrum (Mittelwert beider Zentren)
        center_1 = find_disk_center(image_1)
        center_2 = find_disk_center(image_2)
        center = (
            (center_1[0] + center_2[0]) / 2,
            (center_1[1] + center_2[1]) / 2
        )
    else:
        center = None

    # Berechne Profile
    profile_1 = compute_radial_profile(image_1, n_bins=n_bins, center=center)
    profile_2 = compute_radial_profile(image_2, n_bins=n_bins, center=center)

    # Rekonstruiere Modelle
    model_1 = reconstruct_from_profile(profile_1, image_1.shape)
    model_2 = reconstruct_from_profile(profile_2, image_2.shape)

    # Berechne Residuen
    residual_1 = compute_residual(image_1, model_1, method=method)
    residual_2 = compute_residual(image_2, model_2, method=method)

    info = {
        "profile_1": profile_1,
        "profile_2": profile_2,
        "model_1": model_1,
        "model_2": model_2,
        "center": center if shared_center else (profile_1.center, profile_2.center),
        "method": method
    }

    return residual_1, residual_2, info
