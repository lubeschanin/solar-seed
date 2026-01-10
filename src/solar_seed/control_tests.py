"""
Control Tests for Residual-MI
=============================

Four controls to ensure that the measured MI
is not caused by artifacts:

C1: Time-Shift Null     - Temporal decoupling
C2: Ring-wise Shuffle   - Azimuthal decoupling with preserved radial statistics
C3: PSF/Blur Matching   - Resolution matching
C4: Co-alignment Check  - Verify spatial registration
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Optional
from dataclasses import dataclass
from scipy import ndimage

from solar_seed.mutual_info import mutual_information
from solar_seed.radial_profile import (
    prepare_pair_for_residual_mi,
    find_disk_center
)


# ============================================================================
# C1: TIME-SHIFT NULL
# ============================================================================

@dataclass
class TimeShiftResult:
    """Result of the time-shift test."""
    mi_original: float
    mi_shifted: float
    mi_reduction: float
    mi_reduction_percent: float
    passed: bool  # True if MI drops significantly


def time_shift_null(
    image_1: NDArray[np.float64],
    image_2: NDArray[np.float64],
    seed: int = 42,
    bins: int = 64
) -> TimeShiftResult:
    """
    C1: Time-Shift Null Test.

    Simulates temporal decoupling by completely shuffling channel B.
    For real timeseries: use image from different time.

    Expectation: MI_residual drops significantly because spatial structures
    no longer correlate.

    Args:
        image_1: First image (Channel A)
        image_2: Second image (Channel B)
        seed: Random seed
        bins: Bins for MI calculation

    Returns:
        TimeShiftResult with comparison Original vs. Shifted
    """
    # Original residual MI
    res_1, res_2, _ = prepare_pair_for_residual_mi(image_1, image_2)
    mi_original = mutual_information(res_1, res_2, bins=bins)

    # "Time-shift" by globally shuffling channel B
    rng = np.random.default_rng(seed)
    image_2_shuffled = image_2.ravel().copy()
    rng.shuffle(image_2_shuffled)
    image_2_shuffled = image_2_shuffled.reshape(image_2.shape)

    # Residual MI after shift
    res_1_new, res_2_shifted, _ = prepare_pair_for_residual_mi(
        image_1, image_2_shuffled
    )
    mi_shifted = mutual_information(res_1_new, res_2_shifted, bins=bins)

    # Analysis
    mi_reduction = mi_original - mi_shifted
    mi_reduction_percent = (mi_reduction / mi_original * 100) if mi_original > 0 else 0

    # Test passed if MI drops by at least 50%
    passed = mi_reduction_percent > 50

    return TimeShiftResult(
        mi_original=mi_original,
        mi_shifted=mi_shifted,
        mi_reduction=mi_reduction,
        mi_reduction_percent=mi_reduction_percent,
        passed=passed
    )


# ============================================================================
# C2: RING-WISE SHUFFLE
# ============================================================================

@dataclass
class RingShuffleResult:
    """Result of the ring-shuffle test."""
    mi_original: float
    mi_ring_shuffled: float
    mi_global_shuffled: float
    ring_reduction_percent: float
    global_reduction_percent: float
    ring_stronger: bool  # True if ring-shuffle reduces more


@dataclass
class SectorRingShuffleResult:
    """Result of the extended sector-ring-shuffle test."""
    mi_original: float
    mi_ring_shuffled: float  # Radial only
    mi_sector_shuffled: float  # Radial + sector
    mi_global_shuffled: float  # Completely global

    ring_reduction_percent: float
    sector_reduction_percent: float
    global_reduction_percent: float

    # What explains what?
    radial_contribution: float  # MI_global - MI_ring
    azimuthal_contribution: float  # MI_ring - MI_sector
    local_structure: float  # MI_sector (what remains after everything)


def create_radial_bins(
    shape: Tuple[int, int],
    center: Tuple[float, float],
    n_rings: int = 20
) -> NDArray[np.int64]:
    """
    Creates a map of ring indices.

    Args:
        shape: Image size
        center: (y, x) center
        n_rings: Number of concentric rings

    Returns:
        Array with ring index per pixel
    """
    y, x = np.ogrid[:shape[0], :shape[1]]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    max_r = np.max(r)

    # Bin-Grenzen
    ring_indices = np.clip(
        (r / max_r * n_rings).astype(np.int64),
        0, n_rings - 1
    )

    return ring_indices


def ring_shuffle(
    image: NDArray[np.float64],
    ring_indices: NDArray[np.int64],
    seed: int = 42
) -> NDArray[np.float64]:
    """
    Shuffelt Pixel nur innerhalb gleicher Ringe.

    Erhält die radiale Statistik, zerstört azimutale Korrelationen.

    Args:
        image: Eingabebild
        ring_indices: Ring-Index pro Pixel
        seed: Random seed

    Returns:
        Bild mit ring-weise geshuffelten Pixeln
    """
    rng = np.random.default_rng(seed)
    result = image.copy()
    n_rings = ring_indices.max() + 1

    for ring_idx in range(n_rings):
        mask = ring_indices == ring_idx
        pixels = result[mask].copy()
        rng.shuffle(pixels)
        result[mask] = pixels

    return result


def ring_wise_shuffle_test(
    image_1: NDArray[np.float64],
    image_2: NDArray[np.float64],
    n_rings: int = 20,
    seed: int = 42,
    bins: int = 64
) -> RingShuffleResult:
    """
    C2: Ring-wise Shuffle Test.

    Vergleicht Ring-Shuffle mit globalem Shuffle.
    Ring-Shuffle erhält radiale Statistik aber zerstört azimutale Struktur.

    Erwartung: Ring-Shuffle reduziert MI stärker als globaler Shuffle,
    weil er gezielt die strukturelle Korrelation zerstört.

    Args:
        image_1: Erstes Bild
        image_2: Zweites Bild
        n_rings: Anzahl der Ringe
        seed: Random seed
        bins: Bins für MI

    Returns:
        RingShuffleResult mit Vergleich
    """
    # Finde Zentrum
    center = find_disk_center(image_1)

    # Ring-Indizes
    ring_indices = create_radial_bins(image_1.shape, center, n_rings)

    # Original residual MI
    res_1, res_2, _ = prepare_pair_for_residual_mi(image_1, image_2)
    mi_original = mutual_information(res_1, res_2, bins=bins)

    # Ring-Shuffle auf Kanal B
    image_2_ring = ring_shuffle(image_2, ring_indices, seed=seed)
    res_1_r, res_2_ring, _ = prepare_pair_for_residual_mi(image_1, image_2_ring)
    mi_ring_shuffled = mutual_information(res_1_r, res_2_ring, bins=bins)

    # Globaler Shuffle auf Kanal B
    rng = np.random.default_rng(seed + 1000)
    image_2_global = image_2.ravel().copy()
    rng.shuffle(image_2_global)
    image_2_global = image_2_global.reshape(image_2.shape)
    res_1_g, res_2_global, _ = prepare_pair_for_residual_mi(image_1, image_2_global)
    mi_global_shuffled = mutual_information(res_1_g, res_2_global, bins=bins)

    # Reduktionen
    ring_reduction = (mi_original - mi_ring_shuffled) / mi_original * 100 if mi_original > 0 else 0
    global_reduction = (mi_original - mi_global_shuffled) / mi_original * 100 if mi_original > 0 else 0

    return RingShuffleResult(
        mi_original=mi_original,
        mi_ring_shuffled=mi_ring_shuffled,
        mi_global_shuffled=mi_global_shuffled,
        ring_reduction_percent=ring_reduction,
        global_reduction_percent=global_reduction,
        ring_stronger=ring_reduction > global_reduction
    )


def create_sector_ring_bins(
    shape: Tuple[int, int],
    center: Tuple[float, float],
    n_rings: int = 20,
    n_sectors: int = 16
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Erstellt Ring- und Sektor-Indizes für combined shuffling.

    Args:
        shape: Bildgröße
        center: (y, x) Zentrum
        n_rings: Anzahl konzentrischer Ringe
        n_sectors: Anzahl azimutaler Sektoren

    Returns:
        (ring_indices, sector_indices) - jeweils Array mit Index pro Pixel
    """
    y, x = np.ogrid[:shape[0], :shape[1]]

    # Radiale Indizes
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    max_r = np.max(r)
    ring_indices = np.clip(
        (r / max_r * n_rings).astype(np.int64),
        0, n_rings - 1
    )

    # Azimutale Indizes (Winkel von 0 bis 2π)
    theta = np.arctan2(y - center[0], x - center[1])  # -π to π
    theta_normalized = (theta + np.pi) / (2 * np.pi)  # 0 to 1
    sector_indices = np.clip(
        (theta_normalized * n_sectors).astype(np.int64),
        0, n_sectors - 1
    )

    return ring_indices, sector_indices


def sector_ring_shuffle(
    image: NDArray[np.float64],
    ring_indices: NDArray[np.int64],
    sector_indices: NDArray[np.int64],
    seed: int = 42
) -> NDArray[np.float64]:
    """
    Shuffelt Pixel innerhalb von Ring+Sektor-Kombinationen.

    Erhält sowohl radiale als auch grobe azimutale Statistik,
    zerstört nur lokale (feine) Korrelationen.

    Args:
        image: Eingabebild
        ring_indices: Ring-Index pro Pixel
        sector_indices: Sektor-Index pro Pixel
        seed: Random seed

    Returns:
        Bild mit sector-ring-weise geshuffelten Pixeln
    """
    rng = np.random.default_rng(seed)
    result = image.copy()

    n_rings = ring_indices.max() + 1
    n_sectors = sector_indices.max() + 1

    for ring_idx in range(n_rings):
        for sector_idx in range(n_sectors):
            mask = (ring_indices == ring_idx) & (sector_indices == sector_idx)
            if mask.sum() > 1:
                pixels = result[mask].copy()
                rng.shuffle(pixels)
                result[mask] = pixels

    return result


def sector_ring_shuffle_test(
    image_1: NDArray[np.float64],
    image_2: NDArray[np.float64],
    n_rings: int = 20,
    n_sectors: int = 16,
    seed: int = 42,
    bins: int = 64
) -> SectorRingShuffleResult:
    """
    Erweiterter C2: Sector-Ring Shuffle Test.

    Vergleicht drei Shuffle-Level:
    1. Ring-only: Erhält radiale Statistik
    2. Sector-Ring: Erhält radiale + grobe azimutale Statistik
    3. Global: Zerstört alles

    Damit kann man sauber trennen:
    - Radialer Beitrag zur MI
    - Azimutaler Beitrag
    - Echte lokale Struktur

    Args:
        image_1: Erstes Bild
        image_2: Zweites Bild
        n_rings: Anzahl Ringe
        n_sectors: Anzahl Sektoren (z.B. 16 = 22.5° pro Sektor)
        seed: Random seed
        bins: Bins für MI

    Returns:
        SectorRingShuffleResult mit detaillierter Aufschlüsselung
    """
    # Finde Zentrum
    center = find_disk_center(image_1)

    # Indizes erstellen
    ring_indices, sector_indices = create_sector_ring_bins(
        image_1.shape, center, n_rings, n_sectors
    )

    # Original residual MI
    res_1, res_2, _ = prepare_pair_for_residual_mi(image_1, image_2)
    mi_original = mutual_information(res_1, res_2, bins=bins)

    # 1. Ring-Shuffle (nur radial)
    image_2_ring = ring_shuffle(image_2, ring_indices, seed=seed)
    res_1_r, res_2_ring, _ = prepare_pair_for_residual_mi(image_1, image_2_ring)
    mi_ring_shuffled = mutual_information(res_1_r, res_2_ring, bins=bins)

    # 2. Sector-Ring-Shuffle (radial + azimutal)
    image_2_sector = sector_ring_shuffle(
        image_2, ring_indices, sector_indices, seed=seed+500
    )
    res_1_s, res_2_sector, _ = prepare_pair_for_residual_mi(image_1, image_2_sector)
    mi_sector_shuffled = mutual_information(res_1_s, res_2_sector, bins=bins)

    # 3. Global Shuffle
    rng = np.random.default_rng(seed + 1000)
    image_2_global = image_2.ravel().copy()
    rng.shuffle(image_2_global)
    image_2_global = image_2_global.reshape(image_2.shape)
    res_1_g, res_2_global, _ = prepare_pair_for_residual_mi(image_1, image_2_global)
    mi_global_shuffled = mutual_information(res_1_g, res_2_global, bins=bins)

    # Reduktionen berechnen
    def safe_reduction(original, shuffled):
        return (original - shuffled) / original * 100 if original > 0 else 0

    ring_reduction = safe_reduction(mi_original, mi_ring_shuffled)
    sector_reduction = safe_reduction(mi_original, mi_sector_shuffled)
    global_reduction = safe_reduction(mi_original, mi_global_shuffled)

    # Analyze contributions
    # MI_global is baseline (nearly 0)
    # MI_ring - MI_global = radial contribution
    # MI_sector - MI_ring = azimuthal contribution (negative, since ring > sector)
    # MI_original - MI_sector = local structure
    radial_contribution = mi_ring_shuffled - mi_global_shuffled
    azimuthal_contribution = mi_sector_shuffled - mi_ring_shuffled
    local_structure = mi_original - mi_sector_shuffled

    return SectorRingShuffleResult(
        mi_original=mi_original,
        mi_ring_shuffled=mi_ring_shuffled,
        mi_sector_shuffled=mi_sector_shuffled,
        mi_global_shuffled=mi_global_shuffled,
        ring_reduction_percent=ring_reduction,
        sector_reduction_percent=sector_reduction,
        global_reduction_percent=global_reduction,
        radial_contribution=radial_contribution,
        azimuthal_contribution=azimuthal_contribution,
        local_structure=local_structure
    )


# ============================================================================
# C3: PSF/BLUR MATCHING
# ============================================================================

@dataclass
class BlurMatchResult:
    """Result of the blur-matching test."""
    mi_original: float
    mi_blurred: float
    mi_change: float
    mi_change_percent: float
    blur_sigma: float
    stable: bool  # True if MI changes little (<20%)


def apply_gaussian_blur(
    image: NDArray[np.float64],
    sigma: float
) -> NDArray[np.float64]:
    """
    Wendet Gaussian Blur auf ein Bild an.

    Args:
        image: Eingabebild
        sigma: Standardabweichung des Gauss-Kernels

    Returns:
        Geblurrtes Bild
    """
    return ndimage.gaussian_filter(image, sigma=sigma)


def psf_blur_matching(
    image_1: NDArray[np.float64],
    image_2: NDArray[np.float64],
    sigma: float = 2.0,
    bins: int = 64
) -> BlurMatchResult:
    """
    C3: PSF/Blur Matching Test.

    Bringt beide Kanäle auf gleiche effektive Auflösung durch Blur.
    Testet ob hochfrequente Details die MI treiben.

    Erwartung: Wenn MI stabil bleibt → großskalige Korrelationen dominant.
    Wenn MI stark fällt → hochfrequente Details waren wichtig.

    Args:
        image_1: Erstes Bild
        image_2: Zweites Bild
        sigma: Blur-Stärke (in Pixeln)
        bins: Bins für MI

    Returns:
        BlurMatchResult mit Vergleich vor/nach Blur
    """
    # Original residual MI
    res_1, res_2, _ = prepare_pair_for_residual_mi(image_1, image_2)
    mi_original = mutual_information(res_1, res_2, bins=bins)

    # Blur both channels
    image_1_blur = apply_gaussian_blur(image_1, sigma)
    image_2_blur = apply_gaussian_blur(image_2, sigma)

    # Residual MI after blur
    res_1_blur, res_2_blur, _ = prepare_pair_for_residual_mi(
        image_1_blur, image_2_blur
    )
    mi_blurred = mutual_information(res_1_blur, res_2_blur, bins=bins)

    # Analysis
    mi_change = mi_blurred - mi_original
    mi_change_percent = (mi_change / mi_original * 100) if mi_original > 0 else 0

    # Stable if change < 20%
    stable = abs(mi_change_percent) < 20

    return BlurMatchResult(
        mi_original=mi_original,
        mi_blurred=mi_blurred,
        mi_change=mi_change,
        mi_change_percent=mi_change_percent,
        blur_sigma=sigma,
        stable=stable
    )


# ============================================================================
# C4: CO-ALIGNMENT CHECK
# ============================================================================

@dataclass
class CoAlignmentResult:
    """Result of the co-alignment test."""
    mi_map: NDArray[np.float64]  # MI for each shift
    shifts: List[Tuple[int, int]]  # (dy, dx) Shifts
    max_shift: Tuple[int, int]  # Shift with maximum MI
    mi_at_zero: float  # MI at (0, 0)
    mi_at_max: float  # Maximum MI
    centered: bool  # True if maximum at (0, 0)


def shift_image(
    image: NDArray[np.float64],
    shift: Tuple[int, int]
) -> NDArray[np.float64]:
    """
    Verschiebt ein Bild um (dy, dx) Pixel.

    Args:
        image: Eingabebild
        shift: (dy, dx) Verschiebung

    Returns:
        Verschobenes Bild (mit Nullen aufgefüllt)
    """
    dy, dx = shift
    result = np.zeros_like(image)

    # Quell- und Ziel-Bereiche berechnen
    src_y_start = max(0, -dy)
    src_y_end = min(image.shape[0], image.shape[0] - dy)
    src_x_start = max(0, -dx)
    src_x_end = min(image.shape[1], image.shape[1] - dx)

    dst_y_start = max(0, dy)
    dst_y_end = min(image.shape[0], image.shape[0] + dy)
    dst_x_start = max(0, dx)
    dst_x_end = min(image.shape[1], image.shape[1] + dx)

    result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        image[src_y_start:src_y_end, src_x_start:src_x_end]

    return result


def co_alignment_check(
    image_1: NDArray[np.float64],
    image_2: NDArray[np.float64],
    max_offset: int = 3,
    bins: int = 64
) -> CoAlignmentResult:
    """
    C4: Co-Alignment Check.

    Berechnet Residual-MI als Funktion von kleinen Pixel-Shifts.
    Testet ob die Bilder korrekt ausgerichtet sind.

    Expectation: Maximum at (0, 0) → correctly aligned.
    Maximum woanders → Co-Registration-Fehler treibt MI.

    Args:
        image_1: Erstes Bild (bleibt fix)
        image_2: Zweites Bild (wird verschoben)
        max_offset: Maximaler Shift in Pixeln (±max_offset)
        bins: Bins für MI

    Returns:
        CoAlignmentResult mit MI-Karte über Shifts
    """
    # Alle Shift-Kombinationen
    shifts = []
    for dy in range(-max_offset, max_offset + 1):
        for dx in range(-max_offset, max_offset + 1):
            shifts.append((dy, dx))

    n_shifts = 2 * max_offset + 1
    mi_map = np.zeros((n_shifts, n_shifts), dtype=np.float64)

    for dy in range(-max_offset, max_offset + 1):
        for dx in range(-max_offset, max_offset + 1):
            # Verschiebe Bild 2
            image_2_shifted = shift_image(image_2, (dy, dx))

            # Berechne Residual-MI
            res_1, res_2, _ = prepare_pair_for_residual_mi(
                image_1, image_2_shifted
            )
            mi = mutual_information(res_1, res_2, bins=bins)

            # Speichere in Karte
            map_y = dy + max_offset
            map_x = dx + max_offset
            mi_map[map_y, map_x] = mi

    # Finde Maximum
    max_idx = np.unravel_index(np.argmax(mi_map), mi_map.shape)
    max_dy = max_idx[0] - max_offset
    max_dx = max_idx[1] - max_offset

    # MI at (0, 0) and maximum
    mi_at_zero = mi_map[max_offset, max_offset]
    mi_at_max = mi_map[max_idx]

    # Centered if maximum at (0, 0)
    centered = (max_dy == 0 and max_dx == 0)

    return CoAlignmentResult(
        mi_map=mi_map,
        shifts=shifts,
        max_shift=(max_dy, max_dx),
        mi_at_zero=mi_at_zero,
        mi_at_max=mi_at_max,
        centered=centered
    )


# ============================================================================
# ALLE KONTROLLEN ZUSAMMEN
# ============================================================================

@dataclass
class AllControlsResult:
    """Results of all control tests."""
    c1_time_shift: TimeShiftResult
    c2_ring_shuffle: RingShuffleResult
    c3_blur_match: BlurMatchResult
    c4_co_alignment: CoAlignmentResult

    @property
    def all_passed(self) -> bool:
        """Alle Kontrollen bestanden?"""
        return (
            self.c1_time_shift.passed and
            self.c3_blur_match.stable and
            self.c4_co_alignment.centered
        )


def run_all_controls(
    image_1: NDArray[np.float64],
    image_2: NDArray[np.float64],
    seed: int = 42,
    bins: int = 64,
    verbose: bool = True
) -> AllControlsResult:
    """
    Führt alle vier Kontroll-Tests durch.

    Args:
        image_1: Erstes Bild
        image_2: Zweites Bild
        seed: Random seed
        bins: Bins für MI
        verbose: Ausgabe während der Berechnung

    Returns:
        AllControlsResult mit allen Ergebnissen
    """
    if verbose:
        print("\n  C1: Time-Shift Null...")
    c1 = time_shift_null(image_1, image_2, seed=seed, bins=bins)

    if verbose:
        print("  C2: Ring-wise Shuffle...")
    c2 = ring_wise_shuffle_test(image_1, image_2, seed=seed, bins=bins)

    if verbose:
        print("  C3: PSF/Blur Matching...")
    c3 = psf_blur_matching(image_1, image_2, sigma=2.0, bins=bins)

    if verbose:
        print("  C4: Co-alignment Check...")
    c4 = co_alignment_check(image_1, image_2, max_offset=3, bins=bins)

    return AllControlsResult(
        c1_time_shift=c1,
        c2_ring_shuffle=c2,
        c3_blur_match=c3,
        c4_co_alignment=c4
    )


def print_control_results(result: AllControlsResult) -> None:
    """
    Gibt formatierte Ergebnisse der Kontroll-Tests aus.

    Args:
        result: AllControlsResult
    """
    print("\n" + "="*72)
    print("  KONTROLL-TESTS FÜR RESIDUAL-MI")
    print("="*72)

    # C1: Time-Shift
    c1 = result.c1_time_shift
    status = "✓ BESTANDEN" if c1.passed else "✗ NICHT BESTANDEN"
    print(f"""
  C1: TIME-SHIFT NULL
  ─────────────────────────────────────────────────────────────────────
  Frage: Fällt MI wenn zeitliche Korrelation zerstört wird?

    MI (original):  {c1.mi_original:.4f} bits
    MI (shifted):   {c1.mi_shifted:.4f} bits
    Reduktion:      {c1.mi_reduction_percent:.1f}%

    Status: {status}
    (Bestanden wenn Reduktion > 50%)
""")

    # C2: Ring-Shuffle
    c2 = result.c2_ring_shuffle
    ring_vs_global = "stärker" if c2.ring_stronger else "schwächer"
    print(f"""  C2: RING-WISE SHUFFLE
  ─────────────────────────────────────────────────────────────────────
  Frage: Ist azimutale Struktur wichtiger als radiale Statistik?

    MI (original):      {c2.mi_original:.4f} bits
    MI (ring-shuffle):  {c2.mi_ring_shuffled:.4f} bits  (Reduktion: {c2.ring_reduction_percent:.1f}%)
    MI (global-shuffle): {c2.mi_global_shuffled:.4f} bits  (Reduktion: {c2.global_reduction_percent:.1f}%)

    Ring-Shuffle reduziert {ring_vs_global} als globaler Shuffle.
""")

    # C3: Blur Matching
    c3 = result.c3_blur_match
    status = "✓ STABIL" if c3.stable else "⚠ SENSITIV"
    direction = "erhöht" if c3.mi_change > 0 else "reduziert"
    print(f"""  C3: PSF/BLUR MATCHING (σ = {c3.blur_sigma} px)
  ─────────────────────────────────────────────────────────────────────
  Frage: Sind hochfrequente Details wichtig für MI?

    MI (original):  {c3.mi_original:.4f} bits
    MI (blurred):   {c3.mi_blurred:.4f} bits
    Änderung:       {c3.mi_change_percent:+.1f}% ({direction})

    Status: {status}
    (Stabil wenn Änderung < 20%)
""")

    # C4: Co-alignment
    c4 = result.c4_co_alignment
    status = "✓ ZENTRIERT" if c4.centered else f"⚠ OFFSET bei {c4.max_shift}"
    print(f"""  C4: CO-ALIGNMENT CHECK (±3 px)
  ─────────────────────────────────────────────────────────────────────
  Frage: Ist die räumliche Registrierung korrekt?

    MI bei (0, 0):      {c4.mi_at_zero:.4f} bits
    Maximale MI:        {c4.mi_at_max:.4f} bits  bei Shift {c4.max_shift}

    Status: {status}
    (Bestanden wenn Maximum bei (0, 0))

    MI-Karte über Shifts:
""")

    # ASCII-Darstellung der Co-Alignment Karte
    mi_map = c4.mi_map
    mi_min, mi_max = mi_map.min(), mi_map.max()
    chars = " ·:░▒▓█"

    print("         dx")
    print("       -3-2-1 0+1+2+3")
    for i, dy in enumerate(range(-3, 4)):
        row = "    " + (f"{dy:+d} " if dy != 0 else " 0 ") + "│"
        for j in range(mi_map.shape[1]):
            val = (mi_map[i, j] - mi_min) / (mi_max - mi_min) if mi_max > mi_min else 0
            char_idx = int(val * (len(chars) - 1))
            row += chars[char_idx]
        # Markiere Maximum
        if i == c4.max_shift[0] + 3:
            row += f"│ ← max"
        else:
            row += "│"
        print(row)
    print("    dy")

    # Gesamtergebnis
    print("\n" + "="*72)
    if result.all_passed:
        print("  ✓ ALLE KRITISCHEN KONTROLLEN BESTANDEN")
        print("    → Die Residual-MI ist wahrscheinlich echtes Signal")
    else:
        print("  ⚠ NICHT ALLE KONTROLLEN BESTANDEN")
        print("    → Ergebnisse mit Vorsicht interpretieren")
    print("="*72)
