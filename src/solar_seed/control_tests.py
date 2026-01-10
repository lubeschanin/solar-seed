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

    # Bin boundaries
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
    Shuffles pixels only within the same rings.

    Preserves radial statistics, destroys azimuthal correlations.

    Args:
        image: Input image
        ring_indices: Ring index per pixel
        seed: Random seed

    Returns:
        Image with ring-wise shuffled pixels
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

    Compares ring-shuffle with global shuffle.
    Ring-shuffle preserves radial statistics but destroys azimuthal structure.

    Expectation: Ring-shuffle reduces MI more than global shuffle,
    because it specifically destroys structural correlation.

    Args:
        image_1: First image
        image_2: Second image
        n_rings: Number of rings
        seed: Random seed
        bins: Bins for MI

    Returns:
        RingShuffleResult with comparison
    """
    # Find center
    center = find_disk_center(image_1)

    # Ring indices
    ring_indices = create_radial_bins(image_1.shape, center, n_rings)

    # Original residual MI
    res_1, res_2, _ = prepare_pair_for_residual_mi(image_1, image_2)
    mi_original = mutual_information(res_1, res_2, bins=bins)

    # Ring-shuffle on channel B
    image_2_ring = ring_shuffle(image_2, ring_indices, seed=seed)
    res_1_r, res_2_ring, _ = prepare_pair_for_residual_mi(image_1, image_2_ring)
    mi_ring_shuffled = mutual_information(res_1_r, res_2_ring, bins=bins)

    # Global shuffle on channel B
    rng = np.random.default_rng(seed + 1000)
    image_2_global = image_2.ravel().copy()
    rng.shuffle(image_2_global)
    image_2_global = image_2_global.reshape(image_2.shape)
    res_1_g, res_2_global, _ = prepare_pair_for_residual_mi(image_1, image_2_global)
    mi_global_shuffled = mutual_information(res_1_g, res_2_global, bins=bins)

    # Reductions
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
    Creates ring and sector indices for combined shuffling.

    Args:
        shape: Image size
        center: (y, x) center
        n_rings: Number of concentric rings
        n_sectors: Number of azimuthal sectors

    Returns:
        (ring_indices, sector_indices) - each array with index per pixel
    """
    y, x = np.ogrid[:shape[0], :shape[1]]

    # Radial indices
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    max_r = np.max(r)
    ring_indices = np.clip(
        (r / max_r * n_rings).astype(np.int64),
        0, n_rings - 1
    )

    # Azimuthal indices (angle from 0 to 2pi)
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
    Shuffles pixels within ring+sector combinations.

    Preserves both radial and coarse azimuthal statistics,
    only destroys local (fine) correlations.

    Args:
        image: Input image
        ring_indices: Ring index per pixel
        sector_indices: Sector index per pixel
        seed: Random seed

    Returns:
        Image with sector-ring-wise shuffled pixels
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
    Extended C2: Sector-Ring Shuffle Test.

    Compares three shuffle levels:
    1. Ring-only: Preserves radial statistics
    2. Sector-Ring: Preserves radial + coarse azimuthal statistics
    3. Global: Destroys everything

    This allows clean separation of:
    - Radial contribution to MI
    - Azimuthal contribution
    - True local structure

    Args:
        image_1: First image
        image_2: Second image
        n_rings: Number of rings
        n_sectors: Number of sectors (e.g. 16 = 22.5 deg per sector)
        seed: Random seed
        bins: Bins for MI

    Returns:
        SectorRingShuffleResult with detailed breakdown
    """
    # Find center
    center = find_disk_center(image_1)

    # Create indices
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

    # Calculate reductions
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
    Applies Gaussian blur to an image.

    Args:
        image: Input image
        sigma: Standard deviation of the Gaussian kernel

    Returns:
        Blurred image
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

    Brings both channels to the same effective resolution through blur.
    Tests whether high-frequency details drive the MI.

    Expectation: If MI remains stable -> large-scale correlations dominant.
    If MI drops significantly -> high-frequency details were important.

    Args:
        image_1: First image
        image_2: Second image
        sigma: Blur strength (in pixels)
        bins: Bins for MI

    Returns:
        BlurMatchResult with comparison before/after blur
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
    Shifts an image by (dy, dx) pixels.

    Args:
        image: Input image
        shift: (dy, dx) displacement

    Returns:
        Shifted image (padded with zeros)
    """
    dy, dx = shift
    result = np.zeros_like(image)

    # Calculate source and destination regions
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

    Calculates Residual-MI as a function of small pixel shifts.
    Tests whether the images are correctly aligned.

    Expectation: Maximum at (0, 0) -> correctly aligned.
    Maximum elsewhere -> Co-registration error drives MI.

    Args:
        image_1: First image (remains fixed)
        image_2: Second image (is shifted)
        max_offset: Maximum shift in pixels (+/-max_offset)
        bins: Bins for MI

    Returns:
        CoAlignmentResult with MI map over shifts
    """
    # All shift combinations
    shifts = []
    for dy in range(-max_offset, max_offset + 1):
        for dx in range(-max_offset, max_offset + 1):
            shifts.append((dy, dx))

    n_shifts = 2 * max_offset + 1
    mi_map = np.zeros((n_shifts, n_shifts), dtype=np.float64)

    for dy in range(-max_offset, max_offset + 1):
        for dx in range(-max_offset, max_offset + 1):
            # Shift image 2
            image_2_shifted = shift_image(image_2, (dy, dx))

            # Calculate Residual-MI
            res_1, res_2, _ = prepare_pair_for_residual_mi(
                image_1, image_2_shifted
            )
            mi = mutual_information(res_1, res_2, bins=bins)

            # Store in map
            map_y = dy + max_offset
            map_x = dx + max_offset
            mi_map[map_y, map_x] = mi

    # Find maximum
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
# ALL CONTROLS TOGETHER
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
        """All controls passed?"""
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
    Runs all four control tests.

    Args:
        image_1: First image
        image_2: Second image
        seed: Random seed
        bins: Bins for MI
        verbose: Output during computation

    Returns:
        AllControlsResult with all results
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
    Prints formatted results of the control tests.

    Args:
        result: AllControlsResult
    """
    print("\n" + "="*72)
    print("  CONTROL TESTS FOR RESIDUAL-MI")
    print("="*72)

    # C1: Time-Shift
    c1 = result.c1_time_shift
    status = "PASSED" if c1.passed else "NOT PASSED"
    print(f"""
  C1: TIME-SHIFT NULL
  ─────────────────────────────────────────────────────────────────────
  Question: Does MI drop when temporal correlation is destroyed?

    MI (original):  {c1.mi_original:.4f} bits
    MI (shifted):   {c1.mi_shifted:.4f} bits
    Reduction:      {c1.mi_reduction_percent:.1f}%

    Status: {status}
    (Passed if reduction > 50%)
""")

    # C2: Ring-Shuffle
    c2 = result.c2_ring_shuffle
    ring_vs_global = "more" if c2.ring_stronger else "less"
    print(f"""  C2: RING-WISE SHUFFLE
  ─────────────────────────────────────────────────────────────────────
  Question: Is azimuthal structure more important than radial statistics?

    MI (original):      {c2.mi_original:.4f} bits
    MI (ring-shuffle):  {c2.mi_ring_shuffled:.4f} bits  (Reduction: {c2.ring_reduction_percent:.1f}%)
    MI (global-shuffle): {c2.mi_global_shuffled:.4f} bits  (Reduction: {c2.global_reduction_percent:.1f}%)

    Ring-shuffle reduces {ring_vs_global} than global shuffle.
""")

    # C3: Blur Matching
    c3 = result.c3_blur_match
    status = "STABLE" if c3.stable else "SENSITIVE"
    direction = "increased" if c3.mi_change > 0 else "reduced"
    print(f"""  C3: PSF/BLUR MATCHING (sigma = {c3.blur_sigma} px)
  ─────────────────────────────────────────────────────────────────────
  Question: Are high-frequency details important for MI?

    MI (original):  {c3.mi_original:.4f} bits
    MI (blurred):   {c3.mi_blurred:.4f} bits
    Change:         {c3.mi_change_percent:+.1f}% ({direction})

    Status: {status}
    (Stable if change < 20%)
""")

    # C4: Co-alignment
    c4 = result.c4_co_alignment
    status = "CENTERED" if c4.centered else f"OFFSET at {c4.max_shift}"
    print(f"""  C4: CO-ALIGNMENT CHECK (+/-3 px)
  ─────────────────────────────────────────────────────────────────────
  Question: Is the spatial registration correct?

    MI at (0, 0):       {c4.mi_at_zero:.4f} bits
    Maximum MI:         {c4.mi_at_max:.4f} bits  at shift {c4.max_shift}

    Status: {status}
    (Passed if maximum at (0, 0))

    MI map over shifts:
""")

    # ASCII representation of the Co-Alignment map
    mi_map = c4.mi_map
    mi_min, mi_max = mi_map.min(), mi_map.max()
    chars = " .:+=*#"

    print("         dx")
    print("       -3-2-1 0+1+2+3")
    for i, dy in enumerate(range(-3, 4)):
        row = "    " + (f"{dy:+d} " if dy != 0 else " 0 ") + "|"
        for j in range(mi_map.shape[1]):
            val = (mi_map[i, j] - mi_min) / (mi_max - mi_min) if mi_max > mi_min else 0
            char_idx = int(val * (len(chars) - 1))
            row += chars[char_idx]
        # Mark maximum
        if i == c4.max_shift[0] + 3:
            row += f"| <- max"
        else:
            row += "|"
        print(row)
    print("    dy")

    # Overall result
    print("\n" + "="*72)
    if result.all_passed:
        print("  + ALL CRITICAL CONTROLS PASSED")
        print("    -> The Residual-MI is probably real signal")
    else:
        print("  ! NOT ALL CONTROLS PASSED")
        print("    -> Interpret results with caution")
    print("="*72)
