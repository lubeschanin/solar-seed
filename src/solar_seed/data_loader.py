"""
Data Loader for Solar Seed
===========================

Supports:
- Synthetic test data (for validation)
- SunPy sample data
- Real AIA FITS data (via SunPy)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
from pathlib import Path


# ============================================================================
# SYNTHETIC DATA
# ============================================================================

def generate_pure_noise(
    shape: Tuple[int, int] = (512, 512),
    seed: int | None = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generates COMPLETELY independent data.

    For null model validation.
    Expectation: MI approx MI_null, Z approx 0

    Args:
        shape: Array dimensions
        seed: Random seed

    Returns:
        Tuple of two independent arrays
    """
    rng = np.random.default_rng(seed)

    # Exponential distribution simulates photon counts
    data_1 = rng.exponential(1000, shape)
    data_2 = rng.exponential(1000, shape)

    return data_1, data_2


def generate_correlated_noise(
    shape: Tuple[int, int] = (512, 512),
    correlation: float = 0.5,
    seed: int | None = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generates correlated data WITHOUT common spatial structure.

    For MI calculation validation.
    Expectation: MI >> MI_null, Z >> 3

    Args:
        shape: Array dimensions
        correlation: Pearson correlation between arrays
        seed: Random seed

    Returns:
        Tuple of two correlated arrays
    """
    rng = np.random.default_rng(seed)
    n = shape[0] * shape[1]

    # Correlated normal distributions via Cholesky
    mean = [0, 0]
    cov = [[1, correlation], [correlation, 1]]
    xy = rng.multivariate_normal(mean, cov, n)

    # Transform to positive values
    data_1 = np.exp(xy[:, 0]).reshape(shape) * 1000
    data_2 = np.exp(xy[:, 1]).reshape(shape) * 1000

    return data_1, data_2


def generate_synthetic_sun(
    shape: Tuple[int, int] = (512, 512),
    extra_correlation: float = 0.0,
    n_active_regions: int = 5,
    seed: int | None = 42
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generates realistic synthetic sun data.

    Both channels share:
    - Solar disk geometry (limb darkening)
    - Active regions (with different intensity)

    extra_correlation adds ADDITIONAL common fluctuations.

    Args:
        shape: Array dimensions
        extra_correlation: Additional correlation (0-1)
        n_active_regions: Number of simulated active regions
        seed: Random seed

    Returns:
        Tuple of two "wavelength channels"
    """
    rng = np.random.default_rng(seed)

    # Coordinate grid
    y, x = np.ogrid[:shape[0], :shape[1]]
    center = (shape[0] // 2, shape[1] // 2)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r_max = min(center) * 0.9
    
    # Limb Darkening: I = I_0 * sqrt(1 - (r/R)^2)
    mu = np.sqrt(np.maximum(0, 1 - (r / r_max)**2))
    base = mu * 10000
    base[r > r_max] = 0  # Outside the sun = 0

    # Active regions (common, but different intensity)
    for _ in range(n_active_regions):
        rx = rng.integers(shape[0] // 4, 3 * shape[0] // 4)
        ry = rng.integers(shape[1] // 4, 3 * shape[1] // 4)
        rr = np.sqrt((x - ry)**2 + (y - rx)**2)
        region = np.exp(-rr**2 / 100) * rng.uniform(2000, 4000)
        base += region
    
    # Channel 1: simulated 193 A
    noise_1 = rng.normal(0, 300, shape)
    data_1 = base + noise_1

    # Channel 2: simulated 211 A (different temperature response)
    base_2 = base * 0.8  # Scaling factor
    noise_2 = rng.normal(0, 300, shape)

    if extra_correlation > 0:
        # Common fluctuations (simulates e.g. common plasma dynamics)
        shared_fluct = rng.normal(0, 500, shape)
        data_1 = data_1 + extra_correlation * shared_fluct
        data_2 = base_2 + noise_2 + extra_correlation * shared_fluct
    else:
        data_2 = base_2 + noise_2

    # No negative values (physically unrealistic)
    data_1 = np.maximum(0, data_1)
    data_2 = np.maximum(0, data_2)

    return data_1, data_2


# ============================================================================
# SUNPY DATA
# ============================================================================

def load_sunpy_sample() -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
    """
    Loads SunPy sample data (AIA 171 A).

    Since only one channel is available, the second is simulated.

    Returns:
        Tuple of two arrays, or (None, None) if not available
    """
    try:
        import sunpy.data.sample
        import sunpy.map

        print("  📦 Loading SunPy sample data (AIA 171 A)...")
        aia_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)
        data_1 = aia_map.data.astype(np.float64)

        # Simulate second channel
        print("  🔧 Generating simulated second channel...")
        rng = np.random.default_rng(42)
        data_2 = data_1 * 0.85 + rng.normal(0, data_1.std() * 0.2, data_1.shape)
        data_2 = np.maximum(0, data_2)

        print(f"  ✓ Data loaded: {data_1.shape}")
        return data_1, data_2

    except ImportError:
        print("  ⚠️  SunPy not installed. Install with: pip install sunpy")
        return None, None
    except Exception as e:
        print(f"  ✗ Error loading: {e}")
        return None, None


def load_aia_fits(
    wavelength_1: int = 193,
    wavelength_2: int = 211,
    start_time: str = "2024-01-15T12:00:00",
    end_time: str = "2024-01-15T12:10:00",
    data_dir: str = "data/fits"
) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
    """
    Loads real AIA FITS data via SunPy.

    Args:
        wavelength_1: First wavelength in Angstrom
        wavelength_2: Second wavelength in Angstrom
        start_time: Start time (ISO format)
        end_time: End time (ISO format)
        data_dir: Directory for downloaded data

    Returns:
        Tuple of two arrays, or (None, None) if not available
    """
    try:
        import sunpy.map
        from sunpy.net import Fido, attrs as a
        import astropy.units as u

        Path(data_dir).mkdir(parents=True, exist_ok=True)

        data_arrays = []

        for wl in [wavelength_1, wavelength_2]:
            print(f"  🔍 Searching {wl} A data...")

            result = Fido.search(
                a.Time(start_time, end_time),
                a.Instrument("aia"),
                a.Wavelength(wl * u.angstrom),
            )

            if len(result) == 0 or len(result[0]) == 0:
                print(f"  ⚠️  No data found for {wl} A")
                return None, None

            print(f"  📥 Loading {wl} A...")
            files = Fido.fetch(result[0, 0], path=data_dir + "/{file}")

            if not files:
                return None, None

            aia_map = sunpy.map.Map(files[0])
            data_arrays.append(aia_map.data.astype(np.float64))
            print(f"  ✓ {wl} A loaded: {aia_map.date}")

        return data_arrays[0], data_arrays[1]

    except ImportError:
        print("  ⚠️  SunPy not installed")
        return None, None
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None, None


# ============================================================================
# FITS DIRECT
# ============================================================================

def load_fits_file(filepath: str) -> Optional[NDArray[np.float64]]:
    """
    Loads a single FITS file.

    Args:
        filepath: Path to FITS file

    Returns:
        Numpy array or None
    """
    try:
        from astropy.io import fits

        with fits.open(filepath) as hdul:
            # Try Primary HDU
            data = hdul[0].data

            # If empty, try first extension
            if data is None and len(hdul) > 1:
                data = hdul[1].data

            if data is not None:
                return data.astype(np.float64)

        return None

    except ImportError:
        print("  ⚠️  Astropy not installed")
        return None
    except Exception as e:
        print(f"  ✗ Error loading {filepath}: {e}")
        return None
