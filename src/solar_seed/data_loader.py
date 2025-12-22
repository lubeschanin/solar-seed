"""
Daten-Loader für Solar Seed
===========================

Unterstützt:
- Synthetische Testdaten (für Validierung)
- SunPy Sample-Daten
- Echte AIA FITS-Daten (via SunPy)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
from pathlib import Path


# ============================================================================
# SYNTHETISCHE DATEN
# ============================================================================

def generate_pure_noise(
    shape: Tuple[int, int] = (512, 512),
    seed: int | None = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generiert VOLLSTÄNDIG unabhängige Daten.
    
    Für Validierung des Nullmodells.
    Erwartung: MI ≈ MI_null, Z ≈ 0
    
    Args:
        shape: Array-Dimensionen
        seed: Random Seed
        
    Returns:
        Tuple von zwei unabhängigen Arrays
    """
    rng = np.random.default_rng(seed)
    
    # Exponentialverteilung simuliert Photon-Counts
    data_1 = rng.exponential(1000, shape)
    data_2 = rng.exponential(1000, shape)
    
    return data_1, data_2


def generate_correlated_noise(
    shape: Tuple[int, int] = (512, 512),
    correlation: float = 0.5,
    seed: int | None = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generiert korrelierte Daten OHNE gemeinsame räumliche Struktur.
    
    Für Validierung der MI-Berechnung.
    Erwartung: MI >> MI_null, Z >> 3
    
    Args:
        shape: Array-Dimensionen
        correlation: Pearson-Korrelation zwischen den Arrays
        seed: Random Seed
        
    Returns:
        Tuple von zwei korrelierten Arrays
    """
    rng = np.random.default_rng(seed)
    n = shape[0] * shape[1]
    
    # Korrelierte Normalverteilungen via Cholesky
    mean = [0, 0]
    cov = [[1, correlation], [correlation, 1]]
    xy = rng.multivariate_normal(mean, cov, n)
    
    # Transformiere zu positiven Werten
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
    Generiert realistische synthetische Sonnendaten.
    
    Beide Kanäle teilen:
    - Sonnenscheiben-Geometrie (Limb Darkening)
    - Aktive Regionen (mit unterschiedlicher Intensität)
    
    extra_correlation fügt ZUSÄTZLICHE gemeinsame Fluktuationen hinzu.
    
    Args:
        shape: Array-Dimensionen
        extra_correlation: Zusätzliche Korrelation (0-1)
        n_active_regions: Anzahl simulierter aktiver Regionen
        seed: Random Seed
        
    Returns:
        Tuple von zwei "Wellenlängen-Kanälen"
    """
    rng = np.random.default_rng(seed)
    
    # Koordinaten-Grid
    y, x = np.ogrid[:shape[0], :shape[1]]
    center = (shape[0] // 2, shape[1] // 2)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r_max = min(center) * 0.9
    
    # Limb Darkening: I = I_0 * sqrt(1 - (r/R)^2)
    mu = np.sqrt(np.maximum(0, 1 - (r / r_max)**2))
    base = mu * 10000
    base[r > r_max] = 0  # Außerhalb der Sonne = 0
    
    # Aktive Regionen (gemeinsam, aber unterschiedliche Intensität)
    for _ in range(n_active_regions):
        rx = rng.integers(shape[0] // 4, 3 * shape[0] // 4)
        ry = rng.integers(shape[1] // 4, 3 * shape[1] // 4)
        rr = np.sqrt((x - ry)**2 + (y - rx)**2)
        region = np.exp(-rr**2 / 100) * rng.uniform(2000, 4000)
        base += region
    
    # Kanal 1: 193 Å simuliert
    noise_1 = rng.normal(0, 300, shape)
    data_1 = base + noise_1
    
    # Kanal 2: 211 Å simuliert (andere Temperatur-Response)
    base_2 = base * 0.8  # Skalierungsfaktor
    noise_2 = rng.normal(0, 300, shape)
    
    if extra_correlation > 0:
        # Gemeinsame Fluktuationen (simuliert z.B. gemeinsame Plasma-Dynamik)
        shared_fluct = rng.normal(0, 500, shape)
        data_1 = data_1 + extra_correlation * shared_fluct
        data_2 = base_2 + noise_2 + extra_correlation * shared_fluct
    else:
        data_2 = base_2 + noise_2
    
    # Keine negativen Werte (physikalisch unrealistisch)
    data_1 = np.maximum(0, data_1)
    data_2 = np.maximum(0, data_2)
    
    return data_1, data_2


# ============================================================================
# SUNPY DATEN
# ============================================================================

def load_sunpy_sample() -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
    """
    Lädt SunPy Sample-Daten (AIA 171 Å).
    
    Da nur ein Kanal verfügbar ist, wird der zweite simuliert.
    
    Returns:
        Tuple von zwei Arrays, oder (None, None) wenn nicht verfügbar
    """
    try:
        import sunpy.data.sample
        import sunpy.map
        
        print("  📦 Lade SunPy Sample-Daten (AIA 171 Å)...")
        aia_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)
        data_1 = aia_map.data.astype(np.float64)
        
        # Simuliere zweiten Kanal
        print("  🔧 Generiere simulierten zweiten Kanal...")
        rng = np.random.default_rng(42)
        data_2 = data_1 * 0.85 + rng.normal(0, data_1.std() * 0.2, data_1.shape)
        data_2 = np.maximum(0, data_2)
        
        print(f"  ✓ Daten geladen: {data_1.shape}")
        return data_1, data_2
        
    except ImportError:
        print("  ⚠️  SunPy nicht installiert. Installiere mit: pip install sunpy")
        return None, None
    except Exception as e:
        print(f"  ✗ Fehler beim Laden: {e}")
        return None, None


def load_aia_fits(
    wavelength_1: int = 193,
    wavelength_2: int = 211,
    start_time: str = "2024-01-15T12:00:00",
    end_time: str = "2024-01-15T12:10:00",
    data_dir: str = "data/fits"
) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
    """
    Lädt echte AIA FITS-Daten via SunPy.
    
    Args:
        wavelength_1: Erste Wellenlänge in Angström
        wavelength_2: Zweite Wellenlänge in Angström  
        start_time: Startzeit (ISO Format)
        end_time: Endzeit (ISO Format)
        data_dir: Verzeichnis für heruntergeladene Daten
        
    Returns:
        Tuple von zwei Arrays, oder (None, None) wenn nicht verfügbar
    """
    try:
        import sunpy.map
        from sunpy.net import Fido, attrs as a
        import astropy.units as u
        
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        
        data_arrays = []
        
        for wl in [wavelength_1, wavelength_2]:
            print(f"  🔍 Suche {wl} Å Daten...")
            
            result = Fido.search(
                a.Time(start_time, end_time),
                a.Instrument("aia"),
                a.Wavelength(wl * u.angstrom),
            )
            
            if len(result) == 0 or len(result[0]) == 0:
                print(f"  ⚠️  Keine Daten für {wl} Å gefunden")
                return None, None
            
            print(f"  📥 Lade {wl} Å...")
            files = Fido.fetch(result[0, 0], path=data_dir + "/{file}")
            
            if not files:
                return None, None
            
            aia_map = sunpy.map.Map(files[0])
            data_arrays.append(aia_map.data.astype(np.float64))
            print(f"  ✓ {wl} Å geladen: {aia_map.date}")
        
        return data_arrays[0], data_arrays[1]
        
    except ImportError:
        print("  ⚠️  SunPy nicht installiert")
        return None, None
    except Exception as e:
        print(f"  ✗ Fehler: {e}")
        return None, None


# ============================================================================
# FITS DIREKT
# ============================================================================

def load_fits_file(filepath: str) -> Optional[NDArray[np.float64]]:
    """
    Lädt eine einzelne FITS-Datei.
    
    Args:
        filepath: Pfad zur FITS-Datei
        
    Returns:
        Numpy Array oder None
    """
    try:
        from astropy.io import fits
        
        with fits.open(filepath) as hdul:
            # Versuche Primary HDU
            data = hdul[0].data
            
            # Falls leer, versuche erste Extension
            if data is None and len(hdul) > 1:
                data = hdul[1].data
            
            if data is not None:
                return data.astype(np.float64)
        
        return None
        
    except ImportError:
        print("  ⚠️  Astropy nicht installiert")
        return None
    except Exception as e:
        print(f"  ✗ Fehler beim Laden von {filepath}: {e}")
        return None
