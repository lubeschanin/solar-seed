#!/usr/bin/env python3
"""
Multi-Channel Analyse für Solar Seed
=====================================

Berechnet die Kopplungs-Matrix zwischen allen AIA EUV-Kanälen.

7 Kanäle × 6 / 2 = 21 einzigartige Paare

Kanäle (nach Temperatur sortiert):
    304 Å  →  0.05 MK  →  Chromosphäre
    171 Å  →  0.6 MK   →  Ruhige Korona
    193 Å  →  1.2 MK   →  Korona
    211 Å  →  2.0 MK   →  Aktive Regionen
    335 Å  →  2.5 MK   →  Aktive Regionen (heißer)
    94 Å   →  6.3 MK   →  Flares
    131 Å  →  10 MK    →  Flares (sehr heiß)
"""

# Fix für macOS: Fork-Crash vermeiden bei async-Bibliotheken (SunPy/aiohttp)
# Muss VOR allen anderen Imports stehen!
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=False)
except RuntimeError:
    pass  # Bereits gesetzt

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta

from solar_seed.mutual_info import mutual_information
from solar_seed.radial_profile import prepare_pair_for_residual_mi
from solar_seed.control_tests import sector_ring_shuffle_test


# ============================================================================
# DATA SOURCE METADATA
# ============================================================================

AIA_DATA_SOURCE = {
    "instrument": "SDO/AIA",
    "instrument_full": "Solar Dynamics Observatory / Atmospheric Imaging Assembly",
    "operator": "NASA / Stanford University",
    "data_provider": "JSOC (Joint Science Operations Center)",
    "data_url": "http://jsoc.stanford.edu",
    "launch_date": "2010-02-11",
    "orbit": "Geosynchronous (~36,000 km)",
    "native_resolution": "4096x4096 pixels",
    "native_cadence": "12 seconds",
    "wavelengths_angstrom": [94, 131, 171, 193, 211, 304, 335],
    "spectral_range": "EUV (Extreme Ultraviolet)",
    "reference": "Lemen et al. 2012, Solar Physics, 275, 17-40",
    "doi": "10.1007/s11207-011-9776-8"
}

# AIA Quality Flags (32-bit bitmask in FITS QUALITY keyword)
# Reference: SDO AIA Guide, SDOD0060
AIA_QUALITY_FLAGS = {
    0x00000001: ("ACS_MODE", "ACS mode not SCIENCE"),
    0x00000002: ("ACS_ECLP", "ACS eclipse flag set (Earth/Moon transit)"),
    0x00000004: ("ACS_SUNP", "ACS sun presence flag not set"),
    0x00000008: ("ACS_SAFE", "ACS safehold flag set"),
    0x00000010: ("IMG_TYPE", "Image type not LIGHT"),
    0x00000020: ("HWLTNOMINAL", "HW long term not nominal"),
    0x00000040: ("AIESSION", "AIA ISS loop open"),
    0x00040000: ("AIFCPS", "AIA focus calibration in progress"),
    0x00080000: ("AIHIS", "AIA high-speed instrument sequencer"),
    0x80000000: ("MISSING", "Image missing or corrupt"),
}

# Critical flags that invalidate the data completely
AIA_CRITICAL_FLAGS = {
    0x80000000,  # MISSING
    0x00000002,  # ACS_ECLP (eclipse)
    0x00000008,  # ACS_SAFE (safehold)
}


# ============================================================================
# CHANNEL DEFINITIONS
# ============================================================================

@dataclass
class AIAChannel:
    """Definition eines AIA-Kanals."""
    wavelength: int  # Angström
    temperature: float  # MK (Megakelvin)
    description: str
    color: str  # Für Visualisierung


# Kanäle nach Temperatur sortiert
AIA_CHANNELS = [
    AIAChannel(304, 0.05, "Chromosphäre", "red"),
    AIAChannel(171, 0.6, "Ruhige Korona", "yellow"),
    AIAChannel(193, 1.2, "Korona", "orange"),
    AIAChannel(211, 2.0, "Aktive Regionen", "purple"),
    AIAChannel(335, 2.5, "Aktive Regionen (heiß)", "blue"),
    AIAChannel(94, 6.3, "Flares", "green"),
    AIAChannel(131, 10.0, "Flares (sehr heiß)", "cyan"),
]

WAVELENGTHS = [ch.wavelength for ch in AIA_CHANNELS]
WAVELENGTH_TO_TEMP = {ch.wavelength: ch.temperature for ch in AIA_CHANNELS}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PairResult:
    """Ergebnis für ein Wellenlängen-Paar."""
    wavelength_1: int
    wavelength_2: int
    mi_original: float
    mi_residual: float
    mi_ratio: float
    delta_mi_ring: float
    delta_mi_sector: float
    z_score: float
    temperature_diff: float  # Temperatur-Differenz in MK


@dataclass
class CouplingMatrix:
    """Kopplungs-Matrix zwischen allen Kanälen."""
    wavelengths: List[int]
    matrix: NDArray[np.float64]  # 7x7 symmetrische Matrix
    metric: str  # "delta_mi_sector", "mi_ratio", etc.

    def get_value(self, wl1: int, wl2: int) -> float:
        """Gibt Kopplungswert für ein Paar zurück."""
        i = self.wavelengths.index(wl1)
        j = self.wavelengths.index(wl2)
        return self.matrix[i, j]

    def to_ascii(self, precision: int = 3) -> str:
        """ASCII-Darstellung der Matrix."""
        n = len(self.wavelengths)

        # Header
        header = "      " + "  ".join(f"{wl:>6}" for wl in self.wavelengths)
        lines = [header]
        lines.append("      " + "-" * (7 * n + n - 1))

        # Zeilen
        for i, wl in enumerate(self.wavelengths):
            row_vals = []
            for j in range(n):
                if i == j:
                    row_vals.append("   -  ")
                else:
                    val = self.matrix[i, j]
                    row_vals.append(f"{val:>6.{precision}f}")
            lines.append(f"{wl:>5} |" + "  ".join(row_vals))

        return "\n".join(lines)


@dataclass
class MultiChannelResult:
    """Gesamtergebnis der Multi-Channel-Analyse."""
    timestamp: str
    n_timepoints: int
    hours: float

    # Alle Paar-Ergebnisse
    pair_results: List[PairResult]

    # Verschiedene Kopplungs-Matrizen
    coupling_delta_sector: CouplingMatrix
    coupling_mi_ratio: CouplingMatrix
    coupling_delta_ring: CouplingMatrix

    # Statistiken
    mean_values: Dict[str, float] = field(default_factory=dict)
    std_values: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_multichannel_sun(
    shape: Tuple[int, int] = (256, 256),
    n_active_regions: int = 5,
    seed: int = 42
) -> Dict[int, NDArray[np.float64]]:
    """
    Generiert synthetische Sonnendaten für alle 7 AIA-Kanäle.

    Die Kopplung zwischen Kanälen basiert auf:
    1. Gemeinsame Geometrie (Limb Darkening)
    2. Gemeinsame aktive Regionen (mit temperaturabhängiger Response)
    3. Physikalisch motivierte Kopplungen zwischen benachbarten Temperaturen

    Args:
        shape: Bildgröße
        n_active_regions: Anzahl aktiver Regionen
        seed: Random Seed

    Returns:
        Dict[wavelength] -> image array
    """
    rng = np.random.default_rng(seed)

    # Koordinaten-Grid
    y, x = np.ogrid[:shape[0], :shape[1]]
    center = (shape[0] // 2, shape[1] // 2)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r_max = min(center) * 0.9

    # Basis: Limb Darkening (gemeinsam für alle Kanäle)
    mu = np.sqrt(np.maximum(0, 1 - (r / r_max)**2))
    disk_mask = r <= r_max

    # Aktive Regionen (Position und Größe sind gemeinsam, Intensität variiert)
    active_regions = []
    for _ in range(n_active_regions):
        rx = rng.integers(shape[0] // 4, 3 * shape[0] // 4)
        ry = rng.integers(shape[1] // 4, 3 * shape[1] // 4)
        size = rng.uniform(80, 150)
        intensity = rng.uniform(0.5, 1.5)
        active_regions.append((rx, ry, size, intensity))

    # Gemeinsame Plasma-Fluktuationen (physikalische Kopplung)
    # Diese schaffen Korrelation zwischen benachbarten Temperatur-Schichten
    plasma_base = rng.normal(0, 1, shape)
    plasma_smooth = np.zeros_like(plasma_base)
    # Einfache Glättung
    for di in range(-2, 3):
        for dj in range(-2, 3):
            plasma_smooth += np.roll(np.roll(plasma_base, di, 0), dj, 1) / 25

    channels = {}

    for channel in AIA_CHANNELS:
        wl = channel.wavelength
        temp = channel.temperature

        # Basis-Intensität skaliert mit Temperatur (hotter = different response)
        base_intensity = 10000 * (1.0 + 0.1 * np.log10(temp + 0.1))

        # Limb Darkening ist temperaturabhängig
        # Heißere Kanäle zeigen weniger Limb Darkening
        limb_factor = 0.5 + 0.5 / (1 + temp / 5)
        base = mu ** limb_factor * base_intensity

        # Aktive Regionen Response
        # 304 Å (Chromosphäre): schwache Response auf aktive Regionen
        # 94/131 Å (Flares): starke Response nur bei Flares
        for rx, ry, size, intensity in active_regions:
            rr = np.sqrt((x - ry)**2 + (y - rx)**2)

            # Temperatur-Response-Funktion
            if temp < 0.1:  # 304 Å - Chromosphäre
                response = 0.3 * intensity
            elif temp < 1.0:  # 171 Å - kühle Korona
                response = 0.8 * intensity
            elif temp < 3.0:  # 193, 211, 335 Å - Korona/aktive Regionen
                response = 1.2 * intensity
            else:  # 94, 131 Å - Flare-Temperaturen
                # Nur bei "heißen" Regionen sichtbar
                response = 0.5 * intensity if intensity > 1.0 else 0.1

            region = np.exp(-rr**2 / size) * response * 3000
            base += region

        # Temperatur-spezifisches Rauschen
        noise_level = 200 + 50 * np.log10(temp + 0.1)
        noise = rng.normal(0, noise_level, shape)

        # Plasma-Kopplung zwischen benachbarten Temperaturen
        # Stärker für ähnliche Temperaturen
        plasma_contribution = plasma_smooth * 500

        # Finale Zusammensetzung
        image = base + noise + plasma_contribution
        image[~disk_mask] = 0
        image = np.maximum(0, image)

        channels[wl] = image

    return channels


def generate_multichannel_timeseries(
    n_points: int,
    shape: Tuple[int, int] = (256, 256),
    seed: int = 42,
    cadence_minutes: int = 12
) -> List[Tuple[Dict[int, NDArray], str]]:
    """
    Generiert Zeitreihe für alle Kanäle.

    Args:
        n_points: Anzahl Zeitpunkte
        shape: Bildgröße
        seed: Basis-Seed
        cadence_minutes: Zeitabstand zwischen Bildern

    Returns:
        Liste von (channels_dict, timestamp) Tupeln
    """
    results = []
    base_time = datetime.now()

    for i in range(n_points):
        timestamp = (base_time + timedelta(minutes=cadence_minutes * i)).isoformat()

        # Variiere aktive Regionen über Zeit
        n_regions = 3 + (i % 5)

        channels = generate_multichannel_sun(
            shape=shape,
            n_active_regions=n_regions,
            seed=seed + i
        )

        results.append((channels, timestamp))

    return results


# ============================================================================
# REAL AIA DATA LOADING
# ============================================================================

def load_aia_multichannel(
    time_str: str,
    wavelengths: List[int] = None,
    data_dir: str = "data/aia",
    cleanup: bool = True,
    max_retries: int = 4  # Try JSOC + 3 mirrors (ROB, SDAC, CfA)
) -> Tuple[Optional[Dict[int, NDArray]], dict]:
    """
    Lädt echte AIA-Daten für alle Kanäle zu einem Zeitpunkt.

    Args:
        time_str: Zeitpunkt (ISO format)
        wavelengths: Liste der Wellenlängen (default: alle 7)
        data_dir: Verzeichnis für Downloads
        cleanup: FITS-Dateien nach Laden löschen (default: True)
        max_retries: Maximale Anzahl Versuche bei Download-Fehlern

    Returns:
        (channels_dict, metadata) oder (None, {}) bei Fehler
    """
    if wavelengths is None:
        wavelengths = WAVELENGTHS

    try:
        from sunpy.net import Fido, attrs as a
        import sunpy.map
        import astropy.units as u
        import gc
        import time
        import warnings

        # Parse time
        t = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        t_start = t - timedelta(minutes=2)
        t_end = t + timedelta(minutes=2)

        Path(data_dir).mkdir(parents=True, exist_ok=True)

        channels = {}
        metadata = {"timestamp": time_str, "wavelengths": wavelengths, "files": {}}
        files_to_delete = []

        for wl in wavelengths:
            result = Fido.search(
                a.Time(t_start.isoformat(), t_end.isoformat()),
                a.Instrument("aia"),
                a.Wavelength(wl * u.angstrom),
            )

            if len(result) == 0 or len(result[0]) == 0:
                print(f"    ⚠️  Keine Daten für {wl} Å gefunden")
                # Cleanup bei Fehler
                for f in files_to_delete:
                    try:
                        Path(f).unlink()
                    except Exception:
                        pass
                return None, {}

            # Download mit Retry-Logik und Mirror-Fallback
            aia_map = None
            # Sites to try: default (JSOC), then mirrors
            sites_to_try = [None, 'rob', 'sdac', 'cfa']  # ROB=Belgium, SDAC=NASA, CfA=Harvard

            for attempt in range(max_retries):
                site = sites_to_try[min(attempt, len(sites_to_try) - 1)]
                try:
                    # Lade erstes Ergebnis (mit optionalem Mirror)
                    if site:
                        files = Fido.fetch(result[0, 0], path=data_dir + "/{file}", site=site)
                    else:
                        files = Fido.fetch(result[0, 0], path=data_dir + "/{file}")
                    if not files:
                        continue

                    file_path = files[0]

                    # Check file size (AIA FITS ~7.5MB)
                    file_size = Path(file_path).stat().st_size
                    if file_size < 5_000_000:  # At least 5MB expected
                        print(f"    ⚠️  File too small ({file_size/1e6:.1f}MB), Retry...")
                        Path(file_path).unlink()
                        time.sleep(2)
                        continue

                    # Lade Map mit Unterdrückung von Truncation-Warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings('error')  # Warnings als Exceptions
                        try:
                            aia_map = sunpy.map.Map(file_path)
                        except Warning as w:
                            if "truncated" in str(w).lower():
                                print(f"    ⚠️  Truncated file, Retry...")
                                Path(file_path).unlink()
                                time.sleep(2)
                                continue
                            raise

                    files_to_delete.append(file_path)
                    break  # Erfolg!

                except Exception as e:
                    if attempt < max_retries - 1:
                        next_site = sites_to_try[min(attempt + 1, len(sites_to_try) - 1)]
                        mirror_info = f" (trying {next_site} mirror)" if next_site else ""
                        print(f"    ⚠️  Retry {attempt+1}/{max_retries}{mirror_info}: {str(e)[:60]}")
                        # Lösche fehlerhafte Datei falls vorhanden
                        try:
                            if 'file_path' in locals():
                                Path(file_path).unlink()
                        except Exception:
                            pass
                        time.sleep(2)
                    else:
                        raise

            if aia_map is None:
                print(f"    ✗ Download failed for {wl} Å after {max_retries} attempts")
                for f in files_to_delete:
                    try:
                        Path(f).unlink()
                    except Exception:
                        pass
                return None, {}

            # Quality flag check (AIA QUALITY keyword)
            quality = aia_map.meta.get('QUALITY', 0)
            if quality != 0:
                # Critical flags that should skip the image
                CRITICAL_FLAGS = {
                    0x80000000: "MISSING",      # Bit 31: Image missing
                    0x00000002: "ACS_ECLP",     # Bit 1: Eclipse
                    0x00000008: "ACS_SAFE",     # Bit 3: Safehold
                }

                critical = False
                for flag, name in CRITICAL_FLAGS.items():
                    if quality & flag:
                        print(f"    ✗ Critical quality flag for {wl} Å: {name} (0x{quality:08X})")
                        critical = True
                        break

                if critical:
                    # Skip this entire timepoint
                    for f in files_to_delete:
                        try:
                            Path(f).unlink()
                        except Exception:
                            pass
                    return None, {}

                # Non-critical warning
                print(f"    ⚠️  Quality flag for {wl} Å: 0x{quality:08X}")

            # Store quality in metadata
            if "quality" not in metadata:
                metadata["quality"] = {}
            metadata["quality"][wl] = quality

            channels[wl] = aia_map.data.astype(np.float64)
            metadata["files"][wl] = str(file_path)

            # Explizit Map freigeben
            del aia_map

        # Cleanup: FITS-Dateien löschen um Speicherplatz zu sparen
        if cleanup:
            for f in files_to_delete:
                try:
                    Path(f).unlink()
                except Exception:
                    pass

        # Garbage Collection erzwingen
        gc.collect()

        return channels, metadata

    except ImportError:
        print("  ⚠️  SunPy nicht installiert")
        return None, {}
    except Exception as e:
        print(f"  ✗ Fehler beim Laden: {e}")
        return None, {}


def load_aia_multichannel_timeseries(
    start_time: str,
    n_points: int,
    cadence_minutes: int = 12,
    wavelengths: List[int] = None,
    data_dir: str = "data/aia",
    verbose: bool = True,
    cleanup: bool = True
) -> List[Tuple[Optional[Dict[int, NDArray]], str]]:
    """
    Lädt Zeitreihe echter AIA-Daten für alle Kanäle.

    Args:
        start_time: Startzeit (ISO format)
        n_points: Anzahl Zeitpunkte
        cadence_minutes: Zeitabstand
        wavelengths: Wellenlängen (default: alle 7)
        data_dir: Download-Verzeichnis
        verbose: Ausgabe
        cleanup: FITS-Dateien nach Laden löschen

    Returns:
        Liste von (channels_dict, timestamp) Tupeln
    """
    import gc

    if wavelengths is None:
        wavelengths = WAVELENGTHS

    results = []
    t = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
    failed_count = 0
    max_consecutive_failures = 10

    for i in range(n_points):
        timestamp = t.isoformat()

        if verbose:
            print(f"    📥 Lade Zeitpunkt {i+1}/{n_points}: {timestamp[:19]}...")

        channels, metadata = load_aia_multichannel(
            timestamp,
            wavelengths=wavelengths,
            data_dir=data_dir,
            cleanup=cleanup
        )

        if channels is not None:
            results.append((channels, timestamp))
            failed_count = 0  # Reset bei Erfolg
        else:
            failed_count += 1
            if verbose:
                print(f"    ⚠️  Überspringe Zeitpunkt {timestamp}")

            # Abbruch bei zu vielen aufeinanderfolgenden Fehlern
            if failed_count >= max_consecutive_failures:
                if verbose:
                    print(f"    ✗ Abbruch: {max_consecutive_failures} aufeinanderfolgende Fehler")
                break

        t += timedelta(minutes=cadence_minutes)

        # Periodic garbage collection every 50 timepoints
        if (i + 1) % 50 == 0:
            gc.collect()
            if verbose:
                print(f"    🧹 Memory cleaned ({len(results)} timepoints loaded)")

    return results


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_pair(
    image_1: NDArray[np.float64],
    image_2: NDArray[np.float64],
    wavelength_1: int,
    wavelength_2: int,
    bins: int = 64,
    seed: int = 42
) -> PairResult:
    """
    Analysiert ein Wellenlängen-Paar.

    Berechnet alle Metriken: MI, Residual MI, ΔMI_ring, ΔMI_sector.
    """
    from solar_seed.null_model import compute_null_distribution, compute_z_score

    # Original MI
    mi_original = mutual_information(image_1, image_2, bins=bins)

    # Residual MI
    res_1, res_2, _ = prepare_pair_for_residual_mi(image_1, image_2)
    mi_residual = mutual_information(res_1, res_2, bins=bins)

    # Ratio
    mi_ratio = mi_residual / mi_original if mi_original > 0 else 0.0

    # Sector/Ring Shuffle
    sector_result = sector_ring_shuffle_test(
        image_1, image_2,
        n_rings=20,
        n_sectors=16,
        seed=seed,
        bins=bins
    )

    delta_mi_ring = mi_residual - sector_result.mi_ring_shuffled
    delta_mi_sector = mi_residual - sector_result.mi_sector_shuffled

    # Z-Score
    mi_null_mean, mi_null_std, _ = compute_null_distribution(
        res_1, res_2,
        n_shuffles=30,
        bins=bins,
        seed=seed
    )
    z_score = compute_z_score(mi_residual, mi_null_mean, mi_null_std)

    # Temperatur-Differenz
    temp_1 = WAVELENGTH_TO_TEMP.get(wavelength_1, 1.0)
    temp_2 = WAVELENGTH_TO_TEMP.get(wavelength_2, 1.0)
    temp_diff = abs(temp_1 - temp_2)

    return PairResult(
        wavelength_1=wavelength_1,
        wavelength_2=wavelength_2,
        mi_original=mi_original,
        mi_residual=mi_residual,
        mi_ratio=mi_ratio,
        delta_mi_ring=delta_mi_ring,
        delta_mi_sector=delta_mi_sector,
        z_score=z_score,
        temperature_diff=temp_diff
    )


def build_coupling_matrix(
    pair_results: List[PairResult],
    metric: str = "delta_mi_sector"
) -> CouplingMatrix:
    """
    Baut Kopplungs-Matrix aus Paar-Ergebnissen.

    Args:
        pair_results: Liste von PairResult
        metric: Welche Metrik ("delta_mi_sector", "mi_ratio", "delta_mi_ring")

    Returns:
        Symmetrische Kopplungs-Matrix
    """
    n = len(WAVELENGTHS)
    matrix = np.zeros((n, n))

    for pr in pair_results:
        i = WAVELENGTHS.index(pr.wavelength_1)
        j = WAVELENGTHS.index(pr.wavelength_2)

        if metric == "delta_mi_sector":
            value = pr.delta_mi_sector
        elif metric == "mi_ratio":
            value = pr.mi_ratio
        elif metric == "delta_mi_ring":
            value = pr.delta_mi_ring
        else:
            value = pr.delta_mi_sector

        matrix[i, j] = value
        matrix[j, i] = value  # Symmetrisch

    return CouplingMatrix(
        wavelengths=WAVELENGTHS.copy(),
        matrix=matrix,
        metric=metric
    )


def run_multichannel_analysis(
    n_hours: float = 24.0,
    cadence_minutes: int = 12,
    shape: Tuple[int, int] = (256, 256),
    seed: int = 42,
    output_dir: str = "results/multichannel",
    use_real_data: bool = False,
    start_time_str: Optional[str] = None,
    verbose: bool = True
) -> MultiChannelResult:
    """
    Führt komplette Multi-Channel-Analyse durch.

    Args:
        n_hours: Analysezeitraum in Stunden
        cadence_minutes: Zeitabstand zwischen Bildern
        shape: Bildgröße (nur für synthetische Daten)
        seed: Random Seed
        output_dir: Output-Verzeichnis
        use_real_data: Echte AIA-Daten verwenden
        start_time_str: Startzeit für echte Daten (ISO format)
        verbose: Ausführliche Ausgabe

    Returns:
        MultiChannelResult mit allen Ergebnissen
    """
    import time
    start_time = time.time()

    # Output-Verzeichnis
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n_points = int(n_hours * 60 / cadence_minutes)
    n_pairs = len(WAVELENGTHS) * (len(WAVELENGTHS) - 1) // 2

    data_source = "echten AIA" if use_real_data else "synthetischen"

    if verbose:
        print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                🌞 MULTI-CHANNEL ANALYSE 🌱                              ║
╚═══════════════════════════════════════════════════════════════════════╝

  Konfiguration:
    Datenquelle: {data_source}-Daten
    Kanäle:      {len(WAVELENGTHS)} ({', '.join(str(w) for w in WAVELENGTHS)} Å)
    Paare:       {n_pairs}
    Zeitraum:    {n_hours} Stunden
    Zeitpunkte:  {n_points}
    Kadenz:      {cadence_minutes} min
""")

    # Lade oder generiere Zeitreihe
    if use_real_data:
        if start_time_str is None:
            # Default: vor 24 Stunden starten
            start_time_str = (datetime.now() - timedelta(hours=n_hours + 24)).isoformat()

        if verbose:
            print(f"  📡 Lade echte AIA-Daten ab {start_time_str[:19]}...")

        timeseries = load_aia_multichannel_timeseries(
            start_time=start_time_str,
            n_points=n_points,
            cadence_minutes=cadence_minutes,
            verbose=verbose
        )

        if len(timeseries) == 0:
            print("  ✗ No data loaded. Aborting.")
            raise RuntimeError("No AIA data available")

        if verbose:
            print(f"  ✓ {len(timeseries)} timepoints loaded")
    else:
        if verbose:
            print("  📊 Generating synthetic multi-channel timeseries...")

        timeseries = generate_multichannel_timeseries(
            n_points=n_points,
            shape=shape,
            seed=seed,
            cadence_minutes=cadence_minutes
        )

    # Sammle Ergebnisse für alle Paare über alle Zeitpunkte
    all_pair_results: Dict[Tuple[int, int], List[PairResult]] = {
        pair: [] for pair in combinations(WAVELENGTHS, 2)
    }

    if verbose:
        print(f"\n  🔬 Analysiere {n_points} Zeitpunkte × {n_pairs} Paare...")

    for t_idx, (channels, timestamp) in enumerate(timeseries):
        if verbose and (t_idx + 1) % 10 == 0:
            print(f"     Zeitpunkt {t_idx + 1}/{n_points}...")

        # Analysiere alle Paare für diesen Zeitpunkt
        for wl1, wl2 in combinations(WAVELENGTHS, 2):
            result = analyze_pair(
                channels[wl1], channels[wl2],
                wl1, wl2,
                bins=64,
                seed=seed + t_idx
            )
            all_pair_results[(wl1, wl2)].append(result)

    # Aggregiere über Zeit: Mittelwerte pro Paar
    if verbose:
        print("\n  📈 Aggregiere Ergebnisse...")

    aggregated_pairs = []
    for (wl1, wl2), results in all_pair_results.items():
        # Mittelwerte
        avg_result = PairResult(
            wavelength_1=wl1,
            wavelength_2=wl2,
            mi_original=np.mean([r.mi_original for r in results]),
            mi_residual=np.mean([r.mi_residual for r in results]),
            mi_ratio=np.mean([r.mi_ratio for r in results]),
            delta_mi_ring=np.mean([r.delta_mi_ring for r in results]),
            delta_mi_sector=np.mean([r.delta_mi_sector for r in results]),
            z_score=np.mean([r.z_score for r in results]),
            temperature_diff=results[0].temperature_diff
        )
        aggregated_pairs.append(avg_result)

    # Baue Kopplungs-Matrizen
    coupling_sector = build_coupling_matrix(aggregated_pairs, "delta_mi_sector")
    coupling_ratio = build_coupling_matrix(aggregated_pairs, "mi_ratio")
    coupling_ring = build_coupling_matrix(aggregated_pairs, "delta_mi_ring")

    # Statistiken
    all_sector = [p.delta_mi_sector for p in aggregated_pairs]
    all_ratio = [p.mi_ratio for p in aggregated_pairs]

    duration = time.time() - start_time

    result = MultiChannelResult(
        timestamp=datetime.now().isoformat(),
        n_timepoints=n_points,
        hours=n_hours,
        pair_results=aggregated_pairs,
        coupling_delta_sector=coupling_sector,
        coupling_mi_ratio=coupling_ratio,
        coupling_delta_ring=coupling_ring,
        mean_values={
            "delta_mi_sector": float(np.mean(all_sector)),
            "mi_ratio": float(np.mean(all_ratio)),
        },
        std_values={
            "delta_mi_sector": float(np.std(all_sector)),
            "mi_ratio": float(np.std(all_ratio)),
        }
    )

    # Speichere Ergebnisse
    if verbose:
        print("\n  💾 Speichere Ergebnisse...")

    save_multichannel_results(result, out_path)

    if verbose:
        print_multichannel_summary(result, duration)

    return result


# ============================================================================
# OUTPUT
# ============================================================================

def save_multichannel_results(result: MultiChannelResult, output_dir: Path) -> None:
    """Speichert alle Ergebnisse."""

    # 1. Kopplungs-Matrizen als Text
    with open(output_dir / "coupling_matrices.txt", "w") as f:
        f.write("COUPLING MATRICES (Multi-Channel Analysis)\n")
        f.write("=" * 70 + "\n\n")

        f.write("DATA SOURCE:\n")
        f.write(f"  Instrument:   {AIA_DATA_SOURCE['instrument']}\n")
        f.write(f"  Operator:     {AIA_DATA_SOURCE['operator']}\n")
        f.write(f"  Data:         {AIA_DATA_SOURCE['data_provider']}\n")
        f.write(f"  URL:          {AIA_DATA_SOURCE['data_url']}\n")
        f.write(f"  Reference:    {AIA_DATA_SOURCE['reference']}\n\n")

        f.write(f"Period: {result.hours} hours, {result.n_timepoints} timepoints\n\n")

        f.write("ΔMI_sector (true local structure coupling):\n")
        f.write("-" * 50 + "\n")
        f.write(result.coupling_delta_sector.to_ascii() + "\n\n")

        f.write("MI Ratio (residual/original):\n")
        f.write("-" * 50 + "\n")
        f.write(result.coupling_mi_ratio.to_ascii() + "\n\n")

        f.write("ΔMI_ring (structure beyond radial statistics):\n")
        f.write("-" * 50 + "\n")
        f.write(result.coupling_delta_ring.to_ascii() + "\n\n")

    # 2. Paar-Details als CSV
    with open(output_dir / "pair_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "wavelength_1", "wavelength_2", "temp_diff_MK",
            "mi_original", "mi_residual", "mi_ratio",
            "delta_mi_ring", "delta_mi_sector", "z_score"
        ])

        for pr in sorted(result.pair_results, key=lambda x: -x.delta_mi_sector):
            writer.writerow([
                pr.wavelength_1, pr.wavelength_2, f"{pr.temperature_diff:.2f}",
                f"{pr.mi_original:.4f}", f"{pr.mi_residual:.4f}", f"{pr.mi_ratio:.4f}",
                f"{pr.delta_mi_ring:.4f}", f"{pr.delta_mi_sector:.4f}", f"{pr.z_score:.1f}"
            ])

    # 3. Matrizen als JSON (für weitere Verarbeitung)
    matrix_data = {
        "wavelengths": result.coupling_delta_sector.wavelengths,
        "delta_mi_sector": result.coupling_delta_sector.matrix.tolist(),
        "mi_ratio": result.coupling_mi_ratio.matrix.tolist(),
        "delta_mi_ring": result.coupling_delta_ring.matrix.tolist(),
        "metadata": {
            "timestamp": result.timestamp,
            "n_timepoints": result.n_timepoints,
            "hours": result.hours
        },
        "data_source": AIA_DATA_SOURCE,
        "statistics": {
            "mean_delta_mi_sector": result.mean_values.get("delta_mi_sector", 0),
            "std_delta_mi_sector": result.std_values.get("delta_mi_sector", 0),
            "mean_mi_ratio": result.mean_values.get("mi_ratio", 0),
            "std_mi_ratio": result.std_values.get("mi_ratio", 0),
        }
    }

    with open(output_dir / "coupling_matrices.json", "w") as f:
        json.dump(matrix_data, f, indent=2)

    # 4. Temperatur-Kopplung Analyse
    with open(output_dir / "temperature_coupling.txt", "w") as f:
        f.write("TEMPERATUR-KOPPLUNGS-ANALYSE\n")
        f.write("=" * 70 + "\n\n")

        f.write("Hypothese: Benachbarte Temperaturschichten sollten stärker gekoppelt sein.\n\n")

        f.write("Paare sortiert nach ΔMI_sector (höchste Kopplung zuerst):\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Paar':<15} {'ΔT (MK)':<10} {'ΔMI_sector':<12} {'Interpretation'}\n")
        f.write("-" * 70 + "\n")

        for pr in sorted(result.pair_results, key=lambda x: -x.delta_mi_sector):
            if pr.temperature_diff < 1.0:
                interp = "← benachbart"
            elif pr.temperature_diff < 5.0:
                interp = ""
            else:
                interp = "← weit entfernt"

            f.write(f"{pr.wavelength_1}-{pr.wavelength_2:<7} "
                    f"{pr.temperature_diff:<10.2f} "
                    f"{pr.delta_mi_sector:<12.4f} "
                    f"{interp}\n")


def print_multichannel_summary(result: MultiChannelResult, duration: float) -> None:
    """Gibt Zusammenfassung aus."""

    # Top und Bottom Paare nach Kopplung
    sorted_pairs = sorted(result.pair_results, key=lambda x: -x.delta_mi_sector)

    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║              🌞 MULTI-CHANNEL ERGEBNIS 🌱                               ║
╚═══════════════════════════════════════════════════════════════════════╝

  Zeitraum:    {result.hours} Stunden ({result.n_timepoints} Zeitpunkte)
  Dauer:       {duration:.1f} Sekunden

  ════════════════════════════════════════════════════════════════════════

  KOPPLUNGS-MATRIX (ΔMI_sector in bits):

{result.coupling_delta_sector.to_ascii()}

  ════════════════════════════════════════════════════════════════════════

  TOP 5 STÄRKSTE KOPPLUNGEN:
""")

    for i, pr in enumerate(sorted_pairs[:5], 1):
        ch1 = next(c for c in AIA_CHANNELS if c.wavelength == pr.wavelength_1)
        ch2 = next(c for c in AIA_CHANNELS if c.wavelength == pr.wavelength_2)
        print(f"    {i}. {pr.wavelength_1}-{pr.wavelength_2} Å: "
              f"ΔMI = {pr.delta_mi_sector:.4f} bits "
              f"(ΔT = {pr.temperature_diff:.1f} MK)")

    print(f"""
  BOTTOM 3 SCHWÄCHSTE KOPPLUNGEN:
""")

    for i, pr in enumerate(sorted_pairs[-3:], 1):
        print(f"    {i}. {pr.wavelength_1}-{pr.wavelength_2} Å: "
              f"ΔMI = {pr.delta_mi_sector:.4f} bits "
              f"(ΔT = {pr.temperature_diff:.1f} MK)")

    print(f"""
  ════════════════════════════════════════════════════════════════════════

  STATISTIK:
    Mittlere Kopplung (ΔMI_sector): {result.mean_values.get('delta_mi_sector', 0):.4f} ± {result.std_values.get('delta_mi_sector', 0):.4f} bits
    Mittleres MI Ratio:             {result.mean_values.get('mi_ratio', 0):.4f} ± {result.std_values.get('mi_ratio', 0):.4f}

  OUTPUT-DATEIEN:
    results/multichannel/coupling_matrices.txt
    results/multichannel/coupling_matrices.json
    results/multichannel/pair_results.csv
    results/multichannel/temperature_coupling.txt

═══════════════════════════════════════════════════════════════════════
""")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Hauptfunktion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Channel Analyse für Solar Seed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python -m solar_seed.multichannel --hours 24
  python -m solar_seed.multichannel --hours 6 --cadence 6
  python -m solar_seed.multichannel --real --hours 2 --start "2024-01-15T12:00:00"
        """
    )
    parser.add_argument("--hours", type=float, default=24.0,
                        help="Analysezeitraum in Stunden (default: 24)")
    parser.add_argument("--cadence", type=int, default=12,
                        help="Kadenz in Minuten (default: 12)")
    parser.add_argument("--output", type=str, default="results/multichannel",
                        help="Output-Verzeichnis")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random Seed")
    parser.add_argument("--real", action="store_true",
                        help="Echte AIA-Daten verwenden")
    parser.add_argument("--start", type=str, default=None,
                        help="Startzeit für echte Daten (ISO format)")

    args = parser.parse_args()

    run_multichannel_analysis(
        n_hours=args.hours,
        cadence_minutes=args.cadence,
        seed=args.seed,
        output_dir=args.output,
        use_real_data=args.real,
        start_time_str=args.start,
        verbose=True
    )


if __name__ == "__main__":
    main()
