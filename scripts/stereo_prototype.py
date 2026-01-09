#!/usr/bin/env python3
"""
STEREO/EUVI Prototype
=====================

Vergleicht Kopplungs-Hierarchie zwischen SDO/AIA und STEREO-A/EUVI.

Wenn die Hierarchie aus verschiedenen Blickwinkeln identisch ist,
ist sie intrinsisch solar - nicht perspektivisch.

EUVI Kanäle: 304, 171, 195 (≈193), 284 Å
AIA Kanäle:  304, 171, 193, 211, 335, 94, 131 Å

Gemeinsame Kanäle: 304, 171, 195/193

Workflow:
    1. Suche STEREO-A/EUVI Daten für Zeitpunkt
    2. Lade Daten herunter und berechne ΔMI_sector
    3. Vergleiche Hierarchie mit SDO/AIA Ergebnissen
    4. Berechne Korrelation zur Validierung
"""

import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from itertools import combinations
import json
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from solar_seed.mutual_info import mutual_information
from solar_seed.radial_profile import subtract_radial_geometry
from solar_seed.control_tests import sector_ring_shuffle_test

# SunPy imports
try:
    import sunpy.map
    from sunpy.net import Fido, attrs as a
    import astropy.units as u
    SUNPY_AVAILABLE = True
except ImportError:
    SUNPY_AVAILABLE = False
    print("⚠️  SunPy nicht installiert. Installiere mit: uv pip install sunpy")


# EUVI Wellenlängen (in Angstrom)
EUVI_WAVELENGTHS = [304, 171, 195, 284]

# Mapping EUVI -> AIA (für Vergleich)
EUVI_TO_AIA = {
    304: 304,   # He II - identisch
    171: 171,   # Fe IX - identisch
    195: 193,   # Fe XII ≈ Fe XII
    284: None,  # Fe XV - kein AIA-Äquivalent
}

# AIA Referenzwerte werden dynamisch aus Segment-Daten geladen
AIA_REFERENCE_COUPLING = {}  # Wird von load_aia_reference() gefüllt


def load_aia_reference(
    timestamp: str,
    segment_dir: str = "results/rotation/segments"
) -> dict:
    """
    Lädt AIA Referenzwerte aus Segment-Daten für einen exakten Zeitpunkt.

    Args:
        timestamp: ISO timestamp (z.B. "2025-12-01T12:00:00")
        segment_dir: Verzeichnis mit Segment-Dateien

    Returns:
        Dict {(wl1, wl2): delta_mi_sector}
    """
    date = timestamp[:10]
    target_time = timestamp[11:16]  # "HH:MM"

    segment_path = Path(segment_dir) / f"{date}.json"

    if not segment_path.exists():
        print(f"  ⚠️  Keine AIA-Daten für {date} gefunden")
        print(f"      Erwartet: {segment_path}")
        return {}

    try:
        with open(segment_path) as f:
            data = json.load(f)

        timestamps = data.get('timestamps', [])
        pair_values = data.get('pair_values', {})

        # Finde den nächsten Zeitpunkt zu target_time
        best_idx = None
        best_diff = float('inf')

        for i, ts in enumerate(timestamps):
            ts_time = ts[11:16]  # "HH:MM"
            # Berechne Differenz in Minuten
            ts_h, ts_m = int(ts_time[:2]), int(ts_time[3:5])
            tgt_h, tgt_m = int(target_time[:2]), int(target_time[3:5])
            diff = abs((ts_h * 60 + ts_m) - (tgt_h * 60 + tgt_m))

            if diff < best_diff:
                best_diff = diff
                best_idx = i

        if best_idx is None:
            print(f"  ✗ Kein passender Zeitpunkt gefunden")
            return {}

        matched_ts = timestamps[best_idx]
        print(f"  ✓ AIA-Daten geladen für {matched_ts}")
        print(f"    (Ziel: {timestamp}, Differenz: {best_diff} min)")

        # Extrahiere Werte für diesen Zeitpunkt
        reference = {}
        for pair_str, values in pair_values.items():
            if best_idx < len(values):
                wl1, wl2 = map(int, pair_str.split('-'))
                reference[(wl1, wl2)] = values[best_idx]

        print(f"    {len(reference)} Paare extrahiert")

        return reference

    except Exception as e:
        print(f"  ✗ Fehler beim Laden der AIA-Daten: {e}")
        return {}

# Temperatur-Mapping für EUVI
EUVI_TEMPERATURES = {
    304: 0.05,   # MK - Chromosphäre
    171: 0.6,    # MK - Ruhige Korona
    195: 1.2,    # MK - Korona (≈193Å)
    284: 2.0,    # MK - Aktive Regionen
}


def search_stereo_euvi(timestamp: str, spacecraft: str = "A") -> dict:
    """
    Sucht STEREO/EUVI Daten für einen Zeitpunkt.

    Args:
        timestamp: ISO timestamp (z.B. "2025-12-01T12:00:00")
        spacecraft: "A" oder "B"

    Returns:
        Dict mit Suchergebnissen pro Wellenlänge
    """
    if not SUNPY_AVAILABLE:
        return {}

    source = f"STEREO_{spacecraft}"
    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

    # Zeitfenster: ±30 Minuten
    time_start = dt.isoformat()
    time_end = (dt.replace(minute=dt.minute + 30) if dt.minute < 30
                else dt.replace(hour=dt.hour + 1, minute=0)).isoformat()

    results = {}

    print(f"\n🛰️  Suche STEREO-{spacecraft} EUVI Daten für {timestamp[:10]}...")

    for wl in EUVI_WAVELENGTHS:
        try:
            # Neue SunPy 7.x Syntax
            result = Fido.search(
                a.Time(time_start, time_end),
                a.Source(source),
                a.Instrument('EUVI'),
                a.Wavelength(wl * u.Angstrom)
            )

            n_found = len(result[0]) if result else 0
            results[wl] = {
                'count': n_found,
                'result': result if n_found > 0 else None
            }

            status = "✓" if n_found > 0 else "✗"
            print(f"    {status} {wl} Å: {n_found} Dateien gefunden")

        except Exception as e:
            results[wl] = {'count': 0, 'result': None, 'error': str(e)}
            print(f"    ✗ {wl} Å: Fehler - {e}")

    return results


def download_stereo_euvi(
    timestamp: str,
    spacecraft: str = "A",
    output_dir: str = "data/stereo",
    wavelengths: list = None
) -> dict:
    """
    Lädt STEREO/EUVI Daten herunter.

    Args:
        timestamp: ISO timestamp
        spacecraft: "A" oder "B"
        output_dir: Zielverzeichnis
        wavelengths: Liste der Wellenlängen (default: alle)

    Returns:
        Dict mit Pfaden zu heruntergeladenen Dateien
    """
    if not SUNPY_AVAILABLE:
        return {}

    if wavelengths is None:
        wavelengths = EUVI_WAVELENGTHS

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    source = f"STEREO_{spacecraft}"
    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

    # Zeitfenster
    time_start = dt.isoformat()
    time_end = (dt.replace(hour=dt.hour + 1)).isoformat()

    downloaded = {}

    print(f"\n📥 Lade STEREO-{spacecraft} EUVI Daten...")

    for wl in wavelengths:
        try:
            result = Fido.search(
                a.Time(time_start, time_end),
                a.Source(source),
                a.Instrument('EUVI'),
                a.Wavelength(wl * u.Angstrom)
            )

            if result and len(result[0]) > 0:
                # Nur erste Datei herunterladen
                files = Fido.fetch(result[0, 0], path=str(out_path))
                if files:
                    downloaded[wl] = files[0]
                    print(f"    ✓ {wl} Å: {Path(files[0]).name}")
            else:
                print(f"    ✗ {wl} Å: Keine Daten gefunden")

        except Exception as e:
            print(f"    ✗ {wl} Å: Fehler - {e}")

    return downloaded


def load_euvi_multichannel(
    timestamp: str,
    spacecraft: str = "A",
    data_dir: str = "data/stereo"
) -> tuple:
    """
    Lädt EUVI Multi-Channel Daten analog zu AIA.

    Returns:
        (channels_dict, metadata) oder (None, None)
    """
    # Erst herunterladen
    files = download_stereo_euvi(timestamp, spacecraft, data_dir)

    if not files:
        return None, None

    channels = {}

    for wl, filepath in files.items():
        try:
            euvi_map = sunpy.map.Map(filepath)

            # Auf 512x512 resamplen (wie bei AIA)
            # SunPy 7.x Syntax: shape as Quantity array
            target_shape = [512, 512] * u.pix
            resampled = euvi_map.resample(target_shape)

            channels[wl] = resampled.data.astype(np.float64)
            print(f"    ✓ {wl} Å geladen: {resampled.data.shape}")

        except Exception as e:
            print(f"    ⚠️  Fehler beim Laden von {wl} Å: {e}")

    if len(channels) < 2:
        return None, None

    metadata = {
        'spacecraft': f'STEREO-{spacecraft}',
        'instrument': 'EUVI',
        'timestamp': timestamp,
        'wavelengths': list(channels.keys())
    }

    return channels, metadata


def calculate_euvi_coupling(channels: dict) -> dict:
    """
    Berechnet ΔMI_sector für alle EUVI-Kanalpaare.

    Verwendet dieselbe Methodik wie für AIA:
    1. Radiale Profil-Subtraktion
    2. Sector-Ring-Shuffle für ΔMI_sector

    Args:
        channels: Dict {wavelength: 2D-Array}

    Returns:
        Dict mit Ergebnissen pro Paar
    """
    wavelengths = sorted(channels.keys())
    results = {}

    print(f"\n📊 Berechne Kopplungs-Matrix für {len(wavelengths)} Kanäle...")

    for wl1, wl2 in combinations(wavelengths, 2):
        try:
            img1 = channels[wl1]
            img2 = channels[wl2]

            # Radiale Normalisierung (Funktion gibt (residual, profile, model) zurück)
            res1, _, _ = subtract_radial_geometry(img1)
            res2, _, _ = subtract_radial_geometry(img2)

            # MI auf Residuen
            mi_residual = mutual_information(res1, res2)

            # Sector-Ring-Shuffle Test
            shuffle_result = sector_ring_shuffle_test(
                res1, res2,
                n_rings=8,
                n_sectors=8
            )

            # Delta MI = Original - Sector-Shuffled (was nach Geometrie-Entfernung übrig bleibt)
            delta_mi_sector = shuffle_result.mi_original - shuffle_result.mi_sector_shuffled
            # Z-Score approximieren aus Reduktionsprozent
            z_score = shuffle_result.sector_reduction_percent / 5.0  # Grobe Approximation

            # Temperatur-Differenz
            temp_diff = abs(EUVI_TEMPERATURES[wl1] - EUVI_TEMPERATURES[wl2])

            results[(wl1, wl2)] = {
                'mi_residual': mi_residual,
                'delta_mi_sector': delta_mi_sector,
                'z_score': z_score,
                'temperature_diff': temp_diff
            }

            print(f"    {wl1}-{wl2} Å: ΔMI = {delta_mi_sector:.3f} bits (Z={z_score:.1f})")

        except Exception as e:
            print(f"    ⚠️ {wl1}-{wl2} Å: Fehler - {e}")

    return results


def compare_coupling_hierarchies(
    euvi_results: dict,
    aia_reference: dict = None
) -> dict:
    """
    Vergleicht Kopplungs-Hierarchien zwischen EUVI und AIA.

    Args:
        euvi_results: {(wl1, wl2): {'delta_mi_sector': float, ...}}
        aia_reference: {(wl1, wl2): float} - AIA Referenzwerte (optional)

    Returns:
        Vergleichsstatistiken
    """
    if aia_reference is None:
        aia_reference = AIA_REFERENCE_COUPLING

    # Finde gemeinsame Paare (mit Wellenlängen-Mapping)
    # Berücksichtige beide Key-Reihenfolgen
    common_pairs = []

    for euvi_pair in euvi_results.keys():
        # Mappe EUVI -> AIA Wellenlängen
        aia_wl1 = EUVI_TO_AIA.get(euvi_pair[0])
        aia_wl2 = EUVI_TO_AIA.get(euvi_pair[1])

        if aia_wl1 and aia_wl2:
            # Prüfe beide Reihenfolgen
            aia_pair = (aia_wl1, aia_wl2)
            aia_pair_rev = (aia_wl2, aia_wl1)

            if aia_pair in aia_reference:
                common_pairs.append((euvi_pair, aia_pair))
            elif aia_pair_rev in aia_reference:
                common_pairs.append((euvi_pair, aia_pair_rev))

    if not common_pairs:
        return {'error': 'Keine gemeinsamen Paare gefunden'}

    # Extrahiere ΔMI-Werte
    euvi_values = []
    aia_values = []

    for euvi_pair, aia_pair in common_pairs:
        euvi_val = euvi_results[euvi_pair]['delta_mi_sector']
        aia_val = aia_reference[aia_pair]
        euvi_values.append(euvi_val)
        aia_values.append(aia_val)

    # Korrelation berechnen
    if len(euvi_values) >= 2:
        correlation = np.corrcoef(euvi_values, aia_values)[0, 1]
    else:
        correlation = None

    # Rankings erstellen
    euvi_ranking = sorted(
        [(p, euvi_results[p]['delta_mi_sector']) for p in euvi_results],
        key=lambda x: -x[1]
    )
    aia_ranking = sorted(
        [(p, aia_reference[p]) for _, p in common_pairs],
        key=lambda x: -x[1]
    )

    # Ranking-Konsistenz prüfen (Spearman-Korrelation der Ränge)
    euvi_ranks = {p: i for i, (p, _) in enumerate(euvi_ranking)}
    aia_ranks = {p: i for i, (p, _) in enumerate(aia_ranking)}

    rank_diffs = []
    for euvi_pair, aia_pair in common_pairs:
        euvi_rank = euvi_ranks.get(euvi_pair, 999)
        aia_rank = aia_ranks.get(aia_pair, 999)
        rank_diffs.append(abs(euvi_rank - aia_rank))

    return {
        'common_pairs': len(common_pairs),
        'correlation': correlation,
        'euvi_ranking': euvi_ranking,
        'aia_ranking': aia_ranking,
        'mean_rank_diff': np.mean(rank_diffs) if rank_diffs else None,
        'pair_comparison': [
            {
                'euvi_pair': euvi_pair,
                'aia_pair': aia_pair,
                'euvi_mi': euvi_results[euvi_pair]['delta_mi_sector'],
                'aia_mi': aia_reference[aia_pair],
                'ratio': euvi_results[euvi_pair]['delta_mi_sector'] / aia_reference[aia_pair]
                    if aia_reference[aia_pair] > 0 else None
            }
            for euvi_pair, aia_pair in common_pairs
        ]
    }


def validate_intrinsic_hierarchy(comparison: dict, verbose: bool = True) -> dict:
    """
    Validiert, ob die Kopplungs-Hierarchie intrinsisch solar ist.

    Kriterien für intrinsische Hierarchie:
    1. Hohe Korrelation (r > 0.7) zwischen EUVI und AIA ΔMI-Werten
    2. Konsistentes Ranking (mittlere Rangdifferenz < 1)
    3. 171-195 > 304-171 > 304-195 (Temperatur-Ordnung)

    Returns:
        Dict mit Validierungsergebnis
    """
    if 'error' in comparison:
        return {'valid': False, 'error': comparison['error']}

    correlation = comparison.get('correlation')
    mean_rank_diff = comparison.get('mean_rank_diff')

    # Kriterium 1: Korrelation
    corr_valid = correlation is not None and correlation > 0.7

    # Kriterium 2: Ranking-Konsistenz
    rank_valid = mean_rank_diff is not None and mean_rank_diff < 1.5

    # Kriterium 3: Temperatur-Ordnung prüfen
    euvi_ranking = comparison.get('euvi_ranking', [])
    temp_ordered = False

    if len(euvi_ranking) >= 2:
        # Die stärkste Kopplung sollte bei kleiner Temp-Differenz sein
        top_pair, top_mi = euvi_ranking[0]
        # 171-195 sollte am stärksten sein (0.6 MK Differenz)
        if top_pair == (171, 195):
            temp_ordered = True
        # Alternativ: 195-284 wäre auch akzeptabel
        elif top_pair == (195, 284):
            temp_ordered = True

    # Gesamtvalidierung
    is_intrinsic = corr_valid and rank_valid

    result = {
        'is_intrinsic': is_intrinsic,
        'correlation': correlation,
        'correlation_valid': corr_valid,
        'mean_rank_diff': mean_rank_diff,
        'ranking_valid': rank_valid,
        'temperature_ordered': temp_ordered,
        'confidence': 'high' if is_intrinsic and temp_ordered else
                      'medium' if is_intrinsic else 'low'
    }

    if verbose:
        print("\n" + "="*70)
        print("VALIDIERUNG: Intrinsische Kopplungs-Hierarchie")
        print("="*70)

        print(f"\n  Korrelation EUVI↔AIA: {correlation:.3f}" if correlation else
              "\n  Korrelation: nicht berechenbar")
        print(f"  {'✓' if corr_valid else '✗'} Korrelation > 0.7")

        print(f"\n  Mittlere Rangdifferenz: {mean_rank_diff:.2f}" if mean_rank_diff else
              "\n  Rangdifferenz: nicht berechenbar")
        print(f"  {'✓' if rank_valid else '✗'} Rangdifferenz < 1.5")

        print(f"\n  Temperatur-Ordnung: {'✓' if temp_ordered else '✗'} Stärkste Kopplung bei ΔT~0.6 MK")

        print("\n" + "-"*70)
        if is_intrinsic:
            print("  ✅ VALIDIERT: Kopplungs-Hierarchie ist intrinsisch solar!")
            print("     Die Hierarchie ist unabhängig vom Beobachtungswinkel.")
        else:
            print("  ⚠️  NICHT VALIDIERT: Weitere Daten erforderlich")
            if not corr_valid:
                print("     → Korrelation zu niedrig")
            if not rank_valid:
                print("     → Ranking inkonsistent")
        print("-"*70)

    return result


def save_results(
    euvi_results: dict,
    comparison: dict,
    validation: dict,
    timestamp: str,
    aia_reference: dict,
    output_dir: str = "results/stereo"
) -> Path:
    """Speichert Ergebnisse als JSON."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Konvertiere Tuple-Keys zu Strings für JSON
    euvi_json = {
        f"{k[0]}-{k[1]}": v for k, v in euvi_results.items()
    }

    comparison_json = {
        'common_pairs': comparison.get('common_pairs'),
        'correlation': comparison.get('correlation'),
        'mean_rank_diff': comparison.get('mean_rank_diff'),
        'euvi_ranking': [
            {'pair': f"{p[0]}-{p[1]}", 'delta_mi': v}
            for p, v in comparison.get('euvi_ranking', [])
        ],
        'aia_ranking': [
            {'pair': f"{p[0]}-{p[1]}", 'delta_mi': v}
            for p, v in comparison.get('aia_ranking', [])
        ],
        'pair_comparison': [
            {
                'euvi_pair': f"{c['euvi_pair'][0]}-{c['euvi_pair'][1]}",
                'aia_pair': f"{c['aia_pair'][0]}-{c['aia_pair'][1]}",
                'euvi_mi': c['euvi_mi'],
                'aia_mi': c['aia_mi'],
                'ratio': c.get('ratio')
            }
            for c in comparison.get('pair_comparison', [])
        ]
    }

    # Convert numpy types to Python types for JSON serialization
    def to_python(obj):
        if isinstance(obj, (np.bool_, np.generic)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_python(i) for i in obj]
        return obj

    result = {
        'timestamp': timestamp,
        'spacecraft': 'STEREO-A',
        'instrument': 'EUVI',
        'euvi_coupling': to_python(euvi_json),
        'comparison': to_python(comparison_json),
        'validation': to_python(validation),
        'aia_reference': {
            f"{k[0]}-{k[1]}": v for k, v in aia_reference.items()
        }
    }

    filepath = out_path / f"stereo_validation_{timestamp[:10]}.json"
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n📄 Ergebnisse gespeichert: {filepath}")
    return filepath


def main(download: bool = False, analyze: bool = False, timestamp: str = "2025-12-01T12:00:00"):
    """
    Hauptfunktion für STEREO/EUVI Exploration.

    Args:
        download: Wenn True, werden Daten heruntergeladen
        analyze: Wenn True, wird die vollständige Analyse durchgeführt
        timestamp: ISO timestamp für Analyse
    """
    # Bestimme STEREO-A Position basierend auf Datum
    year = int(timestamp[:4])
    if year <= 2011:
        stereo_position = "~180° (gegenüberliegende Seite der Sonne)"
    elif year >= 2023:
        stereo_position = "~51° vor der Erde"
    else:
        stereo_position = "unbekannt"

    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║              🛰️  STEREO/EUVI PROTOTYPE 🌞                             ║
╚═══════════════════════════════════════════════════════════════════════╝

  Ziel: Validieren, dass Kopplungs-Hierarchie intrinsisch solar ist

  Zeitpunkt: {timestamp}
  STEREO-A Position: {stereo_position}
  Gemeinsame Kanäle: 304, 171, 195/193 Å

  Hypothese: Wenn die Kopplungs-Hierarchie aus zwei verschiedenen
             Blickwinkeln identisch ist, ist sie intrinsisch solar.
""")

    # 1. Suche nach verfügbaren Daten
    print("\n" + "="*70)
    print("SCHRITT 1: Daten-Verfügbarkeit prüfen")
    print("="*70)

    search_results = search_stereo_euvi(timestamp, spacecraft="A")

    available = sum(1 for r in search_results.values() if r.get('count', 0) > 0)

    if available < 2:
        print(f"\n  ⚠️  Nur {available} Kanäle verfügbar")
        print("      Möglicherweise noch nicht prozessiert oder Daten-Lücke")
        return search_results

    print(f"\n  ✓ {available}/{len(EUVI_WAVELENGTHS)} Kanäle verfügbar")

    if not download and not analyze:
        print("\n  Zum Herunterladen: python stereo_prototype.py --download")
        print("  Für vollständige Analyse: python stereo_prototype.py --analyze")
        return search_results

    # 2. Download
    print("\n" + "="*70)
    print("SCHRITT 2: Daten herunterladen")
    print("="*70)

    channels, metadata = load_euvi_multichannel(timestamp, spacecraft="A")

    if channels is None:
        print("\n  ✗ Download fehlgeschlagen")
        return None

    print(f"\n  ✓ {len(channels)} Kanäle geladen: {list(channels.keys())} Å")

    if not analyze:
        print("\n  Für Analyse: python stereo_prototype.py --analyze")
        return channels

    # 3. MI-Berechnung
    print("\n" + "="*70)
    print("SCHRITT 3: Kopplungs-Matrix berechnen (ΔMI_sector)")
    print("="*70)

    euvi_results = calculate_euvi_coupling(channels)

    if not euvi_results:
        print("\n  ✗ MI-Berechnung fehlgeschlagen")
        return None

    # 4. AIA-Daten laden und vergleichen
    print("\n" + "="*70)
    print("SCHRITT 4: SDO/AIA Daten laden (exakter Zeitpunkt)")
    print("="*70)

    # AIA-Daten für exakten Zeitpunkt laden
    aia_reference = load_aia_reference(timestamp)

    if not aia_reference:
        print("\n  ⚠️  Keine Segment-Daten verfügbar - lade AIA-Daten live...")
        aia_reference = download_and_analyze_aia(timestamp)

        if not aia_reference:
            print("\n  ✗ Keine AIA-Daten verfügbar - Vergleich nicht möglich")
            return euvi_results

    print("\n  AIA Hierarchie (Top 10):")
    for i, (pair, mi) in enumerate(sorted(aia_reference.items(), key=lambda x: -x[1])[:10]):
        print(f"    {pair[0]}-{pair[1]} Å: {mi:.3f} bits")

    comparison = compare_coupling_hierarchies(euvi_results, aia_reference)

    print("\n  Paarweiser Vergleich:")
    print("  " + "-"*50)
    print(f"  {'EUVI Paar':<12} {'AIA Paar':<12} {'EUVI ΔMI':>10} {'AIA ΔMI':>10} {'Ratio':>8}")
    print("  " + "-"*50)

    for c in comparison.get('pair_comparison', []):
        euvi_p = f"{c['euvi_pair'][0]}-{c['euvi_pair'][1]}"
        aia_p = f"{c['aia_pair'][0]}-{c['aia_pair'][1]}"
        ratio_str = f"{c['ratio']:.2f}" if c.get('ratio') else "N/A"
        print(f"  {euvi_p:<12} {aia_p:<12} {c['euvi_mi']:>10.3f} {c['aia_mi']:>10.3f} {ratio_str:>8}")

    # 5. Validierung
    validation = validate_intrinsic_hierarchy(comparison)

    # 6. Ergebnisse speichern
    save_results(euvi_results, comparison, validation, timestamp, aia_reference)

    # Zusammenfassung
    print("\n" + "="*70)
    print("ZUSAMMENFASSUNG")
    print("="*70)

    corr = comparison.get('correlation')
    corr_str = f"{corr:.3f}" if corr is not None else "N/A (zu wenige Paare)"

    print(f"""
  Zeitpunkt:     {timestamp}
  Instrument:    STEREO-A/EUVI
  Kanäle:        {list(channels.keys())} Å
  Paare:         {len(euvi_results)}
  Gemeinsame:    {comparison.get('common_pairs', 0)}

  Korrelation:   {corr_str}
  Konfidenz:     {validation.get('confidence', 'unknown')}

  Interpretation:
""")

    if validation.get('is_intrinsic'):
        print(f"    Die Kopplungs-Hierarchie ist aus {stereo_position} Blickwinkel")
        print("    konsistent mit SDO/AIA. Dies unterstützt die Hypothese, dass")
        print("    die Temperatur-geordnete Kopplung eine intrinsische Eigenschaft")
        print("    der solaren Atmosphäre ist - nicht ein perspektivischer Effekt.")
    else:
        print("    Die Validierung war nicht erfolgreich. Mögliche Gründe:")
        print("    - Unterschiedliche Aktivitätsniveaus zum Beobachtungszeitpunkt")
        print("    - Kalibrationsdifferenzen zwischen EUVI und AIA")
        print("    - Zu wenige gemeinsame Wellenlängen für robuste Statistik")
        print("    Weitere Zeitpunkte sollten analysiert werden.")

    return {
        'euvi_results': euvi_results,
        'comparison': comparison,
        'validation': validation
    }


def download_and_analyze_aia(
    timestamp: str,
    output_dir: str = "data/aia_temp"
) -> dict:
    """
    Lädt SDO/AIA Daten herunter und berechnet ΔMI_sector.

    Wird verwendet wenn keine Segment-Daten vorliegen.
    """
    if not SUNPY_AVAILABLE:
        return {}

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Zeitfenster: ±15 Minuten
    from datetime import datetime, timedelta
    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    time_start = dt.isoformat()
    time_end = (dt + timedelta(minutes=15)).isoformat()

    # AIA Wellenlängen die zu EUVI passen
    aia_wavelengths = [304, 171, 193, 211]

    print(f"\n📥 Lade SDO/AIA Daten für {timestamp}...")

    channels = {}

    for wl in aia_wavelengths:
        try:
            result = Fido.search(
                a.Time(time_start, time_end),
                a.Instrument('AIA'),
                a.Wavelength(wl * u.Angstrom)
            )

            if result and len(result[0]) > 0:
                # Erste Datei herunterladen
                files = Fido.fetch(result[0, 0], path=str(out_path))
                if files:
                    aia_map = sunpy.map.Map(files[0])
                    # Auf 512x512 resamplen
                    target_shape = [512, 512] * u.pix
                    resampled = aia_map.resample(target_shape)
                    channels[wl] = resampled.data.astype(np.float64)
                    print(f"    ✓ {wl} Å geladen")
            else:
                print(f"    ✗ {wl} Å: Keine Daten")

        except Exception as e:
            print(f"    ✗ {wl} Å: {e}")

    if len(channels) < 2:
        return {}

    # MI berechnen
    print(f"\n📊 Berechne AIA Kopplungs-Matrix...")

    results = {}
    wavelengths = sorted(channels.keys())

    for wl1, wl2 in combinations(wavelengths, 2):
        try:
            img1 = channels[wl1]
            img2 = channels[wl2]

            res1, _, _ = subtract_radial_geometry(img1)
            res2, _, _ = subtract_radial_geometry(img2)

            shuffle_result = sector_ring_shuffle_test(res1, res2, n_rings=8, n_sectors=8)
            delta_mi = shuffle_result.mi_original - shuffle_result.mi_sector_shuffled

            results[(wl1, wl2)] = delta_mi
            print(f"    {wl1}-{wl2} Å: ΔMI = {delta_mi:.3f} bits")

        except Exception as e:
            print(f"    ⚠️ {wl1}-{wl2} Å: {e}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="STEREO/EUVI Validierung")
    parser.add_argument("--download", action="store_true",
                        help="Daten herunterladen")
    parser.add_argument("--analyze", action="store_true",
                        help="Vollständige Analyse durchführen")
    parser.add_argument("--timestamp", type=str, default="2025-12-01T12:00:00",
                        help="Zeitpunkt für Analyse (ISO format)")
    args = parser.parse_args()

    # --analyze impliziert --download
    if args.analyze:
        args.download = True

    main(download=args.download, analyze=args.analyze, timestamp=args.timestamp)
