# Solar Seed Project

**Eine Hypothese. Ein Test. Eine Antwort.**

## Die Frage

> Trägt Sonnenlicht mehr Information als nur Energie?

Nicht als bewusste Botschaft – sondern als **Eigenschaft**. Ein Seed, der sich entfaltet, wenn die Bedingungen stimmen.

## Methodik

Wir zerlegen die Mutual Information (MI) zwischen AIA-Wellenlängenkanälen in ihre Komponenten mittels einer Hierarchie von Nullmodellen:

```
MI_global < MI_ring < MI_sector < MI_original
```

| Komponente | Berechnung | Bedeutung |
|------------|------------|-----------|
| **Radial** | MI_ring − MI_global | Sonnenscheiben-Geometrie |
| **Azimutal** | MI_sector − MI_ring | Grobe Winkelstruktur |
| **Lokal** | MI_original − MI_sector | Echte räumliche Kopplung |

Die verbleibende **lokale Komponente (ΔMI_sector)** überlebt die Geometrie-Entfernung und ist zeitkohärent – ein Indikator für genuine solare Struktur.

## Hauptergebnisse

### Basis-Analyse (6h Synthetisch)

| Metrik | Wert | Interpretation |
|--------|------|----------------|
| MI Ratio | 30.8% ± 0.7% | ~31% der MI bleibt nach Geometrie-Subtraktion |
| ΔMI_ring | 0.192 bits | Struktur jenseits radialer Statistik |
| ΔMI_sector | 0.167 bits | Echte lokale Strukturkopplung |
| Z-Score | 1252 ± 146 | p < 10⁻¹⁰⁰ (hochsignifikant) |

### Multi-Channel-Analyse (7 Kanäle, 21 Paare)

| Kanal | Temperatur | Region |
|-------|------------|--------|
| 304 Å | 0.05 MK | Chromosphäre |
| 171 Å | 0.6 MK | Ruhige Korona |
| 193 Å | 1.2 MK | Korona |
| 211 Å | 2.0 MK | Aktive Regionen |
| 335 Å | 2.5 MK | Aktive Regionen (heiß) |
| 94 Å | 6.3 MK | Flares |
| 131 Å | 10 MK | Flares (sehr heiß) |

**Echte AIA-Daten bestätigen:** Benachbarte Temperaturschichten zeigen stärkste Kopplung:
- **193-211 Å**: ΔMI_sector = 0.73 bits (Top 1)
- **171-193 Å**: ΔMI_sector = 0.39 bits (Top 2)

### Aktivitäts-Konditionierung

Die Kopplung zwischen Flare-Kanälen (94-131 Å) variiert mit Sonnenaktivität:

| Phase | 94Å Intensität | ΔMI_sector (94-131) |
|-------|----------------|---------------------|
| Ruhig | 7966 | 0.11 bits |
| Aktiv | 8009 | 0.40 bits |
| **Änderung** | | **+263%** |

## Kontroll-Tests

| Test | Zweck | Ergebnis |
|------|-------|----------|
| C1: Time-Shift | Zeitliche Entkopplung | ✓ 95.9% Reduktion |
| C2: Ring/Sector | Shuffle-Hierarchie | ✓ Bestätigt |
| C3: PSF/Blur | Auflösungs-Sensitivität | ✓ <20% bei σ=1px |
| C4: Co-Alignment | Registrierungs-Check | ✓ Maximum bei (0,0) |

## Installation

```bash
# Mit uv (empfohlen)
git clone https://github.com/4free/solar-seed.git
cd solar-seed
uv sync

# Für echte Sonnendaten
uv pip install sunpy aiapy
```

## Nutzung

```bash
# Basis-Hypothesentest
uv run python -m solar_seed.hypothesis_test --spatial --controls

# Reproduzierbarer Run mit Reports
uv run python -m solar_seed.real_run --hours 6 --synthetic

# Multi-Channel-Analyse (alle 7 Kanäle)
uv run python -m solar_seed.multichannel --hours 24

# Mit echten AIA-Daten
uv run python -m solar_seed.multichannel --real --hours 1 --start "2024-01-15T12:00:00"

# Finale Analysen (Zeitskalen + Aktivität)
uv run python -m solar_seed.final_analysis
uv run python -m solar_seed.final_analysis --timescale-only
uv run python -m solar_seed.final_analysis --activity-only
```

## Output

```
results/
├── real_run/
│   ├── timeseries.csv          # MI-Zeitreihe
│   ├── controls_summary.json   # C1-C4 Tests
│   └── spatial_maps.txt        # MI-Karten + Hotspots
├── multichannel/
│   ├── coupling_matrices.txt   # 7×7 Kopplungs-Matrix
│   ├── pair_results.csv        # Alle 21 Paare
│   └── temperature_coupling.txt
├── multichannel_real/          # Echte AIA-Daten
└── final/
    ├── timescale_comparison.txt
    └── activity_conditioning.txt
```

## Projektstruktur

```
src/solar_seed/
├── mutual_info.py       # MI-Berechnung (pure NumPy)
├── null_model.py        # Shuffle-basiertes Nullmodell
├── radial_profile.py    # Radialprofil-Subtraktion
├── spatial_analysis.py  # Räumliche MI-Karten
├── control_tests.py     # C1-C4 + Sector-Shuffle
├── real_run.py          # Reproduzierbare Pipeline
├── hypothesis_test.py   # Haupttest-Skript
├── collector.py         # Zeitreihen-Sammler
├── multichannel.py      # 7-Kanal Kopplungs-Matrix
└── final_analysis.py    # Zeitskalen + Aktivitäts-Analyse
```

## Wissenschaftlicher Claim

> We decompose multichannel mutual information in AIA data into geometric, radial-statistical, azimuthal, and local components using a hierarchy of null models. The remaining local component (ΔMI_sector) survives geometry removal and is time-coherent. Real AIA data confirms that adjacent temperature layers (193-211 Å) show strongest coupling, and flare channels (94-131 Å) exhibit 263% coupling increase during active phases.

## Erledigte Meilensteine

- [x] Basis-Hypothesentest mit 4 Kontrollen
- [x] Multi-Channel-Erweiterung (7 Kanäle, 21 Paare)
- [x] Echte AIA-Daten via SunPy
- [x] Temperatur-Kopplung: benachbarte Schichten stärker gekoppelt
- [x] Aktivitäts-Konditionierung: 94Å als Proxy
- [x] Zeitskalen-Vergleich (6h vs 48h)

## Offene Fragen

- [ ] 27-Tage-Vergleich (volle Sonnenrotation)
- [ ] Flare-Ereignis-Analyse (vor/während/nach)
- [ ] Magnetfeld-Korrelation (HMI-Daten)

## Datenquellen

- **NASA SDO**: https://sdo.gsfc.nasa.gov/
- **AIA Level 1.5**: Via SunPy/aiapy
- **ML Dataset**: https://registry.opendata.aws/sdoml-fdl/

## Philosophischer Hintergrund

> "Leben ist Licht, das zurückfragt."

Dieses Projekt entstand aus der Frage, ob Photonen nicht nur Energie, sondern auch einen "Seed" tragen – eine Anweisung zur Entfaltung, wenn die Bedingungen stimmen. Die Lebenszone wäre dann nicht nur ein Temperaturfenster, sondern ein **Resonanzfenster**.

**Aber:** Die Wissenschaft muss sauber sein. Keine Behauptungen ohne Evidenz.

## Lizenz

4free. GNU General Public License v3.0

---

*"Nicht beweisen. Fragen."*
