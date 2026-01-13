# Wait & Backfill Architecture

## Problem

AIA-Datenquellen haben unterschiedliche Verfügbarkeit und Auflösung:

| Quelle | Auflösung | Dateigröße | Latenz | MI-Genauigkeit |
|--------|-----------|------------|--------|----------------|
| Synoptic | 1024² | ~4 MB | ~2 min | 304Å: +350% Inflation |
| SDAC | 1024² | ~4 MB | ~3 Tage | 304Å: +350% Inflation |
| **JSOC** | **4096²** | **~65 MB** | variabel | **Akkurat** |

**Wichtig:** Nur JSOC liefert echte 4k-Daten!
SDAC und andere Mirror behaupten "FULLDISK" liefern aber nur 1k.
Die Dateigröße zeigt es: ~4 MB = 1k, ~65 MB = 4k.

**Kernproblem:** Für Echtzeit-Monitoring ist nur 1k verfügbar, aber 304Å-MI-Werte
sind bei 1k um Faktor 3.5 aufgebläht (räumliches Aliasing).

## Lösung: Wait & Backfill

```
┌─────────────────────────────────────────────────────────────────┐
│                    REAL-TIME MONITORING                         │
│                                                                 │
│  Synoptic (1k) → MI berechnen → Speichern mit resolution='1k'  │
│                                                                 │
│  ⚠️ 304Å-Werte sind inflated, nur für Trend-Erkennung nutzen   │
│  ✓ 193-211 ist scale-invariant, für Predictions verwenden      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    (warte bis JSOC verfügbar)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       BACKFILL JOB                              │
│                                                                 │
│  1. Finde Messungen mit resolution='1k'                        │
│  2. Prüfe ob JSOC 4k-Daten für diesen Zeitpunkt hat (~65MB)    │
│  3. Lade 4k-Daten, verifiziere 4096x4096                       │
│  4. Berechne MI neu mit voller Auflösung                       │
│  5. Update Messung: resolution='4k', neue MI-Werte             │
└─────────────────────────────────────────────────────────────────┘
```

**Aktueller Status (Jan 2026):**
JSOC ist seit 8. Januar offline (Hardware-Ausfall). Backfill wird möglich
sobald JSOC den Backlog verarbeitet hat.

## Datenbank-Schema-Erweiterung

```sql
-- Neue Spalten für coupling_measurements
ALTER TABLE coupling_measurements ADD COLUMN resolution TEXT DEFAULT '1k';
ALTER TABLE coupling_measurements ADD COLUMN backfilled_at TEXT;
ALTER TABLE coupling_measurements ADD COLUMN original_mi_193_304 REAL;  -- vor Backfill
```

## Implementierung

### 1. Real-Time Monitoring (bestehend, angepasst)

```python
def run_coupling_analysis():
    # Versuche 4k (SDAC), Fallback auf 1k (Synoptic)
    channels, timestamp, quality, source = _load_channels([193, 211, 304])

    resolution = '4k' if source in ['sdac', 'jsoc', 'full-res'] else '1k'

    # MI berechnen
    mi_values = compute_mi_pairs(channels)

    # Speichern mit Resolution-Info
    store_coupling_reading(timestamp, mi_values, resolution=resolution)

    # Predictions: nur 193-211 bei 1k, alle Paare bei 4k
    if resolution == '1k':
        # 304Å-Paare nur für Trend, nicht für Predictions
        predictions = predict_from_193_211_only(mi_values)
    else:
        predictions = predict_from_all_pairs(mi_values)
```

### 2. JSOC Loader (4k-only)

```python
def load_aia_jsoc(timestamp: str, wavelengths: list[int]) -> tuple[dict, dict]:
    """
    Lade 4k AIA-Daten von JSOC für einen historischen Zeitpunkt.

    JSOC ist die EINZIGE Quelle für echte 4096² Auflösung!
    Andere Provider (SDAC, etc.) liefern nur 1k trotz "FULLDISK".

    Erkennung: Dateigröße ~65 MB = 4k, ~4 MB = 1k

    Returns:
        (channels_dict, metadata) oder (None, None) wenn nicht 4k verfügbar
    """
    from sunpy.net import Fido, attrs as a
    import astropy.units as u

    result = Fido.search(
        a.Time(dt - timedelta(minutes=2), dt + timedelta(minutes=2)),
        a.Instrument.aia,
        a.Wavelength(wl * u.Angstrom),
        a.Provider('JSOC')  # NUR JSOC hat 4k!
    )

    # Verifiziere 4k durch Dateigröße oder Extent
    if result[0][0]['Size'] < 50:  # 4k = ~65MB, 1k = ~4MB
        return None, None  # Nicht 4k, ablehnen

    # Download und laden...
```

### 3. Backfill Command

```bash
# JSOC Status prüfen (wann sind 4k-Daten verfügbar?)
uv run python scripts/early_warning.py backfill --check-jsoc

# Backfill Status anzeigen
uv run python scripts/early_warning.py backfill --status

# Prüfen was backfillbar wäre (dry-run)
uv run python scripts/early_warning.py backfill --dry-run --days 14

# Tatsächliches Backfill ausführen
uv run python scripts/early_warning.py backfill --days 14
```

```python
@app.command()
def backfill(
    days: int = typer.Option(7, help="Anzahl Tage zurück"),
    start: str = typer.Option(None, help="Start-Datum (YYYY-MM-DD)"),
    end: str = typer.Option(None, help="End-Datum (YYYY-MM-DD)"),
    status: bool = typer.Option(False, help="Zeige Backfill-Status"),
    dry_run: bool = typer.Option(False, help="Nur prüfen, nicht ändern"),
):
    """
    Backfill 1k-Messungen mit 4k-Daten von SDAC.

    Prüft für jede 1k-Messung ob SDAC 4k-Daten verfügbar sind,
    lädt diese herunter und aktualisiert die MI-Werte.
    """
    db = get_monitoring_db()

    # Finde 1k-Messungen im Zeitraum
    measurements = db.get_measurements_for_backfill(
        min_age_days=3,  # SDAC braucht ~3 Tage
        resolution='1k',
        start=start,
        end=end
    )

    print(f"Gefunden: {len(measurements)} Messungen zum Backfill")

    for m in measurements:
        if dry_run:
            print(f"  [DRY] {m['timestamp']}: würde 4k laden")
            continue

        # Versuche 4k zu laden
        channels, meta = load_aia_sdac(m['timestamp'], [193, 211, 304])

        if channels is None:
            print(f"  [SKIP] {m['timestamp']}: SDAC nicht verfügbar")
            continue

        # MI neu berechnen
        new_mi = compute_mi_pairs(channels)

        # Update in DB
        db.update_measurement_with_backfill(
            timestamp=m['timestamp'],
            new_mi=new_mi,
            resolution='4k',
            original_mi_193_304=m['mi_193_304']
        )

        print(f"  [OK] {m['timestamp']}: 193-304 {m['mi_193_304']:.3f} → {new_mi['193-304']:.3f}")
```

### 4. Cron-Job für automatisches Backfill

```bash
# Täglich um 3:00 Uhr Backfill der letzten 7 Tage
0 3 * * * /opt/homebrew/bin/uv run --project /path/to/solar-seed python scripts/early_warning.py backfill --days 7 >> results/early_warning/backfill.log 2>&1
```

## Prediction-Logik bei 1k

Da 304Å bei 1k um +350% aufgebläht ist, sollten Predictions bei 1k-Daten
**nur auf 193-211** basieren:

```python
def should_predict_from_pair(pair: str, resolution: str) -> bool:
    """
    Entscheidet ob ein Paar für Predictions verwendet werden soll.

    Bei 1k: Nur 193-211 (scale-invariant)
    Bei 4k: Alle Paare
    """
    if resolution == '4k':
        return True

    # Bei 1k: nur scale-invariante Paare
    SCALE_INVARIANT_PAIRS = ['193-211']
    return pair in SCALE_INVARIANT_PAIRS
```

## Metriken nach Backfill

Nach dem Backfill können wir die Prediction-Accuracy neu berechnen:

```python
def recalculate_predictions_after_backfill():
    """
    Re-evaluiert Predictions mit korrigierten 4k-Werten.

    Vergleicht:
    - Precision/Recall mit 1k-Daten
    - Precision/Recall mit 4k-Daten (backfilled)
    """
```

## Zusammenfassung

1. **Real-time**: Synoptic 1k, nur 193-211 für Predictions
2. **Backfill**: Nach 3 Tagen automatisch mit SDAC 4k
3. **Audit-Trail**: Originale 1k-Werte bleiben erhalten
4. **Transparenz**: Resolution-Flag zeigt Datenqualität
