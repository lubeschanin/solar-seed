# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Solar Seed** decomposes mutual information between AIA wavelength channels into geometric, radial, azimuthal, and local components using a hierarchy of null models.

**Repository:** https://github.com/lubeschanin/solar-seed

**Core Insight:** The remaining local component (ΔMI_sector) survives geometry removal and is time-coherent, indicating genuine shared solar structure.

**Key Findings:**
- ΔMI_sector = 0.17 ± 0.02 bits (Z > 1000, p < 10⁻¹⁰⁰)
- Real AIA data confirms: adjacent temperature layers (193-211 Å) show strongest coupling (0.59 ± 0.12 bits over 8 days)
- Temperature-ordered coupling structure consistent with magnetically mediated interactions
- **8-day rotation analysis (960 timepoints):** Coupling hierarchy persists over ~30% of solar rotation
- **STEREO 180° validation:** Cross-hemisphere comparison (AIA vs STEREO-A EUVI) shows 90.6% rank correlation - hierarchy is viewpoint-invariant
- **Chromospheric anchor:** 304 Å shows weakest coupling but highest temporal stability (r > 0.43), suggesting stable magnetic footpoints rather than dynamic energy exchange
- **Flare analysis (X9.0):** Most channel pairs show *reduced* coupling during flares (-25% to -47%), reflecting breakdown of coherent organization during rapid magnetic reconfiguration

**Status:** Paper submission-ready (arXiv: astro-ph.SR)

## Commands

```bash
# Interactive CLI (recommended for new users)
./solar-seed

# Run tests
uv run pytest

# Run single test
uv run pytest tests/test_control_tests.py::TestSectorRingShuffle -v

# Hypothesis test with all features
uv run python -m solar_seed.hypothesis_test --spatial --controls

# Reproducible run with reports
uv run python -m solar_seed.real_run --hours 6 --synthetic

# With real AIA data
uv run python -m solar_seed.hypothesis_test --real-data

# Multi-channel analysis (all 7 AIA channels, 21 pairs)
uv run python -m solar_seed.multichannel --hours 24
uv run python -m solar_seed.multichannel --real --hours 1 --start "2024-01-15T12:00:00"

# Final analyses (timescale comparison + activity conditioning)
uv run python -m solar_seed.final_analysis

# Segment-based rotation analysis (recommended, scalable)
uv run python -m solar_seed.final_analysis --segments --start 2025-12-01 --end 2025-12-27
uv run python -m solar_seed.final_analysis --segment 2025-12-15  # Single day
uv run python -m solar_seed.final_analysis --aggregate           # Combine all segments
uv run python -m solar_seed.final_analysis --convert-checkpoint  # Convert legacy data

# Legacy: Monolithic rotation analysis (with checkpoint/resume)
uv run python -m solar_seed.final_analysis --rotation --start "2024-01-01"

# Render sun images (with timezone support)
uv run python -m solar_seed.render_sun --date "08.03.2012" --time "14:00" --timezone Europe/Berlin

# Generate publication figures
uv run python -m solar_seed.visualize --output figures/

# Generate PDF for arXiv submission
uv run python scripts/generate_pdf.py

# Prominence eruption analysis (Sept 13-14, 2013)
uv run python scripts/prominence_2013-09-13.py           # Compare both prominences
uv run python scripts/prominence_2013-09-13.py --single  # First prominence only

# Early Warning System (Typer + Rich CLI)
uv run python scripts/early_warning.py location                 # Set your location (interactive)
uv run python scripts/early_warning.py location berlin          # Set location directly
uv run python scripts/early_warning.py check -m                 # Minimal operator view
uv run python scripts/early_warning.py check -c                 # Full scientific dashboard
uv run python scripts/early_warning.py check -m -l berlin       # Minimal + personal relevance
uv run python scripts/early_warning.py check -c -s              # Full + STEREO-A
uv run python scripts/early_warning.py monitor -c -i 300        # 5-min monitoring
uv run python scripts/early_warning.py stats                    # Database statistics
uv run python scripts/early_warning.py correlations             # Coupling-flare analysis
uv run python scripts/early_warning.py show-predictions         # Show predictions + verification
uv run python scripts/early_warning.py export                   # Export all tables to CSV
uv run python scripts/early_warning.py import-flares            # Import M/X flares from NASA DONKI
```

## Interactive CLI

Start with `./solar-seed` for a user-friendly menu:

```
  [1]  Quick Test (synthetic data, ~2 min)
  [2]  Multi-Channel Analysis (21 wavelength pairs)
  [3]  Rotation Analysis (segment-based, scalable)
  [4]  Flare Analysis (X9.0 Event)
  [5]  Render Sun Images (download + visualize)
  [6]  Early Warning System (real-time monitoring)
  [7]  Status: Check running analysis
  [8]  View Results
```

Features:
- **Segment-based analysis**: Each day analyzed independently, scalable to 100+ days
- Automatic segment detection with extend/aggregate options
- Legacy checkpoint conversion support
- Auto-push segments for cross-system resume (`--auto-push`)
- Progress bar and status display
- Timezone conversion for local times

## Architecture

```
src/solar_seed/
├── cli.py               # Interactive CLI menu (./solar-seed)
├── render_sun.py        # Sun image rendering with timezone support
├── mutual_info.py       # Core: MI(X,Y) = H(X) + H(Y) - H(X,Y)
├── null_model.py        # Shuffle-based null model
├── radial_profile.py    # Radial geometry subtraction
├── spatial_analysis.py  # 8x8 grid MI maps
├── control_tests.py     # C1-C4 + sector-ring shuffle
├── real_run.py          # Reproducible pipeline with CSV/JSON output
├── hypothesis_test.py   # Main test runner
├── collector.py         # Time series collector
├── multichannel.py      # 7-channel coupling matrix (21 pairs), AIA data loading
├── final_analysis.py    # Timescale + activity + 27-day rotation analysis
├── visualize.py         # Publication figure generation (Figures 1-5)
├── monitoring/          # Early Warning System components
│   ├── db.py            # SQLite monitoring database
│   ├── coupling.py      # CouplingMonitor with baselines & trend analysis
│   ├── detection.py     # AnomalyStatus, BreakType, break detection
│   ├── formatting.py    # Rich terminal output (StatusFormatter)
│   ├── validation.py    # ROI variance, MI measurement validation
│   ├── constants.py     # AnomalyLevel, Phase, thresholds
│   └── relevance.py     # Personal relevance (day/night, aurora)
└── data_sources/        # Data loading modules
    ├── aia.py           # SDO/AIA full-res via VSO
    ├── synoptic.py      # AIA synoptic (1k, direct JSOC access)
    └── stereo.py        # STEREO-A/EUVI loader
```

**Key Metrics:**
- `MI Ratio` = MI_residual / MI_original (what survives geometry removal)
- `ΔMI_ring` = MI_residual - MI_ring_shuffled (structure beyond radial stats)
- `ΔMI_sector` = MI_residual - MI_sector_shuffled (true local structure)

**Shuffle Hierarchy:**
```
MI_global < MI_ring < MI_sector < MI_original
   ↓          ↓          ↓          ↓
 noise    radial    azimuthal    local
```

## AIA Channels

| Channel | Temperature | Region |
|---------|-------------|--------|
| 304 Å | 0.05 MK | Chromosphere |
| 171 Å | 0.6 MK | Quiet Corona |
| 193 Å | 1.2 MK | Corona |
| 211 Å | 2.0 MK | Active Regions |
| 335 Å | 2.5 MK | Active Regions (hot) |
| 94 Å | 6.3 MK | Flares |
| 131 Å | 10 MK | Flares (very hot) |

## Control Tests (C1-C4)

| Test | Purpose | Pass Criterion |
|------|---------|----------------|
| C1: Time-Shift | Temporal decoupling | >50% MI reduction |
| C2: Ring/Sector | Radial+azimuthal hierarchy | Hierarchy confirmed |
| C3: PSF/Blur | Resolution sensitivity | <20% MI change |
| C4: Co-Alignment | Registration check | Max at (0,0) |

## Publication Figures

```
figures/
├── figure1_geometric_normalization.png  # MI before/after radial normalization
├── figure2_spatial_distribution.png     # 8x8 spatial MI maps with hotspots
├── figure3_null_model_decomposition.png # Shuffle hierarchy bar chart
├── figure4_coupling_matrix.png          # 7x7 ΔMI_sector heatmap
└── figure5_flare_phases.png             # X9.0 flare coupling evolution
```

## Output Files

```
results/
├── real_run/
│   ├── timeseries.csv          # timestamp, mi_original, mi_residual, etc.
│   ├── controls_summary.json   # C1-C4 results
│   ├── spatial_maps.txt        # ASCII MI maps + hotspots
│   └── run_metadata.json       # Config for reproducibility
├── multichannel/
│   ├── coupling_matrices.txt   # 7x7 ΔMI_sector matrix
│   ├── coupling_matrices.json  # Machine-readable
│   ├── pair_results.csv        # All 21 pairs ranked
│   └── temperature_coupling.txt
├── multichannel_real/          # Same structure, real AIA data
├── flare/
│   ├── flare_analysis.txt      # X9.0 flare phase comparison
│   └── flare_analysis.json     # Machine-readable
├── rotation/
│   ├── segments/               # Segment-based analysis (1 file per day)
│   │   ├── 2025-12-01.json     # Day 1: raw data + daily stats
│   │   ├── 2025-12-02.json     # Day 2 ...
│   │   └── ...
│   ├── rotation_analysis.txt   # Aggregated results
│   ├── rotation_analysis.json
│   ├── coupling_evolution.csv  # Time series for all pairs
│   └── checkpoint.json         # Legacy checkpoint (optional)
├── final/
│   ├── timescale_comparison.txt/json
│   └── activity_conditioning.txt/json
├── prominence/
│   ├── prominence_2013-09-13.json      # Single event analysis
│   └── double_prominence_comparison.json  # Both events compared
└── early_warning/
    ├── monitoring.db             # SQLite database (persistent storage)
    └── coupling_history.json     # 24h coupling residual history (JSON backup)
```

## Sun Image Rendering

Render beautiful AIA sun images with local time and UTC labels:

```bash
uv run python -m solar_seed.render_sun --date "08.03.2012" --time "14:00" --timezone Europe/Berlin
```

**Timezone Support:**
- Input: Local date/time (DD.MM.YYYY HH:MM) + timezone (Europe/Berlin, America/New_York, etc.)
- Output: Image labeled with both local time and UTC
- Example label: `08.03.2012 14:00 Berlin (13:00 UTC)`

**Output files:** `images/sun_2012-03-08_composite.png`, `sun_2012-03-08_193A.png`, etc.

## AIA Data Download

Downloads use **mirror fallback** for reliability:
1. **JSOC** (Stanford) - Primary source
2. **ROB** (Royal Observatory of Belgium)
3. **SDAC** (NASA)
4. **CfA** (Harvard)

If one server fails, the next mirror is tried automatically.

## Rotation Analysis

### Segment-Based Architecture (Recommended)

The **segment-based** approach analyzes each day independently, enabling:
- **Scalability**: Analyze 16, 27, or 100+ days without memory issues
- **Parallelization**: Run multiple days on different machines
- **Fault tolerance**: Only lose one day on failure, not entire analysis
- **Extensibility**: Add more days later without re-processing

```bash
# Analyze date range (each day = one segment)
uv run python -m solar_seed.final_analysis --segments --start 2025-12-01 --end 2025-12-27

# Analyze single day
uv run python -m solar_seed.final_analysis --segment 2025-12-15

# Aggregate all available segments
uv run python -m solar_seed.final_analysis --aggregate

# Convert legacy checkpoint to segments
uv run python -m solar_seed.final_analysis --convert-checkpoint
```

**Output structure:**
```
results/rotation/
├── segments/
│   ├── 2025-12-01.json  # Day 1 raw data + stats
│   ├── 2025-12-02.json  # Day 2 ...
│   └── ...
├── rotation_analysis.json   # Aggregated results
└── coupling_evolution.csv   # Full time series
```

### Legacy: Monolithic Analysis

For backward compatibility, the old checkpoint-based approach is still available:
- Checkpoint saved after every timepoint to `results/rotation/checkpoint.json`
- Use `--no-resume` to force fresh start
- FITS files are deleted after loading to save disk space

## Early Warning System

Multi-layer solar activity monitoring with pre-flare detection (Typer + Rich CLI):

```
STEREO-A (51° ahead)          → 2-4 days warning (active regions)
         ↓
┌─────────────────────────┐
│  ΔMI COUPLING MONITOR   │   → Hours before flare
│  Residual r(t) tracking │     (coupling drops 25-47% during flares)
│  Theil-Sen trend analysis│
└─────────────────────────┘
         ↓
GOES X-ray + DSCOVR           → Minutes to real-time
```

**CLI Commands (Typer):**

| Command | Description |
|---------|-------------|
| `check` | 🔍 Single status check (use `-m` for minimal, `-c` for coupling) |
| `monitor` | 📡 Continuous monitoring with periodic updates |
| `location` | 📍 Set/show your location for personal relevance |
| `stats` | 📊 Show database statistics |
| `correlations` | 📈 Show coupling-flare correlations |
| `show-predictions` | 📋 Show predictions with verification status |
| `export` | 💾 Export database tables to CSV |
| `import-flares` | ⬇️ Import M/X flares from NASA DONKI |

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--coupling` | `-c` | Include SDO/AIA coupling analysis |
| `--stereo` | `-s` | Include STEREO-A EUVI (~3.9 days ahead) |
| `--minimal` | `-m` | Minimal operator view (only actionable info) |
| `--location` | `-l` | Show personal relevance (or use `location` command) |
| `--interval` | `-i` | Monitoring interval in seconds (default: 60) |
| `--no-db` | | Disable database storage |

**Display Modes:**

| Mode | Command | Shows |
|------|---------|-------|
| Minimal | `check -m` | 193-211 status, trend, CLEAR/CAUTION/BREAK |
| Full | `check -c` | All channels, Anomaly levels, Phase, GOES, Solar Wind |
| Personal | `check -m -l berlin` | + Day/night status, radio/GPS risk |

**Anomaly Level (Statistical, based on |z|):**
- `NORMAL`: |z| < 2σ
- `ELEVATED`: 2-4σ
- `STRONG`: 4-7σ
- `EXTREME`: > 7σ

**Phase (Interpretive, rule-based):**

| Phase | Icon | Meaning | Trigger |
|-------|------|---------|---------|
| `BASELINE` | 🟢 | Thermal & structural quiet | GOES quiet, \|z\| < 3 |
| `ELEVATED-QUIET` | 🟢 | Structurally active but stable | \|z\| > 3, stable trends |
| `POST-EVENT` | 🟣 | Non-flaring but reorganizing | GOES quiet, \|z\| > 5 |
| `RECOVERY` | 🟡 | Decaying activity | GOES falling |
| `PRE-FLARE` | ⚠️ | Destabilization detected | Negative z + GOES rising |
| `ACTIVE` | 🔴 | Ongoing energy release | GOES M/X-class |

**Parallel Classification:**
The system runs two classifiers in parallel for empirical validation:
- **GOES-only**: Traditional flux-based (current operational standard)
- **ΔMI-integrated**: Experimental coupling-based (may detect events GOES misses)

Divergences are logged to `phase_divergence` table for correlation analysis.

**Personal Relevance (--location):**
- Day side: Radio/GPS effects affect you NOW (~8 min latency)
- Night side: Only geomagnetic effects (15-48h latency)
- Aurora possible at high latitudes when Kp ≥ 7
- Location saved to `~/.config/solar-seed/location.txt`

**Data Sources:**
- **GOES X-ray**: Flare classification (A/B/C/M/X classes)
- **DSCOVR L1**: Solar wind speed, density, Bz (geomagnetic risk)
- **SDO/AIA**: ΔMI coupling analysis (pre-flare anomaly detection)
- **STEREO-A EUVI**: 51° ahead of Earth (~3.9 days advance view)
- **NOAA SWPC**: Space weather alerts

**Minimal Mode Status (for operators):**
| Status | Condition | Action |
|--------|-----------|--------|
| 🟢 CLEAR | Coupling nominal, trend stable | No action |
| 🟡 CAUTION | Coupling declining or GOES rising | Watch closely |
| 🔴 BREAK DETECTED | Coupling break + accelerating down | Monitor for flare 0.5-2h |

**Baselines (from 8-day analysis):**
- 193-211 Å: 0.59 ± 0.12 bits
- 193-304 Å: 0.07 ± 0.02 bits
- 171-193 Å: 0.17 ± 0.04 bits

**Trend Analysis (Theil-Sen robust estimator):**
```
193-211 Å: 0.590 bits  r=+0.0σ  ✓ NORMAL
         Trend: → STABLE (+0.5%/h)
                ●● medium confidence | n=6 | 50min window | Theil-Sen
```

### Formal Coupling Break Detection

A **Coupling Break** is detected using robust statistics:
```
Break criterion: ΔMI(t) < median(60min) - k × MAD(60min)
Default k = 2.0 (~95% interval)
```

Output shows z_mad (MADs below median):
```
⚠ COUPLING BREAK detected in 193-211:
   Criterion: ΔMI < median - 2.0×MAD = 0.4823
   Current: 0.4150, Deviation: 2.3 MAD below median
```

### Reviewer-Proof Validation (Artifact Tests)

Three validation checks rule out instrumental artifacts:

| Test | Method | Pass Criterion |
|------|--------|----------------|
| A: Time Sync | Channel timestamp spread | < 60s between channels |
| B: Registration | FFT cross-correlation shift | < 10px shift |
| C: Robustness | 2×2 binning recompute | < 20% MI change |

**Validation Output:**
```
VALIDATION STATUS (Reviewer-Proof)
----------------------------------------
193-211: ✓ VALIDATED BREAK at 0.4150
  Criterion: ΔMI < median - 2.0×MAD = 0.4823
  Deviation: 2.3 MAD below median
  ✓ Registration: OK (1.2px shift)
  ✓ Time sync: OK (12s spread)
  ✓ Robustness: STABLE under 2x2 binning (-3.1% change)

193-304: ✗ VETOED (reason: robustness)
  Deviation: 1.8 MAD below median
  Binning change: 130.6% (>20% = unreliable)
```

**Robustness Veto:** Channels with >20% binning sensitivity are marked UNRELIABLE and excluded from break decisions.

### Sudden Drop Detector

Detects relative drops even when ΔMI is above baseline threshold:

```
Reference: max(last 3 readings)
Drop %: (current - reference) / reference

MODERATE: > 15% drop → status ELEVATED
SEVERE:   > 25% drop → status ELEVATED
```

**Use Case:** M3 pre-flare at 22:09 showed ΔMI=0.714 (above baseline 0.59) but was a 25% drop from recent max 0.953. Without Sudden Drop Detector, this would be missed.

### Special Detection - TRANSFER_STATE

- Triggered when: 193-304 rising (>+3%/h) AND 193-211 falling (<-3%/h)
- Interpretation: Chromospheric anchor strengthening while coronal coupling weakens
- May indicate magnetic stress buildup / energy reorganization

### Paper-Ready Claim Format

```
An abrupt drop in the coronal coupling metric (ΔMI 193–211) was detected
10 min before a 66% GOES flux increase. The break passed validation:
registration shift 1.2px, time sync 12s, stable under 2×2 binning (−3%).
```

### SQLite Database (`monitoring.db`) - Schema v0.6

Location: `results/early_warning/monitoring.db`

**Schema Version:** v0.6 (paper-grade with dimension tables)

**Dimension Tables (Lookup):**

| Table | Purpose |
|-------|---------|
| `channels` | AIA wavelength lookup (304, 171, 193, 211, 335, 94, 131 Å) |
| `pairs` | Normalized pair lookup (ch_a < ch_b constraint) |

**Reproducibility Tables:**

| Table | Purpose |
|-------|---------|
| `runs` | Analysis runs with start/end time, version |
| `run_config` | Key-value config per run (thresholds, intervals) |

**Observation Tables:**

| Table | Purpose |
|-------|---------|
| `goes_xray` | GOES X-ray flux measurements |
| `solar_wind` | DSCOVR solar wind data (speed, Bz, density) |
| `coupling_measurements` | ΔMI readings with pair_id FK, quality flags, run_id |
| `flare_events` | Ground truth flares (M/X class from NOAA) |
| `predictions` | Predictions with trigger_kind, valid_from/to, trigger_measurement_id |
| `prediction_matches` | Many-to-many prediction↔flare evaluation |
| `noaa_alerts` | NOAA Space Weather alerts with Kp tracking |
| `phase_divergence` | GOES-only vs ΔMI-integrated classifier divergences |

**Channels Table (auto-seeded):**
```sql
SELECT * FROM channels;
-- id | wavelength | instrument | name         | temperature_mk | region
-- 1  | 94         | AIA        | Fe XVIII     | 6.3            | Flares
-- 2  | 131        | AIA        | Fe VIII/XXI  | 10.0           | Flares (very hot)
-- 3  | 171        | AIA        | Fe IX        | 0.6            | Quiet Corona
-- ...
```

**Pairs Table (normalized):**
```sql
SELECT * FROM pairs;
-- id | ch_a_id | ch_b_id | pair_name
-- 1  | 1       | 2       | 94-131
-- 2  | 1       | 3       | 94-171
-- ...
-- All 21 pairs, guaranteed ch_a_id < ch_b_id
```

**coupling_measurements Quality Fields:**
```sql
-- New fields for reviewer-proof validation
quality_ok BOOLEAN     -- All validation checks passed
robustness_score REAL  -- Binning sensitivity (< 0.20 = good)
sync_delta_s REAL      -- Time spread between channels (< 60s = good)
run_id INTEGER         -- FK to runs table
pair_id INTEGER        -- FK to pairs table
```

**Predictions with trigger_kind:**
```sql
-- Enhanced trigger tracking for paper analysis
trigger_kind TEXT      -- Z_SCORE_SPIKE, SUDDEN_DROP, BREAK, TREND, THRESHOLD, TRANSFER_STATE
trigger_value REAL     -- Actual value that triggered (e.g., -0.25 for 25% drop)
trigger_threshold REAL -- Threshold used (e.g., -0.15 for MODERATE sudden drop)
trigger_measurement_id INTEGER  -- FK to exact coupling_measurement
valid_from TEXT        -- ISO timestamp when prediction window starts
valid_to TEXT          -- ISO timestamp when window expires (default +90min)
```

**trigger_kind Values (auto-set on ALERT/ELEVATED):**
| Kind | Priority | Meaning |
|------|----------|---------|
| `SUDDEN_DROP` | 1 | 15-25%+ drop from recent max |
| `BREAK` | 2 | z_mad > 2.0 (formal coupling break) |
| `THRESHOLD` | 3 | deviation_pct < -15% or -25% |
| `TREND` | 4 | DECLINING or ACCELERATING_DOWN |
| `TRANSFER_STATE` | 5 | 193-304 rising + 193-211 falling |
| `Z_SCORE_SPIKE` | 6 | z > 4σ (extreme spike) |

**NOAA Alerts with Kp Tracking:**
```
message_code: WARK04, ALTK06, ALTEF3, etc.
alert_type:   WARNING, ALERT, WATCH, SUMMARY, FORECAST
kp_observed:  Current Kp index (0-9)
kp_predicted: Predicted Kp from message code
g_scale:      Geomagnetic storm scale (G0-G5)
s_scale:      Solar radiation scale (S0-S5)
r_scale:      Radio blackout scale (R0-R5)
source_region: Active region (e.g., AR3842)
```

**Prediction Evaluation:**
```sql
-- Many-to-many matching for proper metrics
SELECT p.prediction_time, p.trigger_kind, p.trigger_value,
       f.start_time, m.match_type, m.time_to_peak_min
FROM predictions p
JOIN prediction_matches m ON p.id = m.prediction_id
JOIN flare_events f ON f.id = m.flare_event_id
WHERE m.match_type = 'hit';
```

**Divergence Analysis:**
```bash
uv run python -m solar_seed.monitoring.db --divergence 7  # Last 7 days
```

**Schema Migrations:**
Migrations run automatically on startup. The database self-upgrades from any previous version to v0.6.

## Development Guidelines

- Python >=3.12, <3.14 (avoid unstable 3.14 dev versions)
- Core: numpy, scipy
- Optional: sunpy, aiapy for real data; matplotlib for visualization
- Scientific rigor: null models, control tests, reproducible seeds
