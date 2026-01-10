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
```

## Interactive CLI

Start with `./solar-seed` for a user-friendly menu:

```
  [1]  Quick Test (synthetic data, ~2 min)
  [2]  Multi-Channel Analysis (21 wavelength pairs)
  [3]  Rotation Analysis (segment-based, scalable)
  [4]  Flare Analysis (X9.0 Event)
  [5]  Render Sun Images (download + visualize)
  [6]  Status: Check running analysis
  [7]  View Results
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
└── visualize.py         # Publication figure generation (Figures 1-5)
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
└── prominence/
    ├── prominence_2013-09-13.json      # Single event analysis
    └── double_prominence_comparison.json  # Both events compared
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

## Development Guidelines

- Python >=3.12, <3.14 (avoid unstable 3.14 dev versions)
- Core: numpy, scipy
- Optional: sunpy, aiapy for real data; matplotlib for visualization
- Scientific rigor: null models, control tests, reproducible seeds
