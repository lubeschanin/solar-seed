# Solar Seed

**Geometry-controlled mutual information reveals temperature-ordered coupling in the solar atmosphere**

## Abstract

Understanding how different thermal layers of the solar atmosphere are coupled is central to solar physics and space-weather prediction. While correlations between extreme-ultraviolet (EUV) channels are well known, disentangling genuine physical coupling from geometric and statistical confounders remains challenging.

We introduce a **geometry-controlled mutual information framework** to quantify multichannel coupling in SDO/AIA data. By systematically removing disk geometry, radial intensity statistics, and coarse azimuthal structure through a hierarchy of null models, we isolate a residual **local coupling component**.

Applying this method to seven EUV channels spanning chromospheric to flare temperatures, we find that **neighboring temperature channels exhibit significantly stronger local coupling than thermally distant pairs**. This temperature-ordered structure is stable over time, survives time-shift and alignment controls, and is spatially localized to active regions.

## Key Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| MI Ratio | 30.8% ± 0.7% | ~31% of MI survives geometry removal |
| ΔMI_sector | 0.17 bits | Local structure coupling |
| Z-Score | 1252 ± 146 | p < 10⁻¹⁰⁰ (highly significant) |
| Time-shift control | >95% reduction | Confirms temporal coherence |

### Temperature-Ordered Coupling

Strongest local coupling between thermally adjacent layers:
- **193-211 Å** (1.2-2.0 MK): ΔMI_sector = 0.73 bits
- **171-193 Å** (0.6-1.2 MK): ΔMI_sector = 0.39 bits

Chromospheric (304 Å) and flare channels (94, 131 Å) show weaker, activity-dependent coupling.

## Methods

### Hierarchy of Null Models

```
MI_global < MI_ring < MI_sector < MI_original
    ↓          ↓          ↓           ↓
  noise    radial    azimuthal     local
```

| Component | Calculation | Meaning |
|-----------|-------------|---------|
| Radial | MI_ring − MI_global | Disk geometry |
| Azimuthal | MI_sector − MI_ring | Coarse angular structure |
| Local | MI_original − MI_sector | Genuine spatial coupling |

### Mutual Information

```
I(X;Y) = Σ p(x,y) log₂ [p(x,y) / p(x)p(y)]
```

Estimated via histogram discretization (64 bins), reported in bits.

### Geometric Normalization

Per-frame radial mean profile removal:
```
R(r,θ) = I(r,θ) / ⟨I(r)⟩
```

## AIA Channels

| Wavelength | Temperature | Region |
|------------|-------------|--------|
| 304 Å | 0.05 MK | Chromosphere |
| 171 Å | 0.6 MK | Quiet Corona |
| 193 Å | 1.2 MK | Corona |
| 211 Å | 2.0 MK | Active Regions |
| 335 Å | 2.5 MK | Hot Active Regions |
| 94 Å | 6.3 MK | Flares |
| 131 Å | 10 MK | Flares |

## Installation

```bash
git clone https://github.com/lubeschanin/solar-seed.git
cd solar-seed
uv sync

# For real solar data
uv pip install sunpy aiapy
```

## Usage

```bash
# Hypothesis test with controls
uv run python -m solar_seed.hypothesis_test --spatial --controls

# Reproducible run with reports
uv run python -m solar_seed.real_run --hours 6 --synthetic

# Multi-channel analysis (7 channels, 21 pairs)
uv run python -m solar_seed.multichannel --hours 24

# With real AIA data
uv run python -m solar_seed.multichannel --real --hours 1 --start "2024-01-15T12:00:00"

# Final analyses
uv run python -m solar_seed.final_analysis
```

## Control Tests

| Test | Purpose | Result |
|------|---------|--------|
| C1: Time-Shift | Temporal decoupling | >95% MI reduction |
| C2: Ring/Sector | Shuffle hierarchy | Confirmed |
| C3: PSF/Blur | Resolution sensitivity | <20% at σ=1px |
| C4: Co-Alignment | Registration check | Peak at (0,0) |

## Output

```
results/
├── real_run/
│   ├── timeseries.csv          # MI time series
│   ├── controls_summary.json   # C1-C4 results
│   └── spatial_maps.txt        # MI maps + hotspots
├── multichannel/
│   ├── coupling_matrices.txt   # 7×7 coupling matrix
│   ├── pair_results.csv        # All 21 pairs ranked
│   └── temperature_coupling.txt
└── final/
    ├── timescale_comparison.txt
    └── activity_conditioning.txt
```

## Project Structure

```
src/solar_seed/
├── mutual_info.py       # MI computation (pure NumPy)
├── null_model.py        # Shuffle-based null models
├── radial_profile.py    # Radial profile subtraction
├── spatial_analysis.py  # Spatial MI maps
├── control_tests.py     # C1-C4 + sector shuffle
├── real_run.py          # Reproducible pipeline
├── hypothesis_test.py   # Main test runner
├── collector.py         # Time series collector
├── multichannel.py      # 7-channel coupling matrix
└── final_analysis.py    # Timescale + activity analysis
```

## Limitations

- MI quantifies dependence, not causality
- Histogram MI introduces finite-sample bias (mitigated by null-model comparisons)
- Radial normalization assumes approximate radial symmetry
- ΔMI_sector does not identify mechanisms; additional diagnostics required

## Data Sources

- **NASA SDO**: https://sdo.gsfc.nasa.gov/
- **AIA Level 1.5**: Via SunPy/aiapy
- **ML Dataset**: https://registry.opendata.aws/sdoml-fdl/

## Citation

If you use this code, please cite:

> Geometry-controlled mutual information reveals temperature-ordered coupling in the solar atmosphere. (2024)

## License

GNU General Public License v3.0

---

*Careful separation of geometric, statistical, and local contributions is essential for interpreting multichannel dependencies.*
