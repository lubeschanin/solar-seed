# Geometry-controlled mutual information reveals temperature-ordered coupling in the solar atmosphere

**Draft manuscript**

---

## Abstract

Understanding how different thermal layers of the solar atmosphere are coupled is central to solar physics and space-weather prediction. While correlations between extreme-ultraviolet (EUV) channels are well known, disentangling genuine physical coupling from geometric and statistical confounders remains challenging.

Here we introduce a geometry-controlled mutual information framework to quantify multichannel coupling in Solar Dynamics Observatory / Atmospheric Imaging Assembly (SDO/AIA) data. By systematically removing disk geometry, radial intensity statistics, and coarse azimuthal structure through a hierarchy of null models, we isolate a residual local coupling component.

Applying this method to seven EUV channels spanning chromospheric to flare temperatures, we construct a coupling matrix across thermal layers of the solar atmosphere. We find that neighboring temperature channels exhibit significantly stronger local coupling than thermally distant pairs. This temperature-ordered structure is stable over time, survives time-shift and alignment controls, and is spatially localized to active regions rather than disk geometry.

Our results demonstrate that information-theoretic coupling, when properly controlled for geometric effects, reveals an intrinsic organization of the solar atmosphere consistent with magnetically mediated interactions between adjacent thermal layers. The presented framework provides a general, reproducible approach for analyzing multichannel structure in complex astrophysical systems.

---

## 1. Introduction

The solar atmosphere is a highly structured, multi-layered plasma system spanning several orders of magnitude in temperature, from the chromosphere to the hot flaring corona. Understanding how these thermal layers are coupled is essential for explaining energy transport, magnetic reconnection, and the emergence of solar activity that drives space weather.

Observations from the Solar Dynamics Observatory's Atmospheric Imaging Assembly (SDO/AIA) provide simultaneous imaging of the Sun in multiple extreme-ultraviolet (EUV) wavelengths, each sensitive to plasma at characteristic temperatures. While correlations between AIA channels are well documented, interpreting such correlations as evidence of physical coupling remains challenging. Apparent multichannel similarity can arise from disk geometry, limb brightening, shared radial intensity profiles, instrumental effects, or global morphological structure, rather than genuine interaction between thermal layers.

Traditional approaches to multichannel analysis in solar physics rely on intensity correlations, emission measures, or magnetic field extrapolations. While powerful, these methods do not explicitly separate geometric and statistical confounders from local, physically meaningful coupling. As a result, it remains unclear to what extent observed cross-channel structure reflects intrinsic organization of the solar atmosphere versus projection and sampling effects.

Here we introduce a geometry-controlled mutual information framework designed to isolate genuine multichannel coupling in solar imagery. Mutual information (MI), unlike linear correlation, captures arbitrary statistical dependence between channels, but must be applied with care in spatially structured systems. We therefore combine MI estimation with a hierarchy of null models that progressively remove disk geometry, radial statistics, and coarse azimuthal structure.

Applying this framework to seven AIA EUV channels spanning chromospheric to flare temperatures, we construct a coupling matrix of the solar atmosphere. We show that neighboring thermal layers exhibit significantly stronger local coupling than thermally distant layers, and that this structure is stable over time, spatially localized, and robust to extensive controls. These results reveal an intrinsic, temperature-ordered organization of the solar atmosphere consistent with magnetically mediated interactions between adjacent thermal layers.

---

## 2. Data

### 2.1 Observations

We use data from the Atmospheric Imaging Assembly (AIA) aboard the Solar Dynamics Observatory (SDO). Specifically, we analyze the AIA Synoptic data product (Level 1.5), which provides calibrated, co-registered full-disk solar images in multiple EUV wavelengths at a spatial resolution of 1024 × 1024 pixels.

The synoptic data are free of JPEG compression and visualization artefacts and are suitable for quantitative information-theoretic analysis.

### 2.2 EUV Channels

| Wavelength (Å) | Peak Temperature (MK) | Dominant Emission |
|----------------|----------------------|-------------------|
| 304 | 0.05 | Chromosphere / transition region |
| 171 | 0.6 | Quiet corona |
| 193 | 1.2 | Corona |
| 211 | 2.0 | Active regions |
| 335 | 2.5 | Hot active regions |
| 131 | ~10 | Flares |
| 94 | ~6 | Flares |

### 2.3 Time Windows

We analyze multiple temporal windows, including (i) a 6-hour interval (30 timepoints, 12-minute cadence) used for detailed method validation, and (ii) a 24-hour interval used for multi-channel coupling analysis. All timestamps are handled in UTC. Frames with missing data or quality flags are excluded. Run parameters are logged to ensure reproducibility.

### 2.4 Preprocessing

For each frame, we apply minimal preprocessing: FITS loading and NaN handling, robust intensity scaling, and solar disk masking. No per-frame auto-normalization or contrast enhancement is applied. All preprocessing parameters are fixed and recorded.

---

## 3. Methods

### 3.1 Mutual Information Estimation

We quantify statistical dependence between pairs of EUV channels using mutual information (MI), estimated via histogram discretization with 64 bins and fixed bin ranges. MI is reported in bits.

$$
I(X;Y) = \sum_{x,y} p(x,y) \log_2 \frac{p(x,y)}{p(x)p(y)}
$$

### 3.2 Geometric Normalization

To remove disk geometry and radial intensity gradients, we estimate the per-frame radial mean profile and normalize each image. The residual is defined as:

$$
R(r,\theta) = \frac{I(r,\theta)}{\langle I(r) \rangle}
$$

### 3.3 Hierarchy of Null Models

- **Global shuffle**: permute all pixels (preserve histogram, destroy structure)
- **Ring-wise shuffle**: shuffle within annuli (preserve radial statistics)
- **Sector–ring shuffle**: shuffle within radial+azimuth sectors (preserve coarse geometry)
- **Time-shift null**: pair with large temporal offset (destroy time-coherent overlap)

### 3.4 Local Coupling Metric

Local coupling beyond geometry and coarse structure is quantified as:

$$
\Delta\text{MI}_{\text{sector}}^{(i,j)} = \text{MI}(R_i, R_j) - \text{MI}_{\text{sector-null}}(R_i, R_j)
$$

### 3.5 Spatial Analysis

Images are divided into an 8×8 grid and MI is computed per cell to obtain spatial MI maps and identify hotspots.

### 3.6 Statistical Testing

Null distributions are generated via repeated shuffles. Z-scores and p-values are computed per timepoint and aggregated over time.

---

## 4. Results

### 4.1 Global versus Geometry-Controlled Coupling

We compute MI on original images and on geometry-normalized residuals. Across a 6-hour interval (30 timepoints), the ratio $\text{MI}_{\text{residual}}/\text{MI}_{\text{original}}$ remains stable at **30.8% ± 0.7%** (CV = 2.3%), indicating that most apparent coupling is geometric, while a consistent residual component remains after removal.

The residual coupling is extremely significant relative to shuffle-based null models, with minimum Z-scores exceeding Z = 986 and mean Z = 1252 ± 146 ($p < 10^{-100}$).

### 4.2 Spatial Localization of Residual Coupling

Spatial MI maps (8×8 grid) show strong limb domination in original images. After geometric normalization, limb bias is removed and localized hotspots remain. Mean spatial MI decreases by ~42%, while local maxima persist and collapse under time-shift controls.

### 4.3 Hierarchy of Null Models and Structure Decomposition

Applying the hierarchy of null models yields the consistent ordering:

$$
\text{MI}_{\text{global}} < \text{MI}_{\text{ring}} < \text{MI}_{\text{sector}} < \text{MI}_{\text{residual}}
$$

This enables decomposition into radial, azimuthal, and local structure contributions, with a typical local contribution $\Delta\text{MI}_{\text{sector}} \approx 0.17$ bits.

### 4.4 Temperature-Ordered Coupling Across EUV Channels

Extending to seven EUV channels (21 pairs), we find temperature-ordered local coupling: thermally adjacent channels exhibit substantially stronger $\Delta\text{MI}_{\text{sector}}$ than thermally distant pairs. The strongest links occur among coronal channels (e.g., 193–211, 171–193), while chromospheric (304) and flare channels (94, 131) show weaker and more episodic coupling.

### 4.5 Robustness and Temporal Stability

Time-shift controls reduce coupling by >95%, alignment checks peak at (0,0), and scale-response tests indicate coupling dominated by mid-to-large spatial scales.

---

## 5. Discussion

### 5.1 Physical Interpretation of Temperature-Ordered Coupling

The key result is a temperature-ordered pattern of local coupling that persists after removing geometric and statistical confounders. This ordering is consistent with a stratified atmosphere in which magnetic connectivity, heating, and transport organize neighboring thermal layers more strongly than distant ones.

### 5.2 Relation to Existing Solar Diagnostics

The framework complements correlation- and emission-measure-based diagnostics by capturing nonlinear dependence while explicitly controlling geometry. It provides an orthogonal measurement: how strongly layers are locally organized together, independent of disk-scale morphology.

### 5.3 Channel-Specific Behavior

Chromospheric 304 Å shows weaker coupling to coronal channels, consistent with different plasma regimes. Flare channels (94, 131) show moderate, activity-dependent coupling patterns, consistent with episodic heating rather than persistent structural organization.

### 5.4 Temporal Stability and Implications

The stability of $\Delta\text{MI}_{\text{sector}}$ suggests a measurable baseline of atmospheric organization. This study does not claim predictivity; however, deviations from baseline may be informative for event-focused analyses.

### 5.5 Methodological Generality

The hierarchical null-model approach is general: decompose multichannel dependence by removing dominant structural contributions before interpretation.

### 5.6 Scope and Non-Claims

We do not infer causality or directionality. The metric measures shared structure, not energy transfer pathways.

---

## 6. Limitations and Outlook

### 6.1 Methodological Limitations

MI quantifies dependence but not causality. Histogram MI introduces finite-sample bias and depends on binning; null-model comparisons mitigate this. Radial normalization assumes approximate radial symmetry for disk-scale structure and provides conservative removal of geometry.

### 6.2 Interpretation Limits

$\Delta\text{MI}_{\text{sector}}$ does not identify mechanisms. Additional diagnostics (e.g., magnetograms) are required to connect coupling to topology and transport. Flare channels can produce episodic behavior.

### 6.3 Temporal and Observational Scope

Extending beyond 24 hours to full solar rotations and solar-cycle phases is future work. Higher-cadence datasets may reveal finer structure but require careful handling of exposure and gaps.

### 6.4 Outlook

1. **Long-term studies:** coupling evolution across rotations and the solar cycle
2. **Event-conditioned analysis:** baseline deviations around flares/CMEs
3. **Magnetic integration:** relate coupling to magnetogram-derived topology
4. **Method extensions:** alternative MI estimators and multiscale measures

### 6.5 Concluding Remarks

Careful separation of geometric, statistical, and local contributions is essential for interpreting multichannel dependencies. The presented framework provides a reproducible basis for future multichannel analyses.

---

## Figures

### Figure 1 | Effect of geometric normalization on multichannel mutual information

Global mutual information (MI) between AIA EUV channels before and after geometric normalization. Left: MI computed on original images, dominated by disk geometry and limb brightening. Right: MI computed on residual images after radial profile normalization. Approximately 70% of apparent MI is removed, while a stable residual component remains. Error bars indicate standard deviation across 30 timepoints.

### Figure 2 | Spatial distribution of multichannel coupling

Spatial maps of mutual information between AIA channels (193 Å and 211 Å). Left: MI computed on original images, showing strong limb-dominated structure. Right: MI computed on geometry-normalized residuals. Limb bias is removed, revealing localized regions of enhanced coupling. Values are shown in bits. Hotspots correspond to active regions and persist across timepoints.

### Figure 3 | Decomposition of multichannel coupling using null models

Mutual information values under progressively restrictive null models. Global shuffle destroys all spatial structure; ring-wise shuffle preserves radial statistics; sector–ring shuffle preserves coarse geometry. The remaining difference ($\Delta\text{MI}_{\text{sector}}$) quantifies local coupling beyond geometric and statistical effects. Error bars represent standard deviation across timepoints.

### Figure 4 | Coupling matrix of the solar atmosphere

Geometry-controlled local coupling matrix ($\Delta\text{MI}_{\text{sector}}$) for seven AIA EUV channels spanning chromospheric to flare temperatures. Values represent mean $\Delta\text{MI}_{\text{sector}}$ over the analysis window. Strong coupling is observed between thermally adjacent channels, while distant temperature pairs exhibit weaker coupling. The temperature-ordered structure is consistent with magnetically mediated interactions between neighboring atmospheric layers.
