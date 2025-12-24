# Geometry-controlled mutual information reveals temperature-ordered coupling in the solar atmosphere

**Draft manuscript**

---

## Abstract

Understanding how different thermal layers of the solar atmosphere are coupled is central to solar physics and space-weather prediction. While correlations between extreme-ultraviolet (EUV) channels are well known, disentangling genuine physical coupling from geometric and statistical confounders remains challenging.

Here we introduce a geometry-controlled mutual information framework to quantify multichannel coupling in Solar Dynamics Observatory / Atmospheric Imaging Assembly (SDO/AIA) data. By systematically removing disk geometry, radial intensity statistics, and coarse azimuthal structure through a hierarchy of null models, we isolate a residual local coupling component.

Applying this method to seven EUV channels spanning chromospheric to flare temperatures, we construct a coupling matrix across thermal layers of the solar atmosphere. We find that neighboring temperature channels exhibit significantly stronger local coupling than thermally distant pairs. This temperature-ordered structure is stable over time, survives time-shift and alignment controls, and is spatially localized to active regions rather than disk geometry.

Our results demonstrate that information-theoretic coupling, when properly controlled for geometric effects, reveals an intrinsic organization of the solar atmosphere consistent with magnetically mediated interactions between adjacent thermal layers. Solar flares emerge as regime-switching events in this dynamically coupled system, leaving a persistent imprint on its organizational structure. The presented framework provides a general, reproducible approach for analyzing multichannel structure in complex astrophysical systems.

---

## 1. Introduction

The solar atmosphere is a highly structured, multi-layered plasma system spanning several orders of magnitude in temperature, from the chromosphere to the hot flaring corona. Understanding how these thermal layers are coupled is essential for explaining energy transport, magnetic reconnection, and the emergence of solar activity that drives space weather.

Observations from the Solar Dynamics Observatory's Atmospheric Imaging Assembly (SDO/AIA) provide simultaneous imaging of the Sun in multiple extreme-ultraviolet (EUV) wavelengths, each sensitive to plasma at characteristic temperatures. While correlations between AIA channels are well documented, interpreting such correlations as evidence of physical coupling remains challenging. Apparent multichannel similarity can arise from disk geometry, limb brightening, shared radial intensity profiles, instrumental effects, or global morphological structure, rather than genuine interaction between thermal layers.

Traditional approaches to multichannel analysis in solar physics rely on intensity correlations, emission measures, or magnetic field extrapolations. While powerful, these methods do not explicitly separate geometric and statistical confounders from local, physically meaningful coupling. As a result, it remains unclear to what extent observed cross-channel structure reflects intrinsic organization of the solar atmosphere versus projection and sampling effects.

Here we introduce a geometry-controlled mutual information framework designed to isolate genuine multichannel coupling in solar imagery. Mutual information (MI), unlike linear correlation, captures arbitrary statistical dependence between channels, but must be applied with care in spatially structured systems. We therefore combine MI estimation with a hierarchy of null models that progressively remove disk geometry, radial statistics, and coarse azimuthal structure.

Applying this framework to seven AIA EUV channels spanning chromospheric to flare temperatures, we construct a coupling matrix of the solar atmosphere. We show that neighboring thermal layers exhibit significantly stronger local coupling than thermally distant layers, and that this structure is stable over time, spatially localized, and robust to extensive controls. We investigate multichannel coupling dynamics not only in baseline periods but also across major eruptive events. These results reveal an intrinsic, temperature-ordered organization of the solar atmosphere consistent with magnetically mediated interactions between adjacent thermal layers.

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

I(X;Y) = Σ p(x,y) log₂ [p(x,y) / p(x)p(y)]

**Estimator robustness.** Histogram-based MI estimation introduces binning bias, which we mitigate through fixed bin ranges calibrated to the dynamic range of AIA data. Robustness tests with 32, 64, and 128 bins confirm that relative MI differences between channels and the hierarchy of null models remain stable (coefficient of variation <5% across bin choices). Alternative estimators based on k-nearest neighbors (Kraskov et al. 2004) were evaluated but histogram discretization was retained for computational efficiency and reproducibility, given the large number of pairwise comparisons across timepoints.

### 3.2 Geometric Normalization

To remove disk geometry and radial intensity gradients, we estimate the per-frame radial mean profile and normalize each image. The residual is defined as:

R(r,θ) = I(r,θ) / ⟨I(r)⟩

**Limb treatment.** Pixels beyond r/R☉ > 0.98 are excluded from analysis to avoid limb artifacts and off-disk emission. The radial normalization effectively removes limb brightening within the analyzed disk region; robustness tests confirm that results are stable when the limb exclusion threshold is varied between 0.95 and 0.99.

### 3.3 Hierarchy of Null Models

- **Global shuffle**: permute all pixels (preserve histogram, destroy structure)
- **Ring-wise shuffle**: shuffle within annuli (preserve radial statistics)
- **Sector–ring shuffle**: shuffle within radial+azimuth sectors (preserve coarse geometry)
- **Time-shift null**: pair with large temporal offset (destroy time-coherent overlap)

### 3.4 Local Coupling Metric

Local coupling beyond geometry and coarse structure is quantified as:

ΔMI_sector(i,j) = MI(Rᵢ, Rⱼ) − MI_sector-null(Rᵢ, Rⱼ)

### 3.5 Spatial Analysis

Images are divided into an 8×8 grid and MI is computed per cell to obtain spatial MI maps and identify hotspots.

### 3.6 Statistical Testing

For each null model, we generate empirical null distributions using 100 independent shuffles per timepoint. The observed MI is compared to the null distribution via Z-scores:

Z = (MI_observed − μ_null) / σ_null

where μ_null and σ_null are the mean and standard deviation of the null distribution. P-values are computed assuming normality for large Z (confirmed by shuffle distribution shape). For the primary coupling metric ΔMI_sector, we report time-aggregated statistics (mean ± standard deviation across timepoints) and minimum Z-scores to characterize worst-case significance.

**Multiple comparisons.** When analyzing all 21 channel pairs simultaneously, we apply Bonferroni correction for family-wise error control. Given the extremely high Z-scores observed (Z > 100 for most pairs), all reported significant effects survive correction at α = 0.001.

### 3.7 Flare-Specific Analysis

For major eruptive events, we analyze flare-specific time windows as separate subsets. Time intervals are defined relative to the GOES X-ray peak: pre-flare (−2 to −0.5 hours), flare (−0.5 to +1 hour), and post-flare (+1 to +3 hours). Coupling metrics are computed independently for each phase, enabling comparison of baseline organization with flare-driven dynamics.

### 3.8 State-Space Dimensionality Metrics

To characterize the effective complexity of solar atmospheric dynamics, we construct a state vector from coupling invariants and analyze its dimensionality using the following metrics:

**Participation ratio.** Given the eigenvalue spectrum {λᵢ} of the state-space covariance matrix, the participation ratio quantifies effective dimensionality as:

PR = (Σᵢ λᵢ)² / Σᵢ λᵢ²

A uniform distribution across d dimensions yields PR = d, while concentration on a single mode yields PR = 1.

**State-space volume.** We estimate the occupied volume as the square root of the determinant of the covariance matrix (proportional to the volume of the concentration ellipsoid).

**Spectral entropy.** The Shannon entropy of the normalized eigenvalue spectrum measures the uniformity of variance distribution:

H = −Σᵢ pᵢ log₂ pᵢ,  where pᵢ = λᵢ / Σⱼ λⱼ

Higher entropy indicates variance spread across many dimensions; lower entropy indicates concentration on few modes.

---

## 4. Results

### 4.1 Global versus Geometry-Controlled Coupling

We compute MI on original images and on geometry-normalized residuals. Across a 6-hour interval (30 timepoints), the ratio MI_residual/MI_original remains stable at **30.8% ± 0.7%** (CV = 2.3%), indicating that most apparent coupling is geometric, while a consistent residual component remains after removal.

The residual coupling is extremely significant relative to shuffle-based null models, with minimum Z-scores exceeding Z = 986 and mean Z = 1252 ± 146 (p < 10⁻¹⁰⁰).

### 4.2 Spatial Localization of Residual Coupling

Spatial MI maps (8×8 grid) show strong limb domination in original images. After geometric normalization, limb bias is removed and localized hotspots remain. Mean spatial MI decreases by ~42%, while local maxima persist and collapse under time-shift controls.

### 4.3 Hierarchy of Null Models and Structure Decomposition

Applying the hierarchy of null models yields the consistent ordering:

MI_global < MI_ring < MI_sector < MI_residual

This enables decomposition into radial, azimuthal, and local structure contributions, with a typical local contribution ΔMI_sector ≈ 0.17 bits.

### 4.4 Temperature-Ordered Coupling Across EUV Channels

Extending to seven EUV channels (21 pairs), we find temperature-ordered local coupling: thermally adjacent channels exhibit substantially stronger ΔMI_sector than thermally distant pairs. The strongest links occur among coronal channels (e.g., 193–211, 171–193), while chromospheric (304) and flare channels (94, 131) show weaker and more episodic coupling.

### 4.5 Robustness and Temporal Stability

Time-shift controls reduce coupling by >95%, alignment checks peak at (0,0), and scale-response tests indicate coupling dominated by mid-to-large spatial scales.

### 4.6 Coupling Dynamics During Major Solar Flares

To further examine how geometry-controlled multichannel coupling behaves during extreme solar activity, we analyzed an independent X1.9-class flare that occurred on 1 December 2025 (NOAA, source region AR4294, east limb). The reported flare peak times (e.g., NOAA X-ray flux) are used solely for timeline alignment; our MI metric quantifies structural coupling independently of intensity-based diagnostics.

Figure 5 shows the temporal evolution of the local coupling metric ΔMI_sector for multiple EUV channel pairs before, during, and after the flare peak. A consistent pattern emerges: during the impulsive flare phase, most channel pairs exhibit a statistically significant reduction in coupling, particularly those involving the hottest coronal channels (e.g., 131 Å, 335 Å). Decoupling begins shortly before the reported X-ray peak and persists throughout the flare interval.

Notably, the flare peak itself does not coincide with a maximum in multichannel coupling. Instead, approximately 50–70 minutes after the X-ray peak, the system exhibits a pronounced global recoupling event, during which a majority of channel pairs show simultaneous increases in ΔMI_sector. At this time, several thermally adjacent pairs (e.g., 193–211 Å) reach their highest observed coupling values, while chromosphere–corona pairs (e.g., 304–131 Å) also show enhanced coupling.

This behavior mirrors the coupling dynamics observed for the X9.0 flare analyzed earlier, indicating that major flares are characterized by a transient breakdown of multithermal organization, followed by delayed large-scale reorganization of the solar atmosphere. We emphasize that the observed decoupling is not interpreted as a predictive precursor signal, but rather as a diagnostic signature of rapid magnetic reconfiguration during the impulsive flare phase.

These results demonstrate that geometry-controlled mutual information captures aspects of flare-driven coronal restructuring that are not directly visible in traditional intensity-based diagnostics such as X-ray flux, highlighting its potential as a complementary tool for studying the dynamical organization of the solar atmosphere.

### 4.7 Regime-Switching Dynamics and Transition Operators

To formalize the dynamical behavior of the solar atmosphere identified in Sections 4.4–4.6, we model the evolution of the solar state vector

**S**(t) = (I₁(t), I₂(t), I₃(t), I₄(t), I₅(t))

with regime-dependent linearized transition operators. For quiet (normal) conditions, we define the operator A_N and bias b_N via ordinary least squares (regularized) regression

**S**(t+Δt) ≈ A_N · **S**(t) + b_N

using timepoints outside major eruptive intervals. Similarly, for flare (eruptive) conditions, we fit A_F and b_F using data within the temporal bounds of strong activity.

#### 4.7.1 Regime-Dependent Operator Comparison

The Frobenius norm of the difference between the flare and quiet operators is

‖A_F − A_N‖_F = 2.90

indicating that the linearized dynamics in the two regimes are distinct. A direct comparison of the operators confirms that coupling relationships among the state vector components reorganize under eruptive conditions.

#### 4.7.2 Residual Metric as an Early Indicator

We define the residual of the quiet regime operator as

r(t) = ‖**S**(t+Δt) − (A_N · **S**(t) + b_N)‖

which measures deviation from normal propagation. Over the analyzed event on 1 December 2025, we observe that r(t) increases significantly before the reported X-ray peak:

- Pre-flare baseline: 2.3 ± 1.3
- Flare interval: 5.1 ± 1.4
- Threshold (2σ): 5.3

This suggests that the departure from normal dynamics precedes peak radiative signatures, offering a regime-agnostic diagnostic of imminent reorganization.

#### 4.7.3 Hysteresis in State Transitions

To investigate reversibility, we define operators for transitions from normal to flare (A_NF) and flare to normal (A_FN). The difference

‖A_NF − A_FN‖_F = 4.93

demonstrates that the pathway back to quiet conditions is not a simple inverse of the eruptive transition. This hysteresis supports the interpretation that the corona follows different dynamical rules in and out of the eruptive regime.

#### 4.7.4 Post-Flare State Shift

A comparison of mean component values before and after the eruptive interval reveals persistent shifts in specific state vector components:

| Component | Post-Flare Shift |
|-----------|------------------|
| I₅ (Normalized Scale) | +108% |
| I₃ (Chromosphere) | +27% |
| I₂ (Corona Ratio) | +6% |
| I₄ (Core Stability) | −4% |

This observation indicates that the system does not return to its pre-flare attractor but occupies a reconfigured dynamical basin.

#### 4.7.5 Eigenmode Structure of the Transition Operators

An eigenanalysis of the transition matrices A_N and A_F reveals a spectrum of modes with distinct damping characteristics. The dominant mode (Mode 1, λ ≈ 0.90) exhibits slow decay and is primarily composed of the strongly coupled components I₅, I₂, and I₄, representing the long-timescale structure of the corona. In contrast, the fastest decaying mode (Mode 5, λ ≈ 0.11) is dominated by I₃, consistent with rapid chromospheric fluctuations that damp quickly under both quiet and eruptive dynamics.

These mode structures corroborate the notion of fast and slow manifolds in the solar atmospheric dynamics and provide a basis for reduced-order modeling.

---

## 5. Discussion

The results presented in Figure 6 place the geometry-controlled mutual information framework into a dynamical systems context. Rather than describing solar flares as isolated impulsive events, the analysis demonstrates that eruptive activity corresponds to a regime switch in the propagation dynamics of the solar atmospheric state vector.

### 5.1 Regime-Dependent Dynamics of the Solar Atmosphere

The identification of distinct transition operators for quiet and flare conditions indicates that the solar atmosphere obeys different effective dynamical laws in these regimes. The significant difference between the operators A_N and A_F implies that flares are not merely characterized by increased emission or enhanced coupling amplitudes, but by a reorganization of how structural information propagates across temperature layers.

This observation aligns with the physical picture of magnetic reconnection as a topological transition: the global organization of magnetic connectivity changes, leading to altered pathways for energy and information transfer throughout the corona.

### 5.2 Early Breakdown of Quiet-Regime Propagation

The residual metric r(t) (Figure 6A) reveals that the quiet-regime operator fails to describe the system dynamics prior to the X-ray peak of the flare. Importantly, this deviation is detected without reference to radiative flare diagnostics, suggesting that the atmospheric reorganization begins before peak energy release becomes visible in standard X-ray measurements.

Physically, this behavior is consistent with a gradual destabilization of magnetic structures preceding large-scale reconnection. The rising residual therefore captures the onset of dynamical inconsistency in the quiet regime, offering a system-level indicator of impending eruptive activity.

### 5.3 Hysteresis and Post-Flare Reorganization

The pronounced difference between the forward (normal-to-flare) and backward (flare-to-normal) transition operators demonstrates that the system exhibits hysteresis. The solar atmosphere does not retrace its dynamical trajectory after an eruptive event but instead relaxes into a modified configuration. This is further supported by the persistent post-flare shifts in the state vector components, particularly the strong increase in the normalized scale invariant I₅.

Such hysteresis is a hallmark of nonlinear systems undergoing topological reconfiguration and indicates that major flares leave a lasting imprint on coronal organization, rather than constituting transient perturbations around a fixed equilibrium.

### 5.4 Fast and Slow Manifolds in Atmospheric Dynamics

The eigenmode analysis of the quiet-regime operator reveals a clear separation between slow and fast dynamical components. The dominant slow mode, composed primarily of coronal invariants (I₂, I₄, I₅), governs long-timescale structural evolution, while rapidly damped modes are dominated by chromospheric variability (I₃).

This fast–slow manifold separation provides a natural explanation for the observed behavior during flares: chromospheric fluctuations are rapidly suppressed, while the coronal structure reorganizes on longer timescales. The emergence of eruptive behavior can thus be interpreted as a collapse of hierarchical ordering among slow modes rather than an amplification of fast variability.

### 5.5 Implications for Solar Diagnostics

Together, these findings suggest that major solar flares are best understood as regime transitions in a structured dynamical system, characterized by operator switching, hysteresis, and attractor shifts. The geometry-controlled mutual information framework provides access to these properties by isolating intrinsic coupling dynamics from geometric and instrumental effects.

Beyond flare analysis, the identification of regime-dependent operators and invariant structures opens a pathway toward reduced-order modeling of solar atmospheric dynamics and toward system-level diagnostics that complement traditional intensity-based space-weather indicators.

### 5.6 Network Phase Transitions and Hysteresis in Solar Atmospheric Coupling

The dynamical regime switching identified in the operator-based analysis (Figure 6) is further corroborated by an independent network-theoretic characterization of the coupling structure. By interpreting the pairwise residual coupling metrics as a weighted interaction network between temperature layers, we identify two distinct network phase transitions during the analyzed X-class flare (Figure 7).

#### 5.6.1 Collapse of Network Connectivity During Flare Onset

At the onset of the eruptive phase (02:48 UTC), the coupling network undergoes an abrupt collapse. Network density decreases from 0.17 to 0.14, the clustering coefficient drops from 0.33 to zero, and the largest connected component is reduced from four to three nodes. This collapse indicates a breakdown of coordinated interactions among temperature layers and mirrors the loss of hierarchical ordering captured by the invariant I₁ and the quiet-regime operator residual.

Physically, this phase corresponds to the destabilization of pre-existing magnetic connectivity as the system approaches large-scale reconnection. Rather than increasing global coherence, the flare onset is characterized by fragmentation and decoupling of the atmospheric interaction network.

#### 5.6.2 Explosive Reconnection and Network Reorganization

Approximately one hour after the X-ray peak (03:48 UTC), the system enters a second transition marked by rapid network reconnection. Network density increases explosively from 0.14 to 0.67, achieving full connectivity across all seven temperature nodes. Notably, this connectivity overshoots pre-flare levels, indicating that the post-flare state is not a simple recovery of the original configuration.

This reconnection phase aligns temporally with the recoupling spike observed in the operator-based analysis and reflects the establishment of a new, globally coherent interaction structure following magnetic reconfiguration.

#### 5.6.3 Hysteresis and Post-Flare Attractor Shift

A comparison of pre- and post-flare network metrics reveals pronounced hysteresis. None of the primary network measures—density, mean degree, clustering coefficient, or total weight—return to their pre-flare values. Instead, all metrics increase substantially in the post-flare state, with clustering increasing by 139% and network density by 82%.

This persistent shift demonstrates that the solar atmospheric coupling network relaxes into a new attractor characterized by higher connectivity and stronger inter-layer coupling. Such behavior is consistent with irreversible topological changes in the coronal magnetic field and reinforces the interpretation of major flares as system-wide reorganization events rather than transient perturbations.

#### 5.6.4 Unified Interpretation with Operator Dynamics

Together with the operator-based regime switching, the network analysis establishes a coherent picture of flare dynamics: the onset phase is marked by a collapse of coupling coherence, while the recovery phase produces a reorganized, more strongly connected state. The consistency between the operator-level hysteresis and the network-level attractor shift underscores the robustness of the identified phase transitions.

These results suggest that solar flares can be interpreted as nonequilibrium phase transitions in a dynamically coupled, magnetically mediated network, with lasting consequences for the organization of the solar atmosphere.

### 5.7 Redundancy, Functional Clustering, and Structural Memory

Beyond pairwise coupling strengths, the redundancy analysis reveals a higher-order organization of the solar atmospheric coupling structure. Multiple channel pairs exhibit correlated coupling behavior despite lacking shared wavelengths or overlapping temperature sensitivity, indicating that these relationships are not driven by common radiative response functions.

In particular, statistically significant correlations between chromospheric–coronal pairs (e.g., 304–171 Å) and coronal–flare-channel pairs (e.g., 193–131 Å) point to indirect coupling pathways linking thermally separated layers. The existence of 22 such correlated pair combinations suggests coordinated modulation of coupling strengths across the atmosphere, rather than independent local interactions.

Hierarchical clustering of the coupling matrix identifies four functional clusters: a low-amplitude but stable chromospheric bridge, a dominant coronal backbone, an activity-dependent flare-channel group, and an intermediate transition zone. The coronal backbone cluster, characterized by the pairs 171–193–211 Å and their links to 131 Å, emerges as the most stable and reliable structure across time and activity levels.

A subset of six channel pairs forms a stability backbone, exhibiting both high mutual information and reliability exceeding 90%. These pairs persist across quiet, active, and eruptive conditions, suggesting that they encode fundamental structural relationships within the solar atmosphere. Their persistence mirrors the role of redundancy in biological information systems, where duplicated or correlated pathways provide robustness and error tolerance.

Importantly, this redundancy does not imply static behavior. Instead, correlated coupling pairs evolve coherently during regime transitions, indicating that the system preserves relational structure even as absolute coupling strengths change. In this sense, the redundancy pattern constitutes a form of structural memory: information about prior organizational states is retained in the pattern of interdependencies among layers rather than in any single channel.

From a physical perspective, such redundancy is naturally explained by magnetically mediated connectivity spanning multiple temperature regimes. Magnetic field topology provides a substrate through which information about structural organization is distributed redundantly across layers, enabling coordinated reconfiguration during flares while maintaining global coherence.

These findings complement the operator-based and network-level analyses by demonstrating that solar atmospheric organization is not only hierarchical and dynamic, but also redundantly encoded, enhancing robustness and continuity across regime transitions.

### 5.8 State Space Dimensionality and Constraint During Flares

To quantify how solar atmospheric dynamics reorganize across activity regimes, we analyzed the effective dimensionality of the solar state space spanned by the coupling invariants I₁–I₅ and their temporal derivatives. Although the full embedding space has dimension ten, the global participation ratio indicates an effective dimensionality of 6.8, with 90% of the variance captured by seven components.

When separated by activity regime, pronounced differences emerge. Quiet conditions occupy a comparatively high-dimensional and voluminous region of state space (participation ratio 5.37; state-space volume 1.7×10⁵), whereas active and flare states are progressively more constrained. During flares, the effective dimensionality drops to 3.11 and the occupied volume contracts by more than two orders of magnitude relative to quiet conditions.

Despite this strong contraction, the flare state does not exhibit increased variance or entropy. Instead, the Shannon entropy of the normalized eigenvalue spectrum decreases from 1.88 bits (quiet) to 1.34 bits (flare), indicating that the dynamics collapse onto a small number of dominant modes rather than spreading across additional degrees of freedom.

These results demonstrate that solar flares correspond to a contraction of the accessible state space onto a low-dimensional manifold. Rather than representing an expansion into new dynamical configurations, eruptive events impose strong constraints on the system, channeling its evolution through a reduced set of degrees of freedom.

### 5.9 Physical Interpretation: Constraint-Dominated Dynamics

The observed contraction of state-space volume during flares provides a unifying explanation for several previously identified phenomena. The breakdown of coupling hierarchies, the collapse of network connectivity, and the reduction of effective dimensionality all indicate that eruptive events do not introduce additional freedom into the system. Instead, magnetic reconnection drives the atmosphere into a highly constrained configuration dominated by a small number of collective modes.

In this picture, quiet solar conditions correspond to a loosely constrained, high-dimensional regime in which multiple coupling pathways coexist. As magnetic stress accumulates, the system transitions toward a flare state characterized by reduced freedom and strong channeling of energy and information flow. The flare itself represents the release of stored energy within this constrained manifold, rather than an excursion into a higher-dimensional chaotic state.

The strong reduction in state-space volume further implies that flares leave a lasting imprint on the system. Because the post-flare state occupies a different region of the state space, the dynamics do not simply revert to their pre-flare configuration, consistent with the hysteresis observed in both operator dynamics and network topology.

### 5.10 Limitations and Outlook

Several methodological and physical considerations constrain the interpretation of these results.

**Chromospheric channel (304 Å).** Unlike the optically thin coronal EUV lines, the 304 Å channel (He II) is optically thick and originates from a range of formation heights in the chromosphere and transition region. Coupling involving this channel therefore reflects line-of-sight integration effects and radiative transfer processes that differ fundamentally from coronal emission. The consistently weaker and more variable coupling observed for 304 Å pairs is consistent with this physical distinction, and we interpret the chromospheric layer as a regime boundary rather than a smoothly integrated component of coronal organization.

**Sample size and event generalization.** The flare-regime analysis is based on a limited number of major events (N = 2 X-class flares), with the state-space decomposition relying on relatively few flare-state data points (N ≈ 7). While the observed patterns are internally consistent and statistically significant, generalization to all flare classes and solar cycle phases requires validation across a larger event sample. The checkpoint-enabled analysis pipeline developed here is designed to facilitate such extended studies.

**Cadence and spatial resolution.** The 12-minute synoptic cadence may undersample rapid flare dynamics, particularly during the impulsive phase. Higher-cadence AIA data (12-second full resolution) could reveal finer temporal structure in coupling evolution, though at substantially increased computational cost.

**Outlook.** Future work should extend the analysis across multiple solar rotations and activity cycles to establish baseline variability and long-term trends. Integration with magnetogram data (HMI) could directly test the hypothesis of magnetically mediated coupling, while application to stellar EUV observations may enable comparative studies of atmospheric organization across different stellar types.

---

## 6. Conclusion

In this work, we introduced a geometry-controlled mutual information framework to investigate multichannel coupling in the solar atmosphere using SDO/AIA EUV observations. By explicitly removing geometric and instrumental biases, the method isolates intrinsic coupling structures across temperature layers and enables a system-level characterization of solar atmospheric dynamics.

Our analysis demonstrates that the solar atmosphere exhibits robust, temperature-ordered coupling invariants under quiet conditions, forming a stable hierarchical organization dominated by coronal layers. These invariants persist across time and activity levels, indicating an underlying structural backbone mediated by magnetic connectivity rather than transient radiative effects.

During major solar flares, this organization undergoes a qualitative transformation. Using a low-dimensional solar state vector derived from coupling invariants, we show that eruptive events correspond to regime switches in the system dynamics. This transition is characterized by a breakdown of the quiet-regime propagation operator, detectable prior to the X-ray peak, followed by a distinct flare-regime operator and pronounced hysteresis. The post-flare state does not return to its pre-event configuration but instead occupies a reorganized attractor with persistently higher connectivity.

A complementary network-theoretic analysis reveals that these dynamical regime switches correspond to nonequilibrium phase transitions in the coupling network of the solar atmosphere. Flare onset is marked by a collapse of network coherence, while the recovery phase produces an overshooting reconnection into a more strongly coupled state. The consistency between operator-based dynamics and network-level phase behavior supports a unified interpretation of flares as system-wide reorganizations rather than localized impulsive phenomena.

Together, these results establish geometry-controlled mutual information as a powerful tool for probing the dynamical organization of the solar atmosphere. By focusing on intrinsic coupling structures and their evolution, the framework provides access to early indicators of regime change, hysteresis effects, and post-event reconfiguration that are not readily captured by traditional intensity-based diagnostics.

Beyond solar physics, the methodology is directly applicable to other multichannel, spatially structured systems where disentangling intrinsic interactions from geometric constraints is essential. As such, the approach opens new avenues for reduced-order modeling, comparative stellar studies, and system-level diagnostics of complex plasma environments.

---

## Figures

### Figure 1 — Effect of Geometric Normalization

![Figure 1](figures/figure1_geometric_normalization.png)

*Global mutual information (MI) between AIA EUV channels before and after geometric normalization. Left: MI computed on original images, dominated by disk geometry and limb brightening. Right: MI computed on residual images after radial profile normalization. Approximately 70% of apparent MI is removed, while a stable residual component remains.*

### Figure 2 — Spatial Distribution

![Figure 2](figures/figure2_spatial_distribution.png)

*Spatial maps of mutual information between 193 Å and 211 Å channels on an 8×8 grid. Left: Original MI showing strong limb bias. Right: Residual MI after geometric normalization. Stars indicate top residual MI cells, corresponding to active regions.*

### Figure 3 — Null Model Decomposition

![Figure 3](figures/figure3_null_model_decomposition.png)

*Mutual information values under progressively restrictive null models: global shuffle (destroys all structure), ring shuffle (preserves radial statistics), sector shuffle (preserves coarse geometry). The difference ΔMI_sector quantifies local coupling beyond geometry. Error bars indicate standard deviation over time.*

### Figure 4 — Coupling Matrix

![Figure 4](figures/figure4_coupling_matrix.png)

*Geometry-controlled local coupling matrix (ΔMI_sector) for seven AIA EUV channels. Channels ordered by characteristic formation temperature. Stronger coupling is observed between thermally adjacent channels, consistent with magnetically mediated interactions between neighboring layers.*

### Figure 5 — Flare Event Analysis

![Figure 5](figures/figure5_flare_phases.png)

*Geometry-controlled coupling during an X9.0 solar flare (2024-10-03). Time evolution of the local coupling metric ΔMI_sector for selected EUV channel pairs across pre-flare, flare, and post-flare phases (left). The flare peak is marked by the dashed line. Contrary to a naive expectation of uniformly increased coupling during extreme activity, most channel pairs exhibit reduced coupling during the flare peak. Percentage changes from pre-flare to flare conditions are shown on the right. Only a small subset of thermally adjacent channels (e.g. 171–211 Å) shows enhanced coupling, indicating selective reorganization rather than global amplification of multichannel structure.*

### Figure 6 — Regime-Switching Dynamics

![Figure 6](figures/figure6_operator_dynamics.png)

*Regime-switching dynamics of the solar state vector during an X-class flare. (A) Time evolution of the residual r(t) = ‖S(t+Δt) − (A_N·S(t) + b_N)‖, quantifying deviations from the quiet-regime transition operator. The residual exceeds the 2σ threshold prior to the GOES X-ray peak (red band), indicating an early breakdown of quiet-regime dynamics before peak radiative emission. (B) Difference between the flare and quiet transition operators, A_F − A_N. Colored entries denote changes in linear coupling coefficients between state vector components (I₁–I₅); starred values indicate statistically significant differences. The structured pattern demonstrates a redistribution of dynamical couplings rather than a uniform amplitude change. (C) Eigenvalue spectrum of the quiet-regime operator A_N. The dominant slow mode (λ ≈ 0.90) represents long-timescale coronal organization, while rapidly damped modes (λ ≲ 0.11) are dominated by chromospheric variability, consistent with fast–slow manifold separation. (D) Trajectory of the solar state vector in reduced state space (I₁, I₂, I₅). The system transitions from a pre-flare state (green) through the eruptive phase (red) to a post-flare state (blue), which does not coincide with the pre-flare trajectory, indicating hysteresis and a post-event attractor shift. Together, these panels demonstrate that major solar flares correspond to a regime switch characterized by operator change, early deviation from quiet-regime dynamics, hysteresis, and persistent post-flare reorganization.*

### Figure 7 — Network Phase Transitions

![Figure 7](figures/figure7_phase_transitions.png)

*Network-level phase transitions in solar atmospheric coupling during the X1.9 flare. (A) Number of significant coupling pairs (above thresholds 0.2, 0.3, 0.4) as a function of time, showing abrupt collapse during flare onset and subsequent recovery. (B) Network topology metrics: density (fraction of active edges) and clustering coefficient, both exhibiting sharp transitions coincident with flare dynamics. (C) Phase space trajectory in network coordinates (density vs. total coupling weight), demonstrating pronounced hysteresis—the post-flare state does not return to the pre-flare region but occupies a new attractor with higher connectivity. (D) Rate of change of network metrics, highlighting the timing of phase transitions: collapse onset precedes the X-ray peak, while reconnection occurs approximately one hour later. The network analysis independently confirms the regime-switching dynamics identified through operator methods (Figure 6) and establishes that major flares constitute nonequilibrium phase transitions with lasting topological consequences.*

### Figure 8 — Redundancy Structure and Functional Clustering

![Figure 8](figures/figure8_redundancy_structure.png)

*Redundancy structure and functional clustering of multichannel coupling. (A) Correlation matrix of pairwise coupling strengths across time, revealing coordinated variability among channel pairs without shared wavelengths or temperature response functions. (B) Hierarchical clustering (dendrogram) of coupling relationships, identifying four functional clusters: a chromospheric bridge, a dominant coronal backbone, an activity-dependent flare-channel group, and an intermediate transition zone. (C) Stability backbone composed of channel pairs exhibiting both high mutual information and reliability exceeding 90%. These links persist across quiet, active, and eruptive conditions. (D) Network representation of redundant coupling relationships, illustrating how multiple thermally separated layers participate in coherent, magnetically mediated interaction patterns. Together, these panels demonstrate that solar atmospheric coupling is redundantly encoded across temperature layers, providing robustness and continuity of structural organization during dynamical regime transitions.*

### Figure 9 — State Space Contraction During Flares

![Figure 9](figures/figure9_state_space.png)

*Contraction of the effective solar atmospheric state space during flares. (A) Projection of the solar state vector, constructed from the coupling invariants I₁–I₅ and their temporal derivatives, onto the leading principal components. Quiet (green), active (orange), and flare (red) states occupy progressively smaller regions of the state space. (B) Effective dimensionality quantified by the participation ratio for different activity regimes. While quiet conditions span a higher-dimensional region of the embedding space, flare states collapse onto a reduced number of dominant modes. (C) Normalized state-space volume occupied by each regime, showing a contraction by more than two orders of magnitude during flares relative to quiet conditions. (D) Entropy of the eigenvalue spectrum of the state-space covariance matrix. The reduction in entropy during flares indicates increased dynamical constraint rather than enhanced variability. Together, these panels demonstrate that major solar flares correspond to a contraction of the accessible state space onto a low-dimensional manifold, reflecting a transition from distributed coronal organization to strongly constrained dynamics.*

---

## References

1. Boerner, P., Edwards, C., Lemen, J., et al. (2012). Initial Calibration of the Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO). *Solar Physics*, 275, 41–66. https://doi.org/10.1007/s11207-011-9804-8

2. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience.

3. Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. *Physical Review E*, 69, 066138. https://doi.org/10.1103/PhysRevE.69.066138

4. Lemen, J. R., Title, A. M., Akin, D. J., et al. (2012). The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO). *Solar Physics*, 275, 17–40. https://doi.org/10.1007/s11207-011-9776-8

5. O'Dwyer, B., Del Zanna, G., Mason, H. E., Weber, M. A., & Tripathi, D. (2010). SDO/AIA response to coronal hole, quiet Sun, active region, and flare plasma. *Astronomy & Astrophysics*, 521, A21. https://doi.org/10.1051/0004-6361/201014872

6. Pesnell, W. D., Thompson, B. J., & Chamberlin, P. C. (2012). The Solar Dynamics Observatory (SDO). *Solar Physics*, 275, 3–15. https://doi.org/10.1007/s11207-011-9841-3

7. Shibata, K., & Magara, T. (2011). Solar Flares: Magnetohydrodynamic Processes. *Living Reviews in Solar Physics*, 8, 6. https://doi.org/10.12942/lrsp-2011-6

8. Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27, 379–423. https://doi.org/10.1002/j.1538-7305.1948.tb01338.x

9. Viall, N. M., & Klimchuk, J. A. (2012). Evidence for Widespread Cooling in an Active Region Observed with the SDO Atmospheric Imaging Assembly. *The Astrophysical Journal*, 753, 35. https://doi.org/10.1088/0004-637X/753/1/35
