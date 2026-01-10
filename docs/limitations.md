# Limitations and Caveats

## Sample Size and Generalizability

The presented case study demonstrates a single validated precursor event. While the 27-minute lead time and 9.1 MAD deviation are statistically significant, systematic validation requires:

- A larger sample of confirmed precursor–flare pairs (N > 30)
- Calculation of precision/recall across multiple solar activity levels
- Cross-validation against independent monitoring periods

The current false positive rate remains unquantified. Quiet reconfigurations (coupling breaks without subsequent GOES response) were observed but not systematically catalogued.

## Temporal Resolution

The ~10-minute coupling measurement cadence limits detection of rapid-onset events. Flares with sub-10-minute precursor phases may be missed or detected with insufficient lead time. Higher-cadence monitoring (1-2 min) would improve sensitivity but increases computational cost and artifact susceptibility.

## Channel-Specific Reliability

The 193-304 Å pair exhibits high binning sensitivity (>100% change under 2×2 binning in some frames), rendering it unreliable as a primary trigger. This limits the method to corona-corona pairs (193-211, 171-193), excluding direct chromospheric coupling signatures from the detection logic. The physical origin of this sensitivity—whether instrumental or related to chromospheric fine structure—remains unclear.

## Baseline Stationarity

The rolling 60-minute window assumes quasi-stationary coupling statistics. During periods of sustained activity or gradual magnetic evolution, the adaptive baseline may track physical changes rather than providing a stable reference. This could suppress detection of slow-building precursors or generate false breaks during relaxation phases.

## Causal Interpretation

The observed temporal precedence (coupling break → GOES increase) does not establish causation. Alternative interpretations include:

- **Common driver**: Both coupling reduction and X-ray enhancement result from the same magnetic reconfiguration, with different response timescales
- **Coincidence**: Given continuous monitoring, some break–flare temporal associations occur by chance
- **Selection bias**: The reported event was identified post-hoc; prospective detection statistics may differ

## Operational Constraints

For real-time early warning applications:

- AIA data latency (5-15 minutes) reduces effective lead time
- STEREO-A coverage gaps limit advance warning availability
- Computational requirements (~2 min per coupling measurement) constrain update frequency
- Network/server reliability affects continuous monitoring

## Flare Class Sensitivity

The demonstrated precursor preceded a B-class enhancement (+66%, B5→B9), not a major flare (M/X class). Whether coupling breaks scale with flare magnitude, or whether the method detects only a subset of flare types, requires investigation across the full GOES classification range.

---

## Summary Table

| Limitation | Mitigation | Future Work |
|------------|------------|-------------|
| N=1 case study | Validation framework defined | Multi-event catalog |
| 10-min cadence | Sufficient for 27-min precursor | Test 1-2 min cadence |
| 193-304 unreliable | Excluded via robustness veto | Investigate cause |
| Baseline drift | 60-min rolling window | Adaptive window size |
| Causal ambiguity | Acknowledged explicitly | Multi-wavelength timing |
| Data latency | Documented constraint | Near-real-time pipeline |

---

## LaTeX Version (for paper)

```latex
\subsection{Limitations}

Several caveats constrain interpretation of these results. First, the
presented case study represents a single validated event; systematic
false positive/negative rates require larger samples across varied
activity levels. Second, the 10-minute measurement cadence may miss
rapid-onset precursors. Third, the 193-304~\AA{} pair exhibits excessive
binning sensitivity (>100\% in some frames) and is excluded from
primary triggering, limiting chromospheric coupling signatures. Fourth,
the rolling baseline assumes quasi-stationarity, which may not hold
during sustained activity. Fifth, temporal precedence does not establish
causation; common-driver and coincidence hypotheses remain viable.
Finally, operational deployment faces AIA data latency (5--15~min),
computational overhead, and network reliability constraints that reduce
effective lead time. Extension to M/X-class flares and prospective
validation are required before operational use.
```
