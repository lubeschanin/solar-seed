# Hybrid Resolution Strategy

## Overview

For operational break detection, we employ a two-tier resolution architecture that separates trigger-grade decisions from diagnostic analysis.

```
┌─────────────────────────────────────────┐
│  SYNOPTIC (1k) = INSTRUMENT             │
│  → Break Detection                       │
│  → Trigger Decision                      │
│  → Scale-stable, robust                  │
└─────────────────────────────────────────┘
                    │
                    ▼ (only on break)
┌─────────────────────────────────────────┐
│  FULL-RES (4k) = MICROSCOPE             │
│  → Spatial Localization                  │
│  → Transfer-State Interpretation        │
│  → Physics Diagnostics                   │
└─────────────────────────────────────────┘
```

## Rationale

### JSOC Limitations (Post-Nov 2024)

Following the flooding of Stanford's computer facility in November 2024, JSOC imposes strict rate limits:
- 49 GB per request (hard limit)
- One request at a time (queue-based)
- ~5% of pre-2023.12.23 data still on tape restore

The synoptic archive (`jsoc1.stanford.edu/data/aia/synoptic/`) provides **direct access** without queue limitations, enabling reliable 24/7 monitoring.

### Scale-Stability Analysis

| Pair | Synoptic (1k) | Full-Res (4k) | Δ | Interpretation |
|------|---------------|---------------|---|----------------|
| **193-211** | 0.94 bits | 0.90 bits | +5% | Scale-invariant (large-scale coronal coupling) |
| **193-304** | 0.24 bits | 0.18 bits | +33% | Scale-dependent (footpoint/TR fine structure) |

**Key Finding:** The 193-211 Å coupling is carried by large-scale coronal structures and remains stable across resolutions. This makes it ideal for trigger-grade detection at synoptic resolution.

The 193-304 Å pair shows scale dependence consistent with chromospheric footpoint dominance at high resolution. Fine structures that contribute to MI at 4k are smoothed at 1k, yielding higher apparent coupling.

## Implementation

### Data Source Priority

1. **Primary:** AIA synoptic (1024², 2-min cadence, direct HTTP)
2. **Fallback 1:** Full-res via VSO (4096², may be rate-limited)
3. **Fallback 2:** Direct JSOC export (legacy)

### Scale-Robustness Classification

For pairs showing scale dependence, we apply an additional robustness criterion:

```python
scale_ratio = abs(MI_full - MI_syn) / MI_syn
if scale_ratio > 0.40:  # >40% difference
    classification = "DIAGNOSTIC_ONLY"  # Not trigger-capable
```

This prevents scale-sensitive pairs (like 193-304) from generating false alerts due to resolution-dependent MI variations.

## Methods Text (Paper-Ready)

> For operational break detection, we employ AIA synoptic imagery (1024² pixels, 2-min cadence) as the primary data source. Synoptic resolution provides trigger-grade stability: the 193-211 Å coupling shows scale-invariant behavior (ΔMI_syn = 0.94 ± 0.03 bits vs. ΔMI_full = 0.90 ± 0.02 bits, ~5% difference), indicating that the coupling signal is carried by large-scale coronal structures rather than fine features.
>
> The 193-304 Å pair exhibits scale dependence (+10-70% between resolutions), consistent with chromospheric footpoint dominance at high resolution. We classify this pair as "diagnostic only" when |MI_full - MI_syn| / MI_syn > 40%.
>
> Full-resolution (4096²) imagery is retrieved on-demand for validated breaks, enabling spatial localization and transfer-state interpretation without compromising trigger reliability.

## Summary Table

| Aspect | Synoptic (1k) | Full-Res (4k) |
|--------|---------------|---------------|
| **Role** | Trigger (decides) | Microscope (explains) |
| **Access** | Direct HTTP | JSOC queue |
| **Cadence** | 2 min | Variable |
| **193-211** | Trigger-capable | Diagnostic |
| **193-304** | Trigger-capable* | Diagnostic |
| **Robustness** | High (smoothed) | Lower (fine-structure sensitive) |

*193-304 at synoptic is usable but less reliable than 193-211; apply scale-robustness check when full-res available.
