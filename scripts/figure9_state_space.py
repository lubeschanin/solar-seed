#!/usr/bin/env python3
"""Generate Figure 9: State space contraction during flares."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import svd

# Load data
results_dir = Path("results")
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

# Load coupling evolution
df = pd.read_csv(results_dir / "rotation" / "coupling_evolution.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Load state space analysis results
with open(results_dir / "state_space_analysis.json") as f:
    state_results = json.load(f)

# Build state matrix from coupling pairs
pair_cols = [c for c in df.columns if c != 'timestamp']
X = df[pair_cols].values

# Compute 5 coupling invariants (as done in state space analysis)
# I1: Total coupling (sum)
# I2: Hierarchy (max/mean ratio)
# I3: Chromospheric coupling (304 pairs)
# I4: Coronal backbone (193-211)
# I5: Hot channel activity (94, 131 pairs)

chrom_pairs = [c for c in pair_cols if '304' in c]
hot_pairs = [c for c in pair_cols if '94' in c or '131' in c]

I1 = X.sum(axis=1)
I2 = X.max(axis=1) / (X.mean(axis=1) + 1e-10)
I3 = df[chrom_pairs].mean(axis=1).values
I4 = df['193-211'].values if '193-211' in pair_cols else np.zeros(len(df))
I5 = df[hot_pairs].mean(axis=1).values

# Stack invariants
invariants = np.column_stack([I1, I2, I3, I4, I5])

# Add temporal derivatives
dI = np.gradient(invariants, axis=0)
state_matrix = np.hstack([invariants, dI])

# Clean NaN/inf
state_matrix = np.nan_to_num(state_matrix, nan=0, posinf=0, neginf=0)

# Classify regimes based on activity level
activity = I1 + I5  # Total + hot channel activity
q25, q75, q95 = np.percentile(activity, [25, 75, 95])

regimes = ['quiet'] * len(df)
for i in range(len(df)):
    if activity[i] > q95:
        regimes[i] = 'flare'
    elif activity[i] > q75:
        regimes[i] = 'active'
regimes = np.array(regimes, dtype=object)

# PCA via SVD for visualization
X_centered = state_matrix - state_matrix.mean(axis=0)
U, S, Vt = svd(X_centered, full_matrices=False)
X_pca = U[:, :3] * S[:3]
explained_var = (S**2) / (S**2).sum()
explained_variance_ratio = explained_var[:3]

# Color mapping
colors = {'quiet': '#2ecc71', 'active': '#f39c12', 'flare': '#e74c3c'}
regime_colors = [colors[r] for r in regimes]

# Create figure
fig = plt.figure(figsize=(14, 10))

# Panel A: PCA projection
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
for regime in ['quiet', 'active', 'flare']:
    mask = regimes == regime
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                c=colors[regime], label=regime.capitalize(), alpha=0.6, s=30)
ax1.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)')
ax1.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)')
ax1.set_zlabel(f'PC3 ({explained_variance_ratio[2]*100:.1f}%)')
ax1.set_title('A  State Space Projection', fontweight='bold', loc='left')
ax1.legend(loc='upper right')

# Panel B: Participation ratio
ax2 = fig.add_subplot(2, 2, 2)
regimes_list = ['quiet', 'active', 'flare']
pr_values = [state_results['regimes'][r]['participation_ratio'] for r in regimes_list]
bar_colors = [colors[r] for r in regimes_list]
bars = ax2.bar(regimes_list, pr_values, color=bar_colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Participation Ratio')
ax2.set_title('B  Effective Dimensionality', fontweight='bold', loc='left')
ax2.set_ylim(0, 7)
# Add value labels
for bar, val in zip(bars, pr_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.2f}', ha='center', va='bottom', fontsize=11)
ax2.axhline(y=state_results['global']['participation_ratio'], color='gray',
            linestyle='--', alpha=0.7, label=f"Global: {state_results['global']['participation_ratio']:.2f}")
ax2.legend()

# Panel C: State-space volume (log scale)
ax3 = fig.add_subplot(2, 2, 3)
vol_values = [state_results['regimes'][r]['volume'] for r in regimes_list]
bars = ax3.bar(regimes_list, vol_values, color=bar_colors, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('State-Space Volume')
ax3.set_yscale('log')
ax3.set_title('C  Volume Contraction', fontweight='bold', loc='left')
# Add ratio annotation
vol_ratio = vol_values[2] / vol_values[0]
ax3.annotate(f'{vol_ratio:.1e}×', xy=(2, vol_values[2]), xytext=(1.5, vol_values[2]*5),
             fontsize=11, ha='center',
             arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Panel D: Entropy
ax4 = fig.add_subplot(2, 2, 4)
entropy_values = [state_results['regimes'][r]['entropy'] for r in regimes_list]
bars = ax4.bar(regimes_list, entropy_values, color=bar_colors, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Eigenvalue Spectrum Entropy (bits)')
ax4.set_title('D  Dynamical Constraint', fontweight='bold', loc='left')
ax4.set_ylim(0, 2.5)
# Add value labels
for bar, val in zip(bars, entropy_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.2f}', ha='center', va='bottom', fontsize=11)
# Add reduction annotation
entropy_reduction = (entropy_values[0] - entropy_values[2]) / entropy_values[0] * 100
ax4.annotate(f'-{entropy_reduction:.0f}%', xy=(2, entropy_values[2]),
             xytext=(2.3, entropy_values[2] + 0.3),
             fontsize=11, ha='left',
             arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

plt.tight_layout()
plt.savefig(figures_dir / "figure9_state_space.png", dpi=300, bbox_inches='tight')
plt.savefig(figures_dir / "figure9_state_space.pdf", bbox_inches='tight')
plt.close()

print("Figure 9 saved to figures/figure9_state_space.png and .pdf")

# Print summary
print("\nState Space Analysis Summary:")
print(f"  Global participation ratio: {state_results['global']['participation_ratio']:.2f}")
print(f"  Quiet  → PR: {pr_values[0]:.2f}, Vol: {vol_values[0]:.0f}, Entropy: {entropy_values[0]:.2f}")
print(f"  Active → PR: {pr_values[1]:.2f}, Vol: {vol_values[1]:.0f}, Entropy: {entropy_values[1]:.2f}")
print(f"  Flare  → PR: {pr_values[2]:.2f}, Vol: {vol_values[2]:.0f}, Entropy: {entropy_values[2]:.2f}")
print(f"  Flare/Quiet volume ratio: {vol_ratio:.2e}")
