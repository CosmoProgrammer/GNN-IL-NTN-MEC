# plot_figures.py
# Generates three figures:
#   figures/scaling_curve.pdf
#   figures/ablation_bar.pdf
#   figures/noshare_ablation.pdf

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Shared colors ────────────────────────────────────────────────
C_IL     = "#3B82F6"   # blue
C_GNN    = "#10B981"   # teal
C_CTDE   = "#F59E0B"   # amber
C_RAND   = "#EF4444"   # red
C_LOCAL  = "#6B7280"   # gray
C_NOSHARE = "#8B5CF6"  # purple

# ════════════════════════════════════════════════════════════════
# FIGURE 1 — Scaling Curve
# ════════════════════════════════════════════════════════════════

M_vals = [5, 10, 20, 30, 40]

il_mean  = [0.6035, 0.8007, 0.9192, 1.3088, 1.0350]
il_std   = [0.025,  0.089,  0.052,  0.057,  0.052 ]

gnn_mean = [0.5888, 0.7569, 0.9121, 0.7713, 0.7594]
gnn_std  = [0.014,  0.232,  0.385,  0.219,  0.087 ]

# Median for M=20 (bimodal)
gnn_median_m20 = 0.5998
il_median_m20  = 0.9163

# CTDE single-seed reference points (seed 42)
ctde_M    = [10,    30   ]
ctde_vals = [0.6286, 0.6175]

# Random and Local at M=10 as horizontal reference
random_ref = 1.4585
local_ref  = 0.9513

fig, ax = plt.subplots(figsize=(6.5, 4.2))

# IL
ax.plot(M_vals, il_mean, color=C_IL, marker="o",
        linewidth=2, markersize=6, label="IL (mean)")
ax.fill_between(M_vals,
                [m - s for m, s in zip(il_mean, il_std)],
                [m + s for m, s in zip(il_mean, il_std)],
                color=C_IL, alpha=0.15)

# GNN-IL
ax.plot(M_vals, gnn_mean, color=C_GNN, marker="s",
        linewidth=2, markersize=6, label="GNN-IL (mean)")
ax.fill_between(M_vals,
                [m - s for m, s in zip(gnn_mean, gnn_std)],
                [m + s for m, s in zip(gnn_mean, gnn_std)],
                color=C_GNN, alpha=0.15)

# Median markers at M=20
ax.scatter([20], [gnn_median_m20], color=C_GNN, marker="D",
           s=60, zorder=5, label="GNN-IL (median, M=20)")
ax.scatter([20], [il_median_m20],  color=C_IL,  marker="D",
           s=60, zorder=5, label="IL (median, M=20)")

# CTDE reference
ax.scatter(ctde_M, ctde_vals, color=C_CTDE, marker="^",
           s=70, zorder=5, label="CTDE (seed 42)")

# Random and Local dashed references
ax.axhline(random_ref, color=C_RAND,  linestyle="--",
           linewidth=1.2, alpha=0.7, label=f"Random ({random_ref})")
ax.axhline(local_ref,  color=C_LOCAL, linestyle="--",
           linewidth=1.2, alpha=0.7, label=f"Local ({local_ref})")

# M=20 annotation
ax.annotate("Bimodal\n(median shown)",
            xy=(20, gnn_median_m20), xytext=(22, 0.45),
            fontsize=8, color=C_GNN,
            arrowprops=dict(arrowstyle="->", color=C_GNN, lw=1.0))

ax.set_xlabel("Number of UEs ($M$)")
ax.set_ylabel("Average Cost $\\Psi_m$")
ax.set_title("Scaling Performance: IL vs GNN-IL")
ax.set_xticks(M_vals)
ax.legend(loc="upper left", framealpha=0.9)
ax.set_ylim(0.3, 1.7)

plt.tight_layout()
plt.savefig("figures/scaling_curve.pdf", bbox_inches="tight")
plt.close()
print("Saved: figures/scaling_curve.pdf")


# ════════════════════════════════════════════════════════════════
# FIGURE 2 — Ablation Bar Chart (M=10, seed=42)
# ════════════════════════════════════════════════════════════════

methods = [
    "Random",
    "Local",
    "CTDE\n(w/o $B_n$)",
    "IL",
    "IL+$B_n$",
    "GNN-CTDE",
    "CTDE\n(w/ $B_n$)",
    "GNN-IL\n+Stack",
    "GNN-IL",
]

means = [1.4585, 0.9513, 0.9335, 0.6626, 0.6882,
         0.6933, 0.6286, 0.5464, 0.5650]
stds  = [0.0,    0.0,    0.1454, 0.0681, 0.0479,
         0.0646, 0.0345, 0.0260, 0.0279]

colors = [
    C_RAND,   # Random
    C_LOCAL,  # Local
    C_CTDE,   # CTDE w/o Bn
    C_IL,     # IL
    C_IL,     # IL+Bn (lighter)
    C_GNN,    # GNN-CTDE (lighter)
    C_CTDE,   # CTDE w/ Bn
    C_GNN,    # GNN-IL+Stack
    C_GNN,    # GNN-IL
]

alphas = [1.0, 1.0, 0.6, 1.0, 0.6, 0.6, 1.0, 0.7, 1.0]

# Sort by performance (ascending cost = better)
order = sorted(range(len(means)), key=lambda i: means[i], reverse=True)
methods_s = [methods[i] for i in order]
means_s   = [means[i]   for i in order]
stds_s    = [stds[i]    for i in order]
colors_s  = [colors[i]  for i in order]
alphas_s  = [alphas[i]  for i in order]

fig, ax = plt.subplots(figsize=(8.0, 4.5))

bars = ax.barh(range(len(methods_s)), means_s,
               xerr=stds_s, capsize=3,
               color=[c for c in colors_s],
               alpha=1.0,
               error_kw=dict(elinewidth=1.0, ecolor="black"))

# Apply alpha per bar
for bar, a in zip(bars, alphas_s):
    bar.set_alpha(a)

# Value labels
for i, (m, s) in enumerate(zip(means_s, stds_s)):
    ax.text(m + (s if s > 0 else 0) + 0.02, i,
            f"{m:.4f}", va="center", fontsize=8.5)

ax.set_yticks(range(len(methods_s)))
ax.set_yticklabels(methods_s, fontsize=9.5)
ax.set_xlabel("Average Cost $\\Psi_m$ (lower is better)")
ax.set_title("Baseline Comparison — $M=10$, Seed 42")
ax.set_xlim(0, 1.85)

# Legend patches
legend_elements = [
    mpatches.Patch(color=C_RAND,  label="Non-learning baselines"),
    mpatches.Patch(color=C_IL,    label="IL variants"),
    mpatches.Patch(color=C_CTDE,  label="CTDE variants"),
    mpatches.Patch(color=C_GNN,   label="GNN variants"),
]
ax.legend(handles=legend_elements, loc="lower right",
          framealpha=0.9, fontsize=9)

plt.tight_layout()
plt.savefig("figures/ablation_bar.pdf", bbox_inches="tight")
plt.close()
print("Saved: figures/ablation_bar.pdf")


# ════════════════════════════════════════════════════════════════
# FIGURE 3 — Parameter Sharing Ablation
# ════════════════════════════════════════════════════════════════

M_abl    = [10,     20,     30    ]
il_abl   = [0.6626, 0.8120, 1.3383]
noshare  = [0.7401, 1.2463, 1.0987]
gnn_abl  = [0.5650, 0.5833, 0.6595]

fig, ax = plt.subplots(figsize=(6.0, 4.0))

ax.plot(M_abl, il_abl,  color=C_IL,     marker="o",
        linewidth=2, markersize=7, label="IL")
ax.plot(M_abl, noshare, color="#8B5CF6", marker="^",
        linewidth=2, markersize=7, linestyle="--",
        label="GNN-IL-NoShare")
ax.plot(M_abl, gnn_abl, color=C_GNN,    marker="s",
        linewidth=2, markersize=7, label="GNN-IL")

# Annotate NoShare being worse than IL at M=10 and M=20
ax.annotate("Worse than IL",
            xy=(10, 0.7401), xytext=(12, 0.82),
            fontsize=8, color="#8B5CF6",
            arrowprops=dict(arrowstyle="->", color="#8B5CF6", lw=1.0))

ax.set_xlabel("Number of UEs ($M$)")
ax.set_ylabel("Average Cost $\\Psi_m$")
ax.set_title("Parameter Sharing Ablation — Seed 42")
ax.set_xticks(M_abl)
ax.legend(framealpha=0.9)
ax.set_ylim(0.4, 1.5)

plt.tight_layout()
plt.savefig("figures/noshare_ablation.pdf", bbox_inches="tight")
plt.close()
print("Saved: figures/noshare_ablation.pdf")