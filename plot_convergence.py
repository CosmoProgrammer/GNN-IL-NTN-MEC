# plot_convergence.py
# Reads four .txt files from view_results.py and generates:
#   figures/convergence_curves.pdf
#
# Usage:
#   python plot_convergence.py \
#       --il_m10  il_m10.txt \
#       --gnn_m10 gnn_il_m10.txt \
#       --il_m30  il_m30.txt \
#       --gnn_m30 gnn_il_m30.txt

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_IL  = "#3B82F6"
C_GNN = "#10B981"


def parse_txt(path: str, window: int = 20):
    """
    Parse avg_cost per episode from view_results.py --detail full output.
    Episode log rows look like:
        ep_num   avg_cost   loss   eps
    e.g.:
             1      1.5235      2.0661      1.0000
    """
    raw = []
    in_log = False

    with open(path, "r", encoding="utf-16") as f:
        for line in f:
            # Detect start of episode log
            if "EPISODE LOG" in line:
                in_log = True
                continue
            if not in_log:
                continue
            # Skip header/separator lines
            if line.strip().startswith("Ep") or \
               line.strip().startswith("---") or \
               line.strip().startswith("Showing") or \
               line.strip() == "":
                continue
            # Stop at next section
            if line.startswith("="):
                break
            # Parse data row: ep  avg_cost  loss  eps
            parts = line.split()
            if len(parts) >= 2:
                try:
                    cost = float(parts[1])
                    raw.append(cost)
                except ValueError:
                    continue

    # Rolling average
    smoothed = []
    for i in range(len(raw)):
        lo = max(0, i - window + 1)
        chunk = raw[lo:i+1]
        smoothed.append(float(np.mean(chunk)) if chunk else None)

    episodes = list(range(1, len(smoothed) + 1))
    return episodes, smoothed


def plot_pair(ax, eps, il_costs, gnn_costs, title):
    il_y  = [v if v is not None else np.nan for v in il_costs]
    gnn_y = [v if v is not None else np.nan for v in gnn_costs]

    ax.plot(eps, il_y,  color=C_IL,  linewidth=1.8,
            label="IL",     alpha=0.9)
    ax.plot(eps, gnn_y, color=C_GNN, linewidth=1.8,
            label="GNN-IL", alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Cost (rolling $w=20$)")
    
    ax.legend(framealpha=0.9)
    ax.set_xlim(1, max(eps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--il_m10",  default="il_m10.txt")
    parser.add_argument("--gnn_m10", default="gnn_il_m10.txt")
    parser.add_argument("--il_m30",  default="il_m30.txt")
    parser.add_argument("--gnn_m30", default="gnn_il_m30.txt")
    args = parser.parse_args()

    eps_il10,  il10  = parse_txt(args.il_m10)
    eps_gnn10, gnn10 = parse_txt(args.gnn_m10)
    eps_il30,  il30  = parse_txt(args.il_m30)
    eps_gnn30, gnn30 = parse_txt(args.gnn_m30)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    plot_pair(ax1, eps_il10, il10, gnn10,
              "$M=10$: Moderate Coordination Regime")
    plot_pair(ax2, eps_il30, il30, gnn30,
              "$M=30$: High Coordination Regime")

    fig.suptitle("Training Convergence: IL vs GNN-IL",
                 fontsize=13, y=1.01)

    plt.tight_layout()
    plt.savefig("figures/convergence_curves.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: figures/convergence_curves.pdf")


if __name__ == "__main__":
    main()