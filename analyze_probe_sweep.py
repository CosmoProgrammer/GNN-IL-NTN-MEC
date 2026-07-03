"""
Morning-after analysis for the overnight probe/race/warmstart sweep.

Turns a probe_runs/ tree (any subset of cells present) into:
  * a console + markdown summary (per-cell seed table, collapse rate,
    2x2 contingency verdicts),
  * Fig 1  phi-lift trajectories per cell, seeds coloured by outcome,
  * Fig 2  collapse rate vs exploration budget (the phase-boundary figure),
  * Fig 3  warm-start vs baseline per-seed comparison (if warmstart cells exist),
  * Fig 4  representation-health trajectories (effective rank), coloured by
           outcome (the "not plasticity loss" check).

phi = DELTA-Bn probe LIFT = r2_gnn_delta - r2_congbase_delta  (see probe_gnn.py:
the static Bn probe saturates for a random encoder and MUST NOT be used).
Exploration budget = sum of the actual per-episode eps values in the history
(exact, schedule-agnostic).

Usage
-----
    python analyze_probe_sweep.py                          # ./probe_runs -> ./analysis_out
    python analyze_probe_sweep.py --root probe_runs --out analysis_out
Works on the server or on a locally rsync'd copy of probe_runs/.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Outcome colours (CVD-validated pair; gray is non-data ink only)
C_CONV = "#0072B2"   # converged
C_COLL = "#D55E00"   # collapsed
C_INK  = "#7f7f7f"   # thresholds / reference lines

COST_THRESHOLD = 0.9   # below = CONVERGED (between the ~0.61 / ~1.37 modes)
PHI_THRESHOLD  = 0.05  # phi-lift above = IGNITED (readout convenience)


# --- Loading -----------------------------------------------------------------

def load_runs(root):
    """Return list of dicts, one per completed seed run, with cell metadata."""
    runs = []
    pattern = os.path.join(root, "**", "gnn_il_probe_results.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        rel = os.path.relpath(path, root)
        parts = rel.replace("\\", "/").split("/")[:-1]   # drop filename
        # layout: ue{M}/n{N}[/variant]/seed{S}
        if len(parts) < 3 or not parts[-1].startswith("seed"):
            continue
        variant = parts[2] if len(parts) == 4 else "baseline"
        with open(path) as f:
            res = json.load(f)
        hist = res.get("history", [])
        eps_series = [h["eps"] for h in hist if "eps" in h]
        probed = [h for h in hist if "probe_r2_gnn_delta" in h]
        ep   = np.array([h["episode"] for h in probed])
        lift = np.array([(h["probe_r2_gnn_delta"] or np.nan)
                         - (h.get("probe_r2_congbase_delta") or 0.0)
                         for h in probed], dtype=float)
        rank = np.array([h.get("probe_h_eff_rank", np.nan) for h in probed],
                        dtype=float)
        fp = res.get("final_probe") or {}
        phi_final = ((fp.get("r2_gnn_delta") or np.nan)
                     - (fp.get("r2_congbase_delta") or 0.0))
        cost = np.array([h["avg_cost"] for h in hist], dtype=float)
        runs.append({
            "M": int(parts[0][2:]), "N": int(parts[1][1:]),
            "variant": variant, "seed": int(parts[-1][4:]),
            "cell": f"ue{parts[0][2:]}/n{parts[1][1:]}/{variant}",
            "eval_mean": res["eval_mean"],
            "status": "CONVERGED" if res["eval_mean"] <= COST_THRESHOLD
                      else "COLLAPSED",
            "budget": float(np.sum(eps_series)),
            "probe_ep": ep, "phi_lift": lift, "eff_rank": rank,
            "phi_final": phi_final,
            "encoder": "IGNITED" if (phi_final == phi_final
                                     and phi_final >= PHI_THRESHOLD)
                       else "STALLED",
            "cost_hist": cost,
        })
    return runs


def by_cell(runs):
    cells = defaultdict(list)
    for r in runs:
        cells[r["cell"]].append(r)
    return dict(sorted(cells.items()))


# --- Summary -----------------------------------------------------------------

def summarise(cells, out_dir):
    lines = ["# Probe sweep summary", "",
             f"phi = DELTA-Bn probe lift (r2_gnn_delta - r2_congbase_delta); "
             f"CONVERGED if eval_mean <= {COST_THRESHOLD}; "
             f"IGNITED if final phi >= {PHI_THRESHOLD}.", ""]
    print("\n" + "=" * 96)
    for cell, rs in cells.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        n_coll = sum(r["status"] == "COLLAPSED" for r in rs)
        diag = sum((r["status"] == "CONVERGED") == (r["encoder"] == "IGNITED")
                   for r in rs)
        hdr = (f"{cell}  |  budget~{np.mean([r['budget'] for r in rs]):.0f}  "
               f"|  collapse {n_coll}/{len(rs)}  "
               f"|  2x2 on-diagonal {diag}/{len(rs)}")
        print("\n" + hdr)
        print(f"  {'seed':>5} {'eval':>8} {'status':>10} {'phi_fin':>8} "
              f"{'encoder':>8}")
        lines += [f"## {hdr}", "",
                  "| seed | eval | status | phi_final | encoder |",
                  "|---|---|---|---|---|"]
        for r in rs:
            print(f"  {r['seed']:>5} {r['eval_mean']:>8.4f} {r['status']:>10} "
                  f"{r['phi_final']:>8.3f} {r['encoder']:>8}")
            lines.append(f"| {r['seed']} | {r['eval_mean']:.4f} | {r['status']} "
                         f"| {r['phi_final']:.3f} | {r['encoder']} |")
        lines.append("")
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSummary written to {path}")


# --- Figures -----------------------------------------------------------------

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)


def fig_phi_trajectories(cells, out_dir):
    keys = list(cells)
    ncol = min(4, max(1, len(keys)))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow),
                             squeeze=False, sharey=True)
    for i, cell in enumerate(keys):
        ax = axes[i // ncol][i % ncol]
        for r in cells[cell]:
            c = C_CONV if r["status"] == "CONVERGED" else C_COLL
            if len(r["probe_ep"]):
                # marker so 1-point histories (tiny smoke runs) stay visible
                ms = 2.5 if len(r["probe_ep"]) > 10 else 6
                ax.plot(r["probe_ep"], r["phi_lift"], color=c, lw=1.6,
                        alpha=0.85, marker="o", ms=ms)
        ax.axhline(PHI_THRESHOLD, color=C_INK, lw=1, ls="--")
        ax.set_title(cell, fontsize=9)
        _style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("episode")
    for row in axes:
        row[0].set_ylabel("phi (DELTA-Bn lift)")
    # hide unused panels
    for j in range(len(keys), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.legend(handles=[plt.Line2D([], [], color=C_CONV, lw=2,
                                   label="converged seed"),
                        plt.Line2D([], [], color=C_COLL, lw=2,
                                   label="collapsed seed")],
               loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Encoder ignition: phi-lift trajectories by cell", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_phi_trajectories.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def fig_phase_boundary(cells, out_dir):
    """Collapse rate vs exploration budget, one line per M (race + baseline)."""
    per_m = defaultdict(list)
    for cell, rs in cells.items():
        if "warmstart" in cell:
            continue
        M = rs[0]["M"]
        budget = float(np.mean([r["budget"] for r in rs]))
        rate = np.mean([r["status"] == "COLLAPSED" for r in rs])
        per_m[M].append((budget, rate, cell.split("/")[-1]))
    if not per_m:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    # one line per M, points sorted by budget; direct label at line end
    shades = {M: plt.cm.Blues(0.45 + 0.5 * i / max(1, len(per_m) - 1))
              for i, M in enumerate(sorted(per_m))}
    for M in sorted(per_m):
        pts = sorted(per_m[M])
        xs, ys, tags = zip(*pts)
        ax.plot(xs, ys, "-o", color=shades[M], lw=1.8, ms=6, label=f"M={M}")
        for x, y, t in pts:
            ax.annotate(t, (x, y), textcoords="offset points", xytext=(4, 5),
                        fontsize=7, color="#444444")
    ax.set_xlabel("exploration budget  (sum of eps over training)")
    ax.set_ylabel("collapse rate across seeds")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Race-model phase boundary: collapse rate vs exploration budget")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_phase_boundary.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def fig_warmstart(cells, out_dir):
    """Per-seed eval: baseline vs warmstart at the same M/N (dumbbells)."""
    warm = {c: rs for c, rs in cells.items() if c.endswith("/warmstart")}
    if not warm:
        return
    fig, axes = plt.subplots(1, len(warm), figsize=(5.2 * len(warm), 4.0),
                             squeeze=False)
    for i, (cell, rs) in enumerate(warm.items()):
        base_cell = cell.replace("/warmstart", "/baseline")
        base = {r["seed"]: r for r in cells.get(base_cell, [])}
        ax = axes[0][i]
        for k, r in enumerate(sorted(rs, key=lambda r: r["seed"])):
            b = base.get(r["seed"])
            if b:
                ax.plot([k, k], [b["eval_mean"], r["eval_mean"]],
                        color=C_INK, lw=1, zorder=1)
                ax.scatter([k], [b["eval_mean"]], color=C_COLL, s=42, zorder=2,
                           label="baseline" if k == 0 else None)
            ax.scatter([k], [r["eval_mean"]], color=C_CONV, s=42, zorder=2,
                       label="warm-started" if k == 0 else None)
        ax.axhline(COST_THRESHOLD, color=C_INK, lw=1, ls="--")
        ax.set_xticks(range(len(rs)))
        ax.set_xticklabels([str(r["seed"]) for r in
                            sorted(rs, key=lambda r: r["seed"])])
        ax.set_xlabel("seed")
        ax.set_ylabel("eval mean cost")
        ax.set_title(f"warm start vs baseline  ({cell.rsplit('/', 1)[0]})",
                     fontsize=10)
        ax.legend(frameon=False, fontsize=9)
        _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_warmstart.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def fig_health(cells, out_dir):
    """Effective-rank trajectories coloured by outcome (plasticity check)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    plotted = False
    for cell, rs in cells.items():
        for r in rs:
            if len(r["probe_ep"]) and np.isfinite(r["eff_rank"]).any():
                c = C_CONV if r["status"] == "CONVERGED" else C_COLL
                ms = 2.0 if len(r["probe_ep"]) > 10 else 5
                ax.plot(r["probe_ep"], r["eff_rank"], color=c, lw=1.0,
                        alpha=0.5, marker="o", ms=ms)
                plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("episode")
    ax.set_ylabel("embedding effective rank")
    ax.set_title("Representation health: effective rank "
                 "(plasticity-loss predicts collapse here; we predict it stays healthy)")
    ax.legend(handles=[plt.Line2D([], [], color=C_CONV, lw=2, label="converged"),
                       plt.Line2D([], [], color=C_COLL, lw=2, label="collapsed")],
              frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_health_rank.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# --- CLI ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="probe_runs")
    ap.add_argument("--out",  type=str, default="analysis_out")
    args = ap.parse_args()

    runs = load_runs(args.root)
    if not runs:
        print(f"No gnn_il_probe_results.json under {args.root} - nothing to do.")
        return
    os.makedirs(args.out, exist_ok=True)
    cells = by_cell(runs)
    print(f"Loaded {len(runs)} runs across {len(cells)} cells from {args.root}/")

    summarise(cells, args.out)
    fig_phi_trajectories(cells, args.out)
    fig_phase_boundary(cells, args.out)
    fig_warmstart(cells, args.out)
    fig_health(cells, args.out)
    print(f"Figures written to {args.out}/ "
          f"(fig1 trajectories, fig2 phase boundary, fig3 warmstart, fig4 health)")


if __name__ == "__main__":
    main()
