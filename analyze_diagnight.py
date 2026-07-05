"""
One-command analysis of the session-3 `diagnight` sweep (RESEARCH_LOG S18.4).

    python analyze_diagnight.py --out analysis_diag

Reads (all optional — missing blocks are skipped):
  probe_runs/diag/ue{M}/n{N}/{cell}/seed{S}/gnn_il_probe_results.json
      cells: base, floor0.1..0.4, eps0, freezereplay, freezelearn,
             tgt5, tgt50, eps{D}_ep{E} (extended consolidation cells)
  checkpoints/ue{M}/n{N}/seed{S}/{standard,gnn,ctde,ctde_plain,noshare}/*.json
  probe_runs/{poa_deep,poa_n4deep}/poa_M{M}_N{N}.json

Produces in --out:
  summary_diag.md      per-arm tables, dose-response, blowup ordering,
                       2x2 method table, early-stop recovery, PoA tables
  fig_dose_response.png    collapse rate vs eps floor (per M)
  fig_eval_trajectories.png  greedy CRN eval curves per cell
  fig_blowup_leads.png       which signal moves first before divergence
  fig_2x2.png                method x M eval distributions

Key definitions (documented in the output too):
  COLLAPSED        final eval_mean > 0.9 (same readout as the Jul-2 analysis).
  Divergence onset first diag episode where the greedy CRN eval re-crosses
                   0.9 upward after having been < 0.75 (i.e. the run had
                   genuinely converged, then degraded). Runs that never got
                   below 0.75 are "NEVER-LEARNED" (no onset).
  Signal lead      episodes between a signal's first sustained exceedance of
                   its own converged-window baseline (median + 4*MAD) and the
                   onset. Positive = the signal moved BEFORE the eval did.
  Churn proxy      L1/2 distance between consecutive greedy action
                   distributions on the fixed probe batch (diag_act_frac) —
                   aggregate-level policy churn (Schaul et al. 2206.00730).

Interpretation caveats baked into S18.4/S18.6 (pre-registered):
  * freeze_replay is ASYMMETRIC: only "prevents collapse" or "no change" is
    informative; degradation is confounded with the tandem effect
    (Ostrovski et al. 2110.14020).
  * checkpoints/.../ctde/ is CTDE(Bn) (envWithBL — historical hardcode);
    ctde_plain/ is the information-matched 2x2 cell.
  * freeze_learn's eval_mean evaluates the FROZEN policy — its "collapse
    rate" is really "how often was the trigger-time policy already bad".
"""

import argparse
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#0072B2", "#D55E00"          # CVD-safe (converged/collapsed)
COST_THR   = 0.9
CONV_THR   = 0.75
MAD_K      = 4.0
DIAG_KEYS  = ["eval_cost", "q_mean_abs", "q_max_abs", "td_p50", "td_p95",
              "td_max", "grad_p50", "grad_max", "cong_mean", "cong_max",
              "act_entropy"]
LEAD_SIGNALS = ["q_max_abs", "td_p95", "grad_max", "cong_mean", "act_churn"]

METHOD_FILES = [  # (cell dir, results file, label, note)
    ("standard",   "il_results.json",          "IL",          "no share, no graph"),
    ("ctde_plain", "ctde_results.json",        "CTDE-plain",  "share, no graph (2x2 cell)"),
    ("ctde",       "ctde_results.json",        "CTDE(Bn)",    "share, no graph, +Bn obs (fairness cell)"),
    ("noshare",    "gnn_noshare_results.json", "GNN-NoShare", "no share, graph"),
    ("gnn",        "gnn_il_results.json",      "GNN-IL",      "share, graph"),
]


# ─── Loading ────────────────────────────────────────────────────────────────

def load_diag_runs(diag_root):
    """→ list of dicts: cell, M, N, seed, eval, arm, trigger_ep, best_eval,
    best_eval_ep, ep (diag episodes), series {key: np.array}, cost, loss."""
    runs = []
    pat = re.compile(r"ue(\d+)[/\\]n(\d+)[/\\](.+)[/\\]seed(\d+)$")
    for dirpath, _, filenames in os.walk(diag_root):
        if "gnn_il_probe_results.json" not in filenames:
            continue
        m = pat.search(dirpath)
        if not m:
            continue
        with open(os.path.join(dirpath, "gnn_il_probe_results.json")) as f:
            d = json.load(f)
        hist = d["history"]
        de = [e for e in hist if "diag_eval_cost" in e]
        if not de:
            continue
        ep = np.array([e["episode"] for e in de], dtype=float)
        series = {k: np.array([e.get("diag_" + k, np.nan) for e in de],
                              dtype=float) for k in DIAG_KEYS}
        # churn proxy from consecutive action distributions
        fr = [e.get("diag_act_frac") for e in de]
        churn = [np.nan]
        for a, b in zip(fr[:-1], fr[1:]):
            churn.append(0.5 * np.abs(np.array(b) - np.array(a)).sum()
                         if (a is not None and b is not None) else np.nan)
        series["act_churn"] = np.array(churn, dtype=float)
        runs.append(dict(
            M=int(m.group(1)), N=int(m.group(2)), cell=m.group(3).replace("\\", "/"),
            seed=int(m.group(4)), eval=d["eval_mean"],
            arm=d.get("arm", "none"), trigger_ep=d.get("trigger_ep"),
            best_eval=d.get("diag_best_eval"),
            best_eval_ep=d.get("diag_best_eval_ep"),
            ep=ep, series=series,
            episodes=d["config"].get("episodes", int(ep[-1])),
        ))
    return runs


def load_methods(ckpt_root):
    """→ list of dicts: method, M, N, seed, eval, best_train."""
    rows = []
    pat = re.compile(r"ue(\d+)[/\\]n(\d+)[/\\]seed(\d+)$")
    for dirpath, dirnames, _ in os.walk(ckpt_root):
        m = pat.search(dirpath)
        if not m:
            continue
        for sub, fname, label, note in METHOD_FILES:
            p = os.path.join(dirpath, sub, fname)
            if not os.path.exists(p):
                continue
            with open(p) as f:
                d = json.load(f)
            rows.append(dict(method=label, note=note,
                             M=int(m.group(1)), N=int(m.group(2)),
                             seed=int(m.group(3)), eval=d["eval_mean"],
                             best_train=d.get("best_train")))
    return rows


def load_poa(poa_root):
    rows = []
    for sub in ("poa_deep", "poa_n4deep"):
        d = os.path.join(poa_root, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            mm = re.match(r"poa_M(\d+)_N(\d+)\.json", f)
            if mm:
                with open(os.path.join(d, f)) as fh:
                    rows.append(json.load(fh))
    return rows


# ─── Onset + lead analysis ──────────────────────────────────────────────────

def onset_index(ev):
    """First index where eval re-crosses COST_THR upward after having been
    < CONV_THR. Returns (kind, idx): kind in {'converged','never','onset'}."""
    below = np.where(ev < CONV_THR)[0]
    if len(below) == 0:
        return ("never", None)
    first_conv = below[0]
    up = np.where(ev[first_conv:] >= COST_THR)[0]
    if len(up) == 0:
        return ("converged", None)
    return ("onset", first_conv + up[0])


def signal_leads(run):
    """For a run with an onset: per signal, episodes between its first
    sustained baseline exceedance and the eval onset (positive = leads)."""
    ev = run["series"]["eval_cost"]
    kind, oi = onset_index(ev)
    if kind != "onset":
        return kind, None
    below = np.where(ev < CONV_THR)[0]
    ci = below[0]
    if oi - ci < 3:                       # too few converged points to baseline
        return "onset", {}
    leads = {}
    for sig in LEAD_SIGNALS:
        x = run["series"][sig]
        base = x[ci:oi]
        base = base[~np.isnan(base)]
        if len(base) < 3:
            continue
        med = np.median(base)
        mad = np.median(np.abs(base - med)) or (0.1 * abs(med) + 1e-9)
        thr = med + MAD_K * 1.4826 * mad
        # sustained: exceed at two consecutive diag points (or the last point)
        exc = np.where(x >= thr)[0]
        exc = exc[exc >= ci]
        hit = None
        for i in exc:
            if i + 1 >= len(x) or x[i + 1] >= thr:
                hit = i
                break
        if hit is not None:
            leads[sig] = float(run["ep"][oi] - run["ep"][hit])
    return "onset", leads


# ─── Report ─────────────────────────────────────────────────────────────────

def fmt(x, n=4):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag_root", default=os.path.join("probe_runs", "diag"))
    ap.add_argument("--ckpt_root", default="checkpoints")
    ap.add_argument("--poa_root",  default="probe_runs")
    ap.add_argument("--out",       default="analysis_diag")
    cli = ap.parse_args()
    os.makedirs(cli.out, exist_ok=True)

    runs    = load_diag_runs(cli.diag_root)
    methods = load_methods(cli.ckpt_root)
    poa     = load_poa(cli.poa_root)
    L = [f"# diagnight analysis\n",
         f"COLLAPSED = eval_mean > {COST_THR}; onset/lead definitions in "
         f"analyze_diagnight.py docstring (pre-registered caveats in "
         f"RESEARCH_LOG S18.4/S18.6: freeze_replay asymmetric [tandem], "
         f"ctde/ = CTDE(Bn), freeze_learn eval = frozen policy).\n"]

    # ── 1. per-cell arm table ─────────────────────────────────────────
    cells = sorted({(r["M"], r["N"], r["cell"]) for r in runs})
    L.append("## Arms / cells\n")
    L.append("| M | cell | n | collapse | mean eval | med eval | mean trigger_ep | "
             "best-eval rescue* |")
    L.append("|---|------|---|----------|-----------|----------|-----------------|---|")
    for (M, N, cell) in cells:
        rs = [r for r in runs if (r["M"], r["N"], r["cell"]) == (M, N, cell)]
        ev = np.array([r["eval"] for r in rs])
        coll = ev > COST_THR
        trig = [r["trigger_ep"] for r in rs if r["trigger_ep"]]
        # early-stop rescue: collapsed runs whose best greedy CRN eval was good
        resc = [r for r in rs if r["eval"] > COST_THR and r["best_eval"] is not None
                and r["best_eval"] <= COST_THR]
        L.append(f"| {M} | {cell} | {len(rs)} | {int(coll.sum())}/{len(rs)} | "
                 f"{ev.mean():.4f} | {np.median(ev):.4f} | "
                 f"{fmt(np.mean(trig), 0) if trig else '—'} | "
                 f"{len(resc)}/{int(coll.sum())} |")
    L.append("\n*collapsed runs whose best greedy CRN eval during training was "
             "<= 0.9 — the 'early stopping would have rescued it' count.\n")

    # ── 2. dose-response (floors) ─────────────────────────────────────
    L.append("## Dose-response: collapse rate vs eps floor\n")
    floor_of = {"base": 0.05, "floor0.1": 0.1, "floor0.2": 0.2,
                "floor0.3": 0.3, "floor0.4": 0.4}
    fig, ax = plt.subplots(figsize=(6, 4))
    for M in sorted({r["M"] for r in runs}):
        xs, ys, ns = [], [], []
        for cell, fl in sorted(floor_of.items(), key=lambda kv: kv[1]):
            rs = [r for r in runs if r["M"] == M and r["cell"] == cell]
            if not rs:
                continue
            xs.append(fl)
            ys.append(np.mean([r["eval"] > COST_THR for r in rs]))
            ns.append(len(rs))
        if xs:
            ax.plot(xs, ys, "o-", label=f"M={M}")
            L.append(f"M={M}: " + "  ".join(
                f"floor {x:g} -> {y:.0%} (n={n})" for x, y, n in zip(xs, ys, ns)))
    ax.set_xlabel("epsilon floor")
    ax.set_ylabel(f"collapse rate (eval > {COST_THR})")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.set_title("Late-exploration dose vs collapse")
    fig.tight_layout()
    fig.savefig(os.path.join(cli.out, "fig_dose_response.png"), dpi=150)
    plt.close(fig)
    L.append("")

    # ── 3. blowup ordering ────────────────────────────────────────────
    L.append("## Blowup ordering (which signal moves first before divergence)\n")
    all_leads, shapes = {s: [] for s in LEAD_SIGNALS}, {"never": 0, "converged": 0,
                                                        "onset": 0}
    first_mover = {s: 0 for s in LEAD_SIGNALS}
    for r in runs:
        kind, leads = signal_leads(r)
        shapes[kind] += 1
        if kind == "onset" and leads:
            for s, v in leads.items():
                all_leads[s].append(v)
            fm = max(leads.items(), key=lambda kv: kv[1])
            first_mover[fm[0]] += 1
    L.append(f"run shapes: {shapes['converged']} converged-and-stayed, "
             f"{shapes['onset']} converged-then-diverged (onset defined), "
             f"{shapes['never']} never-learned\n")
    L.append("| signal | n | median lead (eps) | p25 | p75 | first mover |")
    L.append("|--------|---|-------------------|-----|-----|-------------|")
    for s in LEAD_SIGNALS:
        v = np.array(all_leads[s])
        if len(v):
            L.append(f"| {s} | {len(v)} | {np.median(v):.0f} | "
                     f"{np.percentile(v, 25):.0f} | {np.percentile(v, 75):.0f} | "
                     f"{first_mover[s]} |")
        else:
            L.append(f"| {s} | 0 | — | — | — | {first_mover[s]} |")
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [all_leads[s] for s in LEAD_SIGNALS]
    ax.boxplot(data, tick_labels=LEAD_SIGNALS)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("lead over eval onset (episodes; >0 = earlier)")
    ax.set_title("Signal leads before divergence")
    plt.setp(ax.get_xticklabels(), rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(cli.out, "fig_blowup_leads.png"), dpi=150)
    plt.close(fig)
    L.append("")

    # ── 4. eval trajectories per cell ─────────────────────────────────
    plot_cells = [c for c in cells if len([r for r in runs
                  if (r["M"], r["N"], r["cell"]) == c]) > 0]
    if plot_cells:
        ncol = 4
        nrow = (len(plot_cells) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(4 * ncol, 2.8 * nrow), squeeze=False)
        for i, (M, N, cell) in enumerate(plot_cells):
            ax = axes[i // ncol][i % ncol]
            for r in runs:
                if (r["M"], r["N"], r["cell"]) != (M, N, cell):
                    continue
                c = ORANGE if r["eval"] > COST_THR else BLUE
                ax.plot(r["ep"], r["series"]["eval_cost"], color=c, lw=1, alpha=0.8)
                if r["trigger_ep"]:
                    ax.axvline(r["trigger_ep"], color="gray", lw=0.5, alpha=0.5)
            ax.axhline(COST_THR, color="gray", ls="--", lw=0.7)
            ax.set_title(f"M={M} {cell}", fontsize=9)
            ax.set_ylim(0.4, 2.0)
        for j in range(len(plot_cells), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle("Greedy CRN eval (blue=converged, orange=collapsed; "
                     "gray line = arm trigger)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(os.path.join(cli.out, "fig_eval_trajectories.png"), dpi=140)
        plt.close(fig)

    # ── 5. 2x2 method table ───────────────────────────────────────────
    if methods:
        L.append("## Method x M (10-seed, same hardware)\n")
        labels = [lab for _, _, lab, _ in METHOD_FILES]
        Ms = sorted({r["M"] for r in methods})
        L.append("| method | " + " | ".join(f"M={M}" for M in Ms) + " |")
        L.append("|--------|" + "|".join(["---"] * len(Ms)) + "|")
        for lab in labels:
            row = [lab]
            for M in Ms:
                ev = np.array([r["eval"] for r in methods
                               if r["method"] == lab and r["M"] == M])
                row.append(f"{ev.mean():.3f}±{ev.std():.3f} ({int((ev > COST_THR).sum())}/{len(ev)}X)"
                           if len(ev) else "—")
            L.append("| " + " | ".join(row) + " |")
        L.append("\n(mean±std eval over seeds; X = collapsed count. "
                 "CTDE(Bn) sees Bn in obs — fairness cell, not the 2x2 cell.)\n")
        fig, ax = plt.subplots(figsize=(7, 4))
        for k, lab in enumerate(labels):
            pts = [(r["M"], r["eval"]) for r in methods if r["method"] == lab]
            if pts:
                x = [p[0] + (k - 2) * 0.6 for p in pts]
                ax.scatter(x, [p[1] for p in pts], s=12, label=lab, alpha=0.7)
        ax.axhline(COST_THR, color="gray", ls="--", lw=0.7)
        ax.set_xlabel("M")
        ax.set_ylabel("eval cost")
        ax.legend(fontsize=8)
        ax.set_title("Per-seed eval by method (jittered)")
        fig.tight_layout()
        fig.savefig(os.path.join(cli.out, "fig_2x2.png"), dpi=150)
        plt.close(fig)

    # ── 6. PoA tables ─────────────────────────────────────────────────
    if poa:
        L.append("## PoA deep (20 BR restarts)\n")
        L.append("| M | N | opt_est | PNE best | PNE worst | PoA_est | PoS_est |")
        L.append("|---|---|---------|----------|-----------|---------|---------|")
        for d in sorted(poa, key=lambda d: (d["N"], d["M"])):
            L.append(f"| {d['M']} | {d['N']} | {fmt(d.get('opt_est'))} | "
                     f"{fmt(d.get('pne_best'))} | {fmt(d.get('pne_worst'))} | "
                     f"{fmt(d.get('pne_worst', np.nan) / d.get('opt_est', np.nan), 3)} | "
                     f"{fmt(d.get('pne_best', np.nan) / d.get('opt_est', np.nan), 3)} |")
        L.append("")

    out_md = os.path.join(cli.out, "summary_diag.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nSummary -> {out_md}; figures -> {cli.out}/")


if __name__ == "__main__":
    main()
