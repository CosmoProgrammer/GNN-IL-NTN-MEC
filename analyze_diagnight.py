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
  fig_dose_response.png    divergence rate vs eps floor (per M)
  fig_eval_trajectories.png  greedy CRN eval curves per cell
  fig_blowup_leads.png       which signal moves first before divergence
  fig_recovery_split.png     terminal-vs-recovering excursion features (AUC)
  fig_2x2.png                method x M eval distributions

Key definitions (v2, 2026-07-06 — divergence-aware; supersedes the single
final-eval>0.9 "COLLAPSED" label of the Jul-2/Jul-5 analyses):
  Labels are assigned on the SMOOTHED greedy CRN eval trace (rolling
  median-of-3 over diag points, robust to single-point dips/spikes):
    NEVER      smoothed trace never sustained below CONV_THR (0.80) — the run
               never learned; a bad final eval here is a performance reading,
               NOT a divergence event.
    STABLE     converged (smoothed < 0.80) and never re-crossed COST_THR
               (0.90), final eval <= 0.90.
    FATAL      converged, then smoothed trace re-crossed 0.90 (onset), and
               final eval > 0.90 — converged-then-diverged, not rescued.
    RECOVERED  onset occurred but final eval <= 0.90 — diverged then came
               back (the onset-recovery split population).
    END-DEG.   converged, no onset visible in the trace, but final eval
               > 0.90 — divergence inside the last diag interval (or a
               CRN-vs-final gap). Counted with FATAL in divergence rates.
  Hysteresis: convergence threshold 0.80 < divergence threshold 0.90, so
  noise around a single level cannot generate spurious onsets.
  Divergence onset = first smoothed diag point >= 0.90 after convergence.
  Signal lead      episodes between a signal's first sustained exceedance of
                   its own converged-window baseline (median + 4*MAD) and the
                   onset. Positive = the signal moved BEFORE the eval did.
  Churn proxy      L1/2 distance between consecutive greedy action
                   distributions on the fixed probe batch (diag_act_frac) —
                   aggregate-level policy churn (Schaul et al. 2206.00730).

Method-table note (the reason "X" collapse counts were dropped): IL, CTDE and
NoShare runs have NO greedy CRN eval traces, and epsilon-greedy training cost
is not a usable proxy — at high M the queue dynamics make cost swing by ~0.8
for epsilon changes of ~0.05, in BOTH directions (IL M30 training cost RISES
0.68->1.55 as eps decays with no divergence implied; IL M20 trains at ~1.2
while its greedy eval is 0.83). Divergence labels in the method table are
therefore only shown for GNN-IL at M in {20,30}, imported from its
trajectory-identical diag-run twins (verified eval_mean-equal per seed).
Everything else gets distribution statistics (mean, std, range, modes).

Interpretation caveats baked into S18.4/S18.6 (pre-registered):
  * freeze_replay is ASYMMETRIC: only "prevents collapse" or "no change" is
    informative; degradation is confounded with the tandem effect
    (Ostrovski et al. 2110.14020).
  * checkpoints/.../ctde/ is CTDE(Bn) (envWithBL — historical hardcode);
    ctde_plain/ is the information-matched 2x2 cell.
  * freeze_learn's eval_mean evaluates the FROZEN policy — under the v2
    label most of its former "collapses" resolve to NEVER (frozen before
    convergence), which is the correct reading.
"""

import argparse
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COST_THR   = 0.9       # divergence / final-eval threshold
CONV_THR   = 0.80      # sustained-convergence threshold (smoothed trace)
MAD_K      = 4.0
DIAG_KEYS  = ["eval_cost", "q_mean_abs", "q_max_abs", "td_p50", "td_p95",
              "td_max", "grad_p50", "grad_max", "cong_mean", "cong_max",
              "act_entropy"]
LEAD_SIGNALS = ["q_max_abs", "td_p95", "grad_max", "cong_mean", "act_churn"]

# CVD-safe label colors (Okabe-Ito)
LABEL_COLORS = {"stable": "#0072B2", "fatal": "#D55E00",
                "recovered": "#009E73", "end_degraded": "#CC79A7",
                "never": "#999999"}
LABEL_ORDER = ["never", "stable", "recovered", "fatal", "end_degraded"]

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
        # gnn_il_probe_results.json = probe_gnn runs (probe_runs/diag/...);
        # ctde_probe_results.json   = probe_ctde runs (probe_runs/diag_ctde/
        # {plain,bn}/...) — same history/diag_* schema, no probe_* keys.
        fname = next((f for f in ("gnn_il_probe_results.json",
                                  "ctde_probe_results.json")
                      if f in filenames), None)
        if fname is None:
            continue
        m = pat.search(dirpath)
        if not m:
            continue
        with open(os.path.join(dirpath, fname)) as f:
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
            eps_series=np.array([e.get("eps", np.nan) for e in de], dtype=float),
            episodes=d["config"].get("episodes", int(ep[-1])),
        ))
    for r in runs:
        r["smooth"] = smooth3(r["series"]["eval_cost"])
        (r["label"], r["conv_idx"], r["onset_idx"],
         r["term_idx"]) = classify_run(r["series"]["eval_cost"], r["eval"])
        # index the lead analysis anchors to: terminal divergence for fatal
        # runs, first excursion for recovered ones
        r["lead_idx"] = r["term_idx"] if r["label"] == "fatal" else r["onset_idx"]
        r["leads"] = signal_leads(r)
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


# ─── Divergence-aware labeling ──────────────────────────────────────────────

def smooth3(x):
    """Rolling median-of-3 (edges use the available 2-point window)."""
    s = np.empty_like(x)
    for i in range(len(x)):
        s[i] = np.nanmedian(x[max(0, i - 1):i + 2])
    return s


def classify_run(ev, final_eval):
    """Divergence-aware label from the greedy CRN eval trace + final eval.
    → (label, conv_idx, onset_idx, term_idx). onset_idx = first smoothed
    excursion >= COST_THR after convergence (None unless fatal/recovered).
    term_idx (fatal only) = start of the TERMINAL divergence — the final
    smoothed segment >= COST_THR that never comes back; falls back to
    onset_idx if the smoothed trace dips below at the very end. Labels:
    never | stable | fatal | recovered | end_degraded (module docstring)."""
    s = smooth3(np.asarray(ev, dtype=float))
    below = np.where(s < CONV_THR)[0]
    if len(below) == 0:
        return ("never", None, None, None)
    ci = int(below[0])
    up = np.where(s[ci:] >= COST_THR)[0]
    if len(up) == 0:
        if final_eval > COST_THR:
            return ("end_degraded", ci, None, None)
        return ("stable", ci, None, None)
    oi = ci + int(up[0])
    if final_eval <= COST_THR:
        return ("recovered", ci, oi, None)
    ti = oi
    if s[-1] >= COST_THR:
        back = np.where(s[oi:] < COST_THR)[0]
        if len(back):
            ti = oi + int(back[-1]) + 1
    return ("fatal", ci, oi, ti)


def legacy_onset_index(ev):
    """Jul-2 definition (single raw point < 0.75, then raw re-cross of 0.9) —
    kept only for the old-vs-new reconciliation table."""
    below = np.where(ev < 0.75)[0]
    if len(below) == 0:
        return ("never", None)
    first_conv = below[0]
    up = np.where(ev[first_conv:] >= COST_THR)[0]
    if len(up) == 0:
        return ("converged", None)
    return ("onset", first_conv + up[0])


def signal_leads(run):
    """For a run with an onset: per signal, episodes between its first
    sustained baseline exceedance and the eval onset (positive = leads).
    Anchor = terminal divergence for fatal runs, first excursion for
    recovered ones (run['lead_idx'])."""
    ci, oi = run["conv_idx"], run["lead_idx"]
    if oi is None:
        return None
    if oi - ci < 3:                       # too few converged points to baseline
        return {}
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
    return leads


def extract_excursions(run):
    """Maximal smoothed-trace segments >= COST_THR after convergence.
    → list of events dict(start, end, terminal, run). An event is TERMINAL
    iff it runs to the end of the trace in a FATAL run; every other event
    (including earlier excursions of eventually-fatal runs) is RECOVERING."""
    ci = run["conv_idx"]
    if ci is None:
        return []
    s = run["smooth"]
    above = s >= COST_THR
    events, i, n = [], ci, len(s)
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            events.append(dict(start=i, end=j, run=run,
                               terminal=(j == n - 1 and run["label"] == "fatal")))
            i = j + 1
        else:
            i += 1
    return events


# event features for the onset-recovery split (deepener i);
# peak_eval is POST-HOC (max over the whole segment — for terminal events
# that includes the final blowup), kept only as a severity readout. All
# other features are available at event onset.
EXC_FEATURES = (["ep_frac", "eps_at_onset", "onset_height", "rise_rate",
                 "q_max_abs_at"] + [f"z_{s}" for s in LEAD_SIGNALS] +
                ["peak_eval"])


def excursion_features(ev):
    """Features at the START of an excursion event. Signal z-scores are
    robust (median/1.4826*MAD) against the run's own converged baseline,
    with earlier excursion points masked out of the baseline."""
    r, e0, e1 = ev["run"], ev["start"], ev["end"]
    s, ci = r["smooth"], r["conv_idx"]
    f = dict(
        ep_frac=float(r["ep"][e0]) / r["episodes"],
        eps_at_onset=float(r["eps_series"][e0]),
        onset_height=float(s[e0]),
        peak_eval=float(np.nanmax(s[e0:e1 + 1])),
        rise_rate=float(s[e0] - s[e0 - 1]) if e0 > 0 else np.nan,
        q_max_abs_at=float(r["series"]["q_max_abs"][e0]),
    )
    base_idx = np.arange(ci, e0)
    base_idx = base_idx[s[base_idx] < COST_THR]
    for sig in LEAD_SIGNALS:
        x = r["series"][sig]
        base = x[base_idx]
        base = base[~np.isnan(base)]
        if len(base) < 3 or np.isnan(x[e0]):
            f[f"z_{sig}"] = np.nan
            continue
        med = np.median(base)
        mad = np.median(np.abs(base - med)) or (0.1 * abs(med) + 1e-9)
        f[f"z_{sig}"] = float((x[e0] - med) / (1.4826 * mad))
    return f


def auc(a, b):
    """Rank AUC: P(a > b) + 0.5*P(a == b), NaNs dropped."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = (a[:, None] > b[None, :]).mean()
    eq = (a[:, None] == b[None, :]).mean()
    return float(gt + 0.5 * eq)


def gap_split(evs, min_gap=0.25):
    """Multi-modality descriptor: split sorted per-seed evals at every
    adjacent gap > min_gap. → None if unimodal, else list of (k, lo, hi)
    groups (k seeds spanning [lo, hi])."""
    ev = np.sort(np.asarray(evs))
    if len(ev) < 4:
        return None
    cuts = np.where(np.diff(ev) > min_gap)[0]
    if len(cuts) == 0:
        return None
    groups, start = [], 0
    for c in list(cuts) + [len(ev) - 1]:
        groups.append((c - start + 1, float(ev[start]), float(ev[c])))
        start = c + 1
    return groups


# ─── Report ─────────────────────────────────────────────────────────────────

def fmt(x, n=4):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def count_labels(rs):
    return {lab: sum(1 for r in rs if r["label"] == lab) for lab in LABEL_ORDER}


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
    L = [f"# diagnight analysis (v2 labels, divergence-aware)\n",
         f"Labels NEVER/STABLE/RECOVERED/FATAL/END-DEG. assigned on the "
         f"smoothed (median-of-3) greedy CRN eval trace with hysteresis "
         f"(converged < {CONV_THR}, diverged >= {COST_THR}) + final eval; "
         f"full definitions in analyze_diagnight.py docstring. DIV = FATAL + "
         f"END-DEG. Pre-registered caveats unchanged (RESEARCH_LOG "
         f"S18.4/S18.6: freeze_replay asymmetric [tandem], ctde/ = CTDE(Bn), "
         f"freeze_learn eval = frozen policy).\n"]

    # ── 1. per-cell arm table ─────────────────────────────────────────
    cells = sorted({(r["M"], r["N"], r["cell"]) for r in runs})
    L.append("## Arms / cells\n")
    L.append("| M | cell | n | never | stable | recov | fatal | end-deg | "
             "DIV | mean eval | med eval | mean term-div ep | rescue* |")
    L.append("|---|------|---|-------|--------|-------|-------|---------|"
             "----|-----------|----------|---------------|---------|")
    for (M, N, cell) in cells:
        rs = [r for r in runs if (r["M"], r["N"], r["cell"]) == (M, N, cell)]
        ev = np.array([r["eval"] for r in rs])
        c = count_labels(rs)
        div = c["fatal"] + c["end_degraded"]
        onsets = [r["ep"][r["term_idx"]] for r in rs if r["term_idx"] is not None]
        # early-stop rescue: diverged runs whose best greedy CRN eval was good
        resc = [r for r in rs if r["label"] in ("fatal", "end_degraded")
                and r["best_eval"] is not None and r["best_eval"] <= COST_THR]
        L.append(f"| {M} | {cell} | {len(rs)} | {c['never']} | {c['stable']} | "
                 f"{c['recovered']} | {c['fatal']} | {c['end_degraded']} | "
                 f"{div}/{len(rs)} | {ev.mean():.4f} | {np.median(ev):.4f} | "
                 f"{fmt(float(np.mean(onsets)), 0) if onsets else '—'} | "
                 f"{len(resc)}/{div} |")
    L.append("\n*diverged (fatal+end-deg) runs whose best greedy CRN eval during "
             "training was <= 0.9 — the 'early stopping would have rescued it' "
             "count. recov column = onsets that came back on their own.\n")

    # ── 1b. old-vs-new label reconciliation ───────────────────────────
    L.append("## Old-vs-new label reconciliation\n")
    conf = {}
    for r in runs:
        old = "collapsed" if r["eval"] > COST_THR else "ok"
        conf[(old, r["label"])] = conf.get((old, r["label"]), 0) + 1
    L.append("| old (final eval > 0.9) | " + " | ".join(LABEL_ORDER) + " |")
    L.append("|---|" + "|".join(["---"] * len(LABEL_ORDER)) + "|")
    for old in ("collapsed", "ok"):
        L.append(f"| {old} | " + " | ".join(
            str(conf.get((old, lab), 0)) for lab in LABEL_ORDER) + " |")
    n_old = sum(v for (o, _), v in conf.items() if o == "collapsed")
    n_div = sum(1 for r in runs if r["label"] in ("fatal", "end_degraded"))
    n_nev = sum(1 for r in runs if r["label"] == "never")
    L.append(f"\nOld 'collapsed' = {n_old}; of these, genuine divergence "
             f"(fatal/end-deg) = {n_div}, never-learned (performance reading, "
             f"not divergence) = {conf.get(('collapsed', 'never'), 0)}. "
             f"Total never-learned = {n_nev}.\n")
    legacy = {"never": 0, "converged": 0, "onset": 0}
    onset_conf = {}
    for r in runs:
        kind, _ = legacy_onset_index(r["series"]["eval_cost"])
        legacy[kind] += 1
        new = ("onset" if r["onset_idx"] is not None
               else ("never" if r["label"] == "never" else "no-onset"))
        onset_conf[(kind, new)] = onset_conf.get((kind, new), 0) + 1
    L.append(f"Trace-shape counts, legacy (raw single-point) definition: "
             f"{legacy['converged']} converged-and-stayed, {legacy['onset']} "
             f"onset, {legacy['never']} never-learned. Legacy-vs-v2 onset "
             f"cross-tab (legacy kind -> v2 onset/no-onset/never): " +
             ", ".join(f"{k[0]}->{k[1]}: {v}"
                       for k, v in sorted(onset_conf.items())) + "\n")

    # ── 2. dose-response (floors) ─────────────────────────────────────
    L.append("## Dose-response: divergence rate vs eps floor\n")
    floor_of = {"base": 0.05, "floor0.1": 0.1, "floor0.2": 0.2,
                "floor0.3": 0.3, "floor0.4": 0.4}
    fig, ax = plt.subplots(figsize=(6, 4))
    for M in sorted({r["M"] for r in runs}):
        xs, ys, ys_on, ns = [], [], [], []
        for cell, fl in sorted(floor_of.items(), key=lambda kv: kv[1]):
            rs = [r for r in runs if r["M"] == M and r["cell"] == cell]
            if not rs:
                continue
            xs.append(fl)
            ys.append(np.mean([r["label"] in ("fatal", "end_degraded")
                               for r in rs]))
            ys_on.append(np.mean([r["onset_idx"] is not None for r in rs]))
            ns.append(len(rs))
        if xs:
            ln, = ax.plot(xs, ys, "o-", label=f"M={M} div")
            ax.plot(xs, ys_on, "s--", color=ln.get_color(), alpha=0.6,
                    label=f"M={M} any onset")
            L.append(f"M={M}: " + "  ".join(
                f"floor {x:g} -> div {y:.0%} / onset {yo:.0%} (n={n})"
                for x, y, yo, n in zip(xs, ys, ys_on, ns)))
    ax.set_xlabel("epsilon floor")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Late-exploration dose vs divergence (solid) / any onset (dashed)")
    fig.tight_layout()
    fig.savefig(os.path.join(cli.out, "fig_dose_response.png"), dpi=150)
    plt.close(fig)
    L.append("")

    # ── 3. blowup ordering ────────────────────────────────────────────
    L.append("## Blowup ordering (which signal moves first before divergence)\n")
    shapes = count_labels(runs)
    L.append("run shapes: " + ", ".join(
        f"{shapes[lab]} {lab}" for lab in LABEL_ORDER) + "\n")
    L.append("Leads are anchored to the TERMINAL divergence for fatal runs "
             "(start of the final never-recovers segment) and to the first "
             "excursion for recovered runs; the fatal table is the paper "
             "claim, the recovered table is exploratory.\n")
    group_leads = {}
    for group in ("fatal", "recovered"):
        all_leads = {s: [] for s in LEAD_SIGNALS}
        first_mover = {s: 0 for s in LEAD_SIGNALS}
        for r in runs:
            if r["label"] != group:
                continue
            leads = r["leads"]
            if leads:
                for s, v in leads.items():
                    all_leads[s].append(v)
                fm = max(leads.items(), key=lambda kv: kv[1])
                first_mover[fm[0]] += 1
        group_leads[group] = all_leads
        L.append(f"### {group} (n runs with computable leads: "
                 f"{sum(first_mover.values())})\n")
        L.append("| signal | n | median lead (eps) | p25 | p75 | first mover |")
        L.append("|--------|---|-------------------|-----|-----|-------------|")
        for s in LEAD_SIGNALS:
            v = np.array(all_leads[s])
            if len(v):
                L.append(f"| {s} | {len(v)} | {np.median(v):.0f} | "
                         f"{np.percentile(v, 25):.0f} | "
                         f"{np.percentile(v, 75):.0f} | {first_mover[s]} |")
            else:
                L.append(f"| {s} | 0 | — | — | — | {first_mover[s]} |")
        L.append("")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, group in zip(axes, ("fatal", "recovered")):
        data = [group_leads[group][s] for s in LEAD_SIGNALS]
        ax.boxplot(data, tick_labels=LEAD_SIGNALS)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"{group} (anchor: "
                     f"{'terminal divergence' if group == 'fatal' else 'first excursion'})",
                     fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=20)
    axes[0].set_ylabel("lead over eval onset (episodes; >0 = earlier)")
    fig.suptitle("Signal leads before divergence", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(cli.out, "fig_blowup_leads.png"), dpi=150)
    plt.close(fig)
    L.append("")

    # ── 3b. first-mover cross-arm consistency (deepener iv) ───────────
    L.append("## First-mover consistency across cells (fatal runs, "
             "terminal anchor)\n")
    L.append("Checks that the pooled churn-led ordering is not a pooling "
             "(Simpson) artifact. Per-cell n is small (1–7) — read as a "
             "direction-consistency check, not per-cell significance.\n")
    L.append("| M | cell | n leads | top fm | " +
             " | ".join(f"fm {s}" for s in LEAD_SIGNALS) +
             " | med lead churn | med lead q_max |")
    L.append("|---|------|---------|--------|" +
             "|".join(["---"] * len(LEAD_SIGNALS)) + "|---|---|")
    churn_top, churn_pos, cells_with_leads = 0, 0, 0
    for (M, N, cell) in cells:
        rs = [r for r in runs if (r["M"], r["N"], r["cell"]) == (M, N, cell)
              and r["label"] == "fatal" and r["leads"]]
        if not rs:
            continue
        cells_with_leads += 1
        fm = {s: 0 for s in LEAD_SIGNALS}
        cl, ql = [], []
        for r in rs:
            fm[max(r["leads"].items(), key=lambda kv: kv[1])[0]] += 1
            if "act_churn" in r["leads"]:
                cl.append(r["leads"]["act_churn"])
            if "q_max_abs" in r["leads"]:
                ql.append(r["leads"]["q_max_abs"])
        top = max(fm.items(), key=lambda kv: kv[1])
        is_churn_top = fm["act_churn"] == top[1]
        churn_top += is_churn_top
        if cl and np.median(cl) > 0:
            churn_pos += 1
        L.append(f"| {M} | {cell} | {len(rs)} | "
                 f"{'act_churn' if is_churn_top else top[0]} | " +
                 " | ".join(str(fm[s]) for s in LEAD_SIGNALS) +
                 f" | {fmt(np.median(cl), 0) if cl else '—'} | "
                 f"{fmt(np.median(ql), 0) if ql else '—'} |")
    L.append(f"\nact_churn is (co-)top first mover in {churn_top}/"
             f"{cells_with_leads} cells with fatal leads; its median lead is "
             f"positive in {churn_pos}/{cells_with_leads}.\n")

    # ── 3c. onset-recovery split (deepener i) ─────────────────────────
    events = [e for r in runs for e in extract_excursions(r)]
    term = [excursion_features(e) for e in events if e["terminal"]]
    rec  = [excursion_features(e) for e in events if not e["terminal"]]
    if term and rec:
        L.append("## Onset-recovery split (event-level, deepener i)\n")
        n_term_runs = len({id(e['run']) for e in events if e['terminal']})
        n_rec_runs  = len({id(e['run']) for e in events if not e['terminal']})
        durs = [(e["run"]["ep"][e["end"]] - e["run"]["ep"][e["start"]])
                for e in events if not e["terminal"]]
        L.append(f"Excursion events (maximal smoothed segments >= "
                 f"{COST_THR} after convergence): {len(term)} terminal "
                 f"(from {n_term_runs} fatal runs) vs {len(rec)} recovering "
                 f"(from {n_rec_runs} runs — includes earlier excursions of "
                 f"eventually-fatal runs). Median recovering-excursion "
                 f"duration: {np.median(durs):.0f} eps. Features measured at "
                 f"event START; z = robust z vs the run's own converged "
                 f"baseline (earlier excursions masked). AUC = P(terminal > "
                 f"recovering); 0.5 = no separation.\n")
        L.append("| feature | n(term) | n(rec) | med term | med rec | AUC |")
        L.append("|---------|---------|--------|----------|---------|-----|")
        aucs = {}
        for f in EXC_FEATURES:
            a = np.array([t[f] for t in term], dtype=float)
            b = np.array([t[f] for t in rec], dtype=float)
            aucs[f] = auc(a, b)
            L.append(f"| {f} | {int((~np.isnan(a)).sum())} | "
                     f"{int((~np.isnan(b)).sum())} | "
                     f"{fmt(float(np.nanmedian(a)), 3)} | "
                     f"{fmt(float(np.nanmedian(b)), 3)} | "
                     f"{fmt(aucs[f], 3)} |")
        L.append("\n(Caveat: events cluster within runs and cells; AUCs are "
                 "descriptive. ep_frac/eps_at_onset carry the timing-vs-dose "
                 "confound — late excursions face low eps AND less time to "
                 "recover. peak_eval is POST-HOC severity — max over the "
                 "whole segment, not available at onset.)\n")
        fig, ax = plt.subplots(figsize=(6.5, 4))
        feats = sorted(aucs, key=lambda f: aucs[f])
        vals = [aucs[f] for f in feats]
        ax.barh(feats, vals, color=["#D55E00" if abs(v - 0.5) > 0.15
                                    else "#0072B2" for v in vals])
        ax.axvline(0.5, color="gray", lw=0.8)
        ax.set_xlabel("AUC  P(terminal > recovering)")
        ax.set_xlim(0, 1)
        ax.set_title("What separates a terminal excursion from a "
                     "recovering one (at onset)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(cli.out, "fig_recovery_split.png"), dpi=150)
        plt.close(fig)

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
                ax.plot(r["ep"], r["series"]["eval_cost"],
                        color=LABEL_COLORS[r["label"]], lw=1, alpha=0.8)
                if r["lead_idx"] is not None:
                    ax.plot(r["ep"][r["lead_idx"]],
                            r["series"]["eval_cost"][r["lead_idx"]],
                            "x", color=LABEL_COLORS[r["label"]], ms=5)
                if r["trigger_ep"]:
                    ax.axvline(r["trigger_ep"], color="gray", lw=0.5, alpha=0.5)
            ax.axhline(COST_THR, color="gray", ls="--", lw=0.7)
            ax.axhline(CONV_THR, color="gray", ls=":", lw=0.7)
            ax.set_title(f"M={M} {cell}", fontsize=9)
            ax.set_ylim(0.4, 2.0)
        for j in range(len(plot_cells), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle("Greedy CRN eval — blue=stable, orange=fatal, "
                     "green=recovered, pink=end-degraded, gray=never-learned; "
                     "x = onset; gray line = arm trigger", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(os.path.join(cli.out, "fig_eval_trajectories.png"), dpi=140)
        plt.close(fig)

    # ── 5. 2x2 method table ───────────────────────────────────────────
    if methods:
        # diag base runs are trajectory-identical twins of the checkpoint
        # GNN runs (same seed/config; CRN probe is non-invasive) — import
        # their divergence labels where the eval matches exactly.
        twin = {(r["M"], r["N"], r["seed"]): r for r in runs
                if r["cell"] == "base"}
        L.append("## Method x M (10-seed, same hardware)\n")
        labels = [lab for _, _, lab, _ in METHOD_FILES]
        Ms = sorted({r["M"] for r in methods})
        L.append("| method | " + " | ".join(f"M={M}" for M in Ms) + " |")
        L.append("|--------|" + "|".join(["---"] * len(Ms)) + "|")
        for lab in labels:
            row = [lab]
            for M in Ms:
                sel = [r for r in methods if r["method"] == lab and r["M"] == M]
                if not sel:
                    row.append("—")
                    continue
                ev = np.array([r["eval"] for r in sel])
                cell = f"{ev.mean():.3f}±{ev.std():.3f}"
                sp = gap_split(ev)
                if sp:
                    cell += " modes " + "+".join(
                        f"{k}@[{lo:.2f}–{hi:.2f}]" for k, lo, hi in sp)
                if lab == "GNN-IL":
                    tw = [twin.get((r["M"], r["N"], r["seed"])) for r in sel]
                    tw = [t for t, r in zip(tw, sel)
                          if t is not None and abs(t["eval"] - r["eval"]) < 1e-6]
                    if len(tw) == len(sel):
                        c = count_labels(tw)
                        cell += (f" [div {c['fatal'] + c['end_degraded']}, "
                                 f"recov {c['recovered']}, never {c['never']}]")
                row.append(cell)
            L.append("| " + " | ".join(row) + " |")
        L.append("\n(mean±std eval over seeds. Divergence labels [..] only for "
                 "GNN-IL at M with diag-twin traces; other methods have no "
                 "greedy CRN traces and eps-greedy training cost is not a "
                 "usable divergence proxy (see docstring) — their spread/"
                 "modality is reported instead. 'modes k@[a–b]+…' = sorted "
                 "per-seed evals split at every adjacent gap > 0.25. "
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
