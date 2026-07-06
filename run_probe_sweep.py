"""
Make-or-break experiment driver for the "stalled encoder" hypothesis.

Runs GNN-IL + Bₙ probe across the bimodal seed set at M=20 and prints the
2×2 contingency that the whole "optimization, not coordination" spine rests on:

                       │  encoder IGNITED (high φ)  │  encoder STALLED (low φ)
    ───────────────────┼────────────────────────────┼─────────────────────────
    CONVERGED (low cost)│        predicted           │       falsifies
    COLLAPSED (high cost)│       falsifies            │        predicted

The spine predicts the diagonal: converged⇔ignited, collapsed⇔stalled.

Design
------
Training is delegated to probe_gnn.train_with_probe (a non-invasive wrapper
over the original trainGnn loop). Each seed writes its own results JSON, and
aggregation reads those files back from disk — so the SAME report works whether
seeds were run here sequentially or in parallel across GPUs by launch_parallel.py.

Usage
-----
    python run_probe_sweep.py                          # run + report, M=20
    python run_probe_sweep.py --aggregate_only         # just rebuild the table
    python run_probe_sweep.py --seeds 42 52 62 72 82
"""

import argparse
import json
import os
from argparse import Namespace

from probe_gnn import train_with_probe


# Defaults mirror probe_gnn.get_args / trainGnn.py for a faithful comparison.
_BASE = dict(
    n_uavs=2, steps_per_ep=100, task_prob=0.3,
    gnn_hidden=64, gnn_out=32, dqn_hidden=128,
    eval_episodes=20, lr=1e-3, gamma=0.9,
    batch_size=32, buffer_cap=5_000, target_update=20, eps_decay=0.995,
    eps_end=0.05, init_encoder="", init_full="", eps_start=1.0,
    probe_every=5, probe_snapshots=512, probe_seed=12345,
    snapshot_every=25, probe_dataset_size=2000,
    # late-collapse diagnostics (session 3) — off by default here
    diag_every=0, diag_eval_eps=4, diag_eval_seed=990000, diag_states=256,
    arm="none", trigger_cost=0.9, trigger_mode="raw", trigger_min_ep=0,
    target_encoder=False,
    log_every=100,
)


def seed_dir(out_dir, seed):
    return os.path.join(out_dir, f"seed{seed}")


def result_path(out_dir, seed):
    return os.path.join(seed_dir(out_dir, seed), "gnn_il_probe_results.json")


def run_one_seed(seed, n_ues, episodes, out_dir, cpu, skip_existing=True):
    """Train one seed (or skip if its result JSON already exists)."""
    rpath = result_path(out_dir, seed)
    if skip_existing and os.path.exists(rpath):
        print(f"[seed {seed}] result exists — skipping ({rpath})")
        return
    args = Namespace(
        **_BASE,
        n_ues=n_ues,
        episodes=episodes,
        seed=seed,
        save_dir=seed_dir(out_dir, seed),
        cpu=cpu,
    )
    print("\n" + "=" * 64)
    print(f" SEED {seed}  (M={n_ues})")
    print("=" * 64)
    train_with_probe(args)


def aggregate(out_dir, seeds, cost_threshold, phi_threshold):
    """Read per-seed result JSONs from disk and print the contingency table."""
    rows = []
    for seed in seeds:
        rpath = result_path(out_dir, seed)
        if not os.path.exists(rpath):
            print(f"[seed {seed}] MISSING {rpath} — not yet run.")
            continue
        with open(rpath) as f:
            res = json.load(f)
        fp = res["final_probe"]
        # φ = ΔBₙ probe LIFT over the Bₙ mean-reversion baseline. The static
        # Bₙ probe sits near ceiling even for a random encoder (Bₙ is a raw
        # input on the UAV nodes) and raw ΔBₙ R² inherits a mean-reversion
        # floor the same way — the lift is the passthrough-proof signal.
        # Falls back gracefully for result files predating these probes.
        r2gd = fp.get("r2_gnn_delta")
        base = fp.get("r2_congbase_delta", 0.0)
        phi  = (r2gd - base) if r2gd is not None else fp["r2_gnn"]
        rows.append({
            "seed":       seed,
            "eval_mean":  res["eval_mean"],
            "status":     "CONVERGED" if res["eval_mean"] <= cost_threshold else "COLLAPSED",
            "phi_final":  phi,
            "phi_gnn_d":  r2gd if r2gd is not None else float("nan"),
            "phi_base_d": base,
            "r2_static":  fp["r2_gnn"],
            # NaN φ (all-quiet congestion regime) counts as not-ignited.
            "encoder":    "IGNITED" if (phi == phi and phi >= phi_threshold) else "STALLED",
        })

    if not rows:
        print("\nNo results to aggregate yet.")
        return rows

    print("\n" + "=" * 78)
    print(" RESULT  (φ = ΔBₙ-probe LIFT: R²[h_ue→ΔBₙ] − R²[Bₙ→ΔBₙ] baseline)")
    print("=" * 78)
    print(f"{'seed':>5}  {'eval_mean':>9}  {'status':>10}  "
          f"{'φ(lift)':>8}  {'R2Δgnn':>7}  {'R2Δbase':>8}  {'R2stat':>7}  {'encoder':>8}")
    print("-" * 78)
    for r in rows:
        print(f"{r['seed']:>5}  {r['eval_mean']:>9.4f}  {r['status']:>10}  "
              f"{r['phi_final']:>8.3f}  {r['phi_gnn_d']:>7.3f}  "
              f"{r['phi_base_d']:>8.3f}  {r['r2_static']:>7.3f}  {r['encoder']:>8}")

    cells = {("CONVERGED", "IGNITED"): [], ("CONVERGED", "STALLED"): [],
             ("COLLAPSED", "IGNITED"): [], ("COLLAPSED", "STALLED"): []}
    for r in rows:
        cells[(r["status"], r["encoder"])].append(r["seed"])

    print("\n  2×2 contingency (seed lists):")
    print(f"                 │ encoder IGNITED      │ encoder STALLED")
    print(f"    ─────────────┼──────────────────────┼─────────────────────")
    print(f"    CONVERGED    │ {str(cells[('CONVERGED','IGNITED')]):<20} │ "
          f"{cells[('CONVERGED','STALLED')]}")
    print(f"    COLLAPSED    │ {str(cells[('COLLAPSED','IGNITED')]):<20} │ "
          f"{cells[('COLLAPSED','STALLED')]}")

    diag = len(cells[("CONVERGED", "IGNITED")]) + len(cells[("COLLAPSED", "STALLED")])
    off  = len(cells[("CONVERGED", "STALLED")]) + len(cells[("COLLAPSED", "IGNITED")])
    print(f"\n  on-diagonal (spine-consistent): {diag}/{len(rows)}   "
          f"off-diagonal (spine-violating): {off}/{len(rows)}")
    if off == 0:
        print("  → encoder quality perfectly tracks convergence. Spine SUPPORTED.")
    elif diag == 0:
        print("  → anti-correlated. Spine FALSIFIED — rethink.")
    else:
        print("  → mixed. Inspect per-seed φ trajectories before concluding.")

    summary_path = os.path.join(out_dir, "probe_sweep_summary.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"out_dir": out_dir, "rows": rows,
                   "cost_threshold": cost_threshold,
                   "phi_threshold": phi_threshold}, f, indent=2)
    print(f"\nSummary written to {summary_path}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_ues",    type=int, default=20)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seeds",    type=int, nargs="+",
                    default=[42, 52, 62, 72, 82])
    ap.add_argument("--out_dir",  type=str, default="probe_runs/ue20")
    ap.add_argument("--cost_threshold", type=float, default=0.9,
                    help="eval_mean below this = CONVERGED (between the ~0.6 "
                         "converged and ~1.37 collapsed modes from the log)")
    ap.add_argument("--phi_threshold", type=float, default=0.05,
                    help="final ΔBₙ-probe lift (r2_gnn_delta − "
                         "r2_congbase_delta) above this = encoder IGNITED. "
                         "NOTE: a convenience readout only — the real analysis "
                         "is the per-episode φ trajectories in each history")
    ap.add_argument("--aggregate_only", action="store_true",
                    help="skip training; just rebuild the table from disk")
    ap.add_argument("--force", action="store_true",
                    help="re-run seeds even if their result JSON exists")
    ap.add_argument("--cpu", action="store_true")
    cli = ap.parse_args()

    if not cli.aggregate_only:
        for seed in cli.seeds:
            run_one_seed(seed, cli.n_ues, cli.episodes, cli.out_dir,
                         cli.cpu, skip_existing=not cli.force)

    aggregate(cli.out_dir, cli.seeds, cli.cost_threshold, cli.phi_threshold)


if __name__ == "__main__":
    main()
