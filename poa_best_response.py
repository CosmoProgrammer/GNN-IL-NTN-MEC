"""
Dynamic-env Price-of-Anarchy check via best-response dynamics  (Tier-1 #2).

Purpose
-------
The stylized one-shot congestion-game probe (RESEARCH_LOG.md §6/§10) showed
PoA ≈ 1.04–1.08: every Nash equilibrium is near-optimal, so GNN-IL's collapsed
mode (~1.37 ≈ >2× optimum) lies OUTSIDE the entire equilibrium set — i.e. the
bimodal collapse is an optimization failure, not equilibrium (mis)selection.
That probe ignored queue memory and stochastic arrivals. This script re-checks
the claim on the REAL `env.py` dynamics.

Method
------
Strategy space is restricted to STATIONARY TARGET POLICIES (as in Chen et al.
2014, arXiv:1404.3200): UE m commits to one target g_m ∈ {0=local, 1..N=UAV}
for the whole episode and uses it whenever it has a task. This is a singleton
player-specific congestion game over episode-level payoffs:

  * Approximate PNE: asynchronous best-response dynamics (round-robin sweeps,
    each UE deviates iff its own mean episode cost improves > tol), from many
    random initial profiles → samples the equilibrium set.
  * Approximate social optimum: the same sweep machinery but each move is
    chosen to minimise SOCIAL cost (team best response), plus the best PNE.

All profile evaluations use COMMON RANDOM NUMBERS (identical np.random seeds
per eval episode) so deviation comparisons are paired and low-variance; the
cost metric is exactly the training/eval metric (per-slot mean over UEs of
env cost, averaged over slots — `info["avg_cost"]`).

Interpretation targets (M=20 reference points, this codebase's history):
  converged GNN-IL ≈ 0.61 · IL ≈ 0.92 · collapsed GNN-IL ≈ 1.37.
Claim to verify: worst-PNE/opt stays small (≈1.1 or less) at all M, and 1.37
sits far above worst-PNE — while converged GNN-IL is near the PNE band.

Caveats (state these in the paper):
  * Stationary targets are a restricted strategy class — the "equilibrium set"
    here is a proxy; state-dependent policies could in principle do better
    (they cannot make the *optimum estimate* worse, so PoA estimates are
    conservative in the direction that matters).
  * With noisy evals, sweeps converge to tol-approximate PNE only.

Usage
-----
    python poa_best_response.py                          # M ∈ {10,20,30}
    python poa_best_response.py --ues 20 --pne_starts 8
Outputs one JSON per (M,N) under --out (default probe_runs/poa/, gitignored)
plus a printed summary table.
"""

import argparse
import json
import os
import time

import numpy as np

from env import NTNMECEnv, EnvConfig


# ─── Profile evaluation (common random numbers) ────────────────────────────

def eval_profile(env, profile, eval_seeds):
    """
    Mean episode cost of a fixed target profile.

    Returns (social_cost, per_ue_cost):
      social_cost : mean over eval episodes of (mean over slots of the
                    per-slot UE-mean cost)  — identical to eval_mean metric.
      per_ue_cost : (M,) same averaging per UE — the payoff each UE best-responds to.
    """
    M = env.cfg.M
    soc_eps = []
    ue_eps  = np.zeros((len(eval_seeds), M))
    for e, seed in enumerate(eval_seeds):
        np.random.seed(seed)          # CRN: same env randomness for every profile
        env.reset()
        slot_costs = []
        ue_sum = np.zeros(M)
        for _ in range(env.cfg.I):
            _, rewards, done, info = env.step(list(profile))
            slot_costs.append(info["avg_cost"])
            ue_sum += -np.asarray(rewards)          # reward = -cost
            if done:
                break
        soc_eps.append(float(np.mean(slot_costs)))
        ue_eps[e] = ue_sum / len(slot_costs)
    return float(np.mean(soc_eps)), ue_eps.mean(axis=0)


class Evaluator:
    """Memoised profile evaluation (profiles recur across sweeps/starts)."""

    def __init__(self, env, eval_seeds):
        self.env = env
        self.eval_seeds = eval_seeds
        self.cache = {}
        self.calls = 0

    def __call__(self, profile):
        key = tuple(profile)
        if key not in self.cache:
            self.cache[key] = eval_profile(self.env, profile, self.eval_seeds)
            self.calls += 1
        return self.cache[key]


# ─── Best-response / team-best-response sweeps ─────────────────────────────

def sweep_to_fixpoint(ev, profile, n_actions, rng, objective, tol, max_sweeps):
    """
    Asynchronous better/best-response dynamics until a full sweep changes
    nothing (a tol-approximate fixed point) or max_sweeps.

    objective = "own"    → each UE minimises its OWN cost      (→ approx PNE)
    objective = "social" → each move minimises SOCIAL cost     (→ local optimum)
    """
    profile = list(profile)
    M = len(profile)
    for _ in range(max_sweeps):
        changed = False
        for m in rng.permutation(M):
            # Incumbent first: a deviation is accepted only if it beats the
            # current action by > tol (ties never cause spurious switches).
            soc, ue = ev(profile)
            best_a = profile[m]
            best_v = ue[m] if objective == "own" else soc
            for a in range(n_actions):
                if a == profile[m]:
                    continue
                cand = profile.copy()
                cand[m] = a
                soc, ue = ev(cand)
                v = ue[m] if objective == "own" else soc
                if v < best_v - tol:
                    best_a, best_v = a, v
            if best_a != profile[m]:
                profile[m] = best_a
                changed = True
        if not changed:
            return profile, True
    return profile, False


def analyse(M, N, args):
    cfg = EnvConfig(M=M, N=N, I=args.steps_per_ep, P_task=args.task_prob)
    env = NTNMECEnv(cfg)
    n_actions = env.n_actions
    eval_seeds = [args.seed * 1000 + e for e in range(args.eval_eps)]
    ev  = Evaluator(env, eval_seeds)
    rng = np.random.default_rng(args.seed)
    t0  = time.time()

    # Reference profiles
    all_local  = [0] * M
    soc_local, _ = ev(all_local)
    rand_soc = float(np.mean([ev(list(rng.integers(0, n_actions, M)))[0]
                              for _ in range(3)]))

    # Approximate PNE set (best-response dynamics from random starts)
    pne, unconverged = [], 0
    for s in range(args.pne_starts):
        start = list(rng.integers(0, n_actions, M))
        prof, ok = sweep_to_fixpoint(ev, start, n_actions, rng,
                                     "own", args.tol, args.max_sweeps)
        pne.append(prof)                      # keep even if sweep-capped…
        unconverged += (0 if ok else 1)       # …but count it
    pne_costs = [ev(p)[0] for p in pne]

    # Approximate social optimum (team-BR local search + best PNE)
    opt_candidates = list(pne_costs)
    for s in range(args.opt_starts):
        start = list(rng.integers(0, n_actions, M)) if s else list(pne[int(np.argmin(pne_costs))])
        prof, _ = sweep_to_fixpoint(ev, start, n_actions, rng,
                                    "social", args.tol, args.max_sweeps)
        opt_candidates.append(ev(prof)[0])
    opt = min(opt_candidates)

    res = {
        "M": M, "N": N,
        "eval_eps": args.eval_eps, "pne_starts": args.pne_starts,
        "opt_starts": args.opt_starts, "tol": args.tol,
        "unconverged_br_runs": unconverged,
        "opt_est":        opt,
        "pne_costs":      sorted(pne_costs),
        "pne_best":       min(pne_costs),
        "pne_worst":      max(pne_costs),
        "poa_est":        max(pne_costs) / opt,
        "pos_est":        min(pne_costs) / opt,
        "all_local_cost": soc_local,
        "random_cost":    rand_soc,
        "profile_evals":  ev.calls,
        "wall_seconds":   round(time.time() - t0, 1),
    }
    return res


# ─── CLI ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ues",   type=int, nargs="+", default=[10, 20, 30])
    ap.add_argument("--n_uavs", type=int, default=2)
    ap.add_argument("--steps_per_ep", type=int, default=100)
    ap.add_argument("--task_prob",    type=float, default=0.3)
    ap.add_argument("--eval_eps",   type=int, default=4,
                    help="CRN episodes per profile evaluation")
    ap.add_argument("--pne_starts", type=int, default=6,
                    help="random initial profiles for best-response dynamics")
    ap.add_argument("--opt_starts", type=int, default=4,
                    help="starts for the team-BR social-optimum search")
    ap.add_argument("--max_sweeps", type=int, default=25)
    ap.add_argument("--tol",  type=float, default=1e-3,
                    help="min per-UE improvement to accept a deviation")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out",  type=str, default=os.path.join("probe_runs", "poa"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"{'M':>4} {'opt_est':>8} {'PNE best':>9} {'PNE worst':>10} "
          f"{'PoA_est':>8} {'PoS_est':>8} {'local':>7} {'random':>7} {'time':>7}")
    print("-" * 78)
    for M in args.ues:
        res = analyse(M, args.n_uavs, args)
        path = os.path.join(args.out, f"poa_M{M}_N{args.n_uavs}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"{M:>4} {res['opt_est']:>8.4f} {res['pne_best']:>9.4f} "
              f"{res['pne_worst']:>10.4f} {res['poa_est']:>8.3f} "
              f"{res['pos_est']:>8.3f} {res['all_local_cost']:>7.4f} "
              f"{res['random_cost']:>7.4f} {res['wall_seconds']:>6.0f}s"
              + ("  [!some BR runs sweep-capped]" if res["unconverged_br_runs"] else ""))
    # ASCII only: this line crashes under cp1252 when stdout is redirected on
    # Windows (UnicodeEncodeError) — results/JSONs are all written before it.
    print(f"\nJSONs in {args.out}/  - compare PNE band vs learned costs "
          f"(collapsed GNN-IL ~1.37 at M=20 should sit FAR above PNE worst).")


if __name__ == "__main__":
    main()
