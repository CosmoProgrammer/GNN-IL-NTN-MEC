"""
Bₙ linear-probe logger for GNN-IL  (NON-INVASIVE wrapper).

Purpose
-------
Tests the load-bearing claim of the "optimization, not coordination" spine:
that bimodal collapse at intermediate M is a STALLED ENCODER — the GNN never
learns to represent the global congestion signal Bₙ — rather than a bad policy
on top of a good representation.

We measure encoder quality φ_t directly with a **linear probe**:

    h_ue  --(linear)-->  Bₙ            (the per-UAV congestion vector)

High probe R²  =  congestion is linearly decodable from the UE embedding
                  =  the encoder "ignited" (φ high).
Low / stalled R² = embedding carries no congestion info (φ ≈ 0, collapsed).

Why this wrapper does NOT perturb training
-------------------------------------------
The probe is fit entirely OFF-LINE from snapshots already sitting in the
agent's GlobalReplayBuffer.  Each buffer entry stores `uav_feats (N,3)`, whose
3rd column is exactly Bₙ (env.py: congestion = #active-edge-queues / M).  We
re-run the *frozen* encoder on the stored (ue_obs, uav_feats, edge_w) to get
h_ue, then solve a closed-form least-squares probe.

Consequences:
  * No env.step(), no ε-greedy, no torch optimiser step during probing.
  * Buffer subsampling uses a private np.random.Generator, never the global
    RNG that training consumes.
  * Therefore the episode-by-episode training trajectory is BIT-IDENTICAL to
    trainGnn.py for the same seed — collapsed seeds stay collapsed.

The actual rollout / learning is delegated to trainGnn.run_episode, so the
dynamics are literally the original code, imported, not reimplemented.

Usage
-----
    python probe_gnn.py --n_ues 20 --seed 42 --save_dir probe_runs/ue20/seed42
"""

import argparse
import json
import os

import numpy as np
import torch

from env      import NTNMECEnv, EnvConfig
from gnnAgent import GNNILAgent
from trainGnn import run_episode, set_seed   # reuse the EXACT training loop


# ─── Linear probe (closed-form least squares) ──────────────────────────────

def _fit_r2(X_tr, Y_tr, X_te, Y_te):
    """
    Ridgeless linear least squares  X·W ≈ Y  (with bias), evaluated as the
    coefficient of determination R² on a held-out split, averaged over the
    output dimensions.  Output dims with ~zero target variance are skipped
    (an undefined R²), so an all-quiet congestion regime reports NaN rather
    than a misleading 0.
    """
    A_tr = np.hstack([X_tr, np.ones((X_tr.shape[0], 1), dtype=X_tr.dtype)])
    A_te = np.hstack([X_te, np.ones((X_te.shape[0], 1), dtype=X_te.dtype)])

    W, *_ = np.linalg.lstsq(A_tr, Y_tr, rcond=None)
    pred  = A_te @ W

    ss_res = ((Y_te - pred) ** 2).sum(axis=0)
    ss_tot = ((Y_te - Y_te.mean(axis=0)) ** 2).sum(axis=0)

    valid = ss_tot > 1e-8
    if not valid.any():
        return float("nan")
    r2 = 1.0 - ss_res[valid] / ss_tot[valid]
    return float(np.mean(r2))


def evaluate_probe(
    agent:         GNNILAgent,
    max_snapshots: int = 512,
    test_frac:     float = 0.3,
    rng:           np.random.Generator = None,
) -> dict:
    """
    Fit the Bₙ probe on the agent's current replay buffer (frozen encoder).

    Returns
    -------
    dict with:
      r2_gnn     : R² of  h_ue → Bₙ          (encoder quality φ proxy)
      r2_rawobs  : R² of  raw_obs → Bₙ        (control; Bₙ is excluded from
                   raw obs by design, so this is the no-message-passing
                   floor — the GNN's lift over this is the signal)
      target_std : std of Bₙ over the sample  (regime diagnostic; ~0 ⇒ R² NaN)
      n_snapshots: snapshots used
    """
    buf = agent.buffer.buf
    if len(buf) < 16:
        return None

    rng = rng or np.random.default_rng(0)
    idx = rng.permutation(len(buf))[:max_snapshots]

    ue_obs    = np.stack([buf[i][0] for i in idx])     # (S, M, obs_dim)
    uav_feats = np.stack([buf[i][1] for i in idx])     # (S, N, 3)
    edge_w    = np.stack([buf[i][2] for i in idx])     # (S, M, N)

    S, M, obs_dim = ue_obs.shape
    N = uav_feats.shape[1]

    # Frozen-encoder embeddings (no grad, no RNG, no dropout in this net)
    with torch.no_grad():
        h_ue = agent.gnn(
            torch.tensor(ue_obs,    dtype=torch.float32, device=agent.device),
            torch.tensor(uav_feats, dtype=torch.float32, device=agent.device),
            torch.tensor(edge_w,    dtype=torch.float32, device=agent.device),
        ).cpu().numpy()                                  # (S, M, gnn_out)

    # Target: Bₙ congestion vector, identical for every UE in a snapshot
    cong = uav_feats[:, :, 2]                            # (S, N)
    Y    = np.repeat(cong, M, axis=0)                    # (S*M, N)

    X_gnn = h_ue.reshape(S * M, -1)                      # (S*M, gnn_out)
    X_raw = ue_obs.reshape(S * M, obs_dim)               # (S*M, obs_dim) control

    # Snapshot-level split so a snapshot's UEs never straddle train/test
    n_test_snap = max(1, int(test_frac * S))
    test_snap   = np.zeros(S, dtype=bool)
    test_snap[rng.permutation(S)[:n_test_snap]] = True
    row_is_test = np.repeat(test_snap, M)

    tr, te = ~row_is_test, row_is_test
    return {
        "r2_gnn":      _fit_r2(X_gnn[tr], Y[tr], X_gnn[te], Y[te]),
        "r2_rawobs":   _fit_r2(X_raw[tr], Y[tr], X_raw[te], Y[te]),
        "target_std":  float(cong.std()),
        "n_snapshots": int(S),
    }


# ─── Training run with per-episode probe logging ───────────────────────────

def train_with_probe(args) -> dict:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    cfg = EnvConfig(
        M      = args.n_ues,
        N      = args.n_uavs,
        I      = args.steps_per_ep,
        P_task = args.task_prob,
    )
    env = NTNMECEnv(cfg)

    agent = GNNILAgent(
        obs_dim       = env.obs_dim,
        n_uavs        = cfg.N,
        n_actions     = env.n_actions,
        gnn_hidden    = args.gnn_hidden,
        gnn_out       = args.gnn_out,
        dqn_hidden    = args.dqn_hidden,
        lr            = args.lr,
        gamma         = args.gamma,
        batch_size    = args.batch_size,
        buffer_cap    = args.buffer_cap,
        target_update = args.target_update,
        eps_decay     = args.eps_decay,
        device        = device,
    )

    # Private RNG for probe subsampling — NEVER the global RNG training uses.
    probe_rng = np.random.default_rng(args.probe_seed)

    history = []
    best_cost = float("inf")

    print(f"\nGNN-IL + Bₙ probe — {args.n_ues} UEs, {args.n_uavs} UAVs, "
          f"{args.episodes} episodes, seed {args.seed}")
    print(f"{'Ep':>5}  {'AvgCost':>9}  {'Loss':>9}  {'Eps':>6}  "
          f"{'R2_gnn':>7}  {'R2_raw':>7}  {'Bstd':>6}")
    print("-" * 62)

    for ep in range(1, args.episodes + 1):
        stats = run_episode(env, agent, train=True)   # identical to trainGnn
        agent.decay_epsilon()

        # ── Off-line probe — does NOT touch global RNG or the env ──────
        probe = None
        if ep % args.probe_every == 0 or ep == args.episodes:
            probe = evaluate_probe(
                agent,
                max_snapshots=args.probe_snapshots,
                rng=probe_rng,
            )

        record = {
            "episode":  ep,
            "avg_cost": stats["avg_cost"],
            "total_loss": stats["total_loss"],
            "eps":      stats["eps"],
        }
        if probe is not None:
            record.update({
                "probe_r2_gnn":    probe["r2_gnn"],
                "probe_r2_rawobs": probe["r2_rawobs"],
                "probe_bn_std":    probe["target_std"],
            })
        history.append(record)

        if stats["avg_cost"] < best_cost:
            best_cost = stats["avg_cost"]

        if ep % args.log_every == 0 or ep == args.episodes:
            r2g = record.get("probe_r2_gnn", float("nan"))
            r2r = record.get("probe_r2_rawobs", float("nan"))
            bst = record.get("probe_bn_std", float("nan"))
            print(f"{ep:>5}  {stats['avg_cost']:>9.4f}  "
                  f"{stats['total_loss']:>9.4f}  {stats['eps']:>6.3f}  "
                  f"{r2g:>7.3f}  {r2r:>7.3f}  {bst:>6.3f}")

    # Final greedy eval — identical protocol to trainGnn (classifies the seed)
    eval_costs = [run_episode(env, agent, train=False)["avg_cost"]
                  for _ in range(args.eval_episodes)]
    eval_mean = float(np.mean(eval_costs))
    eval_std  = float(np.std(eval_costs))

    # Final probe = the φ we end at
    final_probe = evaluate_probe(agent, max_snapshots=args.probe_snapshots,
                                 rng=probe_rng)
    print(f"\nEval ({args.eval_episodes} eps): mean={eval_mean:.4f} "
          f"std={eval_std:.4f}  |  final R2_gnn="
          f"{final_probe['r2_gnn']:.3f}  R2_raw={final_probe['r2_rawobs']:.3f}")

    results = {
        "method":      "GNN-IL+probe",
        "config":      vars(args),
        "history":     history,
        "eval_mean":   eval_mean,
        "eval_std":    eval_std,
        "best_train":  best_cost,
        "final_probe": final_probe,
    }

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "gnn_il_probe_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.save_dir}/gnn_il_probe_results.json")

    return results


# ─── CLI ───────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()

    # Environment — defaults match trainGnn.py; n_ues defaults to the M=20
    # regime where bimodality lives (the experiment of interest).
    p.add_argument("--n_ues",        type=int,   default=20)
    p.add_argument("--n_uavs",       type=int,   default=2)
    p.add_argument("--steps_per_ep", type=int,   default=100)
    p.add_argument("--task_prob",    type=float, default=0.3)

    p.add_argument("--gnn_hidden",   type=int,   default=64)
    p.add_argument("--gnn_out",      type=int,   default=32)
    p.add_argument("--dqn_hidden",   type=int,   default=128)

    p.add_argument("--episodes",      type=int,   default=500)
    p.add_argument("--eval_episodes", type=int,   default=20)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--gamma",         type=float, default=0.9)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--buffer_cap",    type=int,   default=5_000)
    p.add_argument("--target_update", type=int,   default=20)
    p.add_argument("--eps_decay",     type=float, default=0.995)

    # Probe-specific
    p.add_argument("--probe_every",     type=int, default=5,
                   help="probe every K episodes")
    p.add_argument("--probe_snapshots", type=int, default=512,
                   help="max buffer snapshots per probe fit")
    p.add_argument("--probe_seed",      type=int, default=12345,
                   help="private RNG seed for probe subsampling (RNG-isolated)")

    p.add_argument("--seed",      type=int,  default=42)
    p.add_argument("--log_every", type=int,  default=25)
    p.add_argument("--save_dir",  type=str,  default="probe_runs")
    p.add_argument("--cpu",       action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args    = get_args()
    results = train_with_probe(args)
    print(f"\nGNN-IL+probe — eval {results['eval_mean']:.4f} ± "
          f"{results['eval_std']:.4f}  |  final φ (R2_gnn) "
          f"{results['final_probe']['r2_gnn']:.3f}")
