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

TWO targets are probed:
  * Bₙ(t)  — static echo. CAUTION: Bₙ is an input feature on the UAV nodes,
    so even a RANDOM encoder passes it through linearly → this probe can sit
    near ceiling from episode 1 (verified in smoke tests). Logged for
    completeness, NOT the discriminator.
  * ΔBₙ = Bₙ(t+1) − Bₙ(t) — predictive. Requires the embedding to encode the
    joint offloading dynamics (who is about to queue where): random
    passthrough cannot do this. This is the φ used for ignition calls.

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

Late-collapse diagnostics (session 3, --diag_every > 0)
--------------------------------------------------------
The Jul-3 overnight sweep falsified the ignition/race reading (RESEARCH_LOG
S18): collapse is largely a LATE divergence of an already-good policy. This
wrapper therefore also carries an anatomy suite ("which quantity blows up
first?") plus intervention arms ("which knob prevents it?"):

  Measurements every --diag_every episodes (all RNG-GUARDED — global python/
  numpy/torch RNG states are saved, reseeded to a fixed diag seed, restored —
  so the training trajectory stays bit-identical to trainGnn.py, and every
  eval is CRN-paired across runs, seeds, and arms):
    * greedy CRN eval on a canonical eval env (the TRUE policy curve,
      decoupled from exploration noise) + per-UAV congestion occupancy;
    * mean/max |Q| + greedy action distribution/entropy on a FIXED
      random-policy state batch (identical across all runs);
    * per-sample |TD error| percentiles from a private-RNG buffer sample;
    * PRE-clip grad norms (clip-at-10 may be silently saturating), captured
      by wrapping torch.nn.utils.clip_grad_norm_ — training math unchanged.

  Intervention arms (--arm, fire ONCE when the greedy eval first reaches
  --trigger_cost):
    * eps0          : kill exploration        → "late exploration is the trigger"
    * freeze_replay : stop storing, keep training → data poisoning vs bootstrapping
    * freeze_learn  : stop training, keep acting  → "policy was fine" + early-stop control
  (--eps_end floors and --target_update cells need no trigger — plain CLI args.)

Usage
-----
    python probe_gnn.py --n_ues 20 --seed 42 --save_dir probe_runs/ue20/seed42
    python probe_gnn.py --n_ues 20 --seed 42 --diag_every 10 --arm freeze_replay
"""

import argparse
import json
import os
import random

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

    per_dim = np.full(Y_te.shape[1], np.nan)        # R² per output (per-UAV)
    valid = ss_tot > 1e-8
    per_dim[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
    mean = float(np.nanmean(per_dim)) if valid.any() else float("nan")
    return mean, [float(x) for x in per_dim]


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
      h_eff_rank : effective rank of h_ue (entropy of sq. singular values) —
                   distinguishes "never ignited" (our story) from degenerate
                   feature-rank collapse (the plasticity-loss story)
      h_dormant_frac : fraction of embedding dims with ~zero variance
    """
    buf = agent.buffer.buf
    if len(buf) < 16:
        return None

    rng = rng or np.random.default_rng(0)
    idx = rng.permutation(len(buf))[:max_snapshots]

    ue_obs    = np.stack([buf[i][0] for i in idx])     # (S, M, obs_dim)
    uav_feats = np.stack([buf[i][1] for i in idx])     # (S, N, 3)
    edge_w    = np.stack([buf[i][2] for i in idx])     # (S, M, N)
    nxt_uav   = np.stack([buf[i][6] for i in idx])     # (S, N, 3) next slot

    S, M, obs_dim = ue_obs.shape
    N = uav_feats.shape[1]

    # Frozen-encoder embeddings (no grad, no RNG, no dropout in this net)
    with torch.no_grad():
        h_ue = agent.gnn(
            torch.tensor(ue_obs,    dtype=torch.float32, device=agent.device),
            torch.tensor(uav_feats, dtype=torch.float32, device=agent.device),
            torch.tensor(edge_w,    dtype=torch.float32, device=agent.device),
        ).cpu().numpy()                                  # (S, M, gnn_out)

    # Static target: Bₙ congestion vector, identical for every UE in a snapshot
    cong = uav_feats[:, :, 2]                            # (S, N)
    Y    = np.repeat(cong, M, axis=0)                    # (S*M, N)

    # Predictive target: next-slot congestion CHANGE (passthrough-proof φ)
    d_cong  = nxt_uav[:, :, 2] - cong                    # (S, N)
    Y_delta = np.repeat(d_cong, M, axis=0)               # (S*M, N)

    X_gnn = h_ue.reshape(S * M, -1)                      # (S*M, gnn_out)
    X_raw = ue_obs.reshape(S * M, obs_dim)               # (S*M, obs_dim) control

    # Representation-health metrics (rebuttal ammunition: a stalled encoder in
    # our story has healthy rank but no Bₙ content; plasticity/capacity loss
    # predicts degenerate rank / dormant units instead).
    Xc = X_gnn - X_gnn.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(Xc, compute_uv=False)
    p_sv = sv ** 2 / max(float((sv ** 2).sum()), 1e-12)
    h_eff_rank = float(np.exp(-np.sum(p_sv * np.log(p_sv + 1e-12))))
    h_dormant  = float((X_gnn.std(axis=0) < 1e-5).mean())

    # Snapshot-level split so a snapshot's UEs never straddle train/test
    n_test_snap = max(1, int(test_frac * S))
    test_snap   = np.zeros(S, dtype=bool)
    test_snap[rng.permutation(S)[:n_test_snap]] = True
    row_is_test = np.repeat(test_snap, M)

    tr, te = ~row_is_test, row_is_test
    r2g_mean, r2g_per_uav = _fit_r2(X_gnn[tr], Y[tr], X_gnn[te], Y[te])
    r2r_mean, _           = _fit_r2(X_raw[tr], Y[tr], X_raw[te], Y[te])
    r2gd_mean, r2gd_per   = _fit_r2(X_gnn[tr], Y_delta[tr], X_gnn[te], Y_delta[te])
    r2rd_mean, _          = _fit_r2(X_raw[tr], Y_delta[tr], X_raw[te], Y_delta[te])

    # Mean-reversion baseline: predict ΔBₙ from Bₙ(t) alone. ΔBₙ is negatively
    # correlated with Bₙ (queues drain), and Bₙ passes through even a random
    # encoder — so the encoder's LIFT over this baseline, not raw ΔBₙ R², is
    # the genuine "learned the joint dynamics" signal.
    X_cong = np.repeat(cong, M, axis=0)                  # (S*M, N)
    r2cd_mean, _ = _fit_r2(X_cong[tr], Y_delta[tr], X_cong[te], Y_delta[te])
    return {
        "r2_gnn":         r2g_mean,
        "r2_gnn_per_uav": r2g_per_uav,    # length-N list, NaN where Bₙ ≈ const
        "r2_rawobs":      r2r_mean,
        "r2_gnn_delta":   r2gd_mean,
        "r2_gnn_delta_per_uav": r2gd_per,
        "r2_rawobs_delta": r2rd_mean,     # control: own-queue features only
        "r2_congbase_delta": r2cd_mean,   # control: Bₙ mean-reversion floor
        # φ = r2_gnn_delta − r2_congbase_delta  (the ignition discriminator)
        "target_std":     float(cong.std()),
        "delta_std":      float(d_cong.std()),
        "n_snapshots":    int(S),
        "h_eff_rank":     h_eff_rank,
        "h_dormant_frac": h_dormant,
    }


# ─── Off-line re-probing support ───────────────────────────────────────────

def dump_probe_dataset(agent, path, max_snapshots, rng):
    """
    Save a FIXED set of buffer snapshots (ue_obs, uav_feats, edge_w) to .npz.

    Combined with the per-episode encoder snapshots, this lets ANY future probe
    methodology (nonlinear, per-UAV, held-out, different target) be recomputed
    offline across the whole training trajectory — no training re-run needed.
    Bₙ is recoverable as uav_feats[:, :, 2].
    """
    buf = agent.buffer.buf
    if len(buf) == 0:
        return
    idx = rng.permutation(len(buf))[:max_snapshots]
    np.savez_compressed(
        path,
        ue_obs        = np.stack([buf[i][0] for i in idx]).astype(np.float32),
        uav_feats     = np.stack([buf[i][1] for i in idx]).astype(np.float32),
        edge_w        = np.stack([buf[i][2] for i in idx]).astype(np.float32),
        actions       = np.stack([buf[i][3] for i in idx]).astype(np.int64),
        rewards       = np.stack([buf[i][4] for i in idx]).astype(np.float32),
        next_uav_feats= np.stack([buf[i][6] for i in idx]).astype(np.float32),
        task_mask     = np.stack([buf[i][9] for i in idx]),
    )


# ─── Late-collapse diagnostics (session-3 anatomy suite) ───────────────────

class _RNGGuard:
    """
    Save → reseed → restore ALL global RNG streams (python `random`,
    `np.random`, torch CPU + CUDA). env.py consumes the GLOBAL numpy RNG for
    mobility and task arrivals, so any mid-training rollout would perturb the
    training trajectory without this guard. Inside the guard everything sees
    a fixed seed → evals are CRN-paired across runs/seeds/arms; outside, the
    training streams continue exactly where they left off (bit-identity with
    trainGnn.py preserved for the baseline arm).
    """

    def __init__(self, seed: int):
        self.seed = seed

    def __enter__(self):
        self._py = random.getstate()
        self._np = np.random.get_state()
        self._th = torch.get_rng_state()
        self._cu = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        return self

    def __exit__(self, *exc):
        random.setstate(self._py)
        np.random.set_state(self._np)
        torch.set_rng_state(self._th)
        if self._cu is not None:
            torch.cuda.set_rng_state_all(self._cu)
        return False


class _GradNormRecorder:
    """
    Record PRE-clip total grad norms by wrapping torch.nn.utils.clip_grad_norm_
    (which returns the norm BEFORE clipping). gnnAgent resolves the function
    at call time via `nn.utils.clip_grad_norm_`, so patching the module
    attribute is visible there. Training math is completely unchanged.
    """

    def __init__(self):
        self.norms = []
        self._orig = None

    def install(self):
        self._orig = torch.nn.utils.clip_grad_norm_
        rec = self

        def _wrapper(parameters, max_norm, *a, **kw):
            tn = rec._orig(parameters, max_norm, *a, **kw)
            rec.norms.append(float(tn))
            return tn

        torch.nn.utils.clip_grad_norm_ = _wrapper

    def uninstall(self):
        if self._orig is not None:
            torch.nn.utils.clip_grad_norm_ = self._orig
            self._orig = None

    def drain(self):
        out, self.norms = self.norms, []
        return out


def build_diag_states(cfg, n_states: int, seed: int):
    """
    A FIXED batch of random-policy states (ue_obs, uav_feats, edge_w), built
    under the RNG guard with a fixed seed → identical for every run, seed and
    arm. Q statistics measured on this batch are therefore directly
    comparable everywhere (paired across the whole experiment).
    """
    with _RNGGuard(seed):
        env = NTNMECEnv(cfg)
        ue, uav, ew = [], [], []
        obs_list = env.reset()
        graph = env.get_graph_data()
        while len(ue) < n_states:
            ue.append(np.stack(obs_list).astype(np.float32))
            uav.append(np.asarray(graph["uav_x"], dtype=np.float32).copy())
            ew.append(GNNILAgent.extract_edge_matrix(graph))
            actions = [np.random.randint(env.n_actions)
                       for _ in range(env.n_agents)]
            obs_list, _, done, _ = env.step(actions)
            graph = env.get_graph_data()
            if done:
                obs_list = env.reset()
                graph = env.get_graph_data()
    return np.stack(ue), np.stack(uav), np.stack(ew)


def diag_q_stats(agent: GNNILAgent, diag_states) -> dict:
    """Mean/max |Q|, greedy action fractions and entropy on the fixed batch."""
    ue, uav, ew = diag_states
    with torch.no_grad():
        ue_t  = torch.tensor(ue,  dtype=torch.float32, device=agent.device)
        uav_t = torch.tensor(uav, dtype=torch.float32, device=agent.device)
        ew_t  = torch.tensor(ew,  dtype=torch.float32, device=agent.device)
        h        = agent.gnn(ue_t, uav_t, ew_t)
        enriched = torch.cat([ue_t, h], dim=-1)
        S, M, _  = ue.shape
        q        = agent.policy_net(enriched.view(S * M, -1))    # (S*M, A)
        q_abs    = q.abs()
        acts     = q.argmax(dim=1).cpu().numpy()
    frac = np.bincount(acts, minlength=agent.n_actions) / float(len(acts))
    nz = frac[frac > 0]
    return {
        "q_mean_abs":  float(q_abs.mean()),
        "q_max_abs":   float(q_abs.max()),
        "act_frac":    [float(x) for x in frac],
        "act_entropy": float(-(nz * np.log(nz)).sum()),
    }


def diag_td_stats(agent: GNNILAgent, rng: np.random.Generator,
                  n_samples: int = 128):
    """
    Per-sample |TD error| distribution over a private-RNG buffer sample —
    mirrors train_step()'s Double-DQN target math under no_grad, restricted
    to task-mask rows (exactly the rows the loss trains on).
    """
    buf = agent.buffer.buf
    if len(buf) < 8:
        return None
    idx = rng.permutation(len(buf))[:n_samples]
    batch = [buf[i] for i in idx]
    (ue_obs, uav_feats, edge_w, actions, rewards,
     next_ue_obs, next_uav_feats, next_edge_w, dones, task_masks) = zip(*batch)
    dev = agent.device
    t = lambda x, dt: torch.tensor(np.stack(x), dtype=dt, device=dev)
    ue_obs, uav_feats, edge_w = (t(ue_obs, torch.float32),
                                 t(uav_feats, torch.float32),
                                 t(edge_w, torch.float32))
    next_ue_obs, next_uav_feats, next_edge_w = (
        t(next_ue_obs, torch.float32), t(next_uav_feats, torch.float32),
        t(next_edge_w, torch.float32))
    actions    = t(actions, torch.long)
    rewards    = t(rewards, torch.float32)
    dones      = torch.tensor(np.array(dones), dtype=torch.float32, device=dev)
    task_masks = t(task_masks, torch.bool)

    with torch.no_grad():
        B, M, _ = ue_obs.shape
        h      = agent.gnn(ue_obs, uav_feats, edge_w)
        q_all  = agent.policy_net(
            torch.cat([ue_obs, h], dim=-1).view(B * M, -1))
        q_pred = q_all.gather(1, actions.view(B * M, 1)).squeeze(1)

        h_n    = agent.gnn(next_ue_obs, next_uav_feats, next_edge_w)
        enr_n  = torch.cat([next_ue_obs, h_n], dim=-1).view(B * M, -1)
        na     = agent.policy_net(enr_n).argmax(dim=1)
        nq     = agent.target_net(enr_n).gather(1, na.unsqueeze(1)).squeeze(1)
        dn     = dones.unsqueeze(1).expand(B, M).reshape(B * M)
        target = rewards.view(B * M) + agent.gamma * nq * (1.0 - dn)
        td     = (q_pred - target).abs()[task_masks.view(B * M)]

    if td.numel() == 0:
        return None
    tdn = td.cpu().numpy()
    return {"td_p50": float(np.percentile(tdn, 50)),
            "td_p95": float(np.percentile(tdn, 95)),
            "td_max": float(tdn.max())}


def diag_greedy_eval(agent: GNNILAgent, eval_env: NTNMECEnv,
                     n_eps: int, seed: int) -> dict:
    """
    Greedy rollouts on the canonical eval env under the RNG guard → identical
    episodes every call, in every run (CRN). Also records per-UAV congestion
    occupancy (the queue-blowup / env-feedback signature).
    """
    costs, cong_means, cong_max = [], [], 0.0
    with _RNGGuard(seed):
        for _ in range(n_eps):
            obs_list = eval_env.reset()
            graph = eval_env.get_graph_data()
            step_costs, congs = [], []
            for _ in range(eval_env.cfg.I):
                congs.append(
                    np.asarray(graph["uav_x"], dtype=np.float32)[:, 2])
                acts = agent.select_actions(obs_list, graph, greedy=True)
                obs_list, _, done, info = eval_env.step(acts)
                graph = eval_env.get_graph_data()
                step_costs.append(info["avg_cost"])
                if done:
                    break
            costs.append(float(np.mean(step_costs)))
            c = np.stack(congs)
            cong_means.append(float(c.mean()))
            cong_max = max(cong_max, float(c.max()))
    return {"eval_cost": float(np.mean(costs)),
            "eval_std":  float(np.std(costs)),
            "cong_mean": float(np.mean(cong_means)),
            "cong_max":  cong_max}


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
        eps_start     = getattr(args, "eps_start", 1.0),
        eps_decay     = args.eps_decay,
        eps_end       = getattr(args, "eps_end", 0.05),
        device        = device,
        target_encoder = getattr(args, "target_encoder", False),
    )
    if getattr(args, "target_encoder", False):
        print("Fully-frozen bootstrap target: encoder target copy ON "
              "(cold-review artifact control)")

    # Race-model fix #2 (warm start) / #3 (curriculum): initialise the ENCODER
    # from a trained donor checkpoint. GNN weights are M-agnostic, so a donor
    # trained at a different M (e.g. M=30 → M=20) is the curriculum transfer.
    # The policy net stays randomly initialised — the intervention is purely
    # "start above the ignition threshold φ_c", nothing else.
    init_encoder = getattr(args, "init_encoder", "")
    if init_encoder:
        ckpt = torch.load(init_encoder, map_location=agent.device)
        agent.gnn.load_state_dict(ckpt["gnn"])
        print(f"Warm-started encoder from {init_encoder} "
              f"(policy net remains randomly initialised)")

    # Resurrection (S18.5): restart from a mid-training snapshot — loads gnn +
    # policy + target. Combine with --eps_start set to the eps at the snapshot
    # episode to resume the schedule, and a different --seed for a different
    # RNG stream (deterministic doom vs knife-edge chance). Caveat: the replay
    # buffer is NOT checkpointed — it refills from scratch (weights-only
    # resurrection; document in any writeup).
    init_full = getattr(args, "init_full", "")
    if init_full:
        agent.load(init_full)
        print(f"Resurrection: loaded gnn+policy(+target) from {init_full}, "
              f"eps starts at {agent.eps:.3f}")

    # Private RNG for probe subsampling — NEVER the global RNG training uses.
    probe_rng = np.random.default_rng(args.probe_seed)

    # ── Late-collapse diagnostics setup (S18 anatomy suite) ────────────
    diag_every     = getattr(args, "diag_every", 0)
    arm            = getattr(args, "arm", "none")
    trigger_cost   = getattr(args, "trigger_cost", 0.9)
    trigger_mode   = getattr(args, "trigger_mode", "raw")
    trigger_min_ep = getattr(args, "trigger_min_ep", 0)
    diag_eval_eps  = getattr(args, "diag_eval_eps", 4)
    diag_eval_seed = getattr(args, "diag_eval_seed", 990000)
    n_diag_states  = getattr(args, "diag_states", 256)

    diag_states = eval_env = grad_rec = None
    trigger_ep = None
    recent_evals = []      # rolling CRN evals for the smoothed trigger
    best_diag_eval, best_diag_ep = float("inf"), None
    if diag_every > 0:
        # Fixed probe-state batch + canonical eval env: both built under the
        # guard with fixed seeds, so they are IDENTICAL across runs and arms.
        diag_states = build_diag_states(cfg, n_diag_states, diag_eval_seed + 1)
        with _RNGGuard(diag_eval_seed):
            eval_env = NTNMECEnv(cfg)
        grad_rec = _GradNormRecorder()
        grad_rec.install()
        print(f"Diagnostics ON: every {diag_every} eps, arm={arm}, "
              f"trigger_cost={trigger_cost}")

    # Encoder snapshots over training → enables offline re-probing later.
    snap_dir = os.path.join(args.save_dir, "snapshots") if args.save_dir else None
    if snap_dir and args.snapshot_every > 0:
        os.makedirs(snap_dir, exist_ok=True)
    snapshot_eps = []

    history = []
    best_cost = float("inf")

    print(f"\nGNN-IL + Bₙ probe — {args.n_ues} UEs, {args.n_uavs} UAVs, "
          f"{args.episodes} episodes, seed {args.seed}")
    print(f"{'Ep':>5}  {'AvgCost':>9}  {'Loss':>9}  {'Eps':>6}  "
          f"{'R2_gnn':>7}  {'R2_raw':>7}  {'R2Δgnn':>7}  {'R2Δraw':>7}  {'Bstd':>6}")
    print("-" * 80)

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
                "probe_r2_gnn":        probe["r2_gnn"],
                "probe_r2_gnn_per_uav": probe["r2_gnn_per_uav"],
                "probe_r2_rawobs":     probe["r2_rawobs"],
                "probe_r2_gnn_delta":  probe["r2_gnn_delta"],
                "probe_r2_raw_delta":  probe["r2_rawobs_delta"],
                "probe_r2_congbase_delta": probe["r2_congbase_delta"],
                "probe_bn_std":        probe["target_std"],
                "probe_dbn_std":       probe["delta_std"],
                "probe_h_eff_rank":    probe["h_eff_rank"],
                "probe_h_dormant":     probe["h_dormant_frac"],
            })

        # ── Diagnostics: measure, then (maybe) fire the arm ────────────
        if diag_every > 0 and (ep % diag_every == 0 or ep == args.episodes):
            diag = {}
            diag.update(diag_greedy_eval(agent, eval_env,
                                         diag_eval_eps, diag_eval_seed))
            diag.update(diag_q_stats(agent, diag_states))
            td = diag_td_stats(agent, probe_rng)
            if td is not None:
                diag.update(td)
            g = grad_rec.drain()
            if g:
                diag["grad_p50"] = float(np.percentile(g, 50))
                diag["grad_max"] = float(np.max(g))

            if diag["eval_cost"] < best_diag_eval:
                best_diag_eval, best_diag_ep = diag["eval_cost"], ep

            # Trigger statistic. "raw" = the historical behaviour (single CRN
            # eval ≤ threshold — degenerate at M=20, where a near-untrained
            # policy already evals ≤ 0.9 at ep 10). "smoothed" = median of the
            # last 3 CRN evals, matching the v2-label convergence definition,
            # combined with --trigger_min_ep so the arm can only fire on a
            # genuinely converged policy.
            recent_evals.append(diag["eval_cost"])
            if trigger_mode == "smoothed":
                trig_val = (float(np.median(recent_evals[-3:]))
                            if len(recent_evals) >= 3 else float("inf"))
            else:
                trig_val = diag["eval_cost"]

            if (arm != "none" and trigger_ep is None
                    and ep >= trigger_min_ep
                    and trig_val <= trigger_cost):
                trigger_ep = ep
                if arm == "eps0":
                    agent.eps = 0.0
                    agent.eps_end = 0.0
                elif arm == "freeze_replay":
                    agent.store = lambda *a, **k: None
                elif arm == "freeze_learn":
                    agent.train_step = lambda: None
                print(f"[arm] {arm} TRIGGERED at ep {ep} "
                      f"(greedy eval {diag['eval_cost']:.4f} "
                      f"<= {trigger_cost})")

            record.update({f"diag_{k}": v for k, v in diag.items()})
            print(f"    [diag ep {ep}] eval={diag['eval_cost']:.4f}  "
                  f"|Q|max={diag.get('q_max_abs', float('nan')):.2f}  "
                  f"td95={diag.get('td_p95', float('nan')):.3f}  "
                  f"gradmax={diag.get('grad_max', float('nan')):.1f}  "
                  f"cong={diag.get('cong_mean', float('nan')):.3f}")

        history.append(record)

        # Snapshot the encoder+policy for offline re-probing of the trajectory.
        if snap_dir and args.snapshot_every > 0 and (
                ep % args.snapshot_every == 0 or ep == args.episodes):
            agent.save(os.path.join(snap_dir, f"ep{ep}.pt"))
            snapshot_eps.append(ep)

        if stats["avg_cost"] < best_cost:
            best_cost = stats["avg_cost"]

        if ep % args.log_every == 0 or ep == args.episodes:
            r2g  = record.get("probe_r2_gnn", float("nan"))
            r2r  = record.get("probe_r2_rawobs", float("nan"))
            r2gd = record.get("probe_r2_gnn_delta", float("nan"))
            r2rd = record.get("probe_r2_raw_delta", float("nan"))
            bst  = record.get("probe_bn_std", float("nan"))
            print(f"{ep:>5}  {stats['avg_cost']:>9.4f}  "
                  f"{stats['total_loss']:>9.4f}  {stats['eps']:>6.3f}  "
                  f"{r2g:>7.3f}  {r2r:>7.3f}  {r2gd:>7.3f}  {r2rd:>7.3f}  "
                  f"{bst:>6.3f}")

    if grad_rec is not None:
        grad_rec.uninstall()

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
          f"{final_probe['r2_gnn']:.3f}  R2_raw={final_probe['r2_rawobs']:.3f}"
          f"  R2Δgnn={final_probe['r2_gnn_delta']:.3f}"
          f"  R2Δraw={final_probe['r2_rawobs_delta']:.3f}")

    results = {
        "method":      "GNN-IL+probe",
        "config":      vars(args),
        "history":     history,
        "eval_mean":   eval_mean,
        "eval_std":    eval_std,
        "best_train":  best_cost,
        "final_probe": final_probe,
        "snapshot_eps": snapshot_eps,
    }
    if diag_every > 0:
        results.update({
            "arm":              arm,
            "trigger_ep":       trigger_ep,
            # best greedy CRN eval seen during training + when: this is the
            # "would early stopping have rescued this run?" number.
            "diag_best_eval":    best_diag_eval,
            "diag_best_eval_ep": best_diag_ep,
        })

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        # Final weights — self-contained for offline behaviour/embedding analysis
        # (offload/local/drop rates, t-SNE, etc. are all recomputable from these).
        agent.save(os.path.join(args.save_dir, "gnn_il_probe_final.pt"))
        # Fixed probe dataset → re-probe any snapshot offline on a consistent set.
        dump_probe_dataset(
            agent, os.path.join(args.save_dir, "probe_dataset.npz"),
            max_snapshots=args.probe_dataset_size, rng=probe_rng,
        )
        with open(os.path.join(args.save_dir, "gnn_il_probe_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.save_dir}/  "
              f"(results.json, {len(snapshot_eps)} snapshots, probe_dataset.npz, "
              f"final.pt)")

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
    p.add_argument("--eps_end",       type=float, default=0.05,
                   help="exploration floor (race-model fix #1b: raise it to "
                        "sustain late exploration)")
    p.add_argument("--init_encoder",  type=str,   default="",
                   help="path to a .pt checkpoint whose 'gnn' weights warm-"
                        "start the encoder (race-model fix #2; donor from a "
                        "different M = curriculum test #3)")
    p.add_argument("--init_full",     type=str,   default="",
                   help="resurrection: load gnn+policy(+target) from a mid-"
                        "training snapshot; pair with --eps_start and a "
                        "different --seed (buffer is NOT restored)")
    p.add_argument("--eps_start",     type=float, default=1.0,
                   help="initial epsilon (set to the snapshot-episode eps "
                        "when resurrecting)")

    # Probe-specific
    p.add_argument("--probe_every",     type=int, default=5,
                   help="probe every K episodes")
    p.add_argument("--probe_snapshots", type=int, default=512,
                   help="max buffer snapshots per probe fit")
    p.add_argument("--probe_seed",      type=int, default=12345,
                   help="private RNG seed for probe subsampling (RNG-isolated)")
    p.add_argument("--snapshot_every",  type=int, default=25,
                   help="save encoder+policy weights every K episodes "
                        "(0=off); enables offline re-probing of the trajectory")
    p.add_argument("--probe_dataset_size", type=int, default=2000,
                   help="snapshots saved to probe_dataset.npz for offline probes")

    # Late-collapse diagnostics (session 3 — see module docstring)
    p.add_argument("--diag_every", type=int, default=0,
                   help="run the diagnostics suite every K episodes (0=off). "
                        "Measurements are RNG-guarded: the baseline training "
                        "trajectory stays bit-identical to trainGnn.py")
    p.add_argument("--diag_eval_eps", type=int, default=4,
                   help="greedy CRN eval episodes per diagnostics point")
    p.add_argument("--diag_eval_seed", type=int, default=990000,
                   help="fixed seed for the eval env / probe-state batch "
                        "(same for every run and arm -> paired comparisons)")
    p.add_argument("--diag_states", type=int, default=256,
                   help="size of the fixed random-policy state batch for Q stats")
    p.add_argument("--arm", type=str, default="none",
                   choices=["none", "eps0", "freeze_replay", "freeze_learn"],
                   help="intervention fired ONCE when greedy eval first "
                        "reaches --trigger_cost (see docstring)")
    p.add_argument("--trigger_cost", type=float, default=0.9,
                   help="greedy-eval cost at which the arm fires")
    p.add_argument("--trigger_mode", type=str, default="raw",
                   choices=["raw", "smoothed"],
                   help="'raw' = single CRN eval <= trigger_cost (historical; "
                        "degenerate at M=20 where it fires at ep 10). "
                        "'smoothed' = median of last 3 CRN evals <= "
                        "trigger_cost (matches the v2-label convergence "
                        "definition)")
    p.add_argument("--trigger_min_ep", type=int, default=0,
                   help="arm may not fire before this episode (guards the "
                        "eps0 arm against firing on a near-untrained policy)")
    p.add_argument("--target_encoder", action="store_true",
                   help="keep a FROZEN target copy of the GNN encoder for the "
                        "TD target (synced with target_net). Default off = "
                        "historical behaviour: next-state embeddings from the "
                        "online encoder (half-frozen target). Cold-review "
                        "artifact control.")

    p.add_argument("--seed",      type=int,  default=42)
    p.add_argument("--log_every", type=int,  default=25)
    p.add_argument("--save_dir",  type=str,  default="probe_runs")
    p.add_argument("--cpu",       action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args    = get_args()
    results = train_with_probe(args)
    print(f"\nGNN-IL+probe — eval {results['eval_mean']:.4f} ± "
          f"{results['eval_std']:.4f}  |  final φ (R2Δgnn) "
          f"{results['final_probe']['r2_gnn_delta']:.3f}  "
          f"(static R2_gnn {results['final_probe']['r2_gnn']:.3f})")
