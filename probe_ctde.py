"""
Late-collapse diagnostics wrapper for CTDE (NON-INVASIVE, mirrors probe_gnn).

Purpose (cold-review priority 1 — see RESEARCH_LOG §25)
-------------------------------------------------------
The divergence anatomy (CRN eval trace, |Q|/TD/grad/churn signals, v2 labels)
was only ever measured on GNN-IL. The 2×2 says instability follows parameter
SHARING, but without traces on the shared-MLP cells that claim rests on final-
eval distributions alone. This wrapper runs the SAME diagnostic suite on
CTDEAgent — both env variants:

    --env plain : CTDE-plain (env.py, information-matched to IL/GNN-IL)
    --env bl    : CTDE-Bn    (envWithBL.py, Bₙ in the raw observation)

If churn-led late transient divergence shows up here, the anatomy is a
property of parameter-shared value-based MARL, not a GNN-IL artifact — and
since CTDE has a FULLY frozen target (no online-encoder-in-target path), it
also answers the half-frozen-target objection to the phenomenon's existence.

What is measured (identical protocol/seeds to probe_gnn — CRN-paired)
---------------------------------------------------------------------
Every --diag_every episodes, all RNG-GUARDED (global RNG saved/reseeded/
restored, so the training trajectory is bit-identical to trainCtde.py):
  * greedy CRN eval on a canonical eval env + per-UAV congestion occupancy
    (same --diag_eval_seed 990000 as probe_gnn → for --env plain the eval
    episodes are the SAME episodes the GNN-IL diag runs saw);
  * mean/max |Q| + greedy action distribution/entropy on a FIXED
    random-policy state batch (act_frac → analyzer derives act_churn);
  * per-sample |TD error| percentiles from a private-RNG buffer sample;
  * PRE-clip grad norms via the clip_grad_norm_ wrapper.

Intervention arms (--arm, same trigger machinery as probe_gnn, including the
corrected --trigger_mode smoothed / --trigger_min_ep):
  * eps0          : kill exploration once converged
  * freeze_replay : stop storing, keep training
  * freeze_learn  : stop training, keep acting

Fairness control (cold-review W2): --batch_scale_m multiplies the batch size
by M so CTDE's per-update gradient averaging matches GNN-IL's effective batch
(~32·M masked agent-terms), isolating "sharing" from "batch size".

Output: <save_dir>/ctde_probe_results.json — history records carry the same
diag_* keys as gnn_il_probe_results.json, so analyze_diagnight machinery
(classify_run, signal_leads, excursions) applies unchanged.

Usage
-----
    python probe_ctde.py --env plain --n_ues 20 --seed 42 --diag_every 10 \
        --save_dir probe_runs/diag_ctde/plain/ue20/n2/base/seed42
    python probe_ctde.py --env plain --n_ues 20 --seed 42 --diag_every 10 \
        --arm eps0 --trigger_mode smoothed --trigger_cost 0.8 --trigger_min_ep 100
"""

import argparse
import json
import os

import numpy as np
import torch

from ctdeAgent import CTDEAgent
from trainCtde import run_episode, set_seed, _select_env
# Reuse the exact RNG-guard / pre-clip grad recorder from the GNN probe so the
# measurement mechanics are identical across methods.
from probe_gnn import _RNGGuard, _GradNormRecorder


# ─── Diagnostics (CTDE flavours of the probe_gnn suite) ─────────────────────

def build_diag_states(EnvCls, cfg, n_states: int, seed: int):
    """
    FIXED batch of random-policy states (raw per-UE observations), built under
    the RNG guard with a fixed seed → identical for every run, seed and arm of
    the same env variant. (Not identical to probe_gnn's batch — obs_dim and
    the consumed RNG stream differ — but paired across all CTDE runs.)
    """
    with _RNGGuard(seed):
        env = EnvCls(cfg)
        rows = []
        obs_list = env.reset()
        while len(rows) < n_states:
            rows.append(np.stack(obs_list).astype(np.float32))
            actions = [np.random.randint(env.n_actions)
                       for _ in range(env.n_agents)]
            obs_list, _, done, _ = env.step(actions)
            if done:
                obs_list = env.reset()
    return np.stack(rows)                       # (S, M, obs_dim)


def diag_q_stats(controller: CTDEAgent, diag_states: np.ndarray) -> dict:
    """Mean/max |Q|, greedy action fractions and entropy on the fixed batch."""
    S, M, obs_dim = diag_states.shape
    with torch.no_grad():
        obs_t = torch.tensor(diag_states.reshape(S * M, obs_dim),
                             dtype=torch.float32, device=controller.device)
        q     = controller.policy_net(obs_t)                    # (S*M, A)
        q_abs = q.abs()
        acts  = q.argmax(dim=1).cpu().numpy()
    frac = np.bincount(acts, minlength=controller.n_actions) / float(len(acts))
    nz = frac[frac > 0]
    return {
        "q_mean_abs":  float(q_abs.mean()),
        "q_max_abs":   float(q_abs.max()),
        "act_frac":    [float(x) for x in frac],
        "act_entropy": float(-(nz * np.log(nz)).sum()),
    }


def diag_td_stats(controller: CTDEAgent, rng: np.random.Generator,
                  n_samples: int = 128):
    """
    Per-sample |TD error| over a private-RNG buffer sample — mirrors
    CTDEAgent.train_step()'s Double-DQN target math under no_grad.
    (CTDE only stores task slots, so every row is a trained-on row.)
    """
    buf = controller.buffer.buf
    if len(buf) < 8:
        return None
    idx = rng.permutation(len(buf))[:n_samples]
    batch = [buf[i] for i in idx]
    obs, actions, rewards, next_obs, dones = zip(*batch)
    dev = controller.device
    obs      = torch.tensor(np.stack(obs),      dtype=torch.float32, device=dev)
    next_obs = torch.tensor(np.stack(next_obs), dtype=torch.float32, device=dev)
    actions  = torch.tensor(actions,            dtype=torch.long,    device=dev)
    rewards  = torch.tensor(rewards,            dtype=torch.float32, device=dev)
    dones    = torch.tensor(dones,              dtype=torch.float32, device=dev)

    with torch.no_grad():
        q_pred = controller.policy_net(obs).gather(
            1, actions.unsqueeze(1)).squeeze(1)
        next_actions = controller.policy_net(next_obs).argmax(dim=1)
        next_q = controller.target_net(next_obs).gather(
            1, next_actions.unsqueeze(1)).squeeze(1)
        target = rewards + controller.gamma * next_q * (1.0 - dones)
        td = (q_pred - target).abs()

    tdn = td.cpu().numpy()
    return {"td_p50": float(np.percentile(tdn, 50)),
            "td_p95": float(np.percentile(tdn, 95)),
            "td_max": float(tdn.max())}


def diag_greedy_eval(controller: CTDEAgent, eval_env, n_eps: int,
                     seed: int) -> dict:
    """
    Greedy rollouts on the canonical eval env under the RNG guard → identical
    episodes every call (CRN). Same protocol/seed as probe_gnn, so for --env
    plain these are the very episodes the GNN-IL diag traces were scored on.
    """
    costs, cong_means, cong_max = [], [], 0.0
    with _RNGGuard(seed):
        for _ in range(n_eps):
            obs_list = eval_env.reset()
            step_costs, congs = [], []
            for _ in range(eval_env.cfg.I):
                congs.append(np.asarray(eval_env.get_graph_data()["uav_x"],
                                        dtype=np.float32)[:, 2])
                acts = [controller.select_action(o, greedy=True)
                        for o in obs_list]
                obs_list, _, done, info = eval_env.step(acts)
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


# ─── Training run with diagnostics ──────────────────────────────────────────

def train_with_diag(args) -> dict:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    EnvCls, CfgCls = _select_env(args.env)
    variant = "plain" if args.env == "plain" else "Bn"
    print(f"Env: {'env.py (plain obs, information-matched)' if args.env == 'plain' else 'envWithBL.py (Bn in raw obs)'}")

    cfg = CfgCls(
        M      = args.n_ues,
        N      = args.n_uavs,
        I      = args.steps_per_ep,
        P_task = args.task_prob,
    )
    env = EnvCls(cfg)

    batch_size = args.batch_size * (args.n_ues if args.batch_scale_m else 1)
    if args.batch_scale_m:
        print(f"Fairness control: batch scaled to 32*M = {batch_size} "
              f"(matches GNN-IL's effective per-update averaging)")

    controller = CTDEAgent(
        obs_dim       = env.obs_dim,
        n_actions     = env.n_actions,
        n_agents      = env.n_agents,
        hidden        = args.hidden,
        lr            = args.lr,
        gamma         = args.gamma,
        batch_size    = batch_size,
        target_update = args.target_update,
        eps_decay     = args.eps_decay,
        eps_end       = args.eps_end,
        per_agent_cap = args.per_agent_cap,
        device        = device,
    )

    # Private RNG for TD-probe subsampling — never the global RNG training uses.
    probe_rng = np.random.default_rng(args.probe_seed)

    # ── Diagnostics setup (mirrors probe_gnn) ──────────────────────────
    diag_every     = args.diag_every
    arm            = args.arm
    trigger_cost   = args.trigger_cost
    trigger_mode   = args.trigger_mode
    trigger_min_ep = args.trigger_min_ep
    diag_eval_seed = args.diag_eval_seed

    diag_states = eval_env = grad_rec = None
    trigger_ep = None
    recent_evals = []
    best_diag_eval, best_diag_ep = float("inf"), None
    if diag_every > 0:
        diag_states = build_diag_states(EnvCls, cfg, args.diag_states,
                                        diag_eval_seed + 1)
        with _RNGGuard(diag_eval_seed):
            eval_env = EnvCls(cfg)
        grad_rec = _GradNormRecorder()
        grad_rec.install()
        print(f"Diagnostics ON: every {diag_every} eps, arm={arm}, "
              f"trigger={trigger_mode}<= {trigger_cost} from ep "
              f"{trigger_min_ep}")

    history = []
    best_cost = float("inf")

    print(f"\nCTDE-{variant} + diagnostics — {args.n_ues} UEs, {args.n_uavs} "
          f"UAVs, {args.episodes} episodes, seed {args.seed}\n")
    print(f"{'Episode':>8}  {'AvgCost':>10}  {'Loss':>10}  {'Eps':>6}")
    print("-" * 42)

    for ep in range(1, args.episodes + 1):
        stats = run_episode(env, controller, train=True)  # identical to trainCtde
        controller.decay_epsilon()

        record = {
            "episode":    ep,
            "avg_cost":   stats["avg_cost"],
            "total_loss": stats["total_loss"],
            "eps":        stats["eps"],
        }

        # ── Diagnostics: measure, then (maybe) fire the arm ────────────
        if diag_every > 0 and (ep % diag_every == 0 or ep == args.episodes):
            diag = {}
            diag.update(diag_greedy_eval(controller, eval_env,
                                         args.diag_eval_eps, diag_eval_seed))
            diag.update(diag_q_stats(controller, diag_states))
            td = diag_td_stats(controller, probe_rng)
            if td is not None:
                diag.update(td)
            g = grad_rec.drain()
            if g:
                diag["grad_p50"] = float(np.percentile(g, 50))
                diag["grad_max"] = float(np.max(g))

            if diag["eval_cost"] < best_diag_eval:
                best_diag_eval, best_diag_ep = diag["eval_cost"], ep

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
                    controller.eps = 0.0
                    controller.eps_end = 0.0
                elif arm == "freeze_replay":
                    controller.store = lambda *a, **k: None
                elif arm == "freeze_learn":
                    controller.train_step = lambda: None
                print(f"[arm] {arm} TRIGGERED at ep {ep} "
                      f"({trigger_mode} eval {trig_val:.4f} "
                      f"<= {trigger_cost})")

            record.update({f"diag_{k}": v for k, v in diag.items()})
            print(f"    [diag ep {ep}] eval={diag['eval_cost']:.4f}  "
                  f"|Q|max={diag.get('q_max_abs', float('nan')):.2f}  "
                  f"td95={diag.get('td_p95', float('nan')):.3f}  "
                  f"gradmax={diag.get('grad_max', float('nan')):.1f}  "
                  f"cong={diag.get('cong_mean', float('nan')):.3f}")

        history.append(record)

        if stats["avg_cost"] < best_cost:
            best_cost = stats["avg_cost"]

        if ep % args.log_every == 0 or ep == args.episodes:
            print(f"{ep:>8}  {stats['avg_cost']:>10.4f}  "
                  f"{stats['total_loss']:>10.4f}  {stats['eps']:>6.3f}")

    if grad_rec is not None:
        grad_rec.uninstall()

    # Final greedy eval — identical protocol to trainCtde (classifies the seed)
    eval_costs = [run_episode(env, controller, train=False)["avg_cost"]
                  for _ in range(args.eval_episodes)]
    eval_mean = float(np.mean(eval_costs))
    eval_std  = float(np.std(eval_costs))
    print(f"\nEval ({args.eval_episodes} eps): mean={eval_mean:.4f}  "
          f"std={eval_std:.4f}")

    results = {
        "method":     f"CTDE-{variant}+diag",
        "config":     vars(args),
        "history":    history,
        "eval_mean":  eval_mean,
        "eval_std":   eval_std,
        "best_train": best_cost,
    }
    if diag_every > 0:
        results.update({
            "arm":               arm,
            "trigger_ep":        trigger_ep,
            "diag_best_eval":    best_diag_eval,
            "diag_best_eval_ep": best_diag_ep,
        })

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(controller.policy_net.state_dict(),
                   os.path.join(args.save_dir, "ctde_probe_final.pt"))
        with open(os.path.join(args.save_dir, "ctde_probe_results.json"),
                  "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.save_dir}/")

    return results


# ─── CLI ───────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()

    # Environment — defaults match trainCtde.py
    p.add_argument("--env", type=str, default="plain",
                   choices=["plain", "bl"],
                   help="'plain' = env.py (information-matched, the 2x2 "
                        "sharing cell); 'bl' = envWithBL.py (CTDE-Bn)")
    p.add_argument("--n_ues",        type=int,   default=20)
    p.add_argument("--n_uavs",       type=int,   default=2)
    p.add_argument("--steps_per_ep", type=int,   default=100)
    p.add_argument("--task_prob",    type=float, default=0.3)

    # Training — defaults match trainCtde.py
    p.add_argument("--episodes",      type=int,   default=500)
    p.add_argument("--eval_episodes", type=int,   default=20)
    p.add_argument("--hidden",        type=int,   default=128)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--gamma",         type=float, default=0.9)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--target_update", type=int,   default=20)
    p.add_argument("--eps_decay",     type=float, default=0.995)
    p.add_argument("--eps_end",       type=float, default=0.05)
    p.add_argument("--per_agent_cap", type=int,   default=10_000)
    p.add_argument("--batch_scale_m", action="store_true",
                   help="fairness control: batch = batch_size * M, matching "
                        "GNN-IL's effective per-update averaging")

    # Diagnostics — mirror probe_gnn
    p.add_argument("--diag_every", type=int, default=0,
                   help="run the diagnostics suite every K episodes (0=off); "
                        "RNG-guarded, training trajectory bit-identical to "
                        "trainCtde.py")
    p.add_argument("--diag_eval_eps", type=int, default=4)
    p.add_argument("--diag_eval_seed", type=int, default=990000,
                   help="same default as probe_gnn -> CRN-paired evals "
                        "across methods (for --env plain)")
    p.add_argument("--diag_states", type=int, default=256)
    p.add_argument("--arm", type=str, default="none",
                   choices=["none", "eps0", "freeze_replay", "freeze_learn"])
    p.add_argument("--trigger_cost", type=float, default=0.9)
    p.add_argument("--trigger_mode", type=str, default="raw",
                   choices=["raw", "smoothed"])
    p.add_argument("--trigger_min_ep", type=int, default=0)
    p.add_argument("--probe_seed", type=int, default=12345,
                   help="private RNG seed for TD-probe subsampling")

    # Misc
    p.add_argument("--seed",      type=int,  default=42)
    p.add_argument("--log_every", type=int,  default=50)
    p.add_argument("--save_dir",  type=str,  default="probe_runs/diag_ctde")
    p.add_argument("--cpu",       action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    results = train_with_diag(args)
    print(f"\nCTDE+diag — eval {results['eval_mean']:.4f} ± "
          f"{results['eval_std']:.4f}")
