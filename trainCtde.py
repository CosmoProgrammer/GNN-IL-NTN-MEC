"""
CTDE Training Loop.

Runs one shared Dueling DQN (central controller) that all M UEs
draw actions from and push transitions into — exactly the CTDE-based
framework from the paper.

Usage:
    python trainCTDE.py                        # default config
    python trainCTDE.py --n_ues 10 --n_uavs 2 --episodes 500
"""

import argparse
import numpy as np
import torch
import random
import json
import os

from envWithBL        import NTNMECEnv, EnvConfig
from ctdeAgent import CTDEAgent


# ─── Helpers ───────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_controller(env: NTNMECEnv, device: str, **agent_kwargs) -> CTDEAgent:
    return CTDEAgent(
        obs_dim   = env.obs_dim,
        n_actions = env.n_actions,
        n_agents  = env.n_agents,
        device    = device,
        **agent_kwargs,
    )


# ─── One episode ───────────────────────────────────────────────────────────

def run_episode(
    env:        NTNMECEnv,
    controller: CTDEAgent,
    train:      bool = True,
) -> dict:
    """
    Roll out one full episode.

    Returns
    -------
    dict with:
        avg_cost      — mean cost across all UEs and time slots
        total_loss    — mean training loss (nan if train=False)
        eps           — current epsilon of the controller
    """
    obs_list  = env.reset()
    ep_costs  = []
    ep_losses = []

    for _ in range(env.cfg.I):
        # Capture which agents have a task THIS slot, before step() advances time
        had_task = [t is not None for t in env.tasks]

        # All UEs select actions from the shared policy network
        actions = [
            controller.select_action(obs, greedy=not train)
            for obs in obs_list
        ]

        next_obs_list, rewards, done, info = env.step(actions)

        if train:
            # All UEs push into the ONE shared buffer
            for m in range(env.n_agents):
                if not had_task[m]:
                    # No task this slot — skip storing, same reasoning as IL.
                    continue
                controller.store(
                    obs_list[m], actions[m], rewards[m],
                    next_obs_list[m], float(done)
                )

            # One centralized gradient update per timestep (Algorithm 1)
            loss = controller.train_step()
            if loss is not None:
                ep_losses.append(loss)

        ep_costs.append(info["avg_cost"])
        obs_list = next_obs_list

        if done:
            break

    return {
        "avg_cost":   float(np.mean(ep_costs)),
        "total_loss": float(np.mean(ep_losses)) if ep_losses else float("nan"),
        "eps":        controller.eps,
    }


# ─── Full training run ─────────────────────────────────────────────────────

def train(args) -> dict:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    cfg = EnvConfig(
        M      = args.n_ues,
        N      = args.n_uavs,
        I      = args.steps_per_ep,
        P_task = args.task_prob,
    )
    env        = NTNMECEnv(cfg)
    controller = make_controller(
        env,
        device         = device,
        hidden         = args.hidden,
        lr             = args.lr,
        gamma          = args.gamma,
        batch_size     = args.batch_size,
        target_update  = args.target_update,
        eps_decay      = args.eps_decay,
        per_agent_cap  = args.per_agent_cap,
    )

    history   = []
    best_cost = float("inf")

    print(f"\nTraining CTDE — {args.n_ues} UEs, {args.n_uavs} UAVs, "
          f"{args.episodes} episodes\n")
    print(f"{'Episode':>8}  {'AvgCost':>10}  {'Loss':>10}  {'Eps':>6}")
    print("-" * 42)

    for ep in range(1, args.episodes + 1):
        stats = run_episode(env, controller, train=True)
        history.append(stats)

        if ep % args.log_every == 0:
            print(f"{ep:>8}  {stats['avg_cost']:>10.4f}  "
                  f"{stats['total_loss']:>10.4f}  {stats['eps']:>6.3f}")

        # Decay ε once per episode
        controller.decay_epsilon()

        if stats["avg_cost"] < best_cost:
            best_cost = stats["avg_cost"]
            if args.save_dir:
                _save_controller(controller, args.save_dir, tag="ctde_best")

    # Final greedy eval
    eval_costs = []
    for _ in range(args.eval_episodes):
        s = run_episode(env, controller, train=False)
        eval_costs.append(s["avg_cost"])

    eval_mean = float(np.mean(eval_costs))
    eval_std  = float(np.std(eval_costs))
    print(f"\nEval ({args.eval_episodes} eps): "
          f"mean={eval_mean:.4f}  std={eval_std:.4f}")

    results = {
        "method":     "CTDE",
        "config":     vars(args),
        "history":    history,
        "eval_mean":  eval_mean,
        "eval_std":   eval_std,
        "best_train": best_cost,
    }

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        _save_controller(controller, args.save_dir, tag="ctde_final")
        with open(os.path.join(args.save_dir, "ctde_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.save_dir}/")

    return results


# ─── Persistence ───────────────────────────────────────────────────────────

def _save_controller(controller: CTDEAgent, save_dir: str, tag: str):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{tag}_controller.pt")
    torch.save(controller.policy_net.state_dict(), path)


def load_controller(
    env:      NTNMECEnv,
    save_dir: str,
    tag:      str,
    device:   str = "cpu",
) -> CTDEAgent:
    controller = make_controller(env, device=device)
    path = os.path.join(save_dir, f"{tag}_controller.pt")
    controller.policy_net.load_state_dict(torch.load(path, map_location=device))
    controller.policy_net.eval()
    return controller


# ─── Baselines ─────────────────────────────────────────────────────────────

def run_random_policy(env: NTNMECEnv, n_episodes: int = 20) -> float:
    costs = []
    for _ in range(n_episodes):
        env.reset()
        ep_costs = []
        for _ in range(env.cfg.I):
            actions = [random.randint(0, env.n_actions - 1)
                       for _ in range(env.n_agents)]
            _, _, done, info = env.step(actions)
            ep_costs.append(info["avg_cost"])
            if done:
                break
        costs.append(np.mean(ep_costs))
    return float(np.mean(costs))


def run_local_policy(env: NTNMECEnv, n_episodes: int = 20) -> float:
    costs = []
    for _ in range(n_episodes):
        env.reset()
        ep_costs = []
        for _ in range(env.cfg.I):
            actions = [0] * env.n_agents
            _, _, done, info = env.step(actions)
            ep_costs.append(info["avg_cost"])
            if done:
                break
        costs.append(np.mean(ep_costs))
    return float(np.mean(costs))


# ─── CLI ───────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()

    # Environment
    p.add_argument("--n_ues",        type=int,   default=10)
    p.add_argument("--n_uavs",       type=int,   default=2)
    p.add_argument("--steps_per_ep", type=int,   default=100)
    p.add_argument("--task_prob",    type=float, default=0.3)

    # Training
    p.add_argument("--episodes",      type=int,   default=500)
    p.add_argument("--eval_episodes", type=int,   default=20)
    p.add_argument("--hidden",        type=int,   default=128)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--gamma",         type=float, default=0.9)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--target_update", type=int,   default=20)
    p.add_argument("--eps_decay",     type=float, default=0.995)
    p.add_argument("--per_agent_cap", type=int,   default=10_000)

    # Misc
    p.add_argument("--seed",      type=int,  default=42)
    p.add_argument("--log_every", type=int,  default=50)
    p.add_argument("--save_dir",  type=str,  default="checkpoints")
    p.add_argument("--cpu",       action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    results = train(args)

    # Print naive baselines alongside
    cfg = EnvConfig(M=args.n_ues, N=args.n_uavs)
    env = NTNMECEnv(cfg)
    set_seed(args.seed)

    rnd = run_random_policy(env)
    loc = run_local_policy(env)
    print(f"\nBaselines  —  random: {rnd:.4f}  |  local: {loc:.4f}")
    print(f"CTDE eval  —  {results['eval_mean']:.4f} ± {results['eval_std']:.4f}")