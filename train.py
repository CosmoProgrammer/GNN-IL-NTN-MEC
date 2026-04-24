"""
IL Training Loop.

Runs M independent dueling-DQN agents, each training on its own
local replay buffer — exactly the IL-based framework from the paper.

Usage:
    python train_il.py                        # default config
    python train_il.py --n_ues 10 --n_uavs 2 --episodes 500
"""

import argparse
import numpy as np
import torch
import random
import json
import os
from typing import List

from env   import NTNMECEnv, EnvConfig
from agent import ILAgent


# ─── Helpers ───────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_agents(env: NTNMECEnv, device: str, **agent_kwargs) -> List[ILAgent]:
    return [
        ILAgent(
            obs_dim   = env.obs_dim,
            n_actions = env.n_actions,
            device    = device,
            **agent_kwargs,
        )
        for _ in range(env.n_agents)
    ]


# ─── One episode ───────────────────────────────────────────────────────────

def run_episode(
    env:     NTNMECEnv,
    agents:  List[ILAgent],
    train:   bool = True,
) -> dict:
    """
    Roll out one full episode.

    Returns
    -------
    dict with:
        avg_cost      — mean cost across all UEs and time slots
        total_loss    — mean training loss (nan if train=False)
        eps           — current epsilon of agent 0
    """
    obs_list  = env.reset()
    ep_costs  = []
    ep_losses = []

    for _ in range(env.cfg.I):
        # Capture which agents have a task THIS slot, before step() advances time
        had_task = [t is not None for t in env.tasks]

        actions = [
            agent.select_action(obs, greedy=not train)
            for agent, obs in zip(agents, obs_list)
        ]

        next_obs_list, rewards, done, info = env.step(actions)

        if train:
            for m, agent in enumerate(agents):
                if not had_task[m]:
                    # No task this slot — action had no effect, skip storing.
                    # Avoids polluting the buffer with zero-reward noise and
                    # prevents agents from learning that offloading is "risky"
                    # on slots where nothing actually happens.
                    continue
                agent.store(
                    obs_list[m], actions[m], rewards[m],
                    next_obs_list[m], float(done)
                )
                loss = agent.train_step()
                if loss is not None:
                    ep_losses.append(loss)

        ep_costs.append(info["avg_cost"])
        obs_list = next_obs_list

        if done:
            break

    return {
        "avg_cost":   float(np.mean(ep_costs)),
        "total_loss": float(np.mean(ep_losses)) if ep_losses else float("nan"),
        "eps":        agents[0].eps,
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
    env    = NTNMECEnv(cfg)
    agents = make_agents(
        env,
        device        = device,
        hidden        = args.hidden,
        lr            = args.lr,
        gamma         = args.gamma,
        batch_size    = args.batch_size,
        target_update = args.target_update,
        eps_decay     = args.eps_decay,
    )

    history   = []
    best_cost = float("inf")

    print(f"\nTraining IL — {args.n_ues} UEs, {args.n_uavs} UAVs, "
          f"{args.episodes} episodes\n")
    print(f"{'Episode':>8}  {'AvgCost':>10}  {'Loss':>10}  {'Eps':>6}")
    print("-" * 42)

    for ep in range(1, args.episodes + 1):
        stats = run_episode(env, agents, train=True)
        history.append(stats)

        if ep % args.log_every == 0:
            print(f"{ep:>8}  {stats['avg_cost']:>10.4f}  "
                  f"{stats['total_loss']:>10.4f}  {stats['eps']:>6.3f}")

        # Decay ε once per episode for all agents
        for agent in agents:
            agent.decay_epsilon()

        if stats["avg_cost"] < best_cost:
            best_cost = stats["avg_cost"]
            if args.save_dir:
                _save_agents(agents, args.save_dir, tag="il_best")

    # Final greedy eval
    eval_costs = []
    for _ in range(args.eval_episodes):
        s = run_episode(env, agents, train=False)
        eval_costs.append(s["avg_cost"])

    eval_mean = float(np.mean(eval_costs))
    eval_std  = float(np.std(eval_costs))
    print(f"\nEval ({args.eval_episodes} eps): "
          f"mean={eval_mean:.4f}  std={eval_std:.4f}")

    results = {
        "method":     "IL",
        "config":     vars(args),
        "history":    history,
        "eval_mean":  eval_mean,
        "eval_std":   eval_std,
        "best_train": best_cost,
    }

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        _save_agents(agents, args.save_dir, tag="il_final")
        with open(os.path.join(args.save_dir, "il_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.save_dir}/")

    return results


# ─── Persistence ───────────────────────────────────────────────────────────

def _save_agents(agents: List[ILAgent], save_dir: str, tag: str):
    os.makedirs(save_dir, exist_ok=True)
    for m, agent in enumerate(agents):
        path = os.path.join(save_dir, f"{tag}_agent{m}.pt")
        torch.save(agent.policy_net.state_dict(), path)


def load_agents(
    env:      NTNMECEnv,
    save_dir: str,
    tag:      str,
    device:   str = "cpu",
) -> List[ILAgent]:
    agents = make_agents(env, device=device)
    for m, agent in enumerate(agents):
        path = os.path.join(save_dir, f"{tag}_agent{m}.pt")
        agent.policy_net.load_state_dict(torch.load(path, map_location=device))
        agent.policy_net.eval()
    return agents


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
    print(f"IL eval    —  {results['eval_mean']:.4f} ± {results['eval_std']:.4f}")