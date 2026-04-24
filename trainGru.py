"""
IL + GRU Training Loop.

Mirrors train_il.py exactly except:
  - Uses ILGRUAgent (SequenceReplayBuffer + GRU)
  - Calls agent.reset_hidden() at the start of every episode
  - Calls agent.push_step() once per timestep (stores to episode buffer)
  - Episode is flushed to the replay buffer automatically on done=True

Usage:
    python train_il_gru.py --episodes 500 --log_every 50
"""

import argparse
import numpy as np
import torch
import random
import json
import os
from typing import List

from env       import NTNMECEnv, EnvConfig
from gruAgent import ILGRUAgent


# ─── Seed ──────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ─── Agent factory ─────────────────────────────────────────────────────────

def make_agents(env: NTNMECEnv, device: str, **kwargs) -> List[ILGRUAgent]:
    return [
        ILGRUAgent(
            obs_dim   = env.obs_dim,
            n_actions = env.n_actions,
            device    = device,
            **kwargs,
        )
        for _ in range(env.n_agents)
    ]


# ─── One episode ───────────────────────────────────────────────────────────

def run_episode(
    env:    NTNMECEnv,
    agents: List[ILGRUAgent],
    train:  bool = True,
) -> dict:
    obs_list = env.reset()

    # Reset every agent's hidden state at the start of the episode
    for agent in agents:
        agent.reset_hidden()

    ep_costs  = []
    ep_losses = []

    for _ in range(env.cfg.I):
        had_task = [t is not None for t in env.tasks]

        actions = [
            agent.select_action(obs, greedy=not train)
            for agent, obs in zip(agents, obs_list)
        ]

        next_obs_list, rewards, done, info = env.step(actions)

        if train:
            for m, agent in enumerate(agents):
                # Always push the step (contiguous sequence needed for GRU)
                # task_mask signals whether loss should count this step
                agent.push_step(
                    obs_list[m], actions[m], rewards[m],
                    had_task=had_task[m], done=done,
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
        gru_hidden    = args.gru_hidden,
        dqn_hidden    = args.dqn_hidden,
        lr            = args.lr,
        gamma         = args.gamma,
        batch_size    = args.batch_size,
        chunk_len     = args.chunk_len,
        buf_episodes  = args.buf_episodes,
        target_update = args.target_update,
        eps_decay     = args.eps_decay,
    )

    history   = []
    best_cost = float("inf")

    print(f"\nTraining IL+GRU — {args.n_ues} UEs, {args.n_uavs} UAVs, "
          f"{args.episodes} episodes  |  chunk_len={args.chunk_len}\n")
    print(f"{'Episode':>8}  {'AvgCost':>10}  {'Loss':>10}  {'Eps':>6}")
    print("-" * 42)

    for ep in range(1, args.episodes + 1):
        stats = run_episode(env, agents, train=True)
        history.append(stats)

        for agent in agents:
            agent.decay_epsilon()

        if ep % args.log_every == 0:
            print(f"{ep:>8}  {stats['avg_cost']:>10.4f}  "
                  f"{stats['total_loss']:>10.4f}  {stats['eps']:>6.3f}")

        if stats["avg_cost"] < best_cost:
            best_cost = stats["avg_cost"]
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                agents[0].save(os.path.join(args.save_dir, "il_gru_best.pt"))

    eval_costs = []
    for _ in range(args.eval_episodes):
        s = run_episode(env, agents, train=False)
        eval_costs.append(s["avg_cost"])

    eval_mean = float(np.mean(eval_costs))
    eval_std  = float(np.std(eval_costs))
    print(f"\nEval ({args.eval_episodes} eps): "
          f"mean={eval_mean:.4f}  std={eval_std:.4f}")

    results = {
        "method":     "IL+GRU",
        "config":     vars(args),
        "history":    history,
        "eval_mean":  eval_mean,
        "eval_std":   eval_std,
        "best_train": best_cost,
    }

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "il_gru_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.save_dir}/")

    return results


# ─── CLI ───────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()

    # Environment
    p.add_argument("--n_ues",        type=int,   default=10)
    p.add_argument("--n_uavs",       type=int,   default=2)
    p.add_argument("--steps_per_ep", type=int,   default=100)
    p.add_argument("--task_prob",    type=float, default=0.3)

    # GRU
    p.add_argument("--gru_hidden",   type=int,   default=64)
    p.add_argument("--dqn_hidden",   type=int,   default=64)
    p.add_argument("--chunk_len",    type=int,   default=8)
    p.add_argument("--buf_episodes", type=int,   default=200)

    # Training
    p.add_argument("--episodes",      type=int,   default=500)
    p.add_argument("--eval_episodes", type=int,   default=20)
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
    args    = get_args()
    results = train(args)
    print(f"\nIL+GRU eval  —  {results['eval_mean']:.4f} ± {results['eval_std']:.4f}")