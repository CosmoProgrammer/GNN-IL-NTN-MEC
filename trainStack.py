"""
IL + Framestack Training Loop.

Identical to train_il.py except each agent receives a K=4 stacked observation
instead of the raw single-frame obs.

The ILAgent from agent.py is reused unchanged — its obs_dim is simply set to
K * env.obs_dim instead of env.obs_dim.

Usage:
    python train_il_stack.py --episodes 500 --log_every 50
"""

import argparse
import numpy as np
import torch
import random
import json
import os
from typing import List

from env        import NTNMECEnv, EnvConfig
from agent      import ILAgent
from framestack import FrameStack


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_agents(stacked_dim: int, env: NTNMECEnv, device: str, **kwargs) -> List[ILAgent]:
    return [
        ILAgent(
            obs_dim   = stacked_dim,
            n_actions = env.n_actions,
            device    = device,
            **kwargs,
        )
        for _ in range(env.n_agents)
    ]


def run_episode(
    env:     NTNMECEnv,
    agents:  List[ILAgent],
    K:       int,
    train:   bool = True,
) -> dict:
    raw_obs  = env.reset()

    # Initialise frame stacks — zero-pad then push first obs
    stacks = [FrameStack(env.obs_dim, K) for _ in range(env.n_agents)]
    for stack, obs in zip(stacks, raw_obs):
        stack.reset(obs)

    ep_costs  = []
    ep_losses = []

    for _ in range(env.cfg.I):
        had_task    = [t is not None for t in env.tasks]
        stacked_obs = [stack.get() for stack in stacks]   # (K*obs_dim,) each

        actions = [
            agent.select_action(sobs, greedy=not train)
            for agent, sobs in zip(agents, stacked_obs)
        ]

        next_raw_obs, rewards, done, info = env.step(actions)

        # Advance frame stacks with next observation
        next_stacked = []
        for stack, nobs in zip(stacks, next_raw_obs):
            stack.push(nobs)
            next_stacked.append(stack.get())

        if train:
            for m, agent in enumerate(agents):
                if not had_task[m]:
                    continue
                agent.store(
                    stacked_obs[m], actions[m], rewards[m],
                    next_stacked[m], float(done),
                )
                loss = agent.train_step()
                if loss is not None:
                    ep_losses.append(loss)

        ep_costs.append(info["avg_cost"])
        raw_obs = next_raw_obs

        if done:
            break

    return {
        "avg_cost":   float(np.mean(ep_costs)),
        "total_loss": float(np.mean(ep_losses)) if ep_losses else float("nan"),
        "eps":        agents[0].eps,
    }


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
    env         = NTNMECEnv(cfg)
    stacked_dim = env.obs_dim * args.K

    agents = make_agents(
        stacked_dim,
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

    print(f"\nTraining IL+Stack — {args.n_ues} UEs, {args.n_uavs} UAVs, "
          f"{args.episodes} episodes  |  K={args.K}  stacked_dim={stacked_dim}\n")
    print(f"{'Episode':>8}  {'AvgCost':>10}  {'Loss':>10}  {'Eps':>6}")
    print("-" * 42)

    for ep in range(1, args.episodes + 1):
        stats = run_episode(env, agents, args.K, train=True)
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

    eval_costs = []
    for _ in range(args.eval_episodes):
        s = run_episode(env, agents, args.K, train=False)
        eval_costs.append(s["avg_cost"])

    eval_mean = float(np.mean(eval_costs))
    eval_std  = float(np.std(eval_costs))
    print(f"\nEval ({args.eval_episodes} eps): "
          f"mean={eval_mean:.4f}  std={eval_std:.4f}")

    results = {
        "method":     "IL+Stack",
        "config":     vars(args),
        "history":    history,
        "eval_mean":  eval_mean,
        "eval_std":   eval_std,
        "best_train": best_cost,
    }

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "il_stack_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.save_dir}/")

    return results


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--n_ues",        type=int,   default=10)
    p.add_argument("--n_uavs",       type=int,   default=2)
    p.add_argument("--steps_per_ep", type=int,   default=100)
    p.add_argument("--task_prob",    type=float, default=0.3)

    p.add_argument("--K",            type=int,   default=4)
    p.add_argument("--hidden",       type=int,   default=128)

    p.add_argument("--episodes",      type=int,   default=500)
    p.add_argument("--eval_episodes", type=int,   default=20)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--gamma",         type=float, default=0.9)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--target_update", type=int,   default=20)
    p.add_argument("--eps_decay",     type=float, default=0.995)

    p.add_argument("--seed",      type=int,  default=42)
    p.add_argument("--log_every", type=int,  default=50)
    p.add_argument("--save_dir",  type=str,  default="checkpoints")
    p.add_argument("--cpu",       action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args    = get_args()
    results = train(args)
    print(f"\nIL+Stack eval  —  {results['eval_mean']:.4f} ± {results['eval_std']:.4f}")