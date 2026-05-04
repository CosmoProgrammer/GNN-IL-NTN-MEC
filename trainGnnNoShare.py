"""
GNN-IL-NoShare Training Loop — parameter sharing ablation.

Identical to train_gnn_il.py except it uses GNNILNoShareAgent:
M separate GNN encoders + M separate DQNs, no shared weights.

This isolates the two mechanisms behind GNN-IL's gain:
    IL → GNN-IL-NoShare   : value of richer observations alone
    GNN-IL-NoShare → GNN-IL : value of gradient coupling alone

Usage:
    python trainGnnNoShare.py --n_ues 10 --n_uavs 2 --episodes 500
"""

import argparse
import numpy as np
import torch
import random
import json
import os

from env               import NTNMECEnv, EnvConfig
from gnnNoShareAgent import GNNILNoShareAgent


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_episode(env, agent, train=True) -> dict:
    obs_list   = env.reset()
    graph_data = env.get_graph_data()

    ep_costs  = []
    ep_losses = []

    for _ in range(env.cfg.I):
        had_task = [t is not None for t in env.tasks]
        actions  = agent.select_actions(obs_list, graph_data, greedy=not train)

        next_obs_list, rewards, done, info = env.step(actions)
        next_graph_data = env.get_graph_data()

        if train:
            if any(had_task):
                agent.store(
                    obs_list, graph_data,
                    actions, rewards,
                    next_obs_list, next_graph_data,
                    done, task_mask=had_task,
                )
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

        ep_costs.append(info["avg_cost"])
        obs_list   = next_obs_list
        graph_data = next_graph_data

        if done:
            break

    return {
        "avg_cost":   float(np.mean(ep_costs)),
        "total_loss": float(np.mean(ep_losses)) if ep_losses else float("nan"),
        "eps":        agent.eps,
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
    env = NTNMECEnv(cfg)

    agent = GNNILNoShareAgent(
        obs_dim       = env.obs_dim,
        n_uavs        = cfg.N,
        n_actions     = env.n_actions,
        n_agents      = env.n_agents,
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

    history   = []
    best_cost = float("inf")

    print(f"\nTraining GNN-IL-NoShare — {args.n_ues} UEs, {args.n_uavs} UAVs, "
          f"{args.episodes} episodes  (ablation: no parameter sharing)\n")
    print(f"{'Episode':>8}  {'AvgCost':>10}  {'Loss':>10}  {'Eps':>6}")
    print("-" * 42)

    for ep in range(1, args.episodes + 1):
        stats = run_episode(env, agent, train=True)
        history.append(stats)
        agent.decay_epsilon()

        if ep % args.log_every == 0:
            print(f"{ep:>8}  {stats['avg_cost']:>10.4f}  "
                  f"{stats['total_loss']:>10.4f}  {stats['eps']:>6.3f}")

        if stats["avg_cost"] < best_cost:
            best_cost = stats["avg_cost"]
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                agent.save(os.path.join(args.save_dir, "gnn_noshare_best.pt"))

    eval_costs = []
    for _ in range(args.eval_episodes):
        s = run_episode(env, agent, train=False)
        eval_costs.append(s["avg_cost"])

    eval_mean = float(np.mean(eval_costs))
    eval_std  = float(np.std(eval_costs))
    print(f"\nEval ({args.eval_episodes} eps): "
          f"mean={eval_mean:.4f}  std={eval_std:.4f}")

    results = {
        "method":     "GNN-IL-NoShare",
        "config":     vars(args),
        "history":    history,
        "eval_mean":  eval_mean,
        "eval_std":   eval_std,
        "best_train": best_cost,
    }

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        agent.save(os.path.join(args.save_dir, "gnn_noshare_final.pt"))
        with open(os.path.join(args.save_dir, "gnn_noshare_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.save_dir}/")

    return results


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_ues",         type=int,   default=10)
    p.add_argument("--n_uavs",        type=int,   default=2)
    p.add_argument("--steps_per_ep",  type=int,   default=100)
    p.add_argument("--task_prob",     type=float, default=0.3)
    p.add_argument("--gnn_hidden",    type=int,   default=64)
    p.add_argument("--gnn_out",       type=int,   default=32)
    p.add_argument("--dqn_hidden",    type=int,   default=128)
    p.add_argument("--episodes",      type=int,   default=500)
    p.add_argument("--eval_episodes", type=int,   default=20)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--gamma",         type=float, default=0.9)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--buffer_cap",    type=int,   default=5_000)
    p.add_argument("--target_update", type=int,   default=20)
    p.add_argument("--eps_decay",     type=float, default=0.995)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--log_every",     type=int,   default=50)
    p.add_argument("--save_dir",      type=str,   default="checkpoints")
    p.add_argument("--cpu",           action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args    = get_args()
    results = train(args)
    print(f"\nGNN-IL-NoShare eval  —  "
          f"{results['eval_mean']:.4f} ± {results['eval_std']:.4f}")