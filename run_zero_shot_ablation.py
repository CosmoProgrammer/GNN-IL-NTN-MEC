import argparse
import os
import random
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from env import EnvConfig, NTNMECEnv
from gnnAgent import GNNILAgent
from train import load_agents as load_il_agents
from train import make_agents as make_il_agents
from train import run_episode as run_il_episode
from trainGnn import run_episode as run_gnn_episode


@dataclass
class EvalRow:
    method: str
    train_m: int
    eval_m: int
    status: str
    mean: Optional[float]
    std: Optional[float]
    note: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(force_cpu: bool) -> str:
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_gnn(
    checkpoint_path: str,
    eval_m: int,
    n_uavs: int,
    eval_episodes: int,
    steps_per_ep: int,
    task_prob: float,
    seed: int,
    device: str,
    gnn_hidden: int,
    gnn_out: int,
    dqn_hidden: int,
) -> EvalRow:
    if not os.path.exists(checkpoint_path):
        return EvalRow(
            method="GNN-IL",
            train_m=10,
            eval_m=eval_m,
            status="MISSING",
            mean=None,
            std=None,
            note=f"checkpoint not found: {checkpoint_path}",
        )

    try:
        set_seed(seed)
        cfg = EnvConfig(M=eval_m, N=n_uavs, I=steps_per_ep, P_task=task_prob)
        env = NTNMECEnv(cfg)

        agent = GNNILAgent(
            obs_dim=env.obs_dim,
            n_uavs=n_uavs,
            n_actions=env.n_actions,
            gnn_hidden=gnn_hidden,
            gnn_out=gnn_out,
            dqn_hidden=dqn_hidden,
            device=device,
        )
        agent.load(checkpoint_path)

        costs: List[float] = []
        for _ in range(eval_episodes):
            stats = run_gnn_episode(env, agent, train=False)
            costs.append(stats["avg_cost"])

        return EvalRow(
            method="GNN-IL",
            train_m=10,
            eval_m=eval_m,
            status="OK",
            mean=float(np.mean(costs)),
            std=float(np.std(costs)),
            note="zero-shot" if eval_m != 10 else "in-domain",
        )
    except Exception as exc:  # pragma: no cover
        return EvalRow(
            method="GNN-IL",
            train_m=10,
            eval_m=eval_m,
            status="ERROR",
            mean=None,
            std=None,
            note=str(exc),
        )


def evaluate_il(
    checkpoint_dir: str,
    eval_m: int,
    n_uavs: int,
    eval_episodes: int,
    steps_per_ep: int,
    task_prob: float,
    seed: int,
    device: str,
    cycle_weights: bool,
) -> EvalRow:
    if not os.path.isdir(checkpoint_dir):
        return EvalRow(
            method="IL",
            train_m=10,
            eval_m=eval_m,
            status="MISSING",
            mean=None,
            std=None,
            note=f"checkpoint dir not found: {checkpoint_dir}",
        )

    try:
        set_seed(seed)
        cfg = EnvConfig(M=eval_m, N=n_uavs, I=steps_per_ep, P_task=task_prob)
        env = NTNMECEnv(cfg)

        if cycle_weights:
            agents = _load_il_agents_cycled(env=env, save_dir=checkpoint_dir, tag="il_final", device=device)
        else:
            agents = load_il_agents(env=env, save_dir=checkpoint_dir, tag="il_final", device=device)

        costs: List[float] = []
        for _ in range(eval_episodes):
            stats = run_il_episode(env, agents, train=False)
            costs.append(stats["avg_cost"])

        return EvalRow(
            method="IL",
            train_m=10,
            eval_m=eval_m,
            status="OK",
            mean=float(np.mean(costs)),
            std=float(np.std(costs)),
            note=(
                "zero-shot (cycled weights)"
                if cycle_weights and eval_m != 10
                else ("zero-shot" if eval_m != 10 else "in-domain")
            ),
        )
    except FileNotFoundError as exc:
        return EvalRow(
            method="IL",
            train_m=10,
            eval_m=eval_m,
            status="FAIL",
            mean=None,
            std=None,
            note=f"per-UE checkpoint mismatch: {exc}",
        )


def _load_il_agents_cycled(
    env: NTNMECEnv,
    save_dir: str,
    tag: str,
    device: str,
) -> List:
    pattern = re.compile(rf"^{re.escape(tag)}_agent(\\d+)\\.pt$")
    indices: List[int] = []
    for name in os.listdir(save_dir):
        m = pattern.match(name)
        if m:
            indices.append(int(m.group(1)))

    indices = sorted(set(indices))
    if not indices:
        raise FileNotFoundError(
            f"No IL checkpoints found for tag '{tag}' in {save_dir}"
        )

    agents = make_il_agents(env=env, device=device)
    for m, agent in enumerate(agents):
        src_idx = indices[m % len(indices)]
        path = os.path.join(save_dir, f"{tag}_agent{src_idx}.pt")
        agent.policy_net.load_state_dict(torch.load(path, map_location=device))
        agent.policy_net.eval()

    return agents
    except RuntimeError as exc:
        return EvalRow(
            method="IL",
            train_m=10,
            eval_m=eval_m,
            status="FAIL",
            mean=None,
            std=None,
            note=f"shape/load mismatch: {exc}",
        )
    except Exception as exc:  # pragma: no cover
        return EvalRow(
            method="IL",
            train_m=10,
            eval_m=eval_m,
            status="ERROR",
            mean=None,
            std=None,
            note=str(exc),
        )


def evaluate_retrained(
    method: str,
    train_m: int,
    n_uavs: int,
    eval_episodes: int,
    steps_per_ep: int,
    task_prob: float,
    seed: int,
    device: str,
    checkpoints_root: str,
    use_seed_layout: bool,
    seed_id: int,
    gnn_hidden: int,
    gnn_out: int,
    dqn_hidden: int,
) -> EvalRow:
    ckpt_dir = os.path.join(checkpoints_root, f"ue{train_m}")
    if use_seed_layout:
        ckpt_dir = os.path.join(ckpt_dir, f"seed{seed_id}")

    if method == "GNN-IL":
        if use_seed_layout:
            ckpt = os.path.join(ckpt_dir, "gnn", "gnn_il_final.pt")
        else:
            ckpt = os.path.join(ckpt_dir, "gnn_il_final.pt")
        row = evaluate_gnn(
            checkpoint_path=ckpt,
            eval_m=train_m,
            n_uavs=n_uavs,
            eval_episodes=eval_episodes,
            steps_per_ep=steps_per_ep,
            task_prob=task_prob,
            seed=seed,
            device=device,
            gnn_hidden=gnn_hidden,
            gnn_out=gnn_out,
            dqn_hidden=dqn_hidden,
        )
        row.train_m = train_m
        row.note = "retrained in-domain"
        return row

    il_dir = os.path.join(ckpt_dir, "standard") if use_seed_layout else ckpt_dir
    row = evaluate_il(
        checkpoint_dir=il_dir,
        eval_m=train_m,
        n_uavs=n_uavs,
        eval_episodes=eval_episodes,
        steps_per_ep=steps_per_ep,
        task_prob=task_prob,
        seed=seed,
        device=device,
    )
    row.train_m = train_m
    row.note = "retrained in-domain" if row.status == "OK" else row.note
    return row


def fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def print_rows(title: str, rows: List[EvalRow]) -> None:
    print("\n" + title)
    print("-" * len(title))
    print(f"{'Method':<8} {'TrainM':>6} {'EvalM':>6} {'Status':>8} {'Mean':>10} {'Std':>10}  Note")
    for r in rows:
        print(
            f"{r.method:<8} {r.train_m:>6} {r.eval_m:>6} {r.status:>8} "
            f"{fmt(r.mean):>10} {fmt(r.std):>10}  {r.note}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run zero-shot generalization ablation: train M=10 checkpoints evaluated on M=10/20/30."
    )
    p.add_argument("--checkpoints_root", type=str, default="checkpoints")
    p.add_argument("--use_seed_results", action="store_true")
    p.add_argument("--seed_id", type=int, default=42)
    p.add_argument("--eval_ms", type=int, nargs="+", default=[10, 20, 30])
    p.add_argument("--n_uavs", type=int, default=2)
    p.add_argument("--eval_episodes", type=int, default=20)
    p.add_argument("--steps_per_ep", type=int, default=100)
    p.add_argument("--task_prob", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--gnn_hidden", type=int, default=64)
    p.add_argument("--gnn_out", type=int, default=32)
    p.add_argument("--dqn_hidden", type=int, default=128)

    p.add_argument("--skip_retrained", action="store_true")
    p.add_argument(
        "--il_cycle_weights",
        action="store_true",
        help="For IL zero-shot only: cycle available source checkpoints across EvalM agents.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = pick_device(args.cpu)

    print(f"Device: {device}")
    print("Evaluating zero-shot transfer from TrainM=10 checkpoints.")

    ue10_dir = os.path.join(args.checkpoints_root, "ue10")
    if args.use_seed_results:
        ue10_seed_dir = os.path.join(ue10_dir, f"seed{args.seed_id}")
        gnn_ue10 = os.path.join(ue10_seed_dir, "gnn", "gnn_il_final.pt")
        il_ue10_dir = os.path.join(ue10_seed_dir, "standard")
    else:
        gnn_ue10 = os.path.join(ue10_dir, "gnn_il_final.pt")
        il_ue10_dir = ue10_dir

    zero_shot_rows: List[EvalRow] = []
    for eval_m in args.eval_ms:
        zero_shot_rows.append(
            evaluate_gnn(
                checkpoint_path=gnn_ue10,
                eval_m=eval_m,
                n_uavs=args.n_uavs,
                eval_episodes=args.eval_episodes,
                steps_per_ep=args.steps_per_ep,
                task_prob=args.task_prob,
                seed=args.seed,
                device=device,
                gnn_hidden=args.gnn_hidden,
                gnn_out=args.gnn_out,
                dqn_hidden=args.dqn_hidden,
            )
        )
        zero_shot_rows.append(
            evaluate_il(
                checkpoint_dir=il_ue10_dir,
                eval_m=eval_m,
                n_uavs=args.n_uavs,
                eval_episodes=args.eval_episodes,
                steps_per_ep=args.steps_per_ep,
                task_prob=args.task_prob,
                seed=args.seed,
                device=device,
                cycle_weights=args.il_cycle_weights,
            )
        )

    print_rows("Zero-Shot Ablation (TrainM=10 -> EvalM)", zero_shot_rows)

    if not args.skip_retrained:
        retrained_rows: List[EvalRow] = []
        for m in args.eval_ms:
            retrained_rows.append(
                evaluate_retrained(
                    method="GNN-IL",
                    train_m=m,
                    n_uavs=args.n_uavs,
                    eval_episodes=args.eval_episodes,
                    steps_per_ep=args.steps_per_ep,
                    task_prob=args.task_prob,
                    seed=args.seed,
                    device=device,
                    checkpoints_root=args.checkpoints_root,
                    use_seed_layout=args.use_seed_results,
                    seed_id=args.seed_id,
                    gnn_hidden=args.gnn_hidden,
                    gnn_out=args.gnn_out,
                    dqn_hidden=args.dqn_hidden,
                )
            )
            retrained_rows.append(
                evaluate_retrained(
                    method="IL",
                    train_m=m,
                    n_uavs=args.n_uavs,
                    eval_episodes=args.eval_episodes,
                    steps_per_ep=args.steps_per_ep,
                    task_prob=args.task_prob,
                    seed=args.seed,
                    device=device,
                    checkpoints_root=args.checkpoints_root,
                    use_seed_layout=args.use_seed_results,
                    seed_id=args.seed_id,
                    gnn_hidden=args.gnn_hidden,
                    gnn_out=args.gnn_out,
                    dqn_hidden=args.dqn_hidden,
                )
            )

        print_rows("Retrained In-Domain Reference (TrainM=EvalM)", retrained_rows)

    if args.il_cycle_weights:
        print("\nNote: IL cycling applies only to zero-shot IL rows (TrainM=10 -> EvalM).")


if __name__ == "__main__":
    main()
