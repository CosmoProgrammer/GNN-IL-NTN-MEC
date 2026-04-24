"""
GNN-IL + Framestack Training Loop.

Architecture:
    raw_obs → GNN → cat(raw_obs, gnn_out) = enriched_obs
    K enriched_obs frames stacked → DuelingDQN → Q-values
    DQN input dim = K * (obs_dim + gnn_out)

Gradient flow in train_step():
    loss → DQN(stacked_enriched) → GNN(raw_obs_curr) → raw_obs_curr
    Only the current frame's GNN is in the gradient graph.
    The K-1 history frames are stored as pre-computed enriched vectors
    (stop-gradient on history — standard practice in frame-stacked DQN).

Usage:
    python train_gnn_il_stack.py --episodes 500 --log_every 50
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import json
import os
from typing import List, Optional, Tuple

from env        import NTNMECEnv, EnvConfig
from gnnAgent  import BipartiteGNNEncoder
from agent      import DuelingDQN
from framestack import FrameStack


# ─── Buffer ────────────────────────────────────────────────────────────────

class GNNStackBuffer:
    def __init__(self, capacity: int = 5_000):
        self.buf = deque(maxlen=capacity)

    def push(
        self,
        raw_curr:   np.ndarray,   # (M, obs_dim)
        raw_next:   np.ndarray,   # (M, obs_dim)
        hist_curr:  np.ndarray,   # (M, K-1, enr_dim)
        hist_next:  np.ndarray,   # (M, K-1, enr_dim)
        uav_curr:   np.ndarray,   # (N, 3)
        uav_next:   np.ndarray,   # (N, 3)
        ew_curr:    np.ndarray,   # (M, N)
        ew_next:    np.ndarray,   # (M, N)
        actions:    np.ndarray,   # (M,)
        rewards:    np.ndarray,   # (M,)
        masks:      np.ndarray,   # (M,) bool
        done:       float,
    ):
        self.buf.append((
            raw_curr.astype(np.float32),  raw_next.astype(np.float32),
            hist_curr.astype(np.float32), hist_next.astype(np.float32),
            uav_curr.astype(np.float32),  uav_next.astype(np.float32),
            ew_curr.astype(np.float32),   ew_next.astype(np.float32),
            actions.astype(np.int64),
            rewards.astype(np.float32),
            masks.astype(bool),
            np.float32(done),
        ))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buf, batch_size)
        (rc, rn, hc, hn, uc, un, ec, en, act, rew, mask, done) = zip(*batch)
        return (
            torch.tensor(np.stack(rc),   dtype=torch.float32),  # (B,M,obs)
            torch.tensor(np.stack(rn),   dtype=torch.float32),  # (B,M,obs)
            torch.tensor(np.stack(hc),   dtype=torch.float32),  # (B,M,K-1,enr)
            torch.tensor(np.stack(hn),   dtype=torch.float32),  # (B,M,K-1,enr)
            torch.tensor(np.stack(uc),   dtype=torch.float32),  # (B,N,3)
            torch.tensor(np.stack(un),   dtype=torch.float32),  # (B,N,3)
            torch.tensor(np.stack(ec),   dtype=torch.float32),  # (B,M,N)
            torch.tensor(np.stack(en),   dtype=torch.float32),  # (B,M,N)
            torch.tensor(np.stack(act),  dtype=torch.long),     # (B,M)
            torch.tensor(np.stack(rew),  dtype=torch.float32),  # (B,M)
            torch.tensor(np.stack(mask), dtype=torch.bool),     # (B,M)
            torch.tensor(np.array(done), dtype=torch.float32),  # (B,)
        )

    def __len__(self) -> int:
        return len(self.buf)


# ─── Agent ─────────────────────────────────────────────────────────────────

class GNNStackAgent:
    def __init__(
        self,
        obs_dim:       int,
        n_uavs:        int,
        n_actions:     int,
        K:             int   = 4,
        gnn_hidden:    int   = 64,
        gnn_out:       int   = 32,
        dqn_hidden:    int   = 128,
        lr:            float = 1e-3,
        gamma:         float = 0.9,
        batch_size:    int   = 32,
        buffer_cap:    int   = 5_000,
        target_update: int   = 20,
        eps_start:     float = 1.0,
        eps_end:       float = 0.05,
        eps_decay:     float = 0.995,
        device:        str   = "cpu",
    ):
        self.K             = K
        self.obs_dim       = obs_dim
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update
        self.device        = torch.device(device)
        self.eps       = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay

        self.enriched_dim = obs_dim + gnn_out
        self.stacked_dim  = K * self.enriched_dim

        self.gnn = BipartiteGNNEncoder(
            ue_dim=obs_dim, uav_dim=3, hidden=gnn_hidden, out_dim=gnn_out,
        ).to(self.device)

        self.policy_net = DuelingDQN(self.stacked_dim, n_actions, dqn_hidden).to(self.device)
        self.target_net = DuelingDQN(self.stacked_dim, n_actions, dqn_hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimiser = torch.optim.RMSprop(
            list(self.gnn.parameters()) + list(self.policy_net.parameters()), lr=lr,
        )
        self.buffer = GNNStackBuffer(buffer_cap)
        self.steps  = 0

    @staticmethod
    def _edge_matrix(graph_data: dict) -> np.ndarray:
        M = graph_data["n_ues"]; N = graph_data["n_uavs"]
        src, dst, w = graph_data["ue_uav"]
        mat = np.zeros((M, N), dtype=np.float32)
        for s, d, wt in zip(src, dst, w):
            mat[s, d] = wt
        return mat

    def _enrich(self, obs_list: List[np.ndarray], graph_data: dict) -> np.ndarray:
        """GNN forward under no_grad. Returns (M, enriched_dim)."""
        with torch.no_grad():
            ue  = torch.tensor(np.stack(obs_list), dtype=torch.float32
                               ).unsqueeze(0).to(self.device)
            uav = torch.tensor(graph_data["uav_x"], dtype=torch.float32
                               ).unsqueeze(0).to(self.device)
            ew  = torch.tensor(self._edge_matrix(graph_data), dtype=torch.float32
                               ).unsqueeze(0).to(self.device)
            h   = self.gnn(ue, uav, ew)
            enr = torch.cat([ue, h], dim=-1)
        return enr.squeeze(0).cpu().numpy()

    def select_actions(
        self,
        obs_list:   List[np.ndarray],
        graph_data: dict,
        stacks:     List[FrameStack],
        greedy:     bool = False,
    ) -> List[int]:
        enriched = self._enrich(obs_list, graph_data)
        for m, stack in enumerate(stacks):
            stack.push(enriched[m])

        stacked = np.stack([s.get() for s in stacks])
        with torch.no_grad():
            q = self.policy_net(
                torch.tensor(stacked, dtype=torch.float32).to(self.device)
            )
            greedy_acts = q.argmax(dim=1).cpu().numpy()

        if greedy:
            return [int(a) for a in greedy_acts]
        return [
            random.randint(0, self.n_actions - 1) if random.random() < self.eps
            else int(greedy_acts[m])
            for m in range(len(obs_list))
        ]

    def store(
        self,
        raw_curr:    List[np.ndarray],
        raw_next:    List[np.ndarray],
        graph_curr:  dict,
        graph_next:  dict,
        stacks_curr: List[FrameStack],
        stacks_next: List[FrameStack],
        actions:     List[int],
        rewards:     List[float],
        task_masks:  List[bool],
        done:        bool,
    ):
        def get_hist(stacks):
            # stacks[m].frames is a deque of K enriched frames.
            # History = first K-1 frames (all but the latest).
            hist = [np.stack(list(s.frames)[:-1]) for s in stacks]
            return np.stack(hist)                              # (M, K-1, enr_dim)

        self.buffer.push(
            raw_curr  = np.stack(raw_curr),
            raw_next  = np.stack(raw_next),
            hist_curr = get_hist(stacks_curr),
            hist_next = get_hist(stacks_next),
            uav_curr  = graph_curr["uav_x"],
            uav_next  = graph_next["uav_x"],
            ew_curr   = self._edge_matrix(graph_curr),
            ew_next   = self._edge_matrix(graph_next),
            actions   = np.array(actions),
            rewards   = np.array(rewards),
            masks     = np.array(task_masks),
            done      = float(done),
        )

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        (rc, rn, hc, hn, uc, un, ec, en,
         actions, rewards, masks, dones) = [
            t.to(self.device) for t in self.buffer.sample(self.batch_size)
        ]
        B, M, _ = rc.shape

        # Current: GNN in grad graph
        h_c   = self.gnn(rc, uc, ec)
        enc_c = torch.cat([rc, h_c], dim=-1)                  # (B,M,enr)
        stk_c = torch.cat([hc, enc_c.unsqueeze(2)], dim=2    # (B,M,K,enr)
                           ).view(B * M, -1)                   # (B*M,stacked)

        q_all  = self.policy_net(stk_c)
        act_f  = actions.view(B * M)
        q_pred = q_all.gather(1, act_f.unsqueeze(1)).squeeze(1)

        # Next: no grad
        with torch.no_grad():
            h_n   = self.gnn(rn, un, en)
            enc_n = torch.cat([rn, h_n], dim=-1)
            stk_n = torch.cat([hn, enc_n.unsqueeze(2)], dim=2).view(B * M, -1)

            next_act = self.policy_net(stk_n).argmax(dim=1)
            next_q   = self.target_net(stk_n).gather(
                1, next_act.unsqueeze(1)
            ).squeeze(1)

            rew_f  = rewards.view(B * M)
            done_f = dones.unsqueeze(1).expand(B, M).reshape(B * M)
            q_tgt  = rew_f + self.gamma * next_q * (1.0 - done_f)

        mask_f = masks.view(B * M)
        if mask_f.sum() == 0:
            return None

        loss = F.smooth_l1_loss(q_pred[mask_f], q_tgt[mask_f])
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.gnn.parameters()) + list(self.policy_net.parameters()),
            max_norm=10.0,
        )
        self.optimiser.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def save(self, path: str):
        torch.save({"gnn": self.gnn.state_dict(),
                    "policy_net": self.policy_net.state_dict()}, path)


# ─── Helpers ───────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _copy_stacks(stacks: List[FrameStack]) -> List[FrameStack]:
    copies = []
    for s in stacks:
        c = FrameStack(s.obs_dim, s.K)
        c.frames = deque(s.frames, maxlen=s.K)
        copies.append(c)
    return copies


# ─── Episode ───────────────────────────────────────────────────────────────

def run_episode(env: NTNMECEnv, agent: GNNStackAgent, train: bool = True) -> dict:
    obs_list   = env.reset()
    graph_data = env.get_graph_data()
    M          = env.n_agents

    stacks = [FrameStack(agent.enriched_dim, agent.K) for _ in range(M)]
    first_enr = agent._enrich(obs_list, graph_data)
    for m, s in enumerate(stacks):
        s.reset(first_enr[m])

    ep_costs, ep_losses = [], []

    for _ in range(env.cfg.I):
        had_task     = [t is not None for t in env.tasks]
        stacks_before = _copy_stacks(stacks)

        actions = agent.select_actions(obs_list, graph_data, stacks, greedy=not train)

        next_obs_list, rewards, done, info = env.step(actions)
        next_graph = env.get_graph_data()

        # Build next-step stacks: copy post-action stacks, push next enriched
        stacks_next = _copy_stacks(stacks)
        next_enr = agent._enrich(next_obs_list, next_graph)
        for m, s in enumerate(stacks_next):
            s.push(next_enr[m])

        if train and any(had_task):
            agent.store(
                obs_list, next_obs_list,
                graph_data, next_graph,
                stacks_before, stacks_next,
                actions, rewards, had_task, done,
            )
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

        ep_costs.append(info["avg_cost"])
        obs_list   = next_obs_list
        graph_data = next_graph
        if done:
            break

    return {
        "avg_cost":   float(np.mean(ep_costs)),
        "total_loss": float(np.mean(ep_losses)) if ep_losses else float("nan"),
        "eps":        agent.eps,
    }


# ─── Training ──────────────────────────────────────────────────────────────

def train(args) -> dict:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    cfg = EnvConfig(M=args.n_ues, N=args.n_uavs, I=args.steps_per_ep, P_task=args.task_prob)
    env = NTNMECEnv(cfg)
    agent = GNNStackAgent(
        obs_dim=env.obs_dim, n_uavs=cfg.N, n_actions=env.n_actions,
        K=args.K, gnn_hidden=args.gnn_hidden, gnn_out=args.gnn_out,
        dqn_hidden=args.dqn_hidden, lr=args.lr, gamma=args.gamma,
        batch_size=args.batch_size, buffer_cap=args.buffer_cap,
        target_update=args.target_update, eps_decay=args.eps_decay, device=device,
    )

    history, best_cost = [], float("inf")
    print(f"\nTraining GNN-IL+Stack — {args.n_ues} UEs, {args.n_uavs} UAVs, "
          f"{args.episodes} episodes  |  K={args.K}  stacked_dim={agent.stacked_dim}\n")
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
                agent.save(os.path.join(args.save_dir, "gnn_il_stack_best.pt"))

    eval_costs = [run_episode(env, agent, train=False)["avg_cost"]
                  for _ in range(args.eval_episodes)]
    eval_mean = float(np.mean(eval_costs))
    eval_std  = float(np.std(eval_costs))
    print(f"\nEval ({args.eval_episodes} eps): mean={eval_mean:.4f}  std={eval_std:.4f}")

    results = {"method": "GNN-IL+Stack", "config": vars(args), "history": history,
               "eval_mean": eval_mean, "eval_std": eval_std, "best_train": best_cost}

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        agent.save(os.path.join(args.save_dir, "gnn_il_stack_final.pt"))
        with open(os.path.join(args.save_dir, "gnn_il_stack_results.json"), "w") as f:
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
    p.add_argument("--seed",      type=int,  default=42)
    p.add_argument("--log_every", type=int,  default=50)
    p.add_argument("--save_dir",  type=str,  default="checkpoints")
    p.add_argument("--cpu",       action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args    = get_args()
    results = train(args)
    print(f"\nGNN-IL+Stack eval  —  {results['eval_mean']:.4f} ± {results['eval_std']:.4f}")