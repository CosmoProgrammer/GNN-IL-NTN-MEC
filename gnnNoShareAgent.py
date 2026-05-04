"""
GNN-IL-NoShare Agent — M separate GNN encoders + M separate Dueling DQNs.

This is the parameter-sharing ablation for GNN-IL. Architecturally identical
to GNN-IL except every UE has its own GNN and DQN weights that never sync.

Purpose
-------
GNN-IL's performance gain over IL comes from two separable mechanisms:
  (A) Richer observations — GNN propagates UAV congestion to UE embeddings
  (B) Gradient coupling  — shared weights mean UE 1's loss updates weights
                           used by all other UEs via GNN message passing

This agent preserves (A) but eliminates (B).

Comparison table:
    IL               → no GNN,   no sharing  (baseline)
    GNN-IL-NoShare   → GNN obs,  no sharing  (isolates A)
    GNN-IL           → GNN obs,  sharing     (A + B)

If GNN-IL-NoShare >> IL:          richer observations drive the gain
If GNN-IL >> GNN-IL-NoShare:      gradient coupling adds on top
If GNN-IL ≈ GNN-IL-NoShare:       coupling contributes little

Implementation notes
--------------------
- Each agent m owns: BipartiteGNNEncoder[m], DuelingDQN[m], ReplayBuffer[m],
  Optimiser[m]
- Interaction: agent m runs its own GNN[m] on the full graph → enriched_m
- Storage: raw obs + graph arrays stored; GNN re-run during train_step
  (single-node pass, consistent with GNN-CTDE approach)
- Training: loss.backward() only touches GNN[m] and DQN[m] — no coupling
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
from typing import List, Optional

from agent    import DuelingDQN          # reuse unchanged
from gnnAgent import BipartiteGNNEncoder # reuse unchanged


# ─── Per-agent graph replay buffer ─────────────────────────────────────────

class AgentGraphBuffer:
    """
    Stores (raw_obs, uav_x, edge_row, action, reward,
            next_raw_obs, next_uav_x, next_edge_row, done)
    for a single UE.

    edge_row : (N,) — just this UE's row of the full (M, N) edge matrix.
    We store the UE's own edge row rather than the full matrix to keep
    memory proportional to N, not M*N.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obs:           np.ndarray,   # (obs_dim,)
        uav_x:         np.ndarray,   # (N, 3)
        edge_row:      np.ndarray,   # (N,)
        action:        int,
        reward:        float,
        next_obs:      np.ndarray,
        next_uav_x:    np.ndarray,
        next_edge_row: np.ndarray,
        done:          float,
    ):
        self.buffer.append((
            obs.astype(np.float32),
            uav_x.astype(np.float32),
            edge_row.astype(np.float32),
            int(action),
            float(reward),
            next_obs.astype(np.float32),
            next_uav_x.astype(np.float32),
            next_edge_row.astype(np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        (obs, uav_x, edge_row,
         actions, rewards,
         next_obs, next_uav_x, next_edge_row,
         dones) = zip(*batch)

        return (
            torch.tensor(np.stack(obs),           dtype=torch.float32),
            torch.tensor(np.stack(uav_x),         dtype=torch.float32),
            torch.tensor(np.stack(edge_row),       dtype=torch.float32),
            torch.tensor(actions,                  dtype=torch.long),
            torch.tensor(rewards,                  dtype=torch.float32),
            torch.tensor(np.stack(next_obs),       dtype=torch.float32),
            torch.tensor(np.stack(next_uav_x),     dtype=torch.float32),
            torch.tensor(np.stack(next_edge_row),  dtype=torch.float32),
            torch.tensor(dones,                    dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─── Single agent (one UE) ─────────────────────────────────────────────────

class SingleGNNAgent:
    """
    One UE's private GNN encoder + DQN + buffer + optimiser.
    Enriches its own observation using the full graph (all UE/UAV nodes
    visible) but gradients never leave this agent's own parameters.
    """

    def __init__(
        self,
        obs_dim:      int,
        n_uavs:       int,
        n_actions:    int,
        gnn_hidden:   int   = 64,
        gnn_out:      int   = 32,
        dqn_hidden:   int   = 128,
        lr:           float = 1e-3,
        gamma:        float = 0.9,
        buffer_cap:   int   = 5_000,
        batch_size:   int   = 32,
        target_update:int   = 20,
        device:       torch.device = torch.device("cpu"),
    ):
        self.n_actions     = n_actions
        self.n_uavs        = n_uavs
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update
        self.device        = device
        self.enriched_dim  = obs_dim + gnn_out

        self.gnn = BipartiteGNNEncoder(
            ue_dim  = obs_dim,
            uav_dim = 3,
            hidden  = gnn_hidden,
            out_dim = gnn_out,
        ).to(device)

        self.policy_net = DuelingDQN(
            self.enriched_dim, n_actions, dqn_hidden
        ).to(device)

        self.target_net = DuelingDQN(
            self.enriched_dim, n_actions, dqn_hidden
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimiser covers THIS agent's GNN + DQN only
        self.optimiser = torch.optim.RMSprop(
            list(self.gnn.parameters()) + list(self.policy_net.parameters()),
            lr=lr,
        )

        self.buffer = AgentGraphBuffer(capacity=buffer_cap)
        self.steps  = 0

    @torch.no_grad()
    def enrich(
        self,
        obs:      np.ndarray,   # (obs_dim,)
        uav_x:    np.ndarray,   # (N, 3)
        edge_row: np.ndarray,   # (N,)  — this UE's edge weights
    ) -> np.ndarray:            # (enriched_dim,)
        """Run this agent's GNN as a single-node graph."""
        ue_x   = torch.tensor(obs,      dtype=torch.float32).view(1, 1, -1).to(self.device)
        uav_t  = torch.tensor(uav_x,    dtype=torch.float32).unsqueeze(0).to(self.device)
        edge_t = torch.tensor(edge_row, dtype=torch.float32).view(1, 1, -1).to(self.device)

        h      = self.gnn(ue_x, uav_t, edge_t)                  # (1, 1, gnn_out)
        enr    = torch.cat([ue_x, h], dim=-1)                   # (1, 1, enriched_dim)
        return enr.squeeze().cpu().numpy()                       # (enriched_dim,)

    def _enrich_batch(
        self,
        obs:      torch.Tensor,   # (B, obs_dim)
        uav_x:    torch.Tensor,   # (B, N, 3)
        edge_row: torch.Tensor,   # (B, N)
    ) -> torch.Tensor:            # (B, enriched_dim)
        """Re-run GNN inside computation graph for training."""
        ue_x   = obs.unsqueeze(1)                                # (B, 1, obs_dim)
        edge_t = edge_row.unsqueeze(1)                           # (B, 1, N)
        h      = self.gnn(ue_x, uav_x, edge_t)                  # (B, 1, gnn_out)
        enr    = torch.cat([ue_x, h], dim=-1)                   # (B, 1, enriched_dim)
        return enr.squeeze(1)                                    # (B, enriched_dim)

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        (obs, uav_x, edge_row,
         actions, rewards,
         next_obs, next_uav_x, next_edge_row,
         dones) = [t.to(self.device) for t in self.buffer.sample(self.batch_size)]

        enriched      = self._enrich_batch(obs,      uav_x,      edge_row)
        with torch.no_grad():
            enriched_next = self._enrich_batch(next_obs, next_uav_x, next_edge_row)

        q_values = self.policy_net(enriched)
        q_pred   = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(enriched_next).argmax(dim=1)
            next_q       = self.target_net(enriched_next).gather(
                               1, next_actions.unsqueeze(1)).squeeze(1)
            q_target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_pred, q_target)

        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.gnn.parameters()) + list(self.policy_net.parameters()),
            10.0,
        )
        self.optimiser.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())


# ─── Multi-agent wrapper ───────────────────────────────────────────────────

class GNNILNoShareAgent:
    """
    Wraps M independent SingleGNNAgents.

    Drop-in replacement for GNNILAgent — same public interface:
        select_actions(obs_list, graph_data, greedy)
        store(obs_list, graph_data, actions, rewards,
              next_obs_list, next_graph_data, done, task_mask)
        train_step()
        decay_epsilon()
        save(path) / load(path)

    The training loop (trainGnnNoShare.py) can be identical to
    train_gnn_il.py with only the agent class swapped.
    """

    def __init__(
        self,
        obs_dim:       int,
        n_uavs:        int,
        n_actions:     int,
        n_agents:      int,
        gnn_hidden:    int   = 64,
        gnn_out:       int   = 32,
        dqn_hidden:    int   = 128,
        lr:            float = 1e-3,
        gamma:         float = 0.9,
        buffer_cap:    int   = 5_000,
        batch_size:    int   = 32,
        target_update: int   = 20,
        eps_decay:     float = 0.995,
        eps_start:     float = 1.0,
        eps_end:       float = 0.05,
        device:        str   = "cpu",
    ):
        self.n_agents  = n_agents
        self.n_actions = n_actions
        self.n_uavs    = n_uavs
        self.device    = torch.device(device)

        self.eps       = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay

        self.agents = [
            SingleGNNAgent(
                obs_dim       = obs_dim,
                n_uavs        = n_uavs,
                n_actions     = n_actions,
                gnn_hidden    = gnn_hidden,
                gnn_out       = gnn_out,
                dqn_hidden    = dqn_hidden,
                lr            = lr,
                gamma         = gamma,
                buffer_cap    = buffer_cap,
                batch_size    = batch_size,
                target_update = target_update,
                device        = self.device,
            )
            for _ in range(n_agents)
        ]

    # ── Graph helper ───────────────────────────────────────────────────────

    @staticmethod
    def _edge_matrix(graph_data: dict) -> np.ndarray:
        M   = graph_data["n_ues"]
        N   = graph_data["n_uavs"]
        src, dst, w = graph_data["ue_uav"]
        mat = np.zeros((M, N), dtype=np.float32)
        for s, d, wt in zip(src, dst, w):
            mat[s, d] = wt
        return mat                                               # (M, N)

    # ── Action selection ───────────────────────────────────────────────────

    def select_actions(
        self,
        obs_list:   List[np.ndarray],
        graph_data: dict,
        greedy:     bool = False,
    ) -> List[int]:
        uav_x    = graph_data["uav_x"].astype(np.float32)       # (N, 3)
        edge_mat = self._edge_matrix(graph_data)                 # (M, N)

        actions = []
        for m, agent in enumerate(self.agents):
            enriched = agent.enrich(obs_list[m], uav_x, edge_mat[m])
            with torch.no_grad():
                q = agent.policy_net(
                    torch.tensor(enriched, dtype=torch.float32)
                    .unsqueeze(0).to(self.device)
                )
                greedy_a = int(q.argmax(dim=1).item())

            if greedy or random.random() >= self.eps:
                actions.append(greedy_a)
            else:
                actions.append(random.randint(0, self.n_actions - 1))

        return actions

    # ── Store transitions ──────────────────────────────────────────────────

    def store(
        self,
        obs_list:        List[np.ndarray],
        graph_data:      dict,
        actions:         List[int],
        rewards:         List[float],
        next_obs_list:   List[np.ndarray],
        next_graph_data: dict,
        done:            bool,
        task_mask:       List[bool],
    ):
        uav_x        = graph_data["uav_x"].astype(np.float32)
        edge_mat     = self._edge_matrix(graph_data)
        next_uav_x   = next_graph_data["uav_x"].astype(np.float32)
        next_edge_mat= self._edge_matrix(next_graph_data)

        for m, (agent, had_task) in enumerate(zip(self.agents, task_mask)):
            if not had_task:
                continue
            agent.buffer.push(
                obs_list[m],
                uav_x,
                edge_mat[m],             # this UE's edge row only
                actions[m],
                rewards[m],
                next_obs_list[m],
                next_uav_x,
                next_edge_mat[m],
                float(done),
            )

    # ── Training ───────────────────────────────────────────────────────────

    def train_step(self) -> Optional[float]:
        """Train all M agents independently. Returns mean loss."""
        losses = []
        for agent in self.agents:
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
        return float(np.mean(losses)) if losses else None

    # ── ε schedule ─────────────────────────────────────────────────────────

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            f"agent_{m}": {
                "gnn":        a.gnn.state_dict(),
                "policy_net": a.policy_net.state_dict(),
            }
            for m, a in enumerate(self.agents)
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        for m, agent in enumerate(self.agents):
            agent.gnn.load_state_dict(ckpt[f"agent_{m}"]["gnn"])
            agent.policy_net.load_state_dict(ckpt[f"agent_{m}"]["policy_net"])
            agent.target_net.load_state_dict(ckpt[f"agent_{m}"]["policy_net"])