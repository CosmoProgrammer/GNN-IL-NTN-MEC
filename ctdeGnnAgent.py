"""
GNN-CTDE Agent — shared BipartiteGNNEncoder + shared Dueling DQN,
trained centrally over one shared replay buffer.

Architecture
------------
At interaction time (select_actions):
    GNN(graph_data) → enriched embedding per UE   (obs_dim + gnn_out,)
    Actions selected from shared policy network.

At store time:
    Raw obs, UAV node features, edge weights, and next equivalents are
    stored in a graph-aware replay buffer (GraphReplayBuffer).
    We do NOT pre-enrich — enrichment happens during train_step() so
    that the GNN is inside the computation graph and its weights update.

At training time (train_step):
    Buffer yields raw obs + graph tensors.
    GNN re-runs on the batch → enriched embeddings.
    DQN computes TD loss on enriched embeddings.
    loss.backward() flows through DQN AND GNN.
    Optimiser updates both together.

Key difference from buggy version
-----------------------------------
    Old: optimiser only covered policy_net.parameters()
         GNN was pre-run at store() time → frozen random projector
    New: optimiser covers policy_net + gnn parameters
         GNN re-runs inside train_step() → jointly optimised
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
from typing import List, Optional, Tuple

from agent    import DuelingDQN          # reuse unchanged
from gnnAgent import BipartiteGNNEncoder # reuse unchanged


# ─── Graph-Aware Replay Buffer ─────────────────────────────────────────────

class GraphReplayBuffer:
    """
    Stores raw (obs, graph, action, reward, next_obs, next_graph, done)
    tuples so the GNN can be re-run during training.

    Graph data stored per transition:
        uav_x  : (N, 3)   UAV node features
        edge_w : (M, N)   normalised channel-gain edge weights

    We store uav_x and edge_w rather than the full graph_data dict
    to keep memory layout simple and avoid pickle overhead.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer   = deque(maxlen=capacity)

    def push(
        self,
        obs:          np.ndarray,   # (obs_dim,)
        uav_x:        np.ndarray,   # (N, 3)
        edge_w:       np.ndarray,   # (M, N)  — full graph, not per-UE
        action:       int,
        reward:       float,
        next_obs:     np.ndarray,   # (obs_dim,)
        next_uav_x:   np.ndarray,   # (N, 3)
        next_edge_w:  np.ndarray,   # (M, N)
        ue_idx:       int,          # which UE this transition belongs to
        done:         float,
    ):
        self.buffer.append((
            obs.astype(np.float32),
            uav_x.astype(np.float32),
            edge_w.astype(np.float32),
            int(action),
            float(reward),
            next_obs.astype(np.float32),
            next_uav_x.astype(np.float32),
            next_edge_w.astype(np.float32),
            int(ue_idx),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        (obs, uav_x, edge_w,
         actions, rewards,
         next_obs, next_uav_x, next_edge_w,
         ue_idxs, dones) = zip(*batch)

        return (
            torch.tensor(np.stack(obs),          dtype=torch.float32),
            torch.tensor(np.stack(uav_x),        dtype=torch.float32),
            torch.tensor(np.stack(edge_w),        dtype=torch.float32),
            torch.tensor(actions,                 dtype=torch.long),
            torch.tensor(rewards,                 dtype=torch.float32),
            torch.tensor(np.stack(next_obs),      dtype=torch.float32),
            torch.tensor(np.stack(next_uav_x),    dtype=torch.float32),
            torch.tensor(np.stack(next_edge_w),   dtype=torch.float32),
            torch.tensor(ue_idxs,                 dtype=torch.long),
            torch.tensor(dones,                   dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─── GNN-CTDE Agent ────────────────────────────────────────────────────────

class GNNCTDEAgent:
    """
    Central controller: one GNN encoder + one shared Dueling DQN +
    one shared GraphReplayBuffer (capacity = n_agents × per_agent_cap).

    GNN is jointly optimised with the DQN during centralized training.
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
        per_agent_cap: int   = 5_000,
        batch_size:    int   = 32,
        target_update: int   = 20,
        eps_start:     float = 1.0,
        eps_end:       float = 0.05,
        eps_decay:     float = 0.995,
        device:        str   = "cpu",
    ):
        self.n_actions     = n_actions
        self.n_agents      = n_agents
        self.n_uavs        = n_uavs
        self.obs_dim       = obs_dim
        self.gnn_out       = gnn_out
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update
        self.device        = torch.device(device)

        self.enriched_dim  = obs_dim + gnn_out

        # ε-greedy
        self.eps       = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay

        # GNN encoder
        self.gnn = BipartiteGNNEncoder(
            ue_dim  = obs_dim,
            uav_dim = 3,
            hidden  = gnn_hidden,
            out_dim = gnn_out,
        ).to(self.device)

        # Shared Dueling DQN
        self.policy_net = DuelingDQN(
            self.enriched_dim, n_actions, dqn_hidden
        ).to(self.device)

        self.target_net = DuelingDQN(
            self.enriched_dim, n_actions, dqn_hidden
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # ── Optimiser covers BOTH GNN and DQN ──────────────────────────
        # This is the critical fix — GNN weights update during training.
        self.optimiser = torch.optim.RMSprop(
            list(self.gnn.parameters()) + list(self.policy_net.parameters()),
            lr=lr,
        )

        # Graph-aware shared buffer
        self.buffer = GraphReplayBuffer(capacity=n_agents * per_agent_cap)

        self.steps = 0

    # ── Graph helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_edge_matrix(graph_data: dict) -> np.ndarray:
        """Convert (src, dst, weight) edge lists → dense (M, N) matrix."""
        M   = graph_data["n_ues"]
        N   = graph_data["n_uavs"]
        src, dst, w = graph_data["ue_uav"]
        mat = np.zeros((M, N), dtype=np.float32)
        for s, d, wt in zip(src, dst, w):
            mat[s, d] = wt
        return mat                                               # (M, N)

    # ── GNN enrichment (interaction time, no_grad) ─────────────────────────

    @torch.no_grad()
    def _enrich(
        self,
        obs_list:   List[np.ndarray],
        graph_data: dict,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run GNN for action selection.
        Returns enriched embeddings AND the raw graph arrays needed
        for storing transitions.

        Returns
        -------
        enriched : (M, enriched_dim)
        uav_x    : (N, 3)
        edge_w   : (M, N)
        """
        uav_x_np  = graph_data["uav_x"].astype(np.float32)      # (N, 3)
        edge_w_np = self._extract_edge_matrix(graph_data)        # (M, N)

        ue_x   = torch.tensor(
            np.stack(obs_list), dtype=torch.float32
        ).unsqueeze(0).to(self.device)                           # (1, M, obs_dim)

        uav_x  = torch.tensor(uav_x_np).unsqueeze(0).to(self.device)   # (1, N, 3)
        edge_w = torch.tensor(edge_w_np).unsqueeze(0).to(self.device)   # (1, M, N)

        h_ue     = self.gnn(ue_x, uav_x, edge_w)                # (1, M, gnn_out)
        enriched = torch.cat([ue_x, h_ue], dim=-1)              # (1, M, enriched_dim)

        return (
            enriched.squeeze(0).cpu().numpy(),   # (M, enriched_dim)
            uav_x_np,                            # (N, 3)
            edge_w_np,                           # (M, N)
        )

    # ── GNN enrichment (training time, with grad) ──────────────────────────

    def _enrich_batch(
        self,
        obs:    torch.Tensor,    # (B, obs_dim)
        uav_x:  torch.Tensor,    # (B, N, 3)
        edge_w: torch.Tensor,    # (B, M, N)  — full graph per transition
        ue_idx: torch.Tensor,    # (B,)        — which UE row to extract
    ) -> torch.Tensor:           # (B, enriched_dim)
        """
        Re-run GNN inside the computation graph so gradients flow back
        into the encoder weights.

        The GNN expects (1, M, obs_dim) but our buffer stores one UE's
        raw obs per transition, not the full M×obs_dim matrix.

        Strategy: we only have the single UE's obs in the buffer.
        We reconstruct a pseudo-batch by:
          1. Treating each transition independently: (1, 1, obs_dim)
          2. Using a (1, N, 3) UAV feature and a (1, 1, N) edge row.
        This is equivalent to running one-node GNN message passing:
        each UE gathers from all N UAV neighbours → correct semantics.
        """
        B = obs.shape[0]

        # obs        : (B, obs_dim) → (B, 1, obs_dim)
        ue_x   = obs.unsqueeze(1)

        # uav_x      : (B, N, 3)   — already batched
        # edge_w     : (B, M, N)
        # Extract just this UE's row from the full edge matrix
        # ue_idx     : (B,) — indices into dim 1 of edge_w
        row_idx = ue_idx.view(B, 1, 1).expand(B, 1, self.n_uavs)
        ue_edge = edge_w.gather(1, row_idx)                      # (B, 1, N)

        h_ue     = self.gnn(ue_x, uav_x, ue_edge)               # (B, 1, gnn_out)
        enriched = torch.cat([ue_x, h_ue], dim=-1)              # (B, 1, enriched_dim)
        return enriched.squeeze(1)                               # (B, enriched_dim)

    # ── Action selection ───────────────────────────────────────────────────

    def select_actions(
        self,
        obs_list:   List[np.ndarray],
        graph_data: dict,
        greedy:     bool = False,
    ) -> List[int]:
        enriched, _, _ = self._enrich(obs_list, graph_data)      # (M, enriched_dim)

        with torch.no_grad():
            q_values = self.policy_net(
                torch.tensor(enriched, dtype=torch.float32).to(self.device)
            )                                                    # (M, n_actions)
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()

        if greedy:
            return [int(a) for a in greedy_actions]

        return [
            random.randint(0, self.n_actions - 1)
            if random.random() < self.eps
            else int(greedy_actions[m])
            for m in range(len(obs_list))
        ]

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
        """
        Store raw obs + graph arrays. One entry per UE that had a task.
        We store the full uav_x and edge_w so train_step() can re-run GNN.
        """
        # Extract graph arrays once for current and next timestep
        uav_x_np       = graph_data["uav_x"].astype(np.float32)
        edge_w_np      = self._extract_edge_matrix(graph_data)
        next_uav_x_np  = next_graph_data["uav_x"].astype(np.float32)
        next_edge_w_np = self._extract_edge_matrix(next_graph_data)

        for m, had_task in enumerate(task_mask):
            if not had_task:
                continue
            self.buffer.push(
                obs_list[m],
                uav_x_np,
                edge_w_np,
                actions[m],
                rewards[m],
                next_obs_list[m],
                next_uav_x_np,
                next_edge_w_np,
                m,                    # ue_idx — needed to extract edge row
                float(done),
            )

    # ── Centralized training step ──────────────────────────────────────────

    def train_step(self) -> Optional[float]:
        """
        One gradient update. GNN is re-run here — inside the computation
        graph — so both GNN and DQN weights update together.
        """
        if len(self.buffer) < self.batch_size:
            return None

        (obs, uav_x, edge_w,
         actions, rewards,
         next_obs, next_uav_x, next_edge_w,
         ue_idxs, dones) = [
            t.to(self.device) for t in self.buffer.sample(self.batch_size)
        ]

        # ── Enrich current obs (GNN in computation graph) ──────────────
        enriched      = self._enrich_batch(obs,      uav_x,      edge_w,      ue_idxs)
        with torch.no_grad():
            enriched_next = self._enrich_batch(next_obs, next_uav_x, next_edge_w, ue_idxs)

        # ── Double DQN TD target ────────────────────────────────────────
        q_values = self.policy_net(enriched)                     # (B, n_actions)
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

    # ── ε schedule ─────────────────────────────────────────────────────────

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "gnn":        self.gnn.state_dict(),
            "policy_net": self.policy_net.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.gnn.load_state_dict(ckpt["gnn"])
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["policy_net"])