"""
GNN-IL Agent — shared bipartite GNN encoder + shared Dueling DQN.

Architecture
------------
Graph:  UE nodes ↔ UAV nodes, fully bipartite, edge weight = normalised
        channel gain h_{m,n}.  No UE-UE edges (physically unmotivated).

GNN (2-layer, pure torch — no PyG dependency):

  Layer 1  UE → UAV
    For each UAV n, aggregate UE node features weighted by softmax-
    normalised channel gains → UAV hidden state h_uav  (B, N, gnn_hidden)

  Layer 2  UAV → UE
    For each UE m, aggregate h_uav weighted by softmax-normalised channel
    gains → enriched UE embedding h_ue  (B, M, gnn_out)

DQN:
    input = cat(raw_obs_m, h_ue_m)   dim = obs_dim + gnn_out
    → shared Dueling DQN head (same weights for every UE)

Replay buffer:
    Stores FULL SYSTEM transitions so that the GNN can be re-run inside
    train_step(), keeping the computation graph intact for backward().

    Entry: (ue_obs, uav_feats, edge_w,
            actions, rewards,
            next_ue_obs, next_uav_feats, next_edge_w,
            done)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from typing import List, Optional, Tuple

from agent import DuelingDQN       # reuse the same dueling head


# ─── Bipartite GNN Encoder ─────────────────────────────────────────────────

class BipartiteGNNEncoder(nn.Module):
    """
    Two-layer message passing on a UE ↔ UAV bipartite graph.

    Parameters
    ----------
    ue_dim    : dimension of UE node features  (= obs_dim)
    uav_dim   : dimension of UAV node features (= 3)
    hidden    : hidden size after Layer 1
    out_dim   : output embedding size per UE after Layer 2
    """

    def __init__(
        self,
        ue_dim:  int,
        uav_dim: int,
        hidden:  int = 64,
        out_dim: int = 32,
    ):
        super().__init__()

        # Layer 1: [uav_own_feat || weighted_mean(UE_feats)] → hidden
        self.layer1 = nn.Sequential(
            nn.Linear(uav_dim + ue_dim, hidden),
            nn.ReLU(),
        )

        # Layer 2: [ue_own_feat || weighted_mean(h_uav)] → out_dim
        self.layer2 = nn.Sequential(
            nn.Linear(ue_dim + hidden, out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        ue_x:   torch.Tensor,   # (B, M, ue_dim)
        uav_x:  torch.Tensor,   # (B, N, uav_dim)
        edge_w: torch.Tensor,   # (B, M, N)  raw channel gains
    ) -> torch.Tensor:          # (B, M, out_dim)

        # ── Layer 1: UE → UAV ─────────────────────────────────────────
        # Normalise weights column-wise (over UEs) so each UAV's
        # attention coefficients sum to 1.
        attn1     = F.softmax(edge_w, dim=1)                    # (B, M, N)
        # Weighted mean of UE features arriving at each UAV
        h_agg_uav = torch.bmm(attn1.transpose(1, 2), ue_x)    # (B, N, ue_dim)
        # Concat UAV's own features and project
        h_uav     = self.layer1(
            torch.cat([uav_x, h_agg_uav], dim=-1)              # (B, N, uav_dim+ue_dim)
        )                                                        # (B, N, hidden)

        # ── Layer 2: UAV → UE ─────────────────────────────────────────
        # Normalise weights row-wise (over UAVs) so each UE's
        # attention coefficients sum to 1.
        attn2     = F.softmax(edge_w, dim=2)                    # (B, M, N)
        # Weighted mean of UAV hidden states arriving at each UE
        h_agg_ue  = torch.bmm(attn2, h_uav)                   # (B, M, hidden)
        # Concat UE's own features and project
        h_ue      = self.layer2(
            torch.cat([ue_x, h_agg_ue], dim=-1)               # (B, M, ue_dim+hidden)
        )                                                        # (B, M, out_dim)

        return h_ue


# ─── Global Replay Buffer ──────────────────────────────────────────────────

class GlobalReplayBuffer:
    """
    Stores full-system transitions.  Each entry contains observations,
    graph topology, actions, rewards, and next-step equivalents for
    ALL agents simultaneously.

    This is required so that train_step() can re-run the GNN on the
    stored raw observations and keep the computation graph alive for
    loss.backward() to reach GNN weights.
    """

    def __init__(self, capacity: int = 5_000):
        self.buf = deque(maxlen=capacity)

    def push(
        self,
        ue_obs:        np.ndarray,   # (M, obs_dim)
        uav_feats:     np.ndarray,   # (N, 3)
        edge_w:        np.ndarray,   # (M, N)
        actions:       np.ndarray,   # (M,)  int
        rewards:       np.ndarray,   # (M,)  float
        next_ue_obs:   np.ndarray,   # (M, obs_dim)
        next_uav_feats:np.ndarray,   # (N, 3)
        next_edge_w:   np.ndarray,   # (M, N)
        done:          float,
        task_mask:     np.ndarray,   # (M,)  bool — True if agent had a task
    ):
        self.buf.append((
            ue_obs.astype(np.float32),
            uav_feats.astype(np.float32),
            edge_w.astype(np.float32),
            actions.astype(np.int64),
            rewards.astype(np.float32),
            next_ue_obs.astype(np.float32),
            next_uav_feats.astype(np.float32),
            next_edge_w.astype(np.float32),
            np.float32(done),
            task_mask.astype(bool),
        ))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buf, batch_size)
        (ue_obs, uav_feats, edge_w, actions, rewards,
         next_ue_obs, next_uav_feats, next_edge_w, dones,
         task_masks) = zip(*batch)

        return (
            torch.tensor(np.stack(ue_obs),         dtype=torch.float32),   # (B,M,obs)
            torch.tensor(np.stack(uav_feats),       dtype=torch.float32),   # (B,N,3)
            torch.tensor(np.stack(edge_w),          dtype=torch.float32),   # (B,M,N)
            torch.tensor(np.stack(actions),         dtype=torch.long),      # (B,M)
            torch.tensor(np.stack(rewards),         dtype=torch.float32),   # (B,M)
            torch.tensor(np.stack(next_ue_obs),     dtype=torch.float32),   # (B,M,obs)
            torch.tensor(np.stack(next_uav_feats),  dtype=torch.float32),   # (B,N,3)
            torch.tensor(np.stack(next_edge_w),     dtype=torch.float32),   # (B,M,N)
            torch.tensor(np.array(dones),           dtype=torch.float32),   # (B,)
            torch.tensor(np.stack(task_masks),      dtype=torch.bool),      # (B,M)
        )

    def __len__(self) -> int:
        return len(self.buf)


# ─── GNN-IL Agent ──────────────────────────────────────────────────────────

class GNNILAgent:
    """
    Single shared GNN + DQN serving all M UEs.

    Shared weights mean:
      - All UEs use the same aggregation function (homogeneous agents).
      - Parameter count stays small.
      - One optimizer step updates the policy for the whole fleet.

    Gradient flow (the key fix):
      action selection  →  GNN runs under torch.no_grad()   (fast, no graph)
      train_step()      →  GNN runs WITH grad tracking, then
                           loss.backward() reaches GNN weights correctly.
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
        batch_size:   int   = 32,
        buffer_cap:   int   = 5_000,
        target_update:int   = 20,
        eps_start:    float = 1.0,
        eps_end:      float = 0.05,
        eps_decay:    float = 0.995,
        device:       str   = "cpu",
    ):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update
        self.device        = torch.device(device)

        self.eps       = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay

        self.enriched_dim = obs_dim + gnn_out

        # ── Networks ──────────────────────────────────────────────────
        self.gnn = BipartiteGNNEncoder(
            ue_dim  = obs_dim,
            uav_dim = 3,
            hidden  = gnn_hidden,
            out_dim = gnn_out,
        ).to(self.device)

        # Shared dueling DQN — input is enriched observation
        self.policy_net = DuelingDQN(
            self.enriched_dim, n_actions, dqn_hidden
        ).to(self.device)

        self.target_net = DuelingDQN(
            self.enriched_dim, n_actions, dqn_hidden
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Single optimiser covers BOTH gnn and policy_net parameters
        self.optimiser = torch.optim.RMSprop(
            list(self.gnn.parameters()) + list(self.policy_net.parameters()),
            lr=lr,
        )

        self.buffer = GlobalReplayBuffer(buffer_cap)
        self.steps  = 0

    # ── Graph helper ───────────────────────────────────────────────────────

    @staticmethod
    def extract_edge_matrix(graph_data: dict) -> np.ndarray:
        """
        Convert the (src, dst, weight) edge lists from env.get_graph_data()
        into a dense (M, N) numpy matrix.
        Only UE→UAV edges are used; UE-UE edges are ignored entirely.
        """
        M = graph_data["n_ues"]
        N = graph_data["n_uavs"]
        src, dst, w = graph_data["ue_uav"]
        mat = np.zeros((M, N), dtype=np.float32)
        for s, d, wt in zip(src, dst, w):
            mat[s, d] = wt
        return mat                                               # (M, N)

    # ── Action selection ──────────────────────────────────────────────────

    def select_actions(
        self,
        obs_list:   List[np.ndarray],   # length M, each (obs_dim,)
        graph_data: dict,
        greedy:     bool = False,
    ) -> List[int]:
        """
        Forward pass under no_grad → ε-greedy action per UE.
        GNN runs on the full system observation simultaneously.
        """
        with torch.no_grad():
            ue_x   = torch.tensor(
                np.stack(obs_list), dtype=torch.float32
            ).unsqueeze(0).to(self.device)                      # (1, M, obs_dim)

            uav_x  = torch.tensor(
                graph_data["uav_x"], dtype=torch.float32
            ).unsqueeze(0).to(self.device)                      # (1, N, 3)

            edge_w = torch.tensor(
                self.extract_edge_matrix(graph_data), dtype=torch.float32
            ).unsqueeze(0).to(self.device)                      # (1, M, N)

            h_ue     = self.gnn(ue_x, uav_x, edge_w)           # (1, M, gnn_out)
            enriched = torch.cat([ue_x, h_ue], dim=-1)         # (1, M, enriched_dim)
            enriched = enriched.squeeze(0)                      # (M, enriched_dim)

            q_values      = self.policy_net(enriched)           # (M, n_actions)
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()

        if greedy:
            return [int(a) for a in greedy_actions]

        return [
            random.randint(0, self.n_actions - 1)
            if random.random() < self.eps
            else int(greedy_actions[m])
            for m in range(len(obs_list))
        ]

    # ── Store transition ──────────────────────────────────────────────────

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
        self.buffer.push(
            ue_obs         = np.stack(obs_list),
            uav_feats      = graph_data["uav_x"],
            edge_w         = self.extract_edge_matrix(graph_data),
            actions        = np.array(actions),
            rewards        = np.array(rewards),
            next_ue_obs    = np.stack(next_obs_list),
            next_uav_feats = next_graph_data["uav_x"],
            next_edge_w    = self.extract_edge_matrix(next_graph_data),
            done           = float(done),
            task_mask      = np.array(task_mask),
        )

    # ── Training step ─────────────────────────────────────────────────────

    def train_step(self) -> Optional[float]:
        """
        Sample one global mini-batch and update GNN + policy_net jointly.

        The GNN is called INSIDE this function so that the full computation
        graph — GNN encoder → DQN head → loss — exists when backward() runs.
        Both GNN and DQN weights receive gradients.
        """
        if len(self.buffer) < self.batch_size:
            return None

        (ue_obs, uav_feats, edge_w,
         actions, rewards,
         next_ue_obs, next_uav_feats, next_edge_w,
         dones, task_masks) = [t.to(self.device) for t in self.buffer.sample(self.batch_size)]

        # Shapes:
        #   ue_obs, next_ue_obs  : (B, M, obs_dim)
        #   uav_feats            : (B, N, 3)
        #   edge_w               : (B, M, N)
        #   actions, rewards     : (B, M)
        #   dones                : (B,)
        #   task_masks           : (B, M)  bool

        B, M, _ = ue_obs.shape

        # ── Current Q values ──────────────────────────────────────────
        h_ue     = self.gnn(ue_obs, uav_feats, edge_w)         # (B, M, gnn_out)
        enriched = torch.cat([ue_obs, h_ue], dim=-1)           # (B, M, enriched_dim)
        enriched_flat  = enriched.view(B * M, -1)              # (B*M, enriched_dim)
        q_all          = self.policy_net(enriched_flat)         # (B*M, n_actions)
        actions_flat   = actions.view(B * M)                   # (B*M,)
        q_pred = q_all.gather(
            1, actions_flat.unsqueeze(1)
        ).squeeze(1)                                            # (B*M,)

        # ── Target Q values (Double DQN, no grad) ────────────────────
        with torch.no_grad():
            h_ue_next      = self.gnn(next_ue_obs, next_uav_feats, next_edge_w)
            enriched_next  = torch.cat([next_ue_obs, h_ue_next], dim=-1)
            enriched_nflat = enriched_next.view(B * M, -1)    # (B*M, enriched_dim)

            next_actions   = self.policy_net(enriched_nflat).argmax(dim=1)
            next_q         = self.target_net(enriched_nflat).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)                                        # (B*M,)

            dones_exp    = dones.unsqueeze(1).expand(B, M).reshape(B * M)
            rewards_flat = rewards.view(B * M)
            q_target     = rewards_flat + self.gamma * next_q * (1.0 - dones_exp)

        # ── Loss: only over agents that had a task this slot ──────────
        # task_masks flat: (B*M,) bool
        mask_flat = task_masks.view(B * M)                     # (B*M,) bool
        if mask_flat.sum() == 0:
            return None                                         # nothing to learn from

        loss = F.smooth_l1_loss(q_pred[mask_flat], q_target[mask_flat])

        self.optimiser.zero_grad()
        loss.backward()                # gradients flow through DQN AND GNN
        nn.utils.clip_grad_norm_(
            list(self.gnn.parameters()) + list(self.policy_net.parameters()),
            max_norm=10.0,
        )
        self.optimiser.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    # ── ε schedule ────────────────────────────────────────────────────────

    def decay_epsilon(self):
        """Call once per episode."""
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    # ── Persistence ───────────────────────────────────────────────────────

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