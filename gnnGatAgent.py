"""
GNN-IL Agent — BipartiteGATEncoder + shared Dueling DQN.

Upgrade over gnnAgent.py:
  The fixed softmax(channel_gain) aggregation weights are replaced with
  a learnable Graph Attention (GAT) mechanism in both message-passing layers.

  Layer 1  UE → UAV
    score(m, n) = LeakyReLU( a1ᵀ · [W_q1·ue_x_m || W_k1·uav_x_n] )
    Attention weight α_{m,n} = softmax over UEs arriving at UAV n,
    gated by the raw channel gain e_{m,n} so that zero-gain links
    are masked out.

  Layer 2  UAV → UE
    score(n, m) = LeakyReLU( a2ᵀ · [W_q2·h_uav_n || W_k2·ue_x_m] )
    Attention weight α_{n,m} = softmax over UAVs arriving at UE m,
    again gated by the raw channel gain.

Why channel-gain gating?
  In a sparse UAV network not every UE can reach every UAV.
  Multiplying the attention logit by the normalised channel gain
  (≈ 0 for unreachable links) zeroes out impossible offload targets
  before the softmax, without needing an explicit adjacency mask.

Everything outside BipartiteGATEncoder is identical to gnnAgent.py.
No changes to GlobalReplayBuffer, GNNILAgent, or the training loop.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from typing import List, Optional, Tuple

from agent import DuelingDQN


# ─── Bipartite GAT Encoder ─────────────────────────────────────────────────

class BipartiteGATEncoder(nn.Module):
    """
    Two-layer message passing on a UE ↔ UAV bipartite graph
    with learnable GAT-style attention in both directions.

    Parameters
    ----------
    ue_dim   : UE node feature dimension  (= obs_dim)
    uav_dim  : UAV node feature dimension (= 3)
    hidden   : hidden size after Layer 1  (UAV embeddings)
    out_dim  : output embedding size per UE after Layer 2
    attn_dim : projected dimension for computing attention scores
    """

    def __init__(
        self,
        ue_dim:   int,
        uav_dim:  int,
        hidden:   int = 64,
        out_dim:  int = 32,
        attn_dim: int = 32,
    ):
        super().__init__()

        # ── Layer 1 (UE → UAV) attention parameters ───────────────────
        # Project UE features to query space
        self.W_q1 = nn.Linear(ue_dim,  attn_dim, bias=False)
        # Project UAV features to key space
        self.W_k1 = nn.Linear(uav_dim, attn_dim, bias=False)
        # Scalar attention scorer over concatenated query+key
        self.a1   = nn.Linear(2 * attn_dim, 1, bias=False)

        # Layer 1 projection: [uav_own || weighted_mean(UE_feats)] → hidden
        self.layer1 = nn.Sequential(
            nn.Linear(uav_dim + ue_dim, hidden),
            nn.ReLU(),
        )

        # ── Layer 2 (UAV → UE) attention parameters ───────────────────
        # Project UAV hidden states to query space
        self.W_q2 = nn.Linear(hidden,  attn_dim, bias=False)
        # Project UE features to key space
        self.W_k2 = nn.Linear(ue_dim,  attn_dim, bias=False)
        # Scalar attention scorer
        self.a2   = nn.Linear(2 * attn_dim, 1, bias=False)

        # Layer 2 projection: [ue_own || weighted_mean(h_uav)] → out_dim
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

        B, M, _ = ue_x.shape
        N       = uav_x.shape[1]

        # ── Layer 1: UE → UAV (learnable attention) ────────────────────
        # Project features to attention space
        Q1 = self.W_q1(ue_x)    # (B, M, attn_dim)
        K1 = self.W_k1(uav_x)   # (B, N, attn_dim)

        # Broadcast to (B, M, N, attn_dim) for pairwise concatenation
        Q1_exp = Q1.unsqueeze(2).expand(B, M, N, -1)  # (B, M, N, attn_dim)
        K1_exp = K1.unsqueeze(1).expand(B, M, N, -1)  # (B, M, N, attn_dim)

        # Attention logit: scalar per (UE m, UAV n) pair
        logit1 = self.a1(
            torch.cat([Q1_exp, K1_exp], dim=-1)        # (B, M, N, 2*attn_dim)
        ).squeeze(-1)                                   # (B, M, N)
        logit1 = F.leaky_relu(logit1, negative_slope=0.2)

        # Gate by raw channel gain: links with gain ≈ 0 get logit → -∞
        # so they vanish from the softmax without an explicit mask.
        # Normalise gain to [0,1] per batch to keep logit scale stable.
        gain_norm = edge_w / (edge_w.amax(dim=(1, 2), keepdim=True) + 1e-8)
        logit1 = logit1 + torch.log(gain_norm + 1e-8)                 # (B, M, N)

        # Attention weights: softmax over UEs for each UAV (dim=1)
        attn1     = F.softmax(logit1, dim=1)           # (B, M, N)

        # Aggregate UE features at each UAV
        h_agg_uav = torch.bmm(
            attn1.transpose(1, 2), ue_x               # (B, N, ue_dim)
        )
        h_uav     = self.layer1(
            torch.cat([uav_x, h_agg_uav], dim=-1)     # (B, N, uav_dim+ue_dim)
        )                                               # (B, N, hidden)

        # ── Layer 2: UAV → UE (learnable attention) ────────────────────
        Q2 = self.W_q2(h_uav)   # (B, N, attn_dim)
        K2 = self.W_k2(ue_x)    # (B, M, attn_dim)

        # Broadcast to (B, M, N, attn_dim)
        Q2_exp = Q2.unsqueeze(1).expand(B, M, N, -1)  # (B, M, N, attn_dim)
        K2_exp = K2.unsqueeze(2).expand(B, M, N, -1)  # (B, M, N, attn_dim)

        logit2 = self.a2(
            torch.cat([Q2_exp, K2_exp], dim=-1)        # (B, M, N, 2*attn_dim)
        ).squeeze(-1)                                   # (B, M, N)
        logit2 = F.leaky_relu(logit2, negative_slope=0.2)
        logit2 = logit2 + torch.log(gain_norm + 1e-8)                 # (B, M, N)

        # Attention weights: softmax over UAVs for each UE (dim=2)
        attn2     = F.softmax(logit2, dim=2)           # (B, M, N)

        # Aggregate UAV hidden states at each UE
        h_agg_ue = torch.bmm(attn2, h_uav)            # (B, M, hidden)
        h_ue     = self.layer2(
            torch.cat([ue_x, h_agg_ue], dim=-1)       # (B, M, ue_dim+hidden)
        )                                               # (B, M, out_dim)

        return h_ue


# ─── Global Replay Buffer (unchanged) ──────────────────────────────────────

class GlobalReplayBuffer:
    """Identical to gnnAgent.py — stores full-system transitions."""

    def __init__(self, capacity: int = 5_000):
        self.buf = deque(maxlen=capacity)

    def push(
        self,
        ue_obs:         np.ndarray,
        uav_feats:      np.ndarray,
        edge_w:         np.ndarray,
        actions:        np.ndarray,
        rewards:        np.ndarray,
        next_ue_obs:    np.ndarray,
        next_uav_feats: np.ndarray,
        next_edge_w:    np.ndarray,
        done:           float,
        task_mask:      np.ndarray,
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
         next_ue_obs, next_uav_feats, next_edge_w,
         dones, task_masks) = zip(*batch)
        return (
            torch.tensor(np.stack(ue_obs),         dtype=torch.float32),
            torch.tensor(np.stack(uav_feats),       dtype=torch.float32),
            torch.tensor(np.stack(edge_w),          dtype=torch.float32),
            torch.tensor(np.stack(actions),         dtype=torch.long),
            torch.tensor(np.stack(rewards),         dtype=torch.float32),
            torch.tensor(np.stack(next_ue_obs),     dtype=torch.float32),
            torch.tensor(np.stack(next_uav_feats),  dtype=torch.float32),
            torch.tensor(np.stack(next_edge_w),     dtype=torch.float32),
            torch.tensor(np.array(dones),           dtype=torch.float32),
            torch.tensor(np.stack(task_masks),      dtype=torch.bool),
        )

    def __len__(self) -> int:
        return len(self.buf)


# ─── GAT-IL Agent ──────────────────────────────────────────────────────────

class GATILAgent:
    """
    Drop-in upgrade of GNNILAgent using BipartiteGATEncoder.

    The only difference from GNNILAgent is the encoder class.
    All methods (select_actions, store, train_step, save, load)
    are identical — the training loop (trainGnn.py) works unchanged.
    """

    def __init__(
        self,
        obs_dim:      int,
        n_uavs:       int,
        n_actions:    int,
        gnn_hidden:   int   = 64,
        gnn_out:      int   = 32,
        attn_dim:     int   = 32,    # NEW: attention projection dimension
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
        self.n_actions    = n_actions
        self.gamma        = gamma
        self.batch_size   = batch_size
        self.target_update = target_update
        self.device       = torch.device(device)
        self.enriched_dim = obs_dim + gnn_out

        self.eps       = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay

        # Upgraded encoder — everything else stays the same
        self.gnn = BipartiteGATEncoder(
            ue_dim   = obs_dim,
            uav_dim  = 3,
            hidden   = gnn_hidden,
            out_dim  = gnn_out,
            attn_dim = attn_dim,
        ).to(self.device)

        self.policy_net = DuelingDQN(
            self.enriched_dim, n_actions, dqn_hidden
        ).to(self.device)
        self.target_net = DuelingDQN(
            self.enriched_dim, n_actions, dqn_hidden
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Single optimiser — covers GAT encoder AND policy net
        self.optimiser = torch.optim.RMSprop(
            list(self.gnn.parameters()) + list(self.policy_net.parameters()),
            lr=lr,
        )

        self.buffer = GlobalReplayBuffer(buffer_cap)
        self.steps  = 0

    # ── Graph helper (unchanged) ───────────────────────────────────────────

    @staticmethod
    def extract_edge_matrix(graph_data: dict) -> np.ndarray:
        M   = graph_data["n_ues"]
        N   = graph_data["n_uavs"]
        src, dst, w = graph_data["ue_uav"]
        mat = np.zeros((M, N), dtype=np.float32)
        for s, d, wt in zip(src, dst, w):
            mat[s, d] = wt
        return mat

    # ── Action selection (unchanged logic, uses new encoder) ──────────────

    def select_actions(
        self,
        obs_list:   List[np.ndarray],
        graph_data: dict,
        greedy:     bool = False,
    ) -> List[int]:
        with torch.no_grad():
            ue_x   = torch.tensor(
                np.stack(obs_list), dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            uav_x  = torch.tensor(
                graph_data["uav_x"], dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            edge_w = torch.tensor(
                self.extract_edge_matrix(graph_data), dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            h_ue      = self.gnn(ue_x, uav_x, edge_w)
            enriched  = torch.cat([ue_x, h_ue], dim=-1).squeeze(0)
            q_values  = self.policy_net(enriched)
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()

        if greedy:
            return [int(a) for a in greedy_actions]
        return [
            random.randint(0, self.n_actions - 1)
            if random.random() < self.eps
            else int(greedy_actions[m])
            for m in range(len(obs_list))
        ]

    # ── Store transition (unchanged) ───────────────────────────────────────

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

    # ── Training step (unchanged logic) ───────────────────────────────────

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        (ue_obs, uav_feats, edge_w,
         actions, rewards,
         next_ue_obs, next_uav_feats, next_edge_w,
         dones, task_masks) = [
            t.to(self.device) for t in self.buffer.sample(self.batch_size)
        ]

        B, M, _ = ue_obs.shape

        # Current Q values
        h_ue      = self.gnn(ue_obs, uav_feats, edge_w)
        enriched  = torch.cat([ue_obs, h_ue], dim=-1)
        q_all     = self.policy_net(enriched.view(B * M, -1))
        q_pred    = q_all.gather(
            1, actions.view(B * M).unsqueeze(1)
        ).squeeze(1)

        # Target Q values (Double DQN, no grad)
        with torch.no_grad():
            h_ue_next     = self.gnn(next_ue_obs, next_uav_feats, next_edge_w)
            enriched_next = torch.cat([next_ue_obs, h_ue_next], dim=-1)
            enf           = enriched_next.view(B * M, -1)
            next_actions  = self.policy_net(enf).argmax(dim=1)
            next_q        = self.target_net(enf).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            dones_exp     = dones.unsqueeze(1).expand(B, M).reshape(B * M)
            rewards_flat  = rewards.view(B * M)
            q_target      = rewards_flat + self.gamma * next_q * (1.0 - dones_exp)

        # Loss masked to agents that had a task
        mask_flat = task_masks.view(B * M)
        if mask_flat.sum() == 0:
            return None

        loss = F.smooth_l1_loss(q_pred[mask_flat], q_target[mask_flat])

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

    # ── ε schedule (unchanged) ────────────────────────────────────────────

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    # ── Persistence (unchanged) ───────────────────────────────────────────

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