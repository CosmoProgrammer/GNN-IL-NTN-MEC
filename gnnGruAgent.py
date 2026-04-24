"""
GNN-IL + GRU Agent.

Architecture:
    obs → BipartiteGNN → cat(obs, gnn_out)
        → GRU(enriched_dim → gru_hidden)
        → DuelingHead(gru_hidden → n_actions)

The GRU sits between the GNN and the DQN head.
It provides temporal memory over the GNN-enriched (spatially-aware) observations.

Gradient flow:
    loss → DuelingHead → GRU → cat(obs, gnn_out) → GNN → obs
    All three modules (GNN, GRU, DuelingHead) update in one backward pass.

Hidden state during rollout:
    shape (num_layers=1, M, gru_hidden) — one per UE, evolving across timesteps.
    Reset to zeros at the start of every episode.

GlobalSequenceReplayBuffer:
    Stores complete episodes of system-wide transitions.
    Each step: (ue_obs, uav_feats, edge_w, actions, rewards, task_masks, done)
    Sampling pulls L+1-step windows → L transitions, running GNN+GRU over the chunk.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from typing import List, Optional, Tuple

from agent_gru import DuelingHead          # reuse dueling head
from gnn_agent import BipartiteGNNEncoder, GNNILAgent  # reuse GNN encoder


# ─── Global Sequence Replay Buffer ────────────────────────────────────────

class GlobalSequenceReplayBuffer:
    """
    Stores complete system-wide episodes.
    Each stored step:
        ue_obs    : (M, obs_dim)
        uav_feats : (N, 3)
        edge_w    : (M, N)
        actions   : (M,)
        rewards   : (M,)
        task_masks: (M,)  bool
        done      : float

    Sampling pulls L+1-step chunks, giving L transitions with
    corresponding next-step graph observations for the GNN.
    """

    def __init__(self, capacity_episodes: int = 100, chunk_len: int = 8):
        self.L        = chunk_len
        self.episodes = deque(maxlen=capacity_episodes)
        self._current = []

    def push_step(
        self,
        ue_obs:     np.ndarray,   # (M, obs_dim)
        uav_feats:  np.ndarray,   # (N, 3)
        edge_w:     np.ndarray,   # (M, N)
        actions:    np.ndarray,   # (M,)
        rewards:    np.ndarray,   # (M,)
        task_masks: np.ndarray,   # (M,) bool
        done:       bool,
    ):
        self._current.append((
            ue_obs.astype(np.float32),
            uav_feats.astype(np.float32),
            edge_w.astype(np.float32),
            actions.astype(np.int64),
            rewards.astype(np.float32),
            task_masks.astype(bool),
            float(done),
        ))
        if done:
            self._flush()

    def _flush(self):
        if len(self._current) >= self.L + 1:
            self.episodes.append(list(self._current))
        self._current = []

    def sample(self, batch_size: int) -> Optional[Tuple[torch.Tensor, ...]]:
        if len(self.episodes) < 5:
            return None

        ue_chunks, uav_chunks, ew_chunks = [], [], []
        act_chunks, rew_chunks, mask_chunks, done_chunks = [], [], [], []

        attempts = 0
        while len(ue_chunks) < batch_size and attempts < batch_size * 30:
            attempts += 1
            ep = random.choice(self.episodes)
            if len(ep) < self.L + 1:
                continue
            start = random.randint(0, len(ep) - self.L - 1)
            chunk = ep[start : start + self.L + 1]         # L+1 steps

            # Stack per-field over L+1 steps
            ue_arr   = np.stack([s[0] for s in chunk])     # (L+1, M, obs_dim)
            uav_arr  = np.stack([s[1] for s in chunk])     # (L+1, N, 3)
            ew_arr   = np.stack([s[2] for s in chunk])     # (L+1, M, N)
            act_arr  = np.stack([s[3] for s in chunk[:-1]])# (L, M)
            rew_arr  = np.stack([s[4] for s in chunk[:-1]])# (L, M)
            mask_arr = np.stack([s[5] for s in chunk[:-1]])# (L, M)
            done_arr = np.array([s[6] for s in chunk[:-1]])# (L,)

            ue_chunks.append(ue_arr)
            uav_chunks.append(uav_arr)
            ew_chunks.append(ew_arr)
            act_chunks.append(act_arr)
            rew_chunks.append(rew_arr)
            mask_chunks.append(mask_arr)
            done_chunks.append(done_arr)

        if not ue_chunks:
            return None

        return (
            # L+1 steps of graph data (GNN runs over full window)
            torch.tensor(np.stack(ue_chunks),   dtype=torch.float32),  # (B, L+1, M, obs_dim)
            torch.tensor(np.stack(uav_chunks),  dtype=torch.float32),  # (B, L+1, N, 3)
            torch.tensor(np.stack(ew_chunks),   dtype=torch.float32),  # (B, L+1, M, N)
            # L-step targets
            torch.tensor(np.stack(act_chunks),  dtype=torch.long),     # (B, L, M)
            torch.tensor(np.stack(rew_chunks),  dtype=torch.float32),  # (B, L, M)
            torch.tensor(np.stack(mask_chunks), dtype=torch.bool),     # (B, L, M)
            torch.tensor(np.stack(done_chunks), dtype=torch.float32),  # (B, L)
        )

    def __len__(self) -> int:
        return len(self.episodes)


# ─── GNN-IL + GRU Agent ────────────────────────────────────────────────────

class GNNILGRUAgent:
    """
    Shared GNN + shared GRU + shared DuelingHead for all UEs.

    Hidden state h: (1, M, gru_hidden) — one per UE, advanced each rollout step.
    Reset at start of every episode.

    train_step() gradient flow:
        loss → DuelingHead → GRU → enriched_obs → GNN → raw_obs
    """

    def __init__(
        self,
        obs_dim:       int,
        n_uavs:        int,
        n_actions:     int,
        gnn_hidden:    int   = 64,
        gnn_out:       int   = 32,
        gru_hidden:    int   = 64,
        dqn_hidden:    int   = 64,
        lr:            float = 1e-3,
        gamma:         float = 0.9,
        batch_size:    int   = 32,
        chunk_len:     int   = 8,
        buf_episodes:  int   = 100,
        target_update: int   = 20,
        eps_start:     float = 1.0,
        eps_end:       float = 0.05,
        eps_decay:     float = 0.995,
        device:        str   = "cpu",
    ):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update
        self.gru_hidden    = gru_hidden
        self.device        = torch.device(device)

        self.eps       = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay

        enriched_dim = obs_dim + gnn_out

        # GNN encoder (bipartite, same as GNNILAgent)
        self.gnn = BipartiteGNNEncoder(
            ue_dim  = obs_dim,
            uav_dim = 3,
            hidden  = gnn_hidden,
            out_dim = gnn_out,
        ).to(self.device)

        # GRU: input = enriched obs, hidden = gru_hidden
        # batch_first=True  →  input (batch, seq, features)
        self.gru = nn.GRU(
            input_size  = enriched_dim,
            hidden_size = gru_hidden,
            batch_first = True,
        ).to(self.device)

        self.policy_net = DuelingHead(gru_hidden, n_actions, dqn_hidden).to(self.device)
        self.target_net = DuelingHead(gru_hidden, n_actions, dqn_hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Single optimiser — GNN + GRU + DuelingHead updated jointly
        self.optimiser = torch.optim.RMSprop(
            list(self.gnn.parameters()) +
            list(self.gru.parameters()) +
            list(self.policy_net.parameters()),
            lr=lr,
        )

        self.buffer = GlobalSequenceReplayBuffer(buf_episodes, chunk_len)
        self.steps  = 0

        # Running hidden state — reset each episode
        # shape: (num_layers=1, M, gru_hidden)
        self.h: torch.Tensor = None

    # ── Hidden state ──────────────────────────────────────────────────────

    def reset_hidden(self, n_agents: int):
        """Call at the start of every episode with the number of UEs."""
        self.h = torch.zeros(1, n_agents, self.gru_hidden, device=self.device)

    # ── Graph edge matrix helper ───────────────────────────────────────────

    @staticmethod
    def extract_edge_matrix(graph_data: dict) -> np.ndarray:
        M   = graph_data["n_ues"]
        N   = graph_data["n_uavs"]
        src, dst, w = graph_data["ue_uav"]
        mat = np.zeros((M, N), dtype=np.float32)
        for s, d, wt in zip(src, dst, w):
            mat[s, d] = wt
        return mat

    # ── Action selection ──────────────────────────────────────────────────

    def select_actions(
        self,
        obs_list:   List[np.ndarray],   # length M, each (obs_dim,)
        graph_data: dict,
        greedy:     bool = False,
    ) -> List[int]:
        """
        Single-step forward: GNN → GRU (one step) → Q-values.
        Updates self.h in-place.

        GRU shapes:
            enriched_t : (M, 1, enriched_dim)  — M agents, seq_len=1
            h          : (1, M, gru_hidden)    — GRU h_0 / h_n
            gru_out    : (M, 1, gru_hidden)
            feat       : (M, gru_hidden)
        """
        M = len(obs_list)
        with torch.no_grad():
            ue_x   = torch.tensor(
                np.stack(obs_list), dtype=torch.float32
            ).unsqueeze(0).to(self.device)                     # (1, M, obs_dim)
            uav_x  = torch.tensor(
                graph_data["uav_x"], dtype=torch.float32
            ).unsqueeze(0).to(self.device)                     # (1, N, 3)
            edge_w = torch.tensor(
                self.extract_edge_matrix(graph_data), dtype=torch.float32
            ).unsqueeze(0).to(self.device)                     # (1, M, N)

            h_ue     = self.gnn(ue_x, uav_x, edge_w)          # (1, M, gnn_out)
            enriched = torch.cat([ue_x, h_ue], dim=-1)        # (1, M, enriched_dim)

            # Reshape for GRU: M separate single-step sequences
            # (1, M, enriched_dim) → (M, enriched_dim) → (M, 1, enriched_dim)
            enriched_t = enriched.squeeze(0).unsqueeze(1)      # (M, 1, enriched_dim)

            # self.h : (1, M, gru_hidden) — correct h_0 format for M-batch GRU
            gru_out, self.h = self.gru(enriched_t, self.h)    # (M,1,H), (1,M,H)
            feat = gru_out.squeeze(1)                          # (M, gru_hidden)
            q    = self.policy_net(feat)                       # (M, n_actions)
            greedy_acts = q.argmax(dim=1).cpu().numpy()

        if greedy:
            return [int(a) for a in greedy_acts]
        return [
            random.randint(0, self.n_actions - 1)
            if random.random() < self.eps
            else int(greedy_acts[m])
            for m in range(M)
        ]

    # ── Store step ────────────────────────────────────────────────────────

    def push_step(
        self,
        obs_list:   List[np.ndarray],
        graph_data: dict,
        actions:    List[int],
        rewards:    List[float],
        task_masks: List[bool],
        done:       bool,
    ):
        self.buffer.push_step(
            ue_obs     = np.stack(obs_list),
            uav_feats  = graph_data["uav_x"],
            edge_w     = self.extract_edge_matrix(graph_data),
            actions    = np.array(actions),
            rewards    = np.array(rewards),
            task_masks = np.array(task_masks),
            done       = done,
        )

    # ── Training step ─────────────────────────────────────────────────────

    def train_step(self) -> Optional[float]:
        """
        Sample a chunk batch.
        Run GNN at each of L+1 timesteps → enriched sequence.
        Run GRU over that sequence (zero-init h_0).
        Compute Double-DQN loss over L transitions, masked to task steps.

        Critical shapes (B=batch, L=chunk_len, M=n_agents, N=n_uavs):
            ue_obs_seq   : (B, L+1, M, obs_dim)
            uav_feats_seq: (B, L+1, N, 3)
            edge_w_seq   : (B, L+1, M, N)
            actions      : (B, L, M)
            rewards      : (B, L, M)
            task_masks   : (B, L, M)
            dones        : (B, L)

        After GNN loop  →  enriched_seq : (B, L+1, M, enriched_dim)
        After permute   →  (B, M, L+1, enriched_dim)
        After reshape   →  (B*M, L+1, enriched_dim)  ← GRU batch dimension

        After GRU       →  (B*M, L+1, gru_hidden)
        Current steps   →  (B*M, L, gru_hidden)  → reshape (B*M*L, H)
        Next steps      →  (B*M, L, gru_hidden)  → reshape (B*M*L, H)
                           [using steps 1..L from the L+1 output]

        Loss mask shape →  task_masks (B, L, M)
                           → permute(0,2,1) → (B, M, L)
                           → reshape(B*M*L,)
        """
        result = self.buffer.sample(self.batch_size)
        if result is None:
            return None

        (ue_obs_seq, uav_feats_seq, edge_w_seq,
         actions, rewards, task_masks, dones) = [
            t.to(self.device) for t in result
        ]

        B, Lp1, M, _ = ue_obs_seq.shape   # Lp1 = L+1
        L = Lp1 - 1
        _, _, N, _   = uav_feats_seq.shape

        # ── GNN over all L+1 timesteps ────────────────────────────────
        # Run without in-place ops; collect into a list then stack
        enriched_list = []
        for t in range(Lp1):
            h_ue  = self.gnn(
                ue_obs_seq[:, t],        # (B, M, obs_dim)
                uav_feats_seq[:, t],     # (B, N, 3)
                edge_w_seq[:, t],        # (B, M, N)
            )                            # (B, M, gnn_out)
            enc = torch.cat(
                [ue_obs_seq[:, t], h_ue], dim=-1
            )                            # (B, M, enriched_dim)
            enriched_list.append(enc)

        enriched_seq = torch.stack(enriched_list, dim=1)   # (B, L+1, M, enriched_dim)

        # ── Reshape for GRU: treat each (batch × agent) as a sequence ─
        # (B, L+1, M, D) → permute → (B, M, L+1, D) → reshape → (B*M, L+1, D)
        D_enc = enriched_seq.shape[-1]
        enriched_bm = (
            enriched_seq.permute(0, 2, 1, 3)               # (B, M, L+1, D)
                        .reshape(B * M, Lp1, D_enc)        # (B*M, L+1, D)
        )

        h0 = torch.zeros(1, B * M, self.gru_hidden, device=self.device)
        gru_out, _ = self.gru(enriched_bm, h0)             # (B*M, L+1, gru_hidden)

        # Current steps: 0 .. L-1
        feat_curr = gru_out[:, :L, :].reshape(B * M * L, -1)   # (B*M*L, H)
        q_all     = self.policy_net(feat_curr)                   # (B*M*L, n_actions)

        # actions: (B, L, M) → (B, M, L) → (B*M*L,)
        act_flat  = actions.permute(0, 2, 1).reshape(B * M * L)
        q_pred    = q_all.gather(
            1, act_flat.unsqueeze(1)
        ).squeeze(1)                                             # (B*M*L,)

        # ── Target Q (Double DQN, no grad) ───────────────────────────
        with torch.no_grad():
            # Next steps: 1 .. L
            feat_next = gru_out[:, 1:, :].reshape(B * M * L, -1) # (B*M*L, H)
            next_act  = self.policy_net(feat_next).argmax(dim=1)
            next_q    = self.target_net(feat_next).gather(
                1, next_act.unsqueeze(1)
            ).squeeze(1)                                          # (B*M*L,)

            # rewards: (B, L, M) → (B, M, L) → (B*M*L,)
            rew_flat  = rewards.permute(0, 2, 1).reshape(B * M * L)
            # dones:   (B, L) → unsqueeze → (B, 1, L) → expand → (B, M, L) → (B*M*L,)
            done_flat = (
                dones.unsqueeze(1)
                     .expand(B, M, L)
                     .reshape(B * M * L)
            )
            q_target = rew_flat + self.gamma * next_q * (1.0 - done_flat)

        # ── Masked Huber loss ─────────────────────────────────────────
        # task_masks: (B, L, M) → (B, M, L) → (B*M*L,)
        mask_flat = task_masks.permute(0, 2, 1).reshape(B * M * L)  # bool
        if mask_flat.sum() == 0:
            return None

        loss = F.smooth_l1_loss(q_pred[mask_flat], q_target[mask_flat])

        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.gnn.parameters()) +
            list(self.gru.parameters()) +
            list(self.policy_net.parameters()),
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
        torch.save({
            "gnn":        self.gnn.state_dict(),
            "gru":        self.gru.state_dict(),
            "policy_net": self.policy_net.state_dict(),
        }, path)