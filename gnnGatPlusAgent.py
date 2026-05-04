"""
GNN-IL Agent — BipartiteGATEncoder + Auxiliary Regression Loss + PER.

Builds directly on gnnAgentGAT.py. Two additions:

1. Auxiliary Regression Loss (from JCOSD, Chai et al. IEEE TCCN 2026)
   A small CostRegressorHead sits on top of the GNN embeddings and
   predicts the per-UE cost from the node embedding alone.
   Loss = TD_loss + χ * MSE(predicted_cost, actual_cost)
   Actual cost per UE = -reward  (reward is defined as -Ψm in the env).
   This gives the GNN a direct supervised signal rather than relying
   purely on TD backprop — stabilises early training.

2. Prioritized Experience Replay (PER, Schaul et al. 2016)
   Replaces uniform buffer sampling with TD-error-weighted sampling.
   High-error transitions are replayed more often.
   Importance Sampling (IS) weights correct for the sampling bias.

   Key hyperparameters:
     per_alpha  — how strongly to prioritise (0 = uniform, 1 = full)
     per_beta0  — initial IS exponent (anneals to 1 over training)
     per_eps    — small floor so zero-error transitions are never frozen

   After every train_step(), the priorities of the sampled transitions
   are updated with the freshly computed |TD error|.

Everything outside these two additions is identical to gnnAgentGAT.py.
The training loop (trainGnn.py) works unchanged — GATILAgentPlus
exposes the same public API as GATILAgent.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from typing import List, Optional, Tuple

from agent import DuelingDQN
from gnnGatAgent import BipartiteGATEncoder   # reuse the upgraded encoder


# ─── Sum Tree ──────────────────────────────────────────────────────────────

class SumTree:
    """
    Binary tree where each leaf stores a priority and each internal
    node stores the sum of its subtree.

    Supports O(log n) insertion, O(log n) priority update, and
    O(log n) stratified sampling by value in [0, total].

    Indices:
        Internal nodes: 1 … capacity-1  (1-indexed, root = 1)
        Leaves:         capacity … 2*capacity-1
        Leaf i maps to data buffer index i - capacity
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity, dtype=np.float64)
        self.write    = 0          # next leaf to write to
        self.n_entries = 0

    # ── Internal helpers ───────────────────────────────────────────────────

    def _propagate(self, leaf_idx: int, delta: float):
        """Bubble a priority change up to the root."""
        parent = leaf_idx // 2
        while parent >= 1:
            self.tree[parent] += delta
            parent //= 2

    def _leaf_index(self, data_index: int) -> int:
        return data_index + self.capacity

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def add(self, priority: float):
        """Write the next entry with the given priority."""
        leaf = self._leaf_index(self.write)
        self.update(self.write, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, data_index: int, priority: float):
        """Update the priority for an existing data index."""
        leaf  = self._leaf_index(data_index)
        delta = priority - self.tree[leaf]
        self.tree[leaf] = priority
        self._propagate(leaf, delta)

    def sample(self, value: float) -> Tuple[int, float]:
        """
        Traverse the tree to find the leaf whose prefix sum covers `value`.
        Returns (data_index, priority).
        """
        node = 1   # start at root
        while node < self.capacity:
            left = 2 * node
            if value <= self.tree[left]:
                node = left
            else:
                value -= self.tree[left]
                node   = left + 1
        data_index = node - self.capacity
        return data_index, float(self.tree[node])


# ─── Prioritized Global Replay Buffer ──────────────────────────────────────

class PrioritizedGlobalReplayBuffer:
    """
    Full-system transition buffer with SumTree-based priority sampling.

    Each entry is one timestep of the whole system (all UEs simultaneously),
    identical layout to GlobalReplayBuffer but with a per-entry priority.

    Sampling returns a batch of transitions plus per-sample IS weights
    and data indices (needed to update priorities after the TD step).

    Parameters
    ----------
    capacity  : maximum number of stored transitions
    alpha     : priority exponent  (0 = uniform, 1 = full prioritisation)
    beta0     : initial IS exponent (anneals to 1 over training)
    beta_steps: number of train_step() calls over which β → 1
    eps       : small floor added to |TD error| before raising to alpha
    """

    def __init__(
        self,
        capacity:   int   = 5_000,
        alpha:      float = 0.6,
        beta0:      float = 0.4,
        beta_steps: int   = 50_000,
        eps:        float = 1e-6,
    ):
        self.capacity   = capacity
        self.alpha      = alpha
        self.beta0      = beta0
        self.beta_steps = beta_steps
        self.eps        = eps
        self._step      = 0          # counts update() calls for β annealing

        self.tree = SumTree(capacity)
        self.data: List[Optional[tuple]] = [None] * capacity
        self.write = 0

        # Track the maximum priority seen so far.
        # New transitions are inserted at max priority so they are
        # guaranteed to be sampled at least once before any update.
        self._max_priority = 1.0

    # ── β annealing ────────────────────────────────────────────────────────

    @property
    def beta(self) -> float:
        """IS exponent, linearly annealed from beta0 to 1."""
        frac = min(1.0, self._step / max(1, self.beta_steps))
        return self.beta0 + frac * (1.0 - self.beta0)

    # ── Storage ────────────────────────────────────────────────────────────

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
        """Insert a new system transition at maximum priority."""
        entry = (
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
        )
        self.data[self.tree.write] = entry
        # Insert at max priority so new transitions are definitely sampled
        self.tree.add(self._max_priority ** self.alpha)

    # ── Sampling ───────────────────────────────────────────────────────────

    def sample(
        self, batch_size: int
    ) -> Tuple[Tuple[torch.Tensor, ...], np.ndarray, np.ndarray]:
        """
        Stratified sampling: divide [0, total] into batch_size segments,
        draw one value per segment.

        Returns
        -------
        (tensors, indices, is_weights)
            tensors    — same layout as GlobalReplayBuffer.sample()
            indices    — data indices for priority update after TD step
            is_weights — (batch_size,) float32 tensor, normalised to max=1
        """
        n          = self.tree.n_entries
        segment    = self.tree.total / batch_size
        indices    = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        for i in range(batch_size):
            lo    = segment * i
            hi    = segment * (i + 1)
            value = random.uniform(lo, hi)
            idx, pri      = self.tree.sample(min(value, self.tree.total - 1e-10))
            indices[i]    = idx
            priorities[i] = pri

        # IS weights: w_i = (N · P(i))^{-β}  normalised to max = 1
        prob       = priorities / self.tree.total
        is_weights = (n * prob) ** (-self.beta)
        is_weights = is_weights / is_weights.max()           # normalise
        is_weights = torch.tensor(is_weights, dtype=torch.float32)

        # Collate batch
        batch = [self.data[i] for i in indices]
        (ue_obs, uav_feats, edge_w, actions, rewards,
         next_ue_obs, next_uav_feats, next_edge_w,
         dones, task_masks) = zip(*batch)

        tensors = (
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
        return tensors, indices, is_weights

    # ── Priority update ────────────────────────────────────────────────────

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Called after each train_step() with the per-transition |TD error|.
        Updates the SumTree and tracks the running maximum priority.
        """
        self._step += 1
        for idx, err in zip(indices, td_errors):
            priority = (float(abs(err)) + self.eps) ** self.alpha
            self.tree.update(int(idx), priority)
            self._max_priority = max(self._max_priority, priority)

    def __len__(self) -> int:
        return self.tree.n_entries


# ─── Cost Regressor Head ───────────────────────────────────────────────────

class CostRegressorHead(nn.Module):
    """
    Tiny MLP that predicts scalar cost from a GNN node embedding.
    Sits alongside the DQN — shares the GNN encoder, has its own weights.

    Input:  GNN embedding  (enriched_dim = obs_dim + gnn_out)
    Output: predicted cost (scalar per UE)
    """

    def __init__(self, enriched_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(enriched_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (B*M,) or (M,)


# ─── GAT-IL Agent with Aux Loss + PER ──────────────────────────────────────

class GATILAgentPlus:
    """
    GATILAgent upgraded with:
      - Auxiliary regression loss (JCOSD-style)
      - Prioritized Experience Replay

    Public API is identical to GATILAgent so trainGnn.py works unchanged
    (just swap the import and add the new hyperparameters).
    """

    def __init__(
        self,
        obs_dim:       int,
        n_uavs:        int,
        n_actions:     int,
        gnn_hidden:    int   = 64,
        gnn_out:       int   = 32,
        attn_dim:      int   = 32,
        dqn_hidden:    int   = 128,
        lr:            float = 1e-3,
        gamma:         float = 0.9,
        batch_size:    int   = 32,
        buffer_cap:    int   = 5_000,
        target_update: int   = 20,
        eps_start:     float = 1.0,
        eps_end:       float = 0.05,
        eps_decay:     float = 0.995,
        # ── Auxiliary loss ──────────────────────────────────────────────
        chi:           float = 0.5,    # weight of aux loss (χ in JCOSD eq.49)
        reg_hidden:    int   = 64,     # hidden size of cost regressor
        # ── PER ─────────────────────────────────────────────────────────
        per_alpha:     float = 0.6,    # priority exponent
        per_beta0:     float = 0.4,    # initial IS exponent
        per_beta_steps:int   = 50_000, # steps to anneal β → 1
        per_eps:       float = 1e-6,   # priority floor
        device:        str   = "cpu",
    ):
        self.n_actions    = n_actions
        self.gamma        = gamma
        self.batch_size   = batch_size
        self.target_update = target_update
        self.chi          = chi
        self.device       = torch.device(device)
        self.enriched_dim = obs_dim + gnn_out

        self.eps       = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay

        # ── Networks ───────────────────────────────────────────────────
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

        # Auxiliary cost regressor
        self.regressor = CostRegressorHead(self.enriched_dim, reg_hidden).to(self.device)

        # Single optimiser — covers GNN + DQN + regressor
        self.optimiser = torch.optim.RMSprop(
            list(self.gnn.parameters())
            + list(self.policy_net.parameters())
            + list(self.regressor.parameters()),
            lr=lr,
        )

        # ── Prioritized buffer ─────────────────────────────────────────
        self.buffer = PrioritizedGlobalReplayBuffer(
            capacity   = buffer_cap,
            alpha      = per_alpha,
            beta0      = per_beta0,
            beta_steps = per_beta_steps,
            eps        = per_eps,
        )

        self.steps = 0

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

    # ── Action selection (unchanged) ──────────────────────────────────────

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

            h_ue     = self.gnn(ue_x, uav_x, edge_w)
            enriched = torch.cat([ue_x, h_ue], dim=-1).squeeze(0)
            q_values = self.policy_net(enriched)
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()

        if greedy:
            return [int(a) for a in greedy_actions]
        return [
            random.randint(0, self.n_actions - 1)
            if random.random() < self.eps
            else int(greedy_actions[m])
            for m in range(len(obs_list))
        ]

    # ── Store transition (unchanged) ──────────────────────────────────────

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

    # ── Training step (upgraded) ───────────────────────────────────────────

    def train_step(self) -> Optional[float]:
        """
        One gradient update combining:
          (a) IS-weighted TD loss  (PER correction)
          (b) Auxiliary regression loss  (GNN supervision)

        Returns combined loss or None if buffer too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        # ── Sample with priorities ──────────────────────────────────────
        (tensors, sample_indices, is_weights) = self.buffer.sample(self.batch_size)

        (ue_obs, uav_feats, edge_w,
         actions, rewards,
         next_ue_obs, next_uav_feats, next_edge_w,
         dones, task_masks) = [t.to(self.device) for t in tensors]

        is_weights = is_weights.to(self.device)    # (B,)

        B, M, _ = ue_obs.shape

        # ── GNN forward pass (with grad) ───────────────────────────────
        h_ue      = self.gnn(ue_obs, uav_feats, edge_w)    # (B, M, gnn_out)
        enriched  = torch.cat([ue_obs, h_ue], dim=-1)      # (B, M, enriched)
        flat      = enriched.view(B * M, -1)                # (B*M, enriched)

        # ── TD loss (PER-weighted) ─────────────────────────────────────
        q_all   = self.policy_net(flat)                         # (B*M, n_actions)
        q_pred  = q_all.gather(
            1, actions.view(B * M).unsqueeze(1)
        ).squeeze(1)                                            # (B*M,)

        with torch.no_grad():
            h_ue_next     = self.gnn(next_ue_obs, next_uav_feats, next_edge_w)
            enriched_next = torch.cat([next_ue_obs, h_ue_next], dim=-1)
            nflat         = enriched_next.view(B * M, -1)
            next_actions  = self.policy_net(nflat).argmax(dim=1)
            next_q        = self.target_net(nflat).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            dones_exp    = dones.unsqueeze(1).expand(B, M).reshape(B * M)
            rewards_flat = rewards.view(B * M)
            q_target     = rewards_flat + self.gamma * next_q * (1.0 - dones_exp)

        # Per-element TD error for priority update and loss
        td_errors  = (q_pred - q_target).detach()            # (B*M,)
        mask_flat  = task_masks.view(B * M)                  # (B*M,) bool

        if mask_flat.sum() == 0:
            return None

        # IS weight per element: expand per-transition weight to per-UE
        # Each transition i covers M UEs → all M share the same IS weight
        is_per_ue = is_weights.unsqueeze(1).expand(B, M).reshape(B * M)

        # Weighted Huber loss, restricted to UEs that had tasks
        element_loss = F.smooth_l1_loss(
            q_pred[mask_flat], q_target[mask_flat], reduction="none"
        )                                                     # (n_tasks,)
        td_loss = (is_per_ue[mask_flat] * element_loss).mean()

        # ── Auxiliary regression loss ──────────────────────────────────
        # Predicted cost from GNN embedding (detach from DQN gradient path
        # so aux loss only trains the GNN + regressor, not the DQN head)
        pred_cost  = self.regressor(flat[mask_flat])          # (n_tasks,)
        # Actual cost = -reward  (reward = -Ψm when task exists)
        true_cost  = -rewards_flat[mask_flat]                 # (n_tasks,)
        aux_loss   = F.mse_loss(pred_cost, true_cost)

        # ── Combined loss ──────────────────────────────────────────────
        total_loss = td_loss + self.chi * aux_loss

        self.optimiser.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.gnn.parameters())
            + list(self.policy_net.parameters())
            + list(self.regressor.parameters()),
            max_norm=10.0,
        )
        self.optimiser.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # ── Update priorities ──────────────────────────────────────────
        # Use mean |TD error| per transition as the new priority.
        td_errors_np = td_errors.cpu().numpy().reshape(B, M)  # (B, M)
        mask_np      = task_masks.cpu().numpy()                # (B, M)
        # Mean absolute TD error over UEs that had tasks in each transition;
        # fall back to a small constant if no UE had a task (shouldn't happen).
        per_trans_error = np.array([
            np.abs(td_errors_np[b][mask_np[b]]).mean()
            if mask_np[b].any() else 1e-4
            for b in range(B)
        ])
        self.buffer.update_priorities(sample_indices, per_trans_error)

        return float(total_loss.item())

    # ── ε schedule (unchanged) ────────────────────────────────────────────

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    # ── Persistence (unchanged) ───────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "gnn":        self.gnn.state_dict(),
            "policy_net": self.policy_net.state_dict(),
            "regressor":  self.regressor.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.gnn.load_state_dict(ckpt["gnn"])
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["policy_net"])
        if "regressor" in ckpt:
            self.regressor.load_state_dict(ckpt["regressor"])