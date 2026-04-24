"""
IL + GRU Agent.

Architecture:
    obs → GRU(obs_dim → gru_hidden) → DuelingHead(gru_hidden → n_actions)

Key design decisions:
  - Hidden state maintained across timesteps during rollout (reset at episode start).
  - SequenceReplayBuffer stores complete episodes; sampling pulls L-step chunks.
  - GRU is zero-init'd at the start of every sampled chunk (standard approximation).
  - Loss computed only on timesteps where the agent had a task (task_mask).

Framestack fallback (if GRU proves unstable):
  Replace GRU with:
      self.trunk = nn.Sequential(
          nn.Linear(obs_dim * K, hidden), nn.ReLU(),
          nn.Linear(hidden, hidden),      nn.ReLU(),
      )
  And in the episode loop, maintain a deque(maxlen=K) per agent, concatenate
  the last K observations as input. Flat tuple replay buffer works unchanged.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from typing import List, Optional, Tuple


# ─── Dueling Head (shared by both GRU variants) ────────────────────────────

class DuelingHead(nn.Module):
    """
    V(s) + A(s,a) − mean(A) head.
    Input: fixed-dim feature vector (gru_hidden).
    """
    def __init__(self, in_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.adv = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        V = self.value(x)                               # (..., 1)
        A = self.adv(x)                                 # (..., n_actions)
        return V + A - A.mean(dim=-1, keepdim=True)


# ─── Sequence Replay Buffer ────────────────────────────────────────────────

class SequenceReplayBuffer:
    """
    Stores complete episodes.
    Sampling pulls random L+1-length windows → L transitions.

    Each stored step: (obs, action, reward, had_task, done)
    """

    def __init__(self, capacity_episodes: int = 200, chunk_len: int = 8):
        self.L        = chunk_len
        self.episodes = deque(maxlen=capacity_episodes)
        self._current = []

    def push_step(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        had_task: bool,
        done:     bool,
    ):
        self._current.append((
            obs.astype(np.float32),
            int(action),
            float(reward),
            bool(had_task),
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

        obs_seqs, next_obs_seqs = [], []
        act_seqs, rew_seqs, mask_seqs, done_seqs = [], [], [], []

        attempts = 0
        while len(obs_seqs) < batch_size and attempts < batch_size * 30:
            attempts += 1
            ep = random.choice(self.episodes)
            if len(ep) < self.L + 1:
                continue
            start = random.randint(0, len(ep) - self.L - 1)
            chunk = ep[start : start + self.L + 1]           # L+1 steps

            obs_arr = np.stack([s[0] for s in chunk])        # (L+1, obs_dim)
            obs_seqs.append(obs_arr[:-1])                    # (L, obs_dim) — current
            next_obs_seqs.append(obs_arr[1:])                # (L, obs_dim) — next
            act_seqs.append([s[1] for s in chunk[:-1]])
            rew_seqs.append([s[2] for s in chunk[:-1]])
            mask_seqs.append([s[3] for s in chunk[:-1]])
            done_seqs.append([s[4] for s in chunk[:-1]])

        if not obs_seqs:
            return None

        return (
            torch.tensor(np.stack(obs_seqs),      dtype=torch.float32),  # (B, L, obs_dim)
            torch.tensor(np.stack(next_obs_seqs), dtype=torch.float32),  # (B, L, obs_dim)
            torch.tensor(act_seqs,                dtype=torch.long),     # (B, L)
            torch.tensor(rew_seqs,                dtype=torch.float32),  # (B, L)
            torch.tensor(mask_seqs,               dtype=torch.bool),     # (B, L)
            torch.tensor(done_seqs,               dtype=torch.float32),  # (B, L)
        )

    def __len__(self) -> int:
        return len(self.episodes)


# ─── IL + GRU Agent ────────────────────────────────────────────────────────

class ILGRUAgent:
    """
    Independent learner with GRU temporal memory.
    One instance per UE.  Hidden state maintained across rollout timesteps,
    reset at the start of every episode.
    """

    def __init__(
        self,
        obs_dim:       int,
        n_actions:     int,
        gru_hidden:    int   = 64,
        dqn_hidden:    int   = 64,
        lr:            float = 1e-3,
        gamma:         float = 0.9,
        batch_size:    int   = 32,
        chunk_len:     int   = 8,
        buf_episodes:  int   = 200,
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

        # GRU encoder — batch_first=True throughout
        self.gru = nn.GRU(
            input_size  = obs_dim,
            hidden_size = gru_hidden,
            batch_first = True,
        ).to(self.device)

        self.policy_net = DuelingHead(gru_hidden, n_actions, dqn_hidden).to(self.device)
        self.target_net = DuelingHead(gru_hidden, n_actions, dqn_hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimiser = torch.optim.RMSprop(
            list(self.gru.parameters()) + list(self.policy_net.parameters()),
            lr=lr,
        )

        self.buffer = SequenceReplayBuffer(buf_episodes, chunk_len)
        self.steps  = 0

        # Running hidden state: (num_layers=1, batch=1, gru_hidden)
        self.h: torch.Tensor = None
        self.reset_hidden()

    # ── Hidden state ──────────────────────────────────────────────────────

    def reset_hidden(self):
        """Call at the start of every episode."""
        self.h = torch.zeros(1, 1, self.gru_hidden, device=self.device)

    # ── Action selection ──────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """
        Single-step GRU forward. Updates self.h in-place.

        Shapes:
            input x   : (batch=1, seq=1, obs_dim)
            h_0       : (num_layers=1, batch=1, gru_hidden)
            out       : (batch=1, seq=1, gru_hidden)
            h_new     : (num_layers=1, batch=1, gru_hidden)
        """
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).to(self.device)
            x = x.view(1, 1, -1)                           # (1, 1, obs_dim)
            out, self.h = self.gru(x, self.h)              # (1,1,H), (1,1,H)
            feat = out[:, -1, :]                            # (1, gru_hidden)
            q    = self.policy_net(feat)                    # (1, n_actions)

        if not greedy and random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)
        return int(q.argmax(dim=1).item())

    # ── Store step ────────────────────────────────────────────────────────

    def push_step(
        self, obs: np.ndarray, action: int,
        reward: float, had_task: bool, done: bool
    ):
        self.buffer.push_step(obs, action, reward, had_task, done)

    # ── Training step ─────────────────────────────────────────────────────

    def train_step(self) -> Optional[float]:
        """
        Sample one chunk-batch, run GRU from h=0 over the chunk, update.

        Shapes inside:
            obs_seq, next_obs_seq : (B, L, obs_dim)
            actions, dones        : (B, L)
            rewards, task_mask    : (B, L)
            h0                    : (1, B, gru_hidden)
            gru_out               : (B, L, gru_hidden)
            feat                  : (B*L, gru_hidden)
            q_all                 : (B*L, n_actions)
            q_pred, q_target      : (B*L,)
        """
        result = self.buffer.sample(self.batch_size)
        if result is None:
            return None

        obs_seq, next_obs_seq, actions, rewards, task_mask, dones = [
            t.to(self.device) for t in result
        ]

        B, L, _ = obs_seq.shape
        h0 = torch.zeros(1, B, self.gru_hidden, device=self.device)

        # ── Current Q ─────────────────────────────────────────────────
        gru_out, _ = self.gru(obs_seq, h0)                 # (B, L, gru_hidden)
        feat        = gru_out.reshape(B * L, -1)           # (B*L, gru_hidden)
        q_all       = self.policy_net(feat)                # (B*L, n_actions)
        act_flat    = actions.reshape(B * L)               # (B*L,)
        q_pred      = q_all.gather(
            1, act_flat.unsqueeze(1)
        ).squeeze(1)                                        # (B*L,)

        # ── Target Q (Double DQN) ─────────────────────────────────────
        with torch.no_grad():
            gru_out_n, _ = self.gru(next_obs_seq, h0)     # (B, L, gru_hidden)
            feat_n        = gru_out_n.reshape(B * L, -1)
            next_act      = self.policy_net(feat_n).argmax(dim=1)
            next_q        = self.target_net(feat_n).gather(
                1, next_act.unsqueeze(1)
            ).squeeze(1)                                    # (B*L,)

            rew_flat  = rewards.reshape(B * L)
            done_flat = dones.reshape(B * L)
            q_target  = rew_flat + self.gamma * next_q * (1.0 - done_flat)

        # ── Masked Huber loss ─────────────────────────────────────────
        mask_flat = task_mask.reshape(B * L)               # (B*L,) bool
        if mask_flat.sum() == 0:
            return None

        loss = F.smooth_l1_loss(q_pred[mask_flat], q_target[mask_flat])

        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.gru.parameters()) + list(self.policy_net.parameters()),
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
            "gru":        self.gru.state_dict(),
            "policy_net": self.policy_net.state_dict(),
        }, path)