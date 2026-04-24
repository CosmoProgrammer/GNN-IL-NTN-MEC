"""
Independent Learner (IL) agent — Dueling DQN.

Faithful to the paper:
  - Dueling architecture: V(s) + A(s,a) − mean(A)
  - Target network, hard-updated every C steps
  - Experience replay buffer
  - ε-greedy exploration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from typing import Tuple, Optional


# ─── Replay Buffer ─────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((
            np.array(obs,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_obs, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.tensor(np.stack(obs),      dtype=torch.float32),
            torch.tensor(actions,            dtype=torch.long),
            torch.tensor(rewards,            dtype=torch.float32),
            torch.tensor(np.stack(next_obs), dtype=torch.float32),
            torch.tensor(dones,              dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


# ─── Dueling DQN Network ───────────────────────────────────────────────────

class DuelingDQN(nn.Module):
    """
    Shared trunk → split into Value stream V(s) and Advantage stream A(s,a).
    Q(s,a) = V(s) + A(s,a) − mean_a′ A(s,a′)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()

        # Shared feature extractor
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Advantage stream
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(x)
        V    = self.value_stream(feat)                      # (B, 1)
        A    = self.adv_stream(feat)                        # (B, n_actions)
        Q    = V + A - A.mean(dim=1, keepdim=True)         # (B, n_actions)
        return Q


# ─── IL Agent ──────────────────────────────────────────────────────────────

class ILAgent:
    """
    One independent dueling-DQN agent per UE.
    Trains from its own local replay buffer.
    """

    def __init__(
        self,
        obs_dim:      int,
        n_actions:    int,
        hidden:       int   = 128,
        lr:           float = 1e-3,       # paper: 0.001
        gamma:        float = 0.9,        # paper: 0.9
        buffer_cap:   int   = 10_000,
        batch_size:   int   = 32,         # paper: 32
        target_update:int   = 20,         # hard-copy every C steps
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

        # ε-greedy schedule
        self.eps       = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay

        # Networks
        self.policy_net = DuelingDQN(obs_dim, n_actions, hidden).to(self.device)
        self.target_net = DuelingDQN(obs_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimiser (paper uses RMSProp)
        self.optimiser = torch.optim.RMSprop(self.policy_net.parameters(), lr=lr)

        self.buffer    = ReplayBuffer(buffer_cap)
        self.steps     = 0              # total training steps taken

    # ── Action selection ───────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            q = self.policy_net(
                torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            )
            return int(q.argmax(dim=1).item())

    # ── Store transition ───────────────────────────────────────────────────

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    # ── Training step ──────────────────────────────────────────────────────

    def train_step(self) -> Optional[float]:
        """
        Sample one mini-batch and do one gradient update.
        Returns loss value (float) or None if buffer too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = [
            t.to(self.device) for t in self.buffer.sample(self.batch_size)
        ]

        # Current Q values
        q_values = self.policy_net(obs)                     # (B, n_actions)
        q_pred   = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Target Q values — Double DQN style:
        #   action chosen by policy_net, evaluated by target_net
        with torch.no_grad():
            next_actions = self.policy_net(next_obs).argmax(dim=1)       # (B,)
            next_q       = self.target_net(next_obs).gather(
                               1, next_actions.unsqueeze(1)).squeeze(1)  # (B,)
            q_target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_pred, q_target)

        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimiser.step()

        self.steps += 1

        # Hard target update every C steps
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self):
        """Call once per episode externally — not inside train_step."""
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    # ── CTDE weight sync (used by CTDETrainer, not IL) ─────────────────────

    def set_weights(self, state_dict):
        self.policy_net.load_state_dict(state_dict)

    def get_weights(self):
        return self.policy_net.state_dict()