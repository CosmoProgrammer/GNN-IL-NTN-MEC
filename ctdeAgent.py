"""
Centralized Training, Decentralized Execution (CTDE) agent — Dueling DQN.

Faithful to the paper (Algorithm 1):
  - One shared Dueling DQN trained on a central controller
  - One shared replay buffer (capacity = M × per_agent_capacity)
  - All M UEs select actions from the shared policy network (decentralized execution)
  - All M UEs push transitions into the shared buffer
  - One gradient update per timestep, sampled from the shared buffer
  - Hard target-network update every C steps
  - Weight sync (Algorithm 1, line 17) is implicit: all UEs reference the
    same Python object, so no explicit copying is needed
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from typing import List, Tuple, Optional

# Re-use the network and buffer from the IL agent — same architecture.
from agent import DuelingDQN, ReplayBuffer


# ─── CTDE Agent ────────────────────────────────────────────────────────────

class CTDEAgent:
    """
    Central controller that owns one shared Dueling DQN and one shared
    replay buffer.  All M UEs call select_action / store / train_step
    through this single object, mirroring Algorithm 1 in the paper.
    """

    def __init__(
        self,
        obs_dim:         int,
        n_actions:       int,
        n_agents:        int,               # M — needed to size the buffer
        hidden:          int   = 128,
        lr:              float = 1e-3,      # paper: 0.001
        gamma:           float = 0.9,       # paper: 0.9
        per_agent_cap:   int   = 10_000,    # shared buffer = n_agents × this
        batch_size:      int   = 32,        # paper: 32
        target_update:   int   = 20,        # hard-copy every C steps
        eps_start:       float = 1.0,
        eps_end:         float = 0.05,
        eps_decay:       float = 0.995,
        device:          str   = "cpu",
    ):
        self.n_actions    = n_actions
        self.gamma        = gamma
        self.batch_size   = batch_size
        self.target_update = target_update
        self.device       = torch.device(device)

        # ε-greedy schedule  (one shared schedule for all UEs)
        self.eps      = eps_start
        self.eps_end  = eps_end
        self.eps_decay = eps_decay

        # Shared networks  (one logical "central controller")
        self.policy_net = DuelingDQN(obs_dim, n_actions, hidden).to(self.device)
        self.target_net = DuelingDQN(obs_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimiser  (paper uses RMSProp)
        self.optimiser = torch.optim.RMSprop(self.policy_net.parameters(), lr=lr)

        # Shared replay buffer  (capacity scales with number of agents)
        self.buffer = ReplayBuffer(capacity=n_agents * per_agent_cap)

        self.steps = 0          # total training-step counter

    # ── Action selection (decentralized execution) ─────────────────────────

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """
        Any UE calls this with its local observation.
        All UEs share the same policy network — weight sync is implicit.
        """
        if not greedy and random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            q = self.policy_net(
                torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            )
            return int(q.argmax(dim=1).item())

    # ── Store transition (any UE pushes into the shared buffer) ───────────

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    # ── Centralized training step ──────────────────────────────────────────

    def train_step(self) -> Optional[float]:
        """
        One gradient update on the shared policy network.
        Called once per timestep (after all UEs have stored their transitions).
        Returns loss value (float) or None if buffer too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = [
            t.to(self.device) for t in self.buffer.sample(self.batch_size)
        ]

        # Current Q values
        q_values = self.policy_net(obs)                         # (B, n_actions)
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

        # Hard target update every C steps  (Algorithm 1, line 15)
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self):
        """Call once per episode — mirrors the IL agent interface."""
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    # ── Weight access (for evaluation / checkpointing) ────────────────────

    def get_weights(self):
        return self.policy_net.state_dict()

    def set_weights(self, state_dict):
        self.policy_net.load_state_dict(state_dict)