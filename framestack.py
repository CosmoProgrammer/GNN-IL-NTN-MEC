"""
Observation frame stacking — temporal memory approximation.

Replaces the GRU with a fixed-window concatenation of the last K observations.
Provides temporal context (queue dynamics, load trends) without recurrent
training instability.

Usage in episode loops:
    stacks = [FrameStack(obs_dim, K) for _ in range(n_agents)]
    obs_list = env.reset()
    for stack, obs in zip(stacks, obs_list):
        stack.reset(obs)
    ...
    stacked_obs = [stack.get() for stack in stacks]
    actions = [agent.select_action(stacked_obs[m]) ...]
    ...
    for stack, obs in zip(stacks, next_obs_list):
        stack.push(obs)

The stacked observation is simply:
    np.concatenate([obs_{t-K+1}, ..., obs_{t-1}, obs_t])   shape: (K * obs_dim,)

This is a drop-in replacement for the raw obs in:
  - ILAgent        (agent.py)      — input dim: K * obs_dim
  - GNNILAgent     (gnn_agent.py)  — GNN runs on raw obs_dim; stacking applied
                                     to the enriched observation before DQN

IMPORTANT for GNN-IL+Stack:
    The GNN always receives raw (un-stacked) obs — it needs the original obs_dim
    to build node features correctly.
    The frame stack is applied to the GNN-enriched observation:
        enriched_t = cat(obs_t, gnn_out_t)      (enriched_dim,)
        stacked    = cat(enriched_{t-3}, ..., enriched_t)  (K * enriched_dim,)
    The DQN input dim becomes K * enriched_dim.
"""

import numpy as np
from collections import deque


class FrameStack:
    """
    Maintains a rolling window of K observations.
    Zero-pads at the start of each episode.
    """

    def __init__(self, obs_dim: int, K: int = 4):
        self.obs_dim = obs_dim
        self.K       = K
        self.frames  = deque(maxlen=K)
        self.stacked_dim = obs_dim * K

    def reset(self, obs: np.ndarray):
        """Call at the start of every episode with the first observation."""
        self.frames.clear()
        for _ in range(self.K):
            self.frames.append(np.zeros(self.obs_dim, dtype=np.float32))
        self.frames.append(obs.astype(np.float32))

    def push(self, obs: np.ndarray):
        """Call after every env.step() with the next observation."""
        self.frames.append(obs.astype(np.float32))

    def get(self) -> np.ndarray:
        """Returns the stacked observation: (K * obs_dim,)"""
        return np.concatenate(list(self.frames), axis=0)