"""
NTN-MEC Environment — faithful simplification of:
  Fatima, Saxena, Giambene. "Computation Offloading in NTN-empowered MEC
  using Multi-Agent Distributed Deep Reinforcement Learning." GLOBECOM 2024.

Simplifications vs. paper:
  - No LEO satellite (UAV-only for speed; easy to add back)
  - Smaller default scale: M=10 UEs, N=2 UAVs (paper: 40/4)
  - Same queue/delay/energy/reward math as the paper (Eqs 1-14)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# ─── Configuration ─────────────────────────────────────────────────────────

@dataclass
class EnvConfig:
    # Topology
    M: int   = 10       # number of UEs
    N: int   = 2        # number of UAVs

    # Area & altitude
    L: float = 100.0    # m (area length)
    W: float = 100.0    # m (area width)
    H: float = 100.0    # m (UAV hovering altitude)

    # Time
    delta: float = 0.2  # time slot duration (s)
    I: int       = 100  # time slots per episode

    # Task model
    P_task: float = 0.3    # task generation probability per slot
    D_min:  float = 1.0    # min task size (Mbits)
    D_max:  float = 3.0    # max task size (Mbits)  [paper: 2–5, scaled down]
    sm:     float = 297.0  # CPU cycles per bit
    # Paper quotes 0.297 gigacycles/Mbit; since D_bits = D_mbits * 1e6,
    # the correct per-bit value is 0.297e9 / 1e6 = 297 cycles/bit.
    tau:    int   = 10     # deadline (time slots)

    # Communication (Eq. 1)
    B:       float = 1e6    # bandwidth per UE-UAV link (Hz) — 1 MHz
    Pup_uav: float = 0.1    # uplink power to UAV (W)
    beta0:   float = 1e-3   # path loss constant
    j:       float = 2.0    # path loss exponent
    sigma2:  float = 1e-10  # noise power (W) ≈ -100 dBm

    # Computation
    f_device:   float = 1e9   # UE CPU frequency (cycles/s) — 1 GHz
    f_edge_uav: float = 2.5e9 # UAV edge server frequency (cycles/s)

    # Energy (Eqs. 12, 13)
    kappa: float = 1e-27  # effective switched capacitance

    # Cost weights (Eq. 14)
    w1: float = 0.5   # delay weight
    w2: float = 0.5   # energy weight

    # Penalty for a dropped task
    penalty: float = 20.0

    # UE mobility
    v:    float = 1.0  # speed (m/s)
    tmov: float = 1.0  # movement time per slot (s)


# ─── Environment ───────────────────────────────────────────────────────────

class NTNMECEnv:
    """
    Multi-agent NTN-MEC offloading environment.

    Action space per agent:  {0=local, 1=UAV_0, ..., N=UAV_{N-1}}
    Observation per agent:
        [ue_pos(2) | uav_pos(2N) | task_size(1) | local_wait(1) |
         tran_wait(1) | edge_queue_per_uav(N) | active_queues_per_uav(N)]
    Total obs dim = 5 + 3*N  (13 for N=2)
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        self.cfg = config or EnvConfig()
        cfg = self.cfg

        self.n_agents  = cfg.M
        self.n_actions = cfg.N + 1          # local + N UAVs
        # obs: ue_pos(2) + uav_pos(2N) + task(1) + local_wait(1) + tran_wait(1) + edge_q(N)
        # = 5 + 3*N   (active_q removed — GNN provides that signal instead)
        self.obs_dim   = 5 + 3 * cfg.N

        # Fixed UAV horizontal positions (evenly spaced along x-axis, y=W/2)
        self.uav_pos = np.array([
            [(n + 1) * cfg.L / (cfg.N + 1), cfg.W / 2]
            for n in range(cfg.N)
        ], dtype=np.float32)               # shape (N, 2) — horizontal only

        self.reset()

    # ── Reset ──────────────────────────────────────────────────────────────

    def reset(self) -> List[np.ndarray]:
        cfg = self.cfg
        self.t = 0

        # UE positions: uniform random in area
        self.ue_pos = np.random.uniform(
            [0, 0], [cfg.L, cfg.W], size=(cfg.M, 2)
        ).astype(np.float32)

        # UE movement angles
        self.ue_angles = np.random.uniform(0, 2 * np.pi, cfg.M)

        # Queue state: the time slot at which each queue becomes free.
        #   l_comp[m]    — local computation queue for UE m
        #   l_tran[m]    — transmission queue for UE m
        #   l_edge[n, m] — edge queue at UAV n for UE m
        # Initialised to 0 (free from slot 0 onwards).
        self.l_comp = np.zeros(cfg.M, dtype=np.float32)
        self.l_tran = np.zeros(cfg.M, dtype=np.float32)
        self.l_edge = np.zeros((cfg.N, cfg.M), dtype=np.float32)

        # Generate tasks for the first slot
        self.tasks = self._generate_tasks()

        return self._get_observations()

    # ── Step ───────────────────────────────────────────────────────────────

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], bool, dict]:
        """
        Advance one time slot.

        Parameters
        ----------
        actions : list of int, length M
            Action for each UE: 0=local, 1..N = offload to UAV 0..N-1

        Returns
        -------
        next_obs : list of np.ndarray
        rewards  : list of float
        done     : bool
        info     : dict
        """
        cfg = self.cfg
        rewards = []
        costs   = []
        n_tasks = sum(1 for t in self.tasks if t is not None)

        for m in range(cfg.M):
            cost, _ = self._process_task(m, actions[m])
            rewards.append(-cost)
            costs.append(cost)

        # Move UEs, advance clock, generate new tasks
        self._move_ues()
        self.t += 1
        done = (self.t >= cfg.I)
        self.tasks = self._generate_tasks()

        info = {
            "avg_cost":        float(np.mean(costs)),
            "n_tasks_generated": n_tasks,
            "t": self.t,
        }

        return self._get_observations(), rewards, done, info

    # ── Core physics ───────────────────────────────────────────────────────

    def _process_task(self, m: int, action: int) -> Tuple[float, bool]:
        """
        Compute cost for UE m taking `action` at time slot self.t.
        Updates queue state in-place.
        Returns (cost, completed).
        """
        cfg  = self.cfg
        task = self.tasks[m]

        if task is None:
            return 0.0, True          # no task — zero cost

        D_mbits, deadline = task
        D_bits = D_mbits * 1e6        # convert to bits
        X = deadline                  # last valid completion slot
        i = self.t

        if action == 0:
            # ── Local processing ───────────────────────────────────────
            # Eq. 6: l_comp = min(i + δ_comp + γ_comp − 1, X)
            gamma_comp = int(np.ceil(
                (D_bits * cfg.sm) / (cfg.f_device * cfg.delta)
            ))
            delta_comp = max(0.0, self.l_comp[m] - i)   # waiting slots (Eq. 9, simplified)
            delta_comp = int(np.ceil(delta_comp))

            finish     = i + delta_comp + gamma_comp - 1
            l_new      = min(finish, X)
            self.l_comp[m] = l_new

            completed  = (finish <= X)
            delay      = (l_new - i + 1)                # Eq. 3: t_local

            # Eq. 12: e_local = κ · D · s_m · (f_device · Δ)²
            energy = cfg.kappa * D_bits * cfg.sm * (cfg.f_device * cfg.delta) ** 2

        else:
            # ── Offload to UAV (action 1..N → UAV index 0..N-1) ───────
            n    = action - 1
            rate = self._tx_rate(m, n)                   # bits / s  (Eq. 1)

            if rate < 1e3:                               # degenerate channel
                return cfg.penalty, False

            # Transmission (Eq. 7)
            gamma_tran = int(np.ceil(D_bits / (rate * cfg.delta)))
            delta_tran = max(0.0, self.l_tran[m] - i)
            delta_tran = int(np.ceil(delta_tran))

            finish_tran = i + delta_tran + gamma_tran - 1
            l_tran_new  = min(finish_tran, X)
            self.l_tran[m] = l_tran_new

            # Task arrives at edge at slot i_star = finish_tran + 1
            i_star = finish_tran + 1
            if i_star > X:
                return cfg.penalty, False

            # Edge processing — CPU equally shared among active queues (Eq. 8)
            # Active queues at UAV n at time i_star = those whose l_edge >= i_star
            active = int(np.sum(self.l_edge[n] > i_star)) + 1  # +1 for this task
            f_shared   = cfg.f_edge_uav / active
            gamma_edge = int(np.ceil(
                (D_bits * cfg.sm) / (f_shared * cfg.delta)
            ))
            delta_edge = max(0.0, self.l_edge[n, m] - i_star)
            delta_edge = int(np.ceil(delta_edge))

            finish_edge = i_star + delta_edge + gamma_edge - 1
            l_edge_new  = min(finish_edge, X)
            self.l_edge[n, m] = l_edge_new

            completed = (finish_edge <= X)
            # Eq. 4+5: total delay = t_tran + t_edge
            delay  = (l_tran_new - i + 1) + (l_edge_new - i_star + 1)

            # Eq. 13: e_tran = Pup · D / (r · Δ)
            energy = (cfg.Pup_uav * D_bits) / (rate * cfg.delta)

        if not completed:
            return cfg.penalty, False

        # Eq. 14: Ψ = w1·t + w2·e
        cost = cfg.w1 * delay + cfg.w2 * energy
        return float(cost), True

    # ── Channel model ──────────────────────────────────────────────────────

    def _channel_gain(self, m: int, n: int) -> float:
        """h_{m,n}(i) = β0 / (‖q_n − p_m‖² + H²)^(j/2)  — Eq. 1"""
        cfg = self.cfg
        dx  = self.uav_pos[n, 0] - self.ue_pos[m, 0]
        dy  = self.uav_pos[n, 1] - self.ue_pos[m, 1]
        return cfg.beta0 / ((dx**2 + dy**2 + cfg.H**2) ** (cfg.j / 2))

    def _tx_rate(self, m: int, n: int) -> float:
        """r_{m,n}(i) = B · log2(1 + P_up · h / σ²)  (bits/s)  — Eq. 1"""
        cfg = self.cfg
        snr = cfg.Pup_uav * self._channel_gain(m, n) / cfg.sigma2
        return cfg.B * np.log2(1.0 + snr)

    # ── Observations ───────────────────────────────────────────────────────

    def _get_observations(self) -> List[np.ndarray]:
        """
        Build flat observation vector for each UE.
        All values normalised to [0, 1].
        """
        cfg = self.cfg
        obs_list = []

        for m in range(cfg.M):
            # (1) UE position — normalised
            ue_xy  = self.ue_pos[m] / np.array([cfg.L, cfg.W], dtype=np.float32)

            # (2) UAV horizontal positions — normalised
            uav_xy = (self.uav_pos / np.array([cfg.L, cfg.W], dtype=np.float32)).flatten()

            # (3) Task size (0 if no task)
            if self.tasks[m] is not None:
                task_feat = np.array([self.tasks[m][0] / cfg.D_max], dtype=np.float32)
            else:
                task_feat = np.array([0.0], dtype=np.float32)

            # (4) Local queue wait — normalised by tau
            local_wait = np.array(
                [max(0.0, self.l_comp[m] - self.t) / cfg.tau], dtype=np.float32
            )

            # (5) Transmission queue wait — normalised by tau
            tran_wait = np.array(
                [max(0.0, self.l_tran[m] - self.t) / cfg.tau], dtype=np.float32
            )

            # (6) This UE's queue residual at each UAV — normalised by tau
            edge_q = np.array(
                [max(0.0, self.l_edge[n, m] - self.t) / cfg.tau for n in range(cfg.N)],
                dtype=np.float32
            )

            # NOTE: global congestion signal (fraction of UEs queued per UAV)
            # intentionally excluded from raw obs.  IL agents are blind to it.
            # GNN-IL agents recover equivalent coordination signal through
            # message passing — that is the contribution this code evaluates.

            obs = np.concatenate([ue_xy, uav_xy, task_feat,
                                   local_wait, tran_wait, edge_q])
            obs_list.append(obs)

        return obs_list

    # ── Graph data for GNN layer ────────────────────────────────────────────

    def get_graph_data(self) -> dict:
        """
        Build the bipartite UE ↔ UAV graph for the GNN layer.

        UE  node features : raw obs vector  (obs_dim,)  — local info only
        UAV node features : [norm_x, norm_y, congestion_level]  (3,)
            congestion = fraction of UEs with active queue at this UAV.
            This is the key signal IL agents lack; the GNN propagates it
            back to UE nodes via Layer-2 (UAV → UE) message passing.

        Edge weight : normalised channel gain h_{m,n}
        """
        cfg = self.cfg
        obs = self._get_observations()

        # ── UE node features ───────────────────────────────────────────
        ue_x = np.stack(obs, axis=0)                        # (M, obs_dim)

        # ── UAV node features ──────────────────────────────────────────
        uav_feats = []
        max_gain  = cfg.beta0 / (cfg.H ** cfg.j)
        for n in range(cfg.N):
            pos_norm   = self.uav_pos[n] / np.array([cfg.L, cfg.W])
            congestion = np.sum(self.l_edge[n] > self.t) / cfg.M
            uav_feats.append(np.array([pos_norm[0], pos_norm[1], congestion],
                                       dtype=np.float32))
        uav_x = np.stack(uav_feats, axis=0)                 # (N, 3)

        # ── UE ↔ UAV edges (fully bipartite) ──────────────────────────
        ue_uav_src, ue_uav_dst, ue_uav_w = [], [], []

        for m in range(cfg.M):
            for n in range(cfg.N):
                gain      = self._channel_gain(m, n)
                norm_gain = float(gain / max_gain)
                ue_uav_src.append(m)
                ue_uav_dst.append(n)
                ue_uav_w.append(norm_gain)

        return {
            "ue_x":  ue_x,                                   # (M, obs_dim)
            "uav_x": uav_x,                                  # (N, 3)
            "ue_uav": (ue_uav_src, ue_uav_dst, ue_uav_w),   # M*N edges
            "n_ues":  cfg.M,
            "n_uavs": cfg.N,
            "obs_dim": self.obs_dim,
        }

    # ── Mobility ───────────────────────────────────────────────────────────

    def _move_ues(self):
        """Random-direction walk; reflect at area boundaries."""
        cfg = self.cfg
        # 30% chance of changing direction each slot
        change = np.random.random(cfg.M) < 0.3
        self.ue_angles[change] = np.random.uniform(0, 2 * np.pi, change.sum())

        dx = cfg.v * cfg.tmov * np.cos(self.ue_angles)
        dy = cfg.v * cfg.tmov * np.sin(self.ue_angles)

        new_x = self.ue_pos[:, 0] + dx
        new_y = self.ue_pos[:, 1] + dy

        # Reflect off walls
        over_x  = new_x > cfg.L;  new_x[over_x]  = 2 * cfg.L - new_x[over_x];  self.ue_angles[over_x]  = np.pi - self.ue_angles[over_x]
        under_x = new_x < 0;      new_x[under_x] = -new_x[under_x];             self.ue_angles[under_x] = np.pi - self.ue_angles[under_x]
        over_y  = new_y > cfg.W;  new_y[over_y]  = 2 * cfg.W - new_y[over_y];  self.ue_angles[over_y]  = -self.ue_angles[over_y]
        under_y = new_y < 0;      new_y[under_y] = -new_y[under_y];             self.ue_angles[under_y] = -self.ue_angles[under_y]

        self.ue_pos[:, 0] = np.clip(new_x, 0, cfg.L)
        self.ue_pos[:, 1] = np.clip(new_y, 0, cfg.W)

    # ── Task generation ────────────────────────────────────────────────────

    def _generate_tasks(self):
        """Each UE independently generates a task with probability P_task."""
        cfg   = self.cfg
        tasks = []
        for m in range(cfg.M):
            if np.random.random() < cfg.P_task:
                D        = float(np.random.uniform(cfg.D_min, cfg.D_max))
                deadline = self.t + cfg.tau - 1
                tasks.append((D, deadline))
            else:
                tasks.append(None)
        return tasks


# ─── Sanity check ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    env = NTNMECEnv()

    print(f"Obs dim   : {env.obs_dim}")
    print(f"N actions : {env.n_actions}")
    print(f"N agents  : {env.n_agents}")

    obs = env.reset()
    print(f"\nObs[0] shape : {obs[0].shape}")
    print(f"Obs[0]       : {np.round(obs[0], 3)}")

    # Random policy rollout
    total_cost = 0.0
    for step in range(env.cfg.I):
        actions = [np.random.randint(env.n_actions) for _ in range(env.n_agents)]
        obs, rewards, done, info = env.step(actions)
        total_cost += info["avg_cost"]
        if done:
            break

    print(f"\nRandom policy — mean cost per slot: {total_cost / env.cfg.I:.4f}")

    # Check graph data shapes
    graph = env.get_graph_data()
    print(f"\nGraph: ue_x={graph['ue_x'].shape}, uav_x={graph['uav_x'].shape}")
    print(f"       UE–UAV edges : {len(graph['ue_uav'][0])}")