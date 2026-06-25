"""
Quick wall-clock profiler — run this BEFORE the full sweeps.

Why: these nets are tiny and the env rollout is a pure-Python per-UE loop, so
it is not obvious whether the GPU even helps, nor how many processes to pack per
device. This measures per-episode wall time for IL (train.py) and GNN-IL
(trainGnn.py) on CPU and on CUDA, then projects how long the planned grid will
take at several concurrency levels and prints a recommended launch command.

Method: each method/device is timed at two episode counts (default 5 and 20)
and the per-episode cost is the slope (t_long - t_short)/(Δepisodes), which
cancels interpreter/import/CUDA-init startup. Per-episode time is assumed to
scale ~linearly in M (env loops over UEs), anchored at the profiled M.

Usage
-----
  conda run -n rlProject python profile_experiments.py
  conda run -n rlProject python profile_experiments.py --ues 5 10 20 30 40 \
      --uavs 2 --seeds 42 52 62 72 82 --episodes 500
"""

import argparse
import os
import subprocess
import sys
import time

import torch


def time_run(script, M, N, episodes, device):
    """Wall seconds for one short training run, output suppressed."""
    argv = [sys.executable, script,
            "--n_ues", str(M), "--n_uavs", str(N),
            "--episodes", str(episodes), "--eval_episodes", "1",
            "--seed", "42", "--log_every", "100000", "--save_dir", ""]
    if device == "cpu":
        argv.append("--cpu")
    env = dict(os.environ)
    if device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = "0"
    t0 = time.perf_counter()
    r = subprocess.run(argv, env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        return None
    return dt


def per_episode(script, M, N, device, e_short, e_long):
    t_s = time_run(script, M, N, e_short, device)
    t_l = time_run(script, M, N, e_long, device)
    if t_s is None or t_l is None:
        return None
    return max(1e-6, (t_l - t_s) / (e_long - e_short))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile_m", type=int, default=20,
                    help="M to anchor the per-episode measurement at")
    ap.add_argument("--profile_n", type=int, default=2)
    ap.add_argument("--e_short", type=int, default=5)
    ap.add_argument("--e_long",  type=int, default=20)

    # The grid you intend to run (for the projection)
    ap.add_argument("--ues",   type=int, nargs="+", default=[5, 10, 20, 30, 40])
    ap.add_argument("--uavs",  type=int, nargs="+", default=[2])
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 52, 62, 72, 82])
    ap.add_argument("--episodes", type=int, default=500)
    cli = ap.parse_args()

    have_cuda = torch.cuda.is_available()
    devices = ["cpu"] + (["cuda"] if have_cuda else [])
    methods = [("IL", "train.py"), ("GNN-IL", "trainGnn.py")]

    print(f"CUDA available: {have_cuda} "
          f"({torch.cuda.device_count()} GPU(s))" if have_cuda else
          "CUDA available: False")
    print(f"Profiling at M={cli.profile_m}, N={cli.profile_n}  "
          f"(episodes {cli.e_short} vs {cli.e_long}, startup cancelled)\n")

    # ── Measure per-episode time ──────────────────────────────────────
    pe = {}  # (method, device) -> seconds/episode at profile_m
    print(f"{'method':>8}  {'device':>6}  {'ms/episode':>11}")
    print("-" * 30)
    for name, script in methods:
        for dev in devices:
            s = per_episode(script, cli.profile_m, cli.profile_n, dev,
                            cli.e_short, cli.e_long)
            pe[(name, dev)] = s
            print(f"{name:>8}  {dev:>6}  {1000*s:>9.1f}  " if s else
                  f"{name:>8}  {dev:>6}  {'FAILED':>11}")

    # ── Pick the faster device per method ─────────────────────────────
    def best_dev(name):
        cand = [(pe[(name, d)], d) for d in devices if pe.get((name, d))]
        return min(cand)[1] if cand else "cpu"

    rec_dev = best_dev("GNN-IL")   # GNN-IL is the heavier / bottleneck method
    print(f"\nFaster device for GNN-IL: {rec_dev}")
    if have_cuda and pe.get(("GNN-IL", "cpu")) and pe.get(("GNN-IL", "cuda")):
        ratio = pe[("GNN-IL", "cpu")] / pe[("GNN-IL", "cuda")]
        print(f"  GPU speedup over CPU (single process): {ratio:.2f}×  "
              f"({'GPU worth it' if ratio > 1.3 else 'marginal — CPU-parallel may win'})")

    # ── Project the planned grid ──────────────────────────────────────
    # per-episode scales ~linearly in M, anchored at profile_m.
    def run_secs(name, M):
        base = pe.get((name, rec_dev)) or pe.get((name, "cpu"))
        return base * (M / cli.profile_m) * cli.episodes

    n_cells = len(cli.ues) * len(cli.uavs) * len(cli.seeds)

    scaling_secs = sum(
        run_secs("IL", M) + run_secs("GNN-IL", M)
        for M in cli.ues for _ in cli.uavs for _ in cli.seeds
    )
    probe_secs = sum(
        run_secs("GNN-IL", M)               # probe ≈ GNN-IL + cheap offline fit
        for M in cli.ues for _ in cli.uavs for _ in cli.seeds
    )

    def fmt(sec):
        h = sec / 3600.0
        return f"{sec/60:.0f} min" if h < 1 else f"{h:.1f} h"

    print(f"\nPlanned grid: M={cli.ues}  N={cli.uavs}  "
          f"{len(cli.seeds)} seeds  →  {n_cells} cells, "
          f"{cli.episodes} episodes each")
    print("\nProjected SEQUENTIAL compute (1 process):")
    print(f"  scaling (IL+GNN): {fmt(scaling_secs)}")
    print(f"  probe (GNN only): {fmt(probe_secs)}")

    print("\nProjected WALL-CLOCK at concurrency C (≈ sequential / C, ignores "
          "contention):")
    print(f"  {'C':>3}  {'scaling':>10}  {'probe':>10}  {'both':>10}")
    for C in (1, 2, 4, 8, 12):
        print(f"  {C:>3}  {fmt(scaling_secs/C):>10}  {fmt(probe_secs/C):>10}  "
              f"{fmt((scaling_secs+probe_secs)/C):>10}")

    # ── Recommend a launch line ───────────────────────────────────────
    n_logical = os.cpu_count() or 4
    if rec_dev == "cuda":
        n_gpus = torch.cuda.device_count()
        ppg = max(1, min(4, n_logical // max(1, n_gpus)))
        gpus = ",".join(str(i) for i in range(n_gpus))
        flags = f"--gpus {gpus} --procs_per_gpu {ppg}"
        conc = n_gpus * ppg
    else:
        # CPU-bound: leave a couple cores free
        conc = max(1, n_logical - 2)
        flags = f"--gpus cpu --procs_per_gpu {conc}"

    print(f"\nDetected {n_logical} logical CPUs.")
    print(f"Recommended concurrency ≈ {conc}.")
    print("Suggested launch flags:")
    print(f"  {flags}")
    print("\nNOTE: this is a single-machine estimate; the lab server with 2 GPUs "
          "and more\ncores will be faster. Re-run this there to calibrate before "
          "the full grid.")


if __name__ == "__main__":
    main()
