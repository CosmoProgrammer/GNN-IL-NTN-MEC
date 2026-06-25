"""
Parallel job launcher with GPU distribution.

The runs in this project are embarrassingly parallel across seeds (and across M),
and each individual run is tiny (small nets, batch 32) — the wall-clock cost is
dominated by the env's per-UE Python rollout loop, which is CPU-bound. So the
real speed-up is running MANY runs concurrently across CPU cores, with each
process pinned to one of the GPUs (round-robin via CUDA_VISIBLE_DEVICES) so the
small net ops don't contend. Because the nets are so small you can usually pack
several processes per GPU (--procs_per_gpu) before the GPU is the bottleneck.

This launcher only orchestrates the existing scripts (probe_gnn.py, train.py,
trainGnn.py) as subprocesses — it adds no training logic. Finished jobs are
skipped (resumable), and each job streams to its own log file.

Plans
-----
  probe    : GNN-IL + Bₙ probe, per seed at one M  → probe_runs/ue{M}/seed{S}/
  scaling  : canonical IL + GNN-IL, per (M, seed)  → checkpoints/ue{M}/seed{S}/
  both     : scaling then probe

Examples
--------
  # the make-or-break probe sweep, 5 seeds split across 2 GPUs:
  conda run -n rlProject python launch_parallel.py --plan probe \
      --ues 20 --seeds 42 52 62 72 82 --gpus 0,1 --procs_per_gpu 2

  # re-run the whole IL + GNN-IL scaling grid on the server (one hardware):
  conda run -n rlProject python launch_parallel.py --plan scaling \
      --ues 5 10 20 30 40 --seeds 42 52 62 72 82 --gpus 0,1 --procs_per_gpu 2

  # M/N mechanism test — vary N, watch whether collapse tracks load M/N:
  conda run -n rlProject python launch_parallel.py --plan scaling \
      --ues 10 20 40 --uavs 4 --seeds 42 52 62 72 82 --gpus 0,1 --procs_per_gpu 2

Output layout (N is always encoded in the path):
  checkpoints/ue{M}/n{N}/seed{S}/{standard,gnn}/...
  probe_runs/ue{M}/n{N}/seed{S}/gnn_il_probe_results.json
"""

import argparse
import os
import subprocess
import sys
import time


# ─── Job construction ──────────────────────────────────────────────────────

def probe_jobs(ues, uavs, seeds, episodes, out_root):
    jobs = []
    for M in ues:
        for N in uavs:
            for S in seeds:
                sd = os.path.join(out_root, f"ue{M}", f"n{N}", f"seed{S}")
                jobs.append({
                    "name":  f"probe_M{M}_N{N}_s{S}",
                    "argv":  [sys.executable, "probe_gnn.py",
                              "--n_ues", str(M), "--n_uavs", str(N),
                              "--seed", str(S),
                              "--episodes", str(episodes), "--save_dir", sd],
                    # skip if the result JSON already exists
                    "done":  os.path.join(sd, "gnn_il_probe_results.json"),
                })
    return jobs


def scaling_jobs(ues, uavs, seeds, episodes, ckpt_root):
    """Mirror runMultiSeedGNNAbalation1.bat: IL (train.py) + GNN-IL (trainGnn.py)."""
    jobs = []
    for M in ues:
        for N in uavs:
            for S in seeds:
                base = os.path.join(ckpt_root, f"ue{M}", f"n{N}", f"seed{S}")
                std  = os.path.join(base, "standard")
                gnn  = os.path.join(base, "gnn")
                jobs.append({
                    "name": f"IL_M{M}_N{N}_s{S}",
                    "argv": [sys.executable, "train.py",
                             "--n_ues", str(M), "--n_uavs", str(N),
                             "--seed", str(S),
                             "--episodes", str(episodes), "--log_every", "500",
                             "--save_dir", std],
                    "done": os.path.join(std, "il_results.json"),
                })
                jobs.append({
                    "name": f"GNN_M{M}_N{N}_s{S}",
                    "argv": [sys.executable, "trainGnn.py",
                             "--n_ues", str(M), "--n_uavs", str(N),
                             "--seed", str(S),
                             "--episodes", str(episodes), "--log_every", "500",
                             "--save_dir", gnn],
                    "done": os.path.join(gnn, "gnn_il_results.json"),
                })
    return jobs


# ─── Scheduler ─────────────────────────────────────────────────────────────

def run_pool(jobs, slots, log_dir, force):
    """
    Run `jobs` across `slots` (each slot is a GPU id, or "" for CPU). At most
    len(slots) jobs run at once; each running job owns one slot and is launched
    with CUDA_VISIBLE_DEVICES set to that slot's GPU id.
    """
    os.makedirs(log_dir, exist_ok=True)

    pending = list(jobs)
    free    = list(slots)
    running = []   # (popen, slot, job, logfile)
    failed  = []

    total = len(pending)
    started = finished = skipped = 0

    while pending or running:
        # Fill free slots
        while pending and free:
            job = pending.pop(0)
            if not force and job.get("done") and os.path.exists(job["done"]):
                skipped += 1
                print(f"[skip] {job['name']} (exists: {job['done']})")
                continue
            slot = free.pop(0)
            env  = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = slot
            # CPU slot ("") → make sure the script doesn't try CUDA
            argv = list(job["argv"])
            if slot == "" and "--cpu" not in argv:
                argv.append("--cpu")
            log = open(os.path.join(log_dir, job["name"] + ".log"), "w")
            p = subprocess.Popen(argv, env=env, stdout=log,
                                  stderr=subprocess.STDOUT)
            running.append((p, slot, job, log))
            started += 1
            gpu = slot if slot != "" else "cpu"
            print(f"[start {started}/{total}] {job['name']}  GPU={gpu}  "
                  f"({len(running)} running)")

        # Reap finished
        still = []
        for p, slot, job, log in running:
            rc = p.poll()
            if rc is None:
                still.append((p, slot, job, log))
                continue
            log.close()
            free.append(slot)
            finished += 1
            tag = "ok" if rc == 0 else f"FAIL rc={rc}"
            if rc != 0:
                failed.append(job["name"])
            print(f"[done {finished}] {job['name']}  [{tag}]  "
                  f"log: {os.path.join(log_dir, job['name'] + '.log')}")
        running = still

        if running and not (pending and free):
            time.sleep(1.0)

    print(f"\nLaunched {started}, skipped {skipped}, failed {len(failed)}.")
    if failed:
        print("FAILED jobs (check their logs):", ", ".join(failed))
    return failed


# ─── CLI ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=["probe", "scaling", "both"],
                    default="probe")
    ap.add_argument("--ues",   type=int, nargs="+", default=[20])
    ap.add_argument("--uavs",  type=int, nargs="+", default=[2],
                    help="N values (UAV count) to sweep; load per server ≈ M/N")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 52, 62, 72, 82])
    ap.add_argument("--episodes", type=int, default=500)

    ap.add_argument("--gpus", type=str, default="0,1",
                    help="comma-separated GPU ids, or 'cpu' for CPU-only")
    ap.add_argument("--procs_per_gpu", type=int, default=1,
                    help="concurrent processes per GPU (nets are tiny — 2-4 ok)")

    ap.add_argument("--probe_root", type=str, default="probe_runs")
    ap.add_argument("--ckpt_root",  type=str, default="checkpoints")
    ap.add_argument("--log_dir",    type=str, default="parallel_logs")
    ap.add_argument("--force", action="store_true",
                    help="re-run jobs even if their output already exists")
    cli = ap.parse_args()

    # Build the slot list (each slot = a GPU id string, or "" for CPU)
    if cli.gpus.strip().lower() == "cpu":
        slots = [""] * cli.procs_per_gpu
    else:
        gpu_ids = [g.strip() for g in cli.gpus.split(",") if g.strip() != ""]
        slots   = [g for g in gpu_ids for _ in range(cli.procs_per_gpu)]
    print(f"Concurrency: {len(slots)} slot(s) over GPUs "
          f"[{cli.gpus}] × {cli.procs_per_gpu} proc/gpu\n")

    # Assemble jobs per plan
    jobs = []
    if cli.plan in ("scaling", "both"):
        jobs += scaling_jobs(cli.ues, cli.uavs, cli.seeds, cli.episodes,
                             cli.ckpt_root)
    if cli.plan in ("probe", "both"):
        # probe lives in the M=20 regime by default; honour whatever --ues says
        jobs += probe_jobs(cli.ues, cli.uavs, cli.seeds, cli.episodes,
                           cli.probe_root)

    failed = run_pool(jobs, slots, cli.log_dir, cli.force)

    # Auto-aggregate the probe contingency table per (M, N) cell
    if cli.plan in ("probe", "both"):
        import run_probe_sweep
        for M in cli.ues:
            for N in cli.uavs:
                out_dir = os.path.join(cli.probe_root, f"ue{M}", f"n{N}")
                print("\n" + "#" * 64)
                print(f"# PROBE CONTINGENCY  (M={M}, N={N}, load≈{M/N:.1f})")
                print("#" * 64)
                run_probe_sweep.aggregate(out_dir, cli.seeds,
                                          cost_threshold=0.9, phi_threshold=0.5)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
