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

Live results sync (--sync_every N): spawns sync_results.py beside the pool to
push all results/logs (*.json/*.npz/*.log — never *.pt weights) to the git
branch --sync_branch (default 'results-live') every N seconds + once after
final aggregation, so partial results can be pulled and analysed elsewhere
while the night is still running. See sync_results.py for the mechanism
(orphan commits, worktree/index/HEAD untouched) and the laptop-side pull.

Plans
-----
  probe     : GNN-IL + Bₙ probe            → probe_runs/ue{M}/n{N}/seed{S}/
  scaling   : canonical IL + GNN-IL        → checkpoints/ue{M}/n{N}/seed{S}/
  race      : probe with modified ε schedules (the race model's CAUSAL test)
                                           → probe_runs/ue{M}/n{N}/eps{D}[_floor{E}]/seed{S}/
  warmstart : probe at M=20 with the encoder warm-started from the SAME seed's
              trained M=30 donor (fix #2 + curriculum #3 in one; jobs wait
              for their donor checkpoint) → probe_runs/ue20/n{N}/warmstart/seed{S}/
  both      : scaling then probe
  overnight : everything the race model needs, priority-ordered so an early
              stop still leaves the make-or-break results on disk:
                1. probe M=20 (the 2×2 contingency)      5 jobs
                2. probe M=30 (reliability + donors)     5 jobs
                3. race grid  M=20 × {.999, .9975, .99, floor .3}
                              M=30 × {.99, .98}         30 jobs
                4. warmstart  M=30 → M=20                5 jobs
                5. probe M ∈ {5,10,40} (φ-vs-M curve)   15 jobs
              (ignores --ues; honours --seeds/--uavs/--episodes)
  diagnight : the session-3 run — EVERYTHING launchable now in ONE night
              (RESEARCH_LOG S18.4-18.5); only resurrection (needs an analysed
              pre-divergence checkpoint) is deferred. Priority-ordered blocks:
                1. anatomy arms  probe_gnn --diag_every 10 (greedy CRN eval,
                   Q/TD/grad/congestion logging), M ∈ {20,30} × 10 seeds ×
                   {base, floor .1-.4, eps0, freeze_replay, freeze_learn,
                    tgt 5/50}                                   200 jobs
                2. 2x2 cells     CTDE(Bn) + CTDE-plain (--env plain, the
                   information-matched shared-MLP-IL cell) + GNN-NoShare,
                   M ∈ {5,10,20,30,40} × 10 seeds               150 jobs
                3. consolidation de-confound: eps-decay {.9975,.999} at
                   2x episodes with diagnostics, M=20 × 10 seeds 20 jobs
                4. scaling       IL + GNN-IL same grid as 2 (10-seed,
                   same-hardware baselines)                     100 jobs
                5. PoA deep      best-response, 20 restarts, N ∈ {2,4},
                   M ∈ {5,10,20,30,40}                           10 jobs
              Defaults (when --ues/--seeds are untouched): ~480 jobs
              (the Jul-3 overnight instance predates the ctde_plain jobs
              and ran ~430; a resumable rerun adds only the new 50).
  coldnight : the cold-review controls (RESEARCH_LOG §25) — decides whether
              the divergence anatomy is sharing-generic or a GNN-IL artifact,
              and de-confounds the 2×2. Priority-ordered blocks
              (M ∈ {20,30} × 10 seeds unless noted):
                1. CTDE-plain diag base traces (probe_ctde)       20 jobs
                2. CTDE-Bn    diag base traces                    20 jobs
                3. GNN-IL fully-frozen target (--target_encoder)  20 jobs
                4. CTDE-plain arms: eps0(corrected trigger) +
                   freeze_replay                                  40 jobs
                5. CTDE-Bn    arms: same                          40 jobs
                6. GNN-IL eps0 with CORRECTED trigger (smoothed
                   <=0.8, min ep 100 — the old raw<=0.9 fired at
                   ep 10 in all M=20 seeds)                       20 jobs
                7. fairness: CTDE-plain batch=32*M + GNN-IL
                   buffer window matched to CTDE (cap 50k)        40 jobs
              ~200 jobs; resumable; blocks 1–3 are the thesis-deciders.

Examples
--------
  # THE session-3 anatomy run (~200 jobs, resumable):
  conda run -n rlProject python launch_parallel.py --plan diagnight \
      --gpus 0,1 --procs_per_gpu 4

  # the (falsified) race-model overnight run (~60 jobs, resumable):
  conda run -n rlProject python launch_parallel.py --plan overnight \
      --gpus 0,1 --procs_per_gpu 4

  # just the make-or-break probe sweep, 5 seeds split across 2 GPUs:
  conda run -n rlProject python launch_parallel.py --plan probe \
      --ues 20 --seeds 42 52 62 72 82 --gpus 0,1 --procs_per_gpu 2

  # custom exploration-schedule cells:
  conda run -n rlProject python launch_parallel.py --plan race \
      --ues 20 --eps_decays 0.999 0.99 --eps_ends 0.3 --gpus 0,1

  # M/N mechanism test — vary N, watch whether collapse tracks load M/N:
  conda run -n rlProject python launch_parallel.py --plan scaling \
      --ues 10 20 40 --uavs 4 --seeds 42 52 62 72 82 --gpus 0,1 --procs_per_gpu 2

Output layout (N is always encoded in the path):
  checkpoints/ue{M}/n{N}/seed{S}/{standard,gnn}/...
  probe_runs/ue{M}/n{N}/[eps{D}[_floor{E}]/|warmstart/]seed{S}/gnn_il_probe_results.json
  probe_runs/diag/ue{M}/n{N}/{arm}/seed{S}/gnn_il_probe_results.json
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
                cell = os.path.join(out_root, f"ue{M}", f"n{N}")
                sd   = os.path.join(cell, f"seed{S}")
                jobs.append({
                    "name":  f"probe_M{M}_N{N}_s{S}",
                    "argv":  [sys.executable, "probe_gnn.py",
                              "--n_ues", str(M), "--n_uavs", str(N),
                              "--seed", str(S),
                              "--episodes", str(episodes), "--save_dir", sd],
                    # skip if the result JSON already exists
                    "done":  os.path.join(sd, "gnn_il_probe_results.json"),
                    "cell":  cell,     # contingency table is aggregated per cell
                })
    return jobs


def race_jobs(schedules, uavs, seeds, episodes, out_root):
    """
    Exploration-schedule interventions — the race model's CAUSAL test.

    `schedules` is a list of (M, eps_decay, eps_end) cells. Each cell re-runs
    the probe with a modified ε schedule and everything else identical:
      * slower decay / higher floor → larger exploration budget ∫ε dt →
        predicted to IGNITE the encoder and eliminate collapse (e.g. M=20);
      * faster decay → smaller budget → predicted to INDUCE collapse where the
        standard schedule converges (e.g. M=30) — the sharpest falsifiable
        prediction: the bimodal boundary should MOVE with the budget.
    """
    jobs = []
    for (M, dec, end) in schedules:
        tag = f"eps{dec:g}" + (f"_floor{end:g}" if end != 0.05 else "")
        for N in uavs:
            for S in seeds:
                cell = os.path.join(out_root, f"ue{M}", f"n{N}", tag)
                sd   = os.path.join(cell, f"seed{S}")
                jobs.append({
                    "name":  f"race_M{M}_N{N}_{tag}_s{S}",
                    "argv":  [sys.executable, "probe_gnn.py",
                              "--n_ues", str(M), "--n_uavs", str(N),
                              "--seed", str(S), "--episodes", str(episodes),
                              "--eps_decay", str(dec), "--eps_end", str(end),
                              "--save_dir", sd],
                    "done":  os.path.join(sd, "gnn_il_probe_results.json"),
                    "cell":  cell,
                })
    return jobs


def warmstart_jobs(uavs, seeds, episodes, out_root, target_m=20, donor_m=30):
    """
    Race-model fix #2 (warm start) + #3 (curriculum) in one experiment: train
    at target_m with the encoder initialised from the SAME seed's trained
    donor_m encoder (GNN weights are M-agnostic). Prediction: warm-started
    seeds start above the ignition threshold φ_c, so convergence becomes
    initialisation-independent wherever the donor ignited. A collapsed donor
    is informative too (does collapse transfer?).

    Each job "requires" its donor checkpoint: the scheduler defers it until
    the donor probe run (priority 2 in the overnight plan) has finished.
    """
    jobs = []
    for N in uavs:
        for S in seeds:
            donor = os.path.join(out_root, f"ue{donor_m}", f"n{N}",
                                 f"seed{S}", "gnn_il_probe_final.pt")
            cell  = os.path.join(out_root, f"ue{target_m}", f"n{N}", "warmstart")
            sd    = os.path.join(cell, f"seed{S}")
            jobs.append({
                "name":     f"warm_M{target_m}from{donor_m}_N{N}_s{S}",
                "argv":     [sys.executable, "probe_gnn.py",
                             "--n_ues", str(target_m), "--n_uavs", str(N),
                             "--seed", str(S), "--episodes", str(episodes),
                             "--init_encoder", donor,
                             "--save_dir", sd],
                "done":     os.path.join(sd, "gnn_il_probe_results.json"),
                "requires": donor,
                "cell":     cell,
            })
    return jobs


def diag_jobs(ues, uavs, seeds, episodes, out_root, diag_every=10):
    """
    Session-3 late-collapse anatomy (RESEARCH_LOG S18.4). Every job runs the
    probe with the diagnostics suite ON (--diag_every): greedy CRN eval,
    |Q| stats on a fixed state batch, TD-error percentiles, pre-clip grad
    norms, congestion occupancy. Arms:
      * base / floor* / tgt*  — no trigger, schedule/cadence set from ep 1;
      * eps0, freeze_replay, freeze_learn — fire once greedy eval < 0.9.
    Arm-major priority order: if the night is cut short, completed arms are
    complete across both M and all seeds.
    """
    arms = [
        ("base",         []),                              # ground truth
        ("floor0.1",     ["--eps_end", "0.1"]),            # dose-response...
        ("floor0.2",     ["--eps_end", "0.2"]),
        ("floor0.3",     ["--eps_end", "0.3"]),
        ("floor0.4",     ["--eps_end", "0.4"]),            # ...phase boundary
        ("eps0",         ["--arm", "eps0"]),
        ("freezereplay", ["--arm", "freeze_replay"]),
        ("freezelearn",  ["--arm", "freeze_learn"]),
        ("tgt5",         ["--target_update", "5"]),
        ("tgt50",        ["--target_update", "50"]),
    ]
    jobs = []
    for tag, extra in arms:
        for M in ues:
            for N in uavs:
                for S in seeds:
                    cell = os.path.join(out_root, "diag", f"ue{M}", f"n{N}", tag)
                    sd   = os.path.join(cell, f"seed{S}")
                    jobs.append({
                        "name":  f"diag_{tag}_M{M}_N{N}_s{S}",
                        "argv":  [sys.executable, "probe_gnn.py",
                                  "--n_ues", str(M), "--n_uavs", str(N),
                                  "--seed", str(S), "--episodes", str(episodes),
                                  "--diag_every", str(diag_every),
                                  "--save_dir", sd] + extra,
                        "done":  os.path.join(sd, "gnn_il_probe_results.json"),
                        "cell":  cell,
                    })
    return jobs


def ctde_diag_jobs(env_variant, ues, uavs, seeds, episodes, out_root, arms,
                   diag_every=10):
    """
    Cold-review controls: the SAME diagnostic suite probe_gnn runs, on the
    shared-MLP cells (probe_ctde.py). env_variant: 'plain' (env.py,
    information-matched) or 'bl' (envWithBL, CTDE-Bn). `arms` is a list of
    (tag, extra_argv) cells. No 'cell' key: run_probe_sweep's φ-contingency
    aggregation doesn't apply to CTDE (no encoder probe).
    """
    sub = "plain" if env_variant == "plain" else "bn"
    jobs = []
    for tag, extra in arms:
        for M in ues:
            for N in uavs:
                for S in seeds:
                    sd = os.path.join(out_root, "diag_ctde", sub,
                                      f"ue{M}", f"n{N}", tag, f"seed{S}")
                    jobs.append({
                        "name": f"cdiag_{sub}_{tag}_M{M}_N{N}_s{S}",
                        "argv": [sys.executable, "probe_ctde.py",
                                 "--env", env_variant,
                                 "--n_ues", str(M), "--n_uavs", str(N),
                                 "--seed", str(S), "--episodes", str(episodes),
                                 "--diag_every", str(diag_every),
                                 "--save_dir", sd] + extra,
                        "done": os.path.join(sd, "ctde_probe_results.json"),
                    })
    return jobs


def gnn_control_jobs(cells, ues, uavs, seeds, episodes, out_root,
                     diag_every=10):
    """
    Cold-review GNN-side controls, run through probe_gnn with diagnostics on.
    `cells` is a list of (tag, extra_argv); outputs land beside the existing
    diag arms (probe_runs/diag/ue{M}/n{N}/{tag}/seed{S}/) so the analyzer's
    arm table picks them up automatically.
    """
    jobs = []
    for tag, extra in cells:
        for M in ues:
            for N in uavs:
                for S in seeds:
                    cell = os.path.join(out_root, "diag", f"ue{M}", f"n{N}", tag)
                    sd   = os.path.join(cell, f"seed{S}")
                    jobs.append({
                        "name":  f"gctl_{tag}_M{M}_N{N}_s{S}",
                        "argv":  [sys.executable, "probe_gnn.py",
                                  "--n_ues", str(M), "--n_uavs", str(N),
                                  "--seed", str(S), "--episodes", str(episodes),
                                  "--diag_every", str(diag_every),
                                  "--save_dir", sd] + extra,
                        "done":  os.path.join(sd, "gnn_il_probe_results.json"),
                        "cell":  cell,
                    })
    return jobs


def twobytwo_jobs(ues, uavs, seeds, episodes, ckpt_root):
    """
    The sharing-vs-graph 2x2, multi-seed (RESEARCH_LOG S18.5): does parameter
    sharing ALONE produce the bimodal collapse, or does it need the graph
    coupling? The four cells:
      IL          : no sharing, no graph   (from scaling_jobs)
      CTDE-plain  : sharing,    no graph   (ctdeAgent IS shared-MLP IL:
                                            one shared DQN + shared buffer,
                                            local obs, no central critic —
                                            `--env plain` for information
                                            parity with IL/GNN-IL)
      GNN-NoShare : no sharing, graph
      GNN-IL      : sharing,    graph      (from scaling_jobs)
    This builder contributes the CTDE and NoShare cells. NOTE: trainCtde.py
    was historically hardcoded to envWithBL, so the plain `ctde/` cell run on
    the night of Jul 3 is actually CTDE(Bn) — kept as the multi-seed fairness
    ablation; `ctde_plain/` is the true 2x2 cell (added Jul 3, late).
    """
    jobs = []
    for M in ues:
        for N in uavs:
            for S in seeds:
                base = os.path.join(ckpt_root, f"ue{M}", f"n{N}", f"seed{S}")
                ctde = os.path.join(base, "ctde")
                ctdp = os.path.join(base, "ctde_plain")
                nosh = os.path.join(base, "noshare")
                jobs.append({
                    "name": f"CTDE_M{M}_N{N}_s{S}",
                    "argv": [sys.executable, "trainCtde.py",
                             "--n_ues", str(M), "--n_uavs", str(N),
                             "--seed", str(S),
                             "--episodes", str(episodes), "--log_every", "25",
                             "--save_dir", ctde],
                    "done": os.path.join(ctde, "ctde_results.json"),
                })
                jobs.append({
                    "name": f"CTDEP_M{M}_N{N}_s{S}",
                    "argv": [sys.executable, "trainCtde.py",
                             "--n_ues", str(M), "--n_uavs", str(N),
                             "--seed", str(S), "--env", "plain",
                             "--episodes", str(episodes), "--log_every", "25",
                             "--save_dir", ctdp],
                    "done": os.path.join(ctdp, "ctde_results.json"),
                })
                jobs.append({
                    "name": f"NOSH_M{M}_N{N}_s{S}",
                    "argv": [sys.executable, "trainGnnNoShare.py",
                             "--n_ues", str(M), "--n_uavs", str(N),
                             "--seed", str(S),
                             "--episodes", str(episodes), "--log_every", "25",
                             "--save_dir", nosh],
                    "done": os.path.join(nosh, "gnn_noshare_results.json"),
                })
    return jobs


def extended_jobs(uavs, seeds, episodes, out_root, diag_every=10,
                  cells=((20, 0.9975), (20, 0.999))):
    """
    Consolidation de-confound (RESEARCH_LOG S18.1 item 2): the Jul-2 race grid
    confounded exploration BUDGET with CONSOLIDATION TIME — slow-decay cells
    ended training at eps 0.29-0.61, never getting a near-greedy phase. Re-run
    those schedules at 2x episodes (default 1000) with the diagnostics suite
    on, so the greedy CRN eval curve separates "policy is bad" from "training
    cost is high because eps is high".
    """
    ep_ext = 2 * episodes
    jobs = []
    for (M, dec) in cells:
        tag = f"eps{dec:g}_ep{ep_ext}"
        for N in uavs:
            for S in seeds:
                cell = os.path.join(out_root, "diag", f"ue{M}", f"n{N}", tag)
                sd   = os.path.join(cell, f"seed{S}")
                jobs.append({
                    "name":  f"ext_{tag}_M{M}_N{N}_s{S}",
                    "argv":  [sys.executable, "probe_gnn.py",
                              "--n_ues", str(M), "--n_uavs", str(N),
                              "--seed", str(S), "--episodes", str(ep_ext),
                              "--eps_decay", str(dec),
                              "--diag_every", str(diag_every),
                              "--save_dir", sd],
                    "done":  os.path.join(sd, "gnn_il_probe_results.json"),
                    "cell":  cell,
                })
    return jobs


def poa_jobs(ues, out_root, starts=20):
    """
    PoA hardening: deep best-response equilibrium search (20 restarts) on the
    real env, N=2 and N=4, one job per (M, N). poa_best_response.py has no
    --cpu flag → mark no_cpu_flag so CPU slots don't pass it one.
    """
    jobs = []
    for N, sub in ((2, "poa_deep"), (4, "poa_n4deep")):
        od = os.path.join(out_root, sub)
        for M in ues:
            jobs.append({
                "name": f"poa_M{M}_N{N}_deep",
                "argv": [sys.executable, "poa_best_response.py",
                         "--ues", str(M), "--n_uavs", str(N),
                         "--pne_starts", str(starts), "--out", od],
                "done": os.path.join(od, f"poa_M{M}_N{N}.json"),
                "no_cpu_flag": True,
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
                             "--episodes", str(episodes), "--log_every", "25",
                             "--save_dir", std],
                    "done": os.path.join(std, "il_results.json"),
                })
                jobs.append({
                    "name": f"GNN_M{M}_N{N}_s{S}",
                    "argv": [sys.executable, "trainGnn.py",
                             "--n_ues", str(M), "--n_uavs", str(N),
                             "--seed", str(S),
                             "--episodes", str(episodes), "--log_every", "25",
                             "--save_dir", gnn],
                    "done": os.path.join(gnn, "gnn_il_results.json"),
                })
    return jobs


# ─── Scheduler ─────────────────────────────────────────────────────────────

def _fmt_dur(sec):
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _write_status(path, total, skipped, finished, failed, running, t0, eta):
    """Overwrite a single-glance status file the user can `watch cat`/`tail`."""
    done = skipped + finished
    lines = [
        f"updated : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"progress: {done}/{total} resolved  "
        f"({finished} ran, {skipped} skipped, {len(failed)} failed)",
        f"elapsed : {_fmt_dur(time.time() - t0)}",
        f"eta     : {('~' + _fmt_dur(eta)) if eta is not None else '—'} remaining",
        f"running : {', '.join(j['name'] for _, _, j, _, _ in running) or '(none)'}",
    ]
    if failed:
        lines.append(f"FAILED  : {', '.join(failed)}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_pool(jobs, slots, log_dir, force):
    """
    Run `jobs` across `slots` (each slot is a GPU id, or "" for CPU). At most
    len(slots) jobs run at once; each running job owns one slot and is launched
    with CUDA_VISIBLE_DEVICES set to that slot's GPU id.

    Logging: child stdout/stderr stream UNBUFFERED to per-job log files
    (PYTHONUNBUFFERED=1) so `tail -f log_dir/<job>.log` shows live progress.
    A rolling ETA and a STATUS.txt give overall completion + time remaining.
    """
    os.makedirs(log_dir, exist_ok=True)
    status_path = os.path.join(log_dir, "STATUS.txt")

    pending = list(jobs)
    free    = list(slots)
    running = []   # (popen, slot, job, logfile, start_time)
    failed  = []

    total = len(pending)
    started = finished = skipped = 0
    durations = []                  # wall time of each completed run, for ETA
    t0 = time.time()
    C = max(1, len(slots))

    def eta():
        if not durations:
            return None
        avg = sum(durations) / len(durations)
        remaining = total - skipped - finished      # pending + running runs
        return remaining * avg / C

    while pending or running:
        # Fill free slots. Jobs whose "requires" file is missing (e.g. a
        # warm-start donor checkpoint still training) are deferred to the back
        # of the queue; `attempts` bounds one pass so we can't spin forever.
        attempts = len(pending)
        while pending and free and attempts > 0:
            attempts -= 1
            job = pending.pop(0)
            if not force and job.get("done") and os.path.exists(job["done"]):
                skipped += 1
                print(f"[skip] {job['name']} (exists: {job['done']})", flush=True)
                _write_status(status_path, total, skipped, finished, failed,
                              running, t0, eta())
                continue
            req = job.get("requires")
            if req and not os.path.exists(req):
                pending.append(job)         # producer not done yet — try later
                continue
            slot = free.pop(0)
            env  = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = slot
            env["PYTHONUNBUFFERED"] = "1"     # live, unbuffered child log files
            # CPU slot ("") → make sure the script doesn't try CUDA
            argv = list(job["argv"])
            if slot == "" and "--cpu" not in argv and not job.get("no_cpu_flag"):
                argv.append("--cpu")
            log = open(os.path.join(log_dir, job["name"] + ".log"), "w")
            p = subprocess.Popen(argv, env=env, stdout=log,
                                  stderr=subprocess.STDOUT)
            running.append((p, slot, job, log, time.time()))
            started += 1
            gpu = slot if slot != "" else "cpu"
            print(f"[start {started}/{total}] {job['name']}  GPU={gpu}  "
                  f"({len(running)} running)", flush=True)
            _write_status(status_path, total, skipped, finished, failed,
                          running, t0, eta())

        # Reap finished
        still = []
        for p, slot, job, log, st in running:
            rc = p.poll()
            if rc is None:
                still.append((p, slot, job, log, st))
                continue
            log.close()
            free.append(slot)
            finished += 1
            durations.append(time.time() - st)
            tag = "ok" if rc == 0 else f"FAIL rc={rc}"
            if rc != 0:
                failed.append(job["name"])
            e = eta()
            print(f"[done {skipped+finished}/{total}] {job['name']}  [{tag}]  "
                  f"({_fmt_dur(time.time()-st)})  "
                  f"elapsed {_fmt_dur(time.time()-t0)}  "
                  f"eta {('~'+_fmt_dur(e)) if e is not None else '—'}", flush=True)
            _write_status(status_path, total, skipped, finished, failed,
                          still, t0, e)
        running = still

        # Deadlock: nothing is running and EVERY remaining job is blocked on a
        # "requires" file that no running job can produce any more (its
        # producer failed or was never scheduled). Drop them.
        if pending and not running:
            blocked = [j for j in pending if j.get("requires")
                       and not os.path.exists(j["requires"])]
            if len(blocked) == len(pending):
                for job in pending:
                    failed.append(job["name"])
                    print(f"[drop] {job['name']} — requires "
                          f"{job.get('requires')} which nothing running can "
                          f"produce (producer failed?)", flush=True)
                pending.clear()

        if running and not (pending and free):
            time.sleep(1.0)

    _write_status(status_path, total, skipped, finished, failed, [], t0, 0)
    print(f"\nLaunched {started}, skipped {skipped}, failed {len(failed)}  "
          f"in {_fmt_dur(time.time()-t0)}.", flush=True)
    if failed:
        print("FAILED jobs (check their logs):", ", ".join(failed))
    return failed


# ─── CLI ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=["probe", "scaling", "race", "warmstart",
                                       "both", "overnight", "diagnight",
                                       "coldnight"],
                    default="probe")
    ap.add_argument("--ues",   type=int, nargs="+", default=[20])
    ap.add_argument("--uavs",  type=int, nargs="+", default=[2],
                    help="N values (UAV count) to sweep; load per server ≈ M/N")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 52, 62, 72, 82])
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--eps_decays", type=float, nargs="+", default=[],
                    help="race plan: ε-decay variants to run (floor stays 0.05)")
    ap.add_argument("--eps_ends", type=float, nargs="+", default=[],
                    help="race plan: ε-floor variants to run (decay stays 0.995)")

    ap.add_argument("--gpus", type=str, default="0,1",
                    help="comma-separated GPU ids, or 'cpu' for CPU-only")
    ap.add_argument("--procs_per_gpu", type=int, default=1,
                    help="concurrent processes per GPU (nets are tiny — 2-4 ok)")

    ap.add_argument("--probe_root", type=str, default="probe_runs")
    ap.add_argument("--ckpt_root",  type=str, default="checkpoints")
    ap.add_argument("--log_dir",    type=str, default="parallel_logs")
    ap.add_argument("--force", action="store_true",
                    help="re-run jobs even if their output already exists")
    ap.add_argument("--dry_run", action="store_true",
                    help="print the job list (name + done-file, in priority "
                         "order) and exit without launching anything")
    ap.add_argument("--sync_every", type=int, default=0,
                    help="if >0, spawn sync_results.py beside the pool to "
                         "push results/logs (*.json/*.npz/*.log — never *.pt) "
                         "to the git branch --sync_branch every N seconds, "
                         "plus a final push after aggregation. Requires "
                         "`git push` to be authorized from this clone.")
    ap.add_argument("--sync_branch", type=str, default="results-live")
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
    seeds = list(cli.seeds)     # diagnight may widen this (used in aggregation)
    if cli.plan == "diagnight":
        # Rate comparisons across arms need statistical power: when the CLI
        # defaults are untouched, widen to 10 seeds and M ∈ {20, 30} for the
        # anatomy arms (all_m covers the full 2×2/scaling/PoA sweep).
        ues   = cli.ues if cli.ues != [20] else [20, 30]
        all_m = cli.ues if cli.ues != [20] else [5, 10, 20, 30, 40]
        if seeds == [42, 52, 62, 72, 82]:
            seeds += [92, 102, 112, 122, 132]
        # ONE night, priority-ordered — only resurrection (needs an analysed
        # pre-divergence checkpoint) is deferred to a later run.
        jobs += diag_jobs(ues, cli.uavs, seeds, cli.episodes,
                          cli.probe_root)                    # 1. anatomy arms
        jobs += twobytwo_jobs(all_m, cli.uavs, seeds, cli.episodes,
                              cli.ckpt_root)                 # 2. CTDE+NoShare
        jobs += extended_jobs(cli.uavs, seeds, cli.episodes,
                              cli.probe_root)                # 3. consolidation
        jobs += scaling_jobs(all_m, cli.uavs, seeds, cli.episodes,
                             cli.ckpt_root)                  # 4. IL+GNN 10-seed
        jobs += poa_jobs(all_m, cli.probe_root)              # 5. PoA deep N=2/4
    if cli.plan == "coldnight":
        # Cold-review controls (RESEARCH_LOG §25). Same widening convention
        # as diagnight: untouched CLI → M ∈ {20,30} × 10 seeds.
        ues = cli.ues if cli.ues != [20] else [20, 30]
        if seeds == [42, 52, 62, 72, 82]:
            seeds += [92, 102, 112, 122, 132]
        # Corrected trigger: fire on the smoothed (median-of-3) CRN eval
        # crossing 0.8 — the v2-label convergence bar — and never before
        # ep 100. The historical raw<=0.9 trigger fired at ep 10 in ALL ten
        # M=20 eps0 seeds (a near-untrained policy already evals <=0.9).
        corrected = ["--trigger_mode", "smoothed",
                     "--trigger_cost", "0.8",
                     "--trigger_min_ep", "100"]
        base_arm  = [("base", [])]
        ctde_arms = [("eps0",         ["--arm", "eps0"] + corrected),
                     ("freezereplay", ["--arm", "freeze_replay"] + corrected)]
        jobs += ctde_diag_jobs("plain", ues, cli.uavs, seeds, cli.episodes,
                               cli.probe_root, base_arm)     # 1. thesis-decider
        jobs += ctde_diag_jobs("bl", ues, cli.uavs, seeds, cli.episodes,
                               cli.probe_root, base_arm)     # 2. info-matched
        jobs += gnn_control_jobs([("tgtenc", ["--target_encoder"])],
                                 ues, cli.uavs, seeds, cli.episodes,
                                 cli.probe_root)             # 3. artifact ctrl
        jobs += ctde_diag_jobs("plain", ues, cli.uavs, seeds, cli.episodes,
                               cli.probe_root, ctde_arms)    # 4. CTDE arms
        jobs += ctde_diag_jobs("bl", ues, cli.uavs, seeds, cli.episodes,
                               cli.probe_root, ctde_arms)    # 5. CTDE-Bn arms
        jobs += gnn_control_jobs([("eps0v2",
                                   ["--arm", "eps0"] + corrected)],
                                 ues, cli.uavs, seeds, cli.episodes,
                                 cli.probe_root)             # 6. eps0 redone
        jobs += ctde_diag_jobs("plain", ues, cli.uavs, seeds, cli.episodes,
                               cli.probe_root,
                               [("batchM", ["--batch_scale_m"])])  # 7a. fairness
        jobs += gnn_control_jobs([("bigbuf", ["--buffer_cap", "50000"])],
                                 ues, cli.uavs, seeds, cli.episodes,
                                 cli.probe_root)             # 7b. fairness
    if cli.plan in ("scaling", "both"):
        jobs += scaling_jobs(cli.ues, cli.uavs, cli.seeds, cli.episodes,
                             cli.ckpt_root)
    if cli.plan in ("probe", "both"):
        # probe lives in the M=20 regime by default; honour whatever --ues says
        jobs += probe_jobs(cli.ues, cli.uavs, cli.seeds, cli.episodes,
                           cli.probe_root)
    if cli.plan == "race":
        scheds = ([(M, d, 0.05)  for M in cli.ues for d in cli.eps_decays] +
                  [(M, 0.995, e) for M in cli.ues for e in cli.eps_ends])
        if not scheds:
            ap.error("--plan race needs --eps_decays and/or --eps_ends")
        jobs += race_jobs(scheds, cli.uavs, cli.seeds, cli.episodes,
                          cli.probe_root)
    if cli.plan == "warmstart":
        jobs += warmstart_jobs(cli.uavs, cli.seeds, cli.episodes,
                               cli.probe_root)
    if cli.plan == "overnight":
        # Priority-ordered (FIFO scheduler): if the night runs short, the
        # make-or-break results are already on disk. Resumable via re-run.
        jobs += probe_jobs([20], cli.uavs, cli.seeds, cli.episodes,
                           cli.probe_root)              # 1. the 2×2 contingency
        jobs += probe_jobs([30], cli.uavs, cli.seeds, cli.episodes,
                           cli.probe_root)              # 2. donors + reliability
        scheds = [(20, 0.999,  0.05),   # budget ∫ε ≈ 393 ep-units (vs 184 std)
                  (20, 0.9975, 0.05),   # ≈ 285
                  (20, 0.99,   0.05),   # ≈  99  → predicted MORE collapse
                  (20, 0.995,  0.30),   # ≈ 218, late-shaped: floor-vs-seed test
                  (30, 0.99,   0.05),   # ≈  99  → predicted to INDUCE collapse
                  (30, 0.98,   0.05)]   # ≈  65  → stronger induction
        jobs += race_jobs(scheds, cli.uavs, cli.seeds, cli.episodes,
                          cli.probe_root)               # 3. causal test
        jobs += warmstart_jobs(cli.uavs, cli.seeds, cli.episodes,
                               cli.probe_root)          # 4. fix #2/#3
        jobs += probe_jobs([5, 10, 40], cli.uavs, cli.seeds, cli.episodes,
                           cli.probe_root)              # 5. φ-vs-M curve

    if cli.dry_run:
        print(f"DRY RUN — {len(jobs)} jobs in priority order:\n")
        for i, j in enumerate(jobs, 1):
            state = "DONE" if (j.get("done") and os.path.exists(j["done"])) \
                    else "pending"
            print(f"{i:>4}. [{state:>7}] {j['name']}")
        n_done = sum(1 for j in jobs
                     if j.get("done") and os.path.exists(j["done"]))
        print(f"\n{len(jobs) - n_done} to run, {n_done} already resolved.")
        return

    # Live results sync: push-only sidecar, safe beside the pool (orphan
    # commits on a dedicated branch; never touches worktree/index/HEAD).
    sync_proc = None
    sync_argv = [sys.executable, "sync_results.py", "--mode", "push",
                 "--branch", cli.sync_branch,
                 "--roots", cli.probe_root, cli.ckpt_root, cli.log_dir]
    if cli.sync_every > 0:
        sync_proc = subprocess.Popen(
            sync_argv + ["--loop", "--interval", str(cli.sync_every)])
        print(f"Results sync: every {cli.sync_every}s -> branch "
              f"'{cli.sync_branch}' (pull on the laptop with: "
              f"python sync_results.py --mode pull)\n")

    failed = run_pool(jobs, slots, cli.log_dir, cli.force)

    # Auto-aggregate the probe contingency table for every probe-style cell
    cells = []
    for j in jobs:
        c = j.get("cell")
        if c and c not in cells:
            cells.append(c)
    if cells:
        import run_probe_sweep
        for c in cells:
            print("\n" + "#" * 64)
            print(f"# PROBE CONTINGENCY  ({c})")
            print("#" * 64)
            run_probe_sweep.aggregate(c, seeds,
                                      cost_threshold=0.9, phi_threshold=0.05)

    # Final sync AFTER aggregation so the per-cell summary JSONs ship too.
    if cli.sync_every > 0:
        if sync_proc is not None:
            sync_proc.terminate()
            try:
                sync_proc.wait(timeout=30)
            except Exception:
                pass
        subprocess.call(sync_argv)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
