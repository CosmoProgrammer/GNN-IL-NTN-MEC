# Coldnight run — server launch instructions (2026-07-06)

The cold-review controls (RESEARCH_LOG §25): CTDE diagnostic traces to decide
whether the divergence anatomy is sharing-generic or a GNN-IL artifact, the
fully-frozen-target artifact control, the corrected eps0 arm, and the
batch/buffer fairness controls. 200 jobs, priority-ordered, resumable.

## 1. Laptop — this is the ONLY step (listener.sh is live on the server)

`listener.sh` polls origin/main every 5 min and executes `run_experiments.sh`
whenever that file changes on the remote. One push does everything:

```bash
git add probe_ctde.py probe_gnn.py gnnAgent.py launch_parallel.py run_probe_sweep.py analyze_diagnight.py run_experiments.sh SERVER_COLDNIGHT.md
git commit -m "coldnight: CTDE diag probe, frozen-encoder target, corrected eps0 trigger, fairness controls + listener trigger"
git push
```

Within ≤5 min the server pulls and fires `run_experiments.sh`, which:
- refuses to start if a `launch_parallel.py` pool is already running
  (pushes that touch the trigger file while a pool runs are safe no-ops);
- auto-detects free GPUs via nvidia-smi (<2 GB used and <20% util = free):
  both free → 2×5 = 10 procs; one free → 8 procs on it (the proven setting);
  none free → 6 procs on the least-loaded GPU with a loud warning;
- launches `--plan coldnight --sync_every 900` with output in
  `parallel_logs/coldnight.out` and its GPU decision in
  `parallel_logs/coldnight_launch_info.txt` — both synced, so you can read
  them from home.

Concurrency note: the runs are CPU-bound (tiny nets) — TOTAL processes is the
knob, not per-GPU count. 8 total ran the entire 480-job program; 10 is a safe
bump when the box is all ours. Don't push past that blind: if the second GPU
is busy, its owner is probably also using CPU.

## 2. Monitoring / recovery

From home: `python sync_results.py --mode pull`, then read
`parallel_logs/STATUS.txt`, `coldnight_launch_info.txt`, `coldnight.out`
(first sync lands ≤15 min after launch). On the server tmux:
`cat parallel_logs/STATUS.txt` or `tail -f launch.log`.

If the night runs short or anything dies: **touch and re-push
`run_experiments.sh`** (e.g. bump a comment) — the listener re-fires, the
double-launch guard steps aside if a pool is still alive, and finished jobs
are skipped automatically.

## 3. Laptop (any time — the run pushes results every 15 min)

```bash
python sync_results.py --mode pull
conda run -n rlProject python analyze_diagnight.py --out analysis_diag
```

CTDE traces land in `probe_runs/diag_ctde/{plain,bn}/ue{M}/n{N}/{cell}/seed{S}/
ctde_probe_results.json`; GNN controls beside the existing arms in
`probe_runs/diag/ue{M}/n{N}/{tgtenc,eps0v2,bigbuf}/seed{S}/`.
Analyze CTDE cells with `load_diag_runs("probe_runs/diag_ctde/plain")` etc.

## What each block answers (priority order)

| # | jobs | cell | question |
|---|---|---|---|
| 1 | 20 | CTDE-plain base traces, M∈{20,30}×10 seeds | does churn-led late transient divergence appear in a shared MLP? (thesis-decider) |
| 2 | 20 | CTDE-Bn base traces | same, information-advantaged variant |
| 3 | 20 | GNN-IL `--target_encoder` | does a fully-frozen bootstrap target remove the phenomenon? (artifact control — CTDE already has full targets, so 1+3 together close it) |
| 4 | 40 | CTDE-plain eps0 + freeze_replay (corrected trigger) | do the interventions transfer to the shared MLP? |
| 5 | 40 | CTDE-Bn same arms | ditto |
| 6 | 20 | GNN-IL eps0 with corrected trigger (smoothed ≤0.8, min ep 100) | the honest explore-then-commit test — the old raw ≤0.9 trigger fired at ep 10 in all M=20 seeds |
| 7 | 40 | CTDE-plain batch=32·M + GNN-IL buffer cap 50k | batch-size and replay-window fairness confounds |

## Post-run checks (laptop, tomorrow)

- **Twin check**: `diag_ctde/{plain,bn}/.../base/seed{S}` eval_mean must equal
  the checkpoint `ctde_plain`/`ctde` runs' eval_mean exactly per seed
  (same-hardware CRN guard verification, same as the GNN twins).
- eps0v2/CTDE-arm `trigger_ep` must be ≥ 100 and post-convergence — check a
  few JSONs before quoting anything.
