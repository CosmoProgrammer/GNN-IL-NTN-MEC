#!/usr/bin/env bash
#
# Master experiment runner for the AAAI "optimization, not coordination" study.
#
# Run it under tmux/screen (or nohup) so it survives an SSH disconnect:
#     tmux new -s rl
#     conda activate rlproject
#     bash run_all.sh 2>&1 | tee run_all.$(date +%Y%m%d_%H%M).log
#
# Every phase is resumable: launch_parallel.py skips any run whose result JSON
# already exists, so if the box reboots just re-run this script.
#
# >>> RUN profile_experiments.py FIRST <<< and paste its suggested flags below.

set -euo pipefail

# ─── EDIT after profiling ──────────────────────────────────────────────────
GPUS="0,1"            # comma-separated GPU ids, or "cpu" for CPU-only
PROCS=2               # processes per GPU (or total processes if GPUS="cpu")
SEEDS="42 52 62 72 82"
EPISODES=500
# ───────────────────────────────────────────────────────────────────────────

run() {
  echo
  echo "############################################################"
  echo "# launch_parallel.py $*"
  echo "# start: $(date)"
  echo "############################################################"
  python launch_parallel.py "$@" \
      --gpus "$GPUS" --procs_per_gpu "$PROCS" \
      --seeds $SEEDS --episodes "$EPISODES"
}

# ── Phase 1 — headline scaling grid (IL vs GNN-IL), N=2 ─────────────────────
#    Regenerates the multi-seed scaling table on THIS hardware so every paper
#    number comes from one device (fixes the cross-hardware FP-sensitivity).
run --plan scaling --ues 5 10 20 30 40 --uavs 2

# ── Phase 2 — MAKE-OR-BREAK probe at the bimodal regime (M=20, N=2) ─────────
#    Tests the stalled-encoder hypothesis. If you want this answer first,
#    move this block above Phase 1 — the probe self-classifies each seed, so
#    it does not depend on Phase 1.
run --plan probe --ues 20 --uavs 2

# ── Phase 3 (recommended) — probe across M at N=2 → φ-vs-M ──────────────────
#    Race model predicts ignition gets MORE reliable as M grows. Combined with
#    Phase 2 this gives the full φ(M) curve. Comment out to skip.
run --plan probe --ues 5 10 30 40 --uavs 2

# ── Phase 4 — M/N mechanism test (vary N) ──────────────────────────────────
#    Race model predicts collapse tracks LOAD per server ≈ M/N, not M.
#    N=4 loads:  M20→5, M40→10  (compare to N=2: M10→5, M20→10 from Phase 1/2).
run --plan scaling --ues 10 20 40 --uavs 4
run --plan probe   --ues 20 40    --uavs 4

echo
echo "############################################################"
echo "# ALL PHASES DONE: $(date)"
echo "# scaling tables : checkpoints/ue*/n*/seed*/{standard,gnn}/*_results.json"
echo "# probe tables   : probe_runs/ue*/n*/probe_sweep_summary.json"
echo "############################################################"
