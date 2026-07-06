#!/bin/bash
# run_experiments.sh — coldnight launch (2026-07-06), executed by listener.sh
# when this file changes on origin/main. See SERVER_COLDNIGHT.md for the plan.

set -u
cd "$(dirname "$0")"
mkdir -p parallel_logs
INFO="parallel_logs/coldnight_launch_info.txt"   # synced -> readable from home

log() { echo "[$(date)] $*" | tee -a "$INFO"; }

# ── Guard: never stack two pools ────────────────────────────────────────────
# The listener re-executes this script on EVERY push that touches it.
# launch_parallel is resumable but not concurrency-safe (two pools would
# double-run pending jobs), so bail if one is already going.
if pgrep -f "launch_parallel.py" > /dev/null 2>&1; then
    log "launch_parallel already running — not starting a second pool."
    exit 0
fi

# ── Conda ────────────────────────────────────────────────────────────────────
if ! command -v conda > /dev/null 2>&1; then
    for c in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
        if [ -f "$c/etc/profile.d/conda.sh" ]; then
            source "$c/etc/profile.d/conda.sh"
            break
        fi
    done
fi
if ! command -v conda > /dev/null 2>&1; then
    log "ERROR: conda not found on PATH — aborting."
    exit 1
fi

# Pick the env that actually exists on this machine (server = gnn_il).
CONDA_ENV=""
for e in gnn_il rlproject rlProject; do
    if conda env list | awk '{print $1}' | grep -qx "$e"; then
        CONDA_ENV="$e"
        break
    fi
done
if [ -z "$CONDA_ENV" ]; then
    log "ERROR: none of gnn_il/rlproject/rlProject found in conda env list — aborting."
    conda env list >> "$INFO"
    exit 1
fi
log "conda env: $CONDA_ENV"
PY="conda run -n $CONDA_ENV --no-capture-output python"

# ── GPU auto-detect ─────────────────────────────────────────────────────────
# A GPU counts as free if it has <2 GB in use and <20% utilization right now.
# The runs themselves are CPU-bound (tiny nets), so the TOTAL process count is
# the real knob, not processes-per-GPU: 8 total is the proven-good setting
# (the whole 480-job program ran at 8), 10 is a safe modest bump when the box
# is all ours. If a GPU is busy, whatever owns it is probably also eating CPU,
# so we stay at the proven 8 on the free GPU instead of pushing our luck.
FREE=""
while IFS=',' read -r idx mem util; do
    idx=$(echo "$idx" | tr -d ' '); mem=$(echo "$mem" | tr -d ' ')
    util=$(echo "$util" | tr -d ' ')
    if [ "$mem" -lt 2000 ] && [ "$util" -lt 20 ]; then
        FREE="${FREE:+$FREE,}$idx"
    fi
done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
         --format=csv,noheader,nounits)

if [ -z "$FREE" ]; then
    NFREE=0
else
    NFREE=$(( $(echo "$FREE" | tr -cd ',' | wc -c) + 1 ))
fi

if [ "$NFREE" -ge 2 ]; then
    GPUS="$FREE"; PPG=5          # 2 GPUs x 5 = 10 total
elif [ "$NFREE" -eq 1 ]; then
    GPUS="$FREE"; PPG=8          # 1 GPU  x 8 = 8 total (proven setting)
else
    # Both busy: take the least-loaded GPU, run light, and say so loudly.
    GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
           | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
    PPG=6
    log "WARNING: no free GPU found — running 6 procs on least-loaded GPU $GPUS."
fi

TOTAL=$PPG
[ "$NFREE" -ge 1 ] && TOTAL=$((PPG * NFREE))
log "coldnight launch: GPUs=$GPUS procs_per_gpu=$PPG (total $TOTAL procs)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv >> "$INFO"

# ── Launch (resumable; rerun-safe; syncs to results-live every 15 min) ──────
$PY launch_parallel.py --plan coldnight \
    --gpus "$GPUS" --procs_per_gpu "$PPG" --sync_every 900 \
    > parallel_logs/coldnight.out 2>&1
RC=$?
log "coldnight pool exited rc=$RC — see parallel_logs/STATUS.txt / coldnight.out."
