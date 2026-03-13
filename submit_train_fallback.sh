#!/bin/bash
#
# Submit training with best-GPU-first fallback.
# Run on the HPC login node:  ./submit_train_fallback.sh
# (or: bash submit_train_fallback.sh)
#
# Tries each GPU type in order (best first). Submits a job; if it's still
# pending after PENDING_TIMEOUT_SEC, cancels and tries the next GPU type.
# Last resort: request generic gpu:1 (any available GPU).
#
# GPU order (most powerful first, per Explorer inventory):
#   H200 > H100 > A100 > V100-SXM2 > V100 > V100-PCIE > L40S > L40 > A6000 > A5000 > A30 > T4 > P100
#

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# How long to wait before deciding "this GPU isn't free" and trying the next (seconds)
PENDING_TIMEOUT_SEC="${PENDING_TIMEOUT_SEC:-120}"

# Best-to-worst order for training (Slurm GRES type names on Explorer)
GPU_PRIORITY=(
  h200
  h100
  a100
  v100-sxm2
  v100
  v100-pcie
  l40s
  l40
  a6000
  a5000
  a30
  t4
  p100
)

try_submit() {
  local gpu_type="$1"
  local job_id
  job_id=$(sbatch --parsable --partition=gpu --gres=gpu:${gpu_type}:1 --mem=16G --cpus-per-task=4 --time=08:00:00 --output=train_%j.log \
    --job-name=microdlm_train \
    "$SCRIPT_DIR/train.sh" 2>/dev/null) || return 1
  echo "$job_id"
}

get_state() {
  squeue -j "$1" -h -o "%t" 2>/dev/null || echo ""
}

echo "=== MicroDLM training: best-GPU-first fallback ==="
echo "Will try GPU types in order; if job still pending after ${PENDING_TIMEOUT_SEC}s, cancel and try next."
echo ""

for gpu_type in "${GPU_PRIORITY[@]}"; do
  echo "Trying GPU type: $gpu_type ..."
  job_id=$(try_submit "$gpu_type") || { echo "  (submit failed, skipping)"; continue; }
  echo "  Submitted job $job_id"

  waited=0
  while [ "$waited" -lt "$PENDING_TIMEOUT_SEC" ]; do
    state=$(get_state "$job_id")
    if [ -z "$state" ]; then
      echo "  Job $job_id no longer in queue (finished or cancelled). Exiting."
      exit 0
    fi
    if [ "$state" = "RUNNING" ] || [ "$state" = "R" ]; then
      echo "  Job $job_id is RUNNING on $gpu_type. Done."
      echo "  Monitor: squeue -u \$USER  or  tail -f train_${job_id}.log"
      exit 0
    fi
    sleep 15
    waited=$((waited + 15))
  done

  echo "  Still pending after ${PENDING_TIMEOUT_SEC}s; cancelling and trying next GPU type."
  scancel "$job_id" 2>/dev/null || true
done

echo ""
echo "No preferred GPU became available. Submitting with generic gpu:1 (any GPU) ..."
job_id=$(sbatch --parsable --partition=gpu --gres=gpu:1 --mem=16G --cpus-per-task=4 --time=08:00:00 --output=train_%j.log \
  --job-name=microdlm_train \
  "$SCRIPT_DIR/train.sh")
echo "Submitted job $job_id (will run when any GPU is free)."
echo "Monitor: squeue -u \$USER  or  tail -f train_${job_id}.log"
exit 0
