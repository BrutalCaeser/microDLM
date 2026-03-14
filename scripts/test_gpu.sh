#!/bin/bash
# Phase 0a: GPU sanity check — verify CUDA, PyTorch, and environment work.
# Usage: sbatch scripts/test_gpu.sh
#
# This does NOT run training. It just confirms the environment is correctly
# set up before submitting any real compute jobs.

#SBATCH --job-name=microdlm-gpu-test
#SBATCH --partition=gpu-short
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/test_gpu_%j.log
#SBATCH --error=logs/test_gpu_%j.err

set -e

echo "=== microDLM GPU Environment Test ==="
echo "Date:   $(date)"
echo "Node:   $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# ---- modules ----------------------------------------------------------------
module load anaconda3/2024.06
source activate microdlm

cd ~/microDLM
mkdir -p logs

# ---- hardware check ---------------------------------------------------------
echo ""
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ---- PyTorch check ----------------------------------------------------------
echo ""
echo "--- PyTorch / CUDA ---"
python -c "
import torch
print(f'PyTorch:   {torch.__version__}')
print(f'CUDA:      {torch.version.cuda}')
print(f'Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU:       {props.name}')
    print(f'Memory:    {props.total_memory / 1e9:.1f} GB')
    # Quick tensor op to confirm compute works
    a = torch.randn(1024, 1024, device='cuda')
    b = torch.randn(1024, 1024, device='cuda')
    c = a @ b
    torch.cuda.synchronize()
    print(f'MatMul:    OK (1024x1024 on GPU)')
"

# ---- repo check -------------------------------------------------------------
echo ""
echo "--- Repo files ---"
ls -lh diffusion.py gpt.py data/shakespeare.txt weights/diffusion.pt 2>/dev/null || true

# ---- import check -----------------------------------------------------------
echo ""
echo "--- Import check ---"
python -c "
import torch, math, json, os, sys
print('stdlib + torch: OK')
# Make sure the model definition loads without errors
# We only import the module-level code (no training)
import importlib.util, types
spec = importlib.util.spec_from_file_location('diffusion', 'diffusion.py')
# Just verify the file parses cleanly
with open('diffusion.py') as f:
    compile(f.read(), 'diffusion.py', 'exec')
print('diffusion.py:   parses OK')
with open('gpt.py') as f:
    compile(f.read(), 'gpt.py', 'exec')
print('gpt.py:         parses OK')
"

echo ""
echo "=== Test PASSED — environment is ready for training ==="
echo "Next step: sbatch scripts/train_shakespeare.sh"
