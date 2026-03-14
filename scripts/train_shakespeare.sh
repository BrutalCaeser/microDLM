#!/bin/bash
# Phase 0b: Validate existing microDLM training on HPC — Shakespeare dataset.
#
# Goal: confirm the code runs end-to-end on a real GPU with zero modifications.
# This is a VALIDATION run (10 000 iters, same as local training).
# If this completes and produces weights/, Phase 0 is done — then scale.
#
# Usage: sbatch scripts/train_shakespeare.sh
#
# Expected output (A100, ~10 min):
#   step     0: train ~3.50, val ~3.50
#   step  9999: train ~1.50, val ~1.58
#   Saved weights/diffusion.pt
#   Saved weights/gpt.pt

#SBATCH --job-name=microdlm-shakespeare
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/shakespeare_%j.log
#SBATCH --error=logs/shakespeare_%j.err

set -e

echo "=== microDLM Shakespeare Training (Phase 0 Validation) ==="
echo "Date:   $(date)"
echo "Node:   $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# ---- modules ----------------------------------------------------------------
module load anaconda3/2024.06
conda activate microdlm

cd ~/microDLM
mkdir -p logs weights

# ---- hardware info ----------------------------------------------------------
echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- ensure Shakespeare data is present ------------------------------------
echo ""
echo "--- Data ---"
if [ ! -f data/shakespeare.txt ]; then
    echo "Downloading Shakespeare..."
    mkdir -p data
    wget -q -O data/shakespeare.txt \
        https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    echo "Downloaded: $(wc -c < data/shakespeare.txt) bytes"
else
    echo "data/shakespeare.txt present ($(wc -c < data/shakespeare.txt) bytes)"
fi

# ---- train diffusion model -------------------------------------------------
echo ""
echo "=== Training Diffusion LM ==="
python diffusion.py --data shakespeare --train
echo "Diffusion training done."

# ---- train GPT baseline ----------------------------------------------------
echo ""
echo "=== Training GPT (baseline) ==="
python gpt.py --data shakespeare --train
echo "GPT training done."

# ---- verify weights saved --------------------------------------------------
echo ""
echo "--- Weights ---"
ls -lh weights/

# ---- quick generation sample -----------------------------------------------
echo ""
echo "=== Generation Sample (Diffusion) ==="
python diffusion.py --data shakespeare --generate --tokens 200 --steps 40

echo ""
echo "=== Generation Sample (GPT) ==="
python gpt.py --data shakespeare --generate --tokens 200

echo ""
echo "=== Phase 0 COMPLETE ==="
echo "End time: $(date)"
echo "Both models trained and generating. Proceed to Phase 1 (BPE tokenizer)."
