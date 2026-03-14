#!/bin/bash
# Train microDLM on FineWeb-Edu dataset
# Usage: sbatch scripts/train_fineweb.sh
#
# This job trains both Diffusion LM and GPT baseline on FineWeb-Edu.
# Runtime: ~8 hours on A100 (max partition limit)
# Checkpointing: TODO - implement for multi-job training chains

#SBATCH --job-name=fineweb-train
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/fineweb_train_%j.log
#SBATCH --error=logs/fineweb_train_%j.err

set -e

echo "=== microDLM Training on FineWeb-Edu ==="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Load modules
module load anaconda3/2024.06
source $(conda info --base)/etc/profile.d/conda.sh
conda activate microdlm

cd ~/microDLM

# Create directories
mkdir -p logs weights

# Export environment variable for dataset cache
export HF_DATASETS_CACHE="$HOME/microDLM/data/fineweb"

echo ""
echo "=== Training Diffusion LM ==="
python diffusion.py --data fineweb --train

echo ""
echo "=== Training GPT ==="
python gpt.py --data fineweb --train

echo ""
echo "End time: $(date)"
echo "=== Training complete ==="
