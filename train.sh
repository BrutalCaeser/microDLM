#!/bin/bash
#SBATCH --job-name=microdlm_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=train_%j.log

# Train diffusion and GPT on Tiny Shakespeare (data/shakespeare.txt).
# Uses GPU-efficient pipeline: CPU preload -> single model.to(device) -> pinned non_blocking batches.

set -e
module load cuda/12.1.1
module load anaconda3/2024.06
source $(conda info --base)/etc/profile.d/conda.sh
conda activate microdlm

cd ~/microDLM

echo "=== Training Diffusion LM ==="
python diffusion.py --train

echo ""
echo "=== Training GPT ==="
python gpt.py --train

echo ""
echo "=== Done ==="
