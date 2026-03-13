#!/bin/bash
#SBATCH --job-name=microdlm_train
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=train_%j.log

# Same as train.sh but uses courses-gpu partition (24h limit, P100 GPUs).
# Only works if your account has access to courses-gpu (e.g. courses group).
# Use when gpu partition is busy and you have courses-gpu access.

set -e
module load cuda/12.1.1
module load anaconda3/2024.06
source $(conda info --base)/etc/profile.d/conda.sh
conda activate microdlm

cd ~/microDLM
export HF_DATASETS_CACHE="$HOME/microDLM/data/openwebtext_cache"

echo "=== Training Diffusion LM (OpenWebText) [courses-gpu] ==="
python diffusion.py --data openwebtext --train

echo ""
echo "=== Training GPT (OpenWebText) [courses-gpu] ==="
python gpt.py --data openwebtext --train

echo ""
echo "=== Done ==="
