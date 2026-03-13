#!/bin/bash
#SBATCH --job-name=microdlm_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=train_%j.log

# Train diffusion and GPT on OpenWebText (cache: ~/microDLM/data/openwebtext_cache).
# Default: any GPU (--gres=gpu:1). For best-GPU-first fallback, run submit_train_fallback.sh
# instead; it overrides --gres with specific types (h200, h100, a100, ...).
# Uses GPU-efficient pipeline: CPU preload -> single model.to(device) -> pinned non_blocking batches.

set -e
module load cuda/12.1.1
module load anaconda3/2024.06
source $(conda info --base)/etc/profile.d/conda.sh
conda activate microdlm

cd ~/microDLM
export HF_DATASETS_CACHE="$HOME/microDLM/data/openwebtext_cache"

echo "=== Training Diffusion LM (OpenWebText) ==="
python diffusion.py --data openwebtext --train

echo ""
echo "=== Training GPT (OpenWebText) ==="
python gpt.py --data openwebtext --train

echo ""
echo "=== Done ==="
