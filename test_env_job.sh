#!/bin/bash
#SBATCH --job-name=microdlm_test
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=test_env_%j.log

set -e
module load cuda/12.1.1
module load anaconda3/2024.06
source $(conda info --base)/etc/profile.d/conda.sh
conda activate microdlm

cd ~/microDLM

echo "=== Environment ==="
echo "Host: $(hostname)"
echo "Python: $(which python)"
python --version
echo ""

echo "=== PyTorch + CUDA ==="
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
"
echo ""

echo "=== Project imports ==="
python -c "
import diffusion
import gpt
print('diffusion OK')
print('gpt OK')
"
echo ""

echo "=== Test complete ==="
