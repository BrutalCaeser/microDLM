#!/bin/bash
#SBATCH --job-name=microdlm_setup
#SBATCH --partition=short
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=setup_env_%j.log

set -e
module load cuda/12.1.1
module load anaconda3/2024.06
source $(conda info --base)/etc/profile.d/conda.sh
conda activate microdlm

echo "Python: $(which python)"
python --version
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "PyTorch install done. Checking CUDA..."
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
echo "Setup complete."
