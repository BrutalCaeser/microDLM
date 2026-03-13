#!/bin/bash
#SBATCH --job-name=download_owt
#SBATCH --partition=short
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=download_dataset_%j.log

set -e
module load anaconda3/2024.06
source $(conda info --base)/etc/profile.d/conda.sh
conda activate microdlm

cd ~/microDLM

# Install Hugging Face datasets if not present
pip install -q datasets huggingface_hub

echo "Starting OpenWebText download (cache: ~/microDLM/data/openwebtext_cache)..."
python download_openwebtext.py
echo "Done."
