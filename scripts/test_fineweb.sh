#!/bin/bash
# Test FineWeb-Edu data loader on HPC
# Usage: sbatch scripts/test_fineweb.sh

#SBATCH --job-name=fineweb-test
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/fineweb_test_%j.log
#SBATCH --error=logs/fineweb_test_%j.err

set -e

echo "=== FineWeb-Edu Loader Test ==="
echo "Start time: $(date)"

# Load modules
module load anaconda3/2024.06
source $(conda info --base)/etc/profile.d/conda.sh
conda activate microdlm

cd ~/microDLM

# Create log directory
mkdir -p logs

# Test the data loader
echo ""
echo "Testing FineWeb-Edu streaming loader..."
python data_fineweb.py

echo ""
echo "End time: $(date)"
echo "=== Test complete ==="
