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
export PYTHONUNBUFFERED=1

echo "=== FineWeb-Edu Loader Test ==="
echo "Start time: $(date)"

# Load modules




cd ~/microDLM

# Create log directory
mkdir -p logs

# Test the data loader
echo ""
echo "Testing FineWeb-Edu streaming loader..."
/home/gupta.yashv/.conda/envs/microdlm/bin/python data_fineweb.py

echo ""
echo "End time: $(date)"
echo "=== Test complete ==="
