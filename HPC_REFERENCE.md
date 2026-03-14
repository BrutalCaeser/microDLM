# HPC Quick Reference — Northeastern Explorer

**Cluster:** `explorer.northeastern.edu`
**User:** `gupta.yashv`
**Project dir:** `~/microDLM`

---

## Access

```bash
ssh gupta.yashv@explorer.northeastern.edu
```

---

## Environment Setup (one-time)

```bash
# Load anaconda — must do this before any conda commands
module load anaconda3/2024.06

# Create environment
conda create -n microdlm python=3.11 -y

# Activate (interactive shell only)
conda activate microdlm

# Install PyTorch with CUDA 12.1
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install HuggingFace dependencies (needed for FineWeb/OpenWebText)
python -m pip install datasets huggingface_hub tokenizers

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> **Note:** SLURM batch scripts must use the full Python path
> `/home/gupta.yashv/.conda/envs/microdlm/bin/python`
> because `conda activate` does not work in non-interactive batch jobs.

---

## Git

```bash
# First time clone
git clone https://github.com/BrutalCaeser/microDLM.git ~/microDLM
cd ~/microDLM
git checkout feature/hpc-shakespeare

# Pull latest changes
cd ~/microDLM
git pull

# Check current branch
git branch
```

---

## Job Submission

```bash
# Submit a job
sbatch scripts/test_gpu.sh
sbatch scripts/train_shakespeare.sh
sbatch scripts/train_fineweb.sh

# Check job status
squeue -u gupta.yashv

# Cancel a job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u gupta.yashv
```

### Job states
| State | Meaning |
|-------|---------|
| `PD` | Pending — waiting in queue |
| `R`  | Running |
| `CG` | Completing |
| *(gone)* | Finished — check logs |

---

## Monitoring

```bash
# Watch a running job's output live (Ctrl+C to stop, does NOT cancel job)
tail -f logs/<log_file>.log

# Check both log and error files
cat logs/shakespeare_<JOB_ID>.log
cat logs/shakespeare_<JOB_ID>.err

# See all log files
ls -lth logs/
```

---

## GPU / Hardware

```bash
# Check GPU availability across cluster
sinfo -o "%N %T %G"

# Check specific partition
sinfo -p gpu
sinfo -p gpu-short

# GPU info on a running node (from within a job)
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# Interactive GPU session (2h max, good for debugging)
srun --partition=gpu-interactive --gres=gpu:v100:1 --mem=16G --cpus-per-task=4 --pty /bin/bash
srun --partition=gpu-interactive --gres=gpu:a100:1 --mem=32G --cpus-per-task=4 --pty /bin/bash
```

---

## Storage

```bash
# Check your quota
squota

# Check disk usage in project dir
du -sh ~/microDLM/
du -sh ~/microDLM/data/
du -sh ~/microDLM/weights/

# Check available space
df -h ~
```

---

## Partitions Summary

| Partition | Max Time | GPU Pool | Notes |
|-----------|----------|----------|-------|
| `gpu` | 8 hours | 102 GPUs | Main training partition |
| `gpu-short` | 2 hours | 102 GPUs | Quick tests, faster queue |
| `gpu-interactive` | 2 hours | 102 GPUs | Interactive `srun` sessions |
| `short` | 48 hours | CPU only | Data preprocessing |
| `courses-gpu` | 24 hours | 34 GPUs | Needs `rc/courses` group access |

---

## Target GPUs

| GPU | Memory | Node(s) | Notes |
|-----|--------|---------|-------|
| A100 | 40/80 GB | d3146, d3203 | Best for training — idle |
| V100-SXM2 | 32 GB | d3091–d3098 | Good fallback |
| V100-PCIE | 16 GB | c2204–c2207 | Limited memory |

Request specific GPU:
```bash
#SBATCH --gres=gpu:a100:1   # 1x A100
#SBATCH --gres=gpu:v100:4   # 4x V100
```

---

## Useful SLURM Directives

```bash
#SBATCH --job-name=my-job
#SBATCH --partition=gpu
#SBATCH --time=07:45:00          # HH:MM:SS — leave buffer before 8h limit
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/job_%j.log  # %j = job ID
#SBATCH --error=logs/job_%j.err
#SBATCH --signal=B:USR1@300       # Signal 5 min before timeout (for checkpoint-resume)
```

---

## Phase Checklist

- [x] Phase 0a — GPU test (`test_gpu.sh`) — **PASSED** on V100-SXM2-32GB
- [ ] Phase 0b — Shakespeare training (`train_shakespeare.sh`)
- [ ] Phase 1 — BPE tokenizer
- [ ] Phase 2 — Dataset → FineWeb-Edu (100M–1B tokens)
- [ ] Phase 3 — Scale to 124M params
- [ ] Phase 4 — Checkpoint-resume (mandatory before any job > 8h)
- [ ] Phase 5 — Multi-GPU DDP (4x A100)
