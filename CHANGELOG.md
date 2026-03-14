# Changelog — microDLM

---

## 2026-03-13 — HPC Setup & Phase 0

### Goal
Get the existing 10.7M param diffusion model training on Northeastern Explorer HPC,
starting from the Shakespeare dataset with zero architecture changes.

---

### Files Created

**`scripts/test_gpu.sh`**
- 30-min GPU-short SLURM job to verify CUDA, PyTorch, and repo integrity before any training
- Checks: nvidia-smi, torch version, CUDA availability, matmul on GPU, file presence, syntax parse of diffusion.py and gpt.py

**`scripts/train_shakespeare.sh`**
- 2-hour A100 SLURM job for Phase 0 validation
- Trains both diffusion and GPT on Shakespeare, then generates samples to confirm working weights

**`scripts/test_fineweb.sh`**
- CPU-only job to test the FineWeb-Edu streaming data loader

**`scripts/train_fineweb.sh`**
- 8-hour A100 job for FineWeb-Edu training (future use)

**`data_fineweb.py`**
- FineWeb-Edu streaming data loader (character-level, HuggingFace datasets)
- Supports streaming mode (no full download) and local binary shard caching
- Builds vocab from first 5K documents, returns train/val DataLoaders

**`HPC_REFERENCE.md`**
- Quick reference for all HPC commands: SSH, environment setup, job submission, monitoring, GPU selection, partitions, storage

**`HPC_DATASET_SETUP_LOG.md`**
- Detailed setup log with dataset strategy, environment steps, and execution checklist

**`hpc_gpu_inventory.md`**
- Full inventory of Explorer cluster GPUs, partitions, and availability

**`scaling_plan_hpc.md`**
- Complete scaling roadmap from 10.7M (Shakespeare) to 124M+ (FineWeb-Edu)
- Covers BPE tokenizer, dataset upgrade, architecture scaling, checkpoint-resume, multi-GPU DDP

---

### Bugs Fixed

**`diffusion.py` — `--data fineweb` routed to wrong loader**
- `choices` included `"fineweb"` but the `else` branch called `load_openwebtext`
- Fixed: split into `elif DATA_SOURCE == "openwebtext"` and `else: # fineweb` with correct `load_fineweb_edu` call

**`gpt.py` — same bug + missing `fineweb` in choices**
- `choices` only had `["shakespeare", "openwebtext"]`
- Fixed: added `"fineweb"` to choices, split else into proper elif/else branches

---

### Environment Issues Encountered

**Issue 1: Conda env directory corrupted**
- Symptom: `conda create` said env exists; `conda activate` succeeded but `pip` not found; `python -m pip` gave `Fatal Python error: init_fs_encoding`
- Root cause: env was created without `module load anaconda3/2024.06` first — Python stdlib missing
- Fix: `rm -rf ~/.conda/envs/microdlm` then recreate with module loaded first

**Issue 2: `conda activate` fails in SLURM batch jobs**
- Symptom: `CondaError: Run 'conda init' before 'conda activate'`
- Tried: `source activate microdlm` → still failed
- Tried: `source $(conda info --base)/etc/profile.d/conda.sh` → still failed
- Root cause: SLURM batch jobs run in a non-interactive shell; conda shell hooks are not available regardless of init method
- Fix: removed all conda activation from scripts, replaced all `python` calls with full path `/home/gupta.yashv/.conda/envs/microdlm/bin/python`

**Issue 3: `git checkout` blocked by untracked files on HPC**
- Symptom: `error: The following untracked working tree files would be overwritten by checkout: HPC_DATASET_SETUP_LOG.md hpc_gpu_inventory.md`
- Root cause: files had been manually copied to HPC before being committed; git refused to overwrite them
- Fix: `rm HPC_DATASET_SETUP_LOG.md hpc_gpu_inventory.md` then re-checkout

---

### Phase 0 Results

| Test | Status | Details |
|------|--------|---------|
| GPU test (`test_gpu.sh`) | **PASSED** | V100-SXM2-32GB, 34.1GB, CUDA 12.1, PyTorch 2.5.1 |
| Shakespeare training | In progress | Job 5069979, step 0 evaluation running |

---

## Branch History

| Branch | Purpose |
|--------|---------|
| `main` | Stable, Shakespeare-trained 10.7M model |
| `feature/training-scripts` | Earlier training script work |
| `feature/hpc-shakespeare` | HPC Phase 0 setup — current work |
