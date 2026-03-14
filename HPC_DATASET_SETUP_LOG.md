# HPC Dataset Setup Log

## Project: microDLM - FineWeb-Edu Dataset on Northeastern HPC

**Last Updated:** 2026-03-13
**User:** ydg (gupta.yashv)
**HPC Cluster:** Northeastern Explorer (explorer.northeastern.edu)
**Dataset:** FineWeb-Edu (Hugging Face)
**Primary Goal:** Scale microDLM from 10.7M params (Shakespeare) to 100M+ params on FineWeb-Edu

---

## Project Context: What We're Building

### The microDLM Architecture

| Component | Diffusion LM | GPT (Baseline) |
|-----------|--------------|----------------|
| **Attention** | Bidirectional (sees all tokens) | Causal (left-to-right only) |
| **Training Objective** | Predict masked tokens | Predict next token |
| **Loss Scope** | Masked positions only | All positions |
| **Generation** | Parallel unmasking (confidence-based) | Sequential (one token at a time) |
| **Generation Speed** | ~40 steps for 240 tokens | 240 steps for 240 tokens |

Both models share: 10.7M params, 6 layers, 6 heads, 384 embedding, RoPE, RMSNorm, ReluSquared MLP.

### Scaling Roadmap (from scaling_plan_hpc.md)

| Stage | Target | Params | Dataset | Tokenizer | Status |
|-------|--------|--------|---------|-----------|--------|
| Current | Shakespeare baseline | 10.7M | Tiny Shakespeare (1.1MB) | Char-level (67 vocab) | DONE |
| Phase 0 | HPC validation | 10.7M | Tiny Shakespeare | Char-level | PENDING |
| Phase 1 | BPE tokenizer | 10.7M | Shakespeare/OpenWebText | BPE (4K-50K vocab) | PENDING |
| Phase 2 | Dataset upgrade | 10.7M | FineWeb-Edu subset (100M-1B tokens) | BPE | THIS WORK |
| Phase 3 | Architecture scale | 124M | FineWeb-Edu | BPE (50K) | PENDING |
| Phase 4 | Checkpoint-resume | 124M | FineWeb-Edu | BPE | PENDING |
| Phase 5 | Multi-GPU DDP | 124M+ | FineWeb-Edu | BPE | PENDING |

---

## Dataset Strategy: Why FineWeb-Edu?

### FineWeb-Edu vs OpenWebText

| Dataset | Size | Quality | Use Case |
|---------|------|---------|----------|
| **OpenWebText** | ~9.7B tokens, ~38GB | High (filtered web text) | MDLM/SEDD baseline |
| **FineWeb-Edu** | ~1.3T tokens total (use subset) | Very high (educational content) | Better quality, newer |
| **Tiny Shakespeare** | ~1.1MB | N/A (literary) | Debugging, education |

### Why a Subset?

Training a 10.7M model on 1B tokens takes ~10-20 hours on 4x A100. The full FineWeb-Edu is impractical:
- Full dataset: 1.3T tokens (too large)
- **Target subset: 100M-1B tokens** (manageable, meaningful)

### Streaming vs Download

FineWeb-Edu supports **streaming** from HuggingFace - no need to download 10TB:
```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
```

For HPC efficiency, we'll:
1. **Stream during preprocessing** - no full download needed
2. **Cache preprocessed shards** - ~2GB for 1B tokens (uint16 binary)
3. **Store in /home** - small enough for home directory quota

---

## Phase 1: HPC Environment Setup

### Step 1.1: Verify HPC Access

```bash
# SSH into cluster
ssh gupta.yashv@explorer.northeastern.edu

# Check GPU availability
sinfo -o "%N %T %G"

# Check quota
squota

# Verify Python/conda
module load anaconda3/2024.06
conda --version
```

### Step 1.2: Create Conda Environment

```bash
# On HPC
cd ~/microDLM

# Create environment
module load anaconda3/2024.06
conda create -n microdlm python=3.11 -y
conda activate microdlm

# Install PyTorch with CUDA (check CUDA version first)
nvidia-smi  # Check CUDA version
# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install HuggingFace dependencies
pip install datasets huggingface_hub tokenizers

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

### Step 1.3: Test GPU Job

Create `scripts/test_gpu.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=microdlm-test
#SBATCH --partition=gpu-short
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/test_%j.log

module load anaconda3/2024.06
conda activate microdlm

cd ~/microDLM
python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

Submit: `sbatch scripts/test_gpu.sh`

---

## Phase 2: Dataset Download Strategy

### Option A: Streaming (Recommended for Initial Testing)

Stream FineWeb-Edu directly without downloading:
- No storage required
- Slower (network-bound)
- Good for testing pipeline

### Option B: Preprocessed Shards (Recommended for Production)

Preprocess to binary shards for fast training:
- ~2GB for 1B tokens
- Memory-mapped loading
- Fast random access

### Dataset Size Reality Check

| Component | Size | Location |
|-----------|------|----------|
| Full FineWeb-Edu (raw) | ~1.3T tokens | HuggingFace (streaming) |
| 1B token subset (uint16) | ~2GB | `/home/gupta.yashv/microDLM/data/fineweb/` |
| Model checkpoints (10.7M) | ~45MB each | `/home/gupta.yashv/microDLM/weights/` |
| Training logs | <100MB | `/home/gupta.yashv/microDLM/logs/` |

**Total: ~3-5GB** - fits comfortably in home directory quota.

---

## Phase 3: Create FineWeb-Edu Data Loader

### Step 3.1: Create `data_fineweb.py`

Create new file (don't modify existing OpenWebText loader):

```python
#!/usr/bin/env python3
"""
FineWeb-Edu data loader for microDLM.
Uses streaming from HuggingFace or preprocessed binary shards.
"""
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader

def _default_cache_dir():
    return os.path.expanduser("~/microDLM/data/fineweb")

def load_fineweb_edu(
    block_size=256,
    batch_size=64,
    val_frac=0.01,
    num_workers=0,
    pin_memory=True,
    use_streaming=True,
    max_tokens=1_000_000_000,  # 1B tokens max
):
    """
    Load FineWeb-Edu dataset.

    Args:
        block_size: Context window size
        batch_size: Batch size for training
        val_frac: Fraction for validation
        num_workers: DataLoader workers
        pin_memory: Pin memory for GPU transfer
        use_streaming: If True, stream from HuggingFace (no download)
        max_tokens: Maximum tokens to use (for subset)

    Returns:
        train_loader, val_loader, stoi, itos, vocab_size, mask_token_id, encode, decode
    """
    from datasets import load_dataset

    cache_dir = _default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)

    # Load dataset (streaming or cached)
    if use_streaming:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
            cache_dir=cache_dir
        )
    else:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            cache_dir=cache_dir
        )

    # Build vocabulary (character-level for now)
    # TODO: Add BPE tokenizer support (Phase 1 of scaling plan)
    MASK_CHAR = "_"
    UNK_CHAR = "<UNK>"

    # Sample documents for vocab building
    chars = set()
    n_sample = 0
    for example in ds:
        if isinstance(example.get("text", ""), str):
            chars.update(example["text"])
        n_sample += 1
        if n_sample >= 5000:  # Build vocab from 5K documents
            break

    # Create vocab
    if MASK_CHAR in chars:
        chars.remove(MASK_CHAR)
    if UNK_CHAR in chars:
        chars.remove(UNK_CHAR)

    itos = [MASK_CHAR, UNK_CHAR] + sorted(chars)
    stoi = {ch: i for i, ch in enumerate(itos)}
    vocab_size = len(itos)
    mask_token_id = stoi[MASK_CHAR]
    unk_id = stoi[UNK_CHAR]

    encode_fn = lambda s: [stoi.get(c, unk_id) for c in s]
    decode_fn = lambda ids: "".join([itos[i] if i < len(itos) else "?" for i in ids])

    # Create dataset class
    class FineWebDataset(Dataset):
        def __init__(self, ds, stoi, block_size, max_tokens=max_tokens):
            self.ds = ds
            self.stoi = stoi
            self.block_size = block_size
            self.max_tokens = max_tokens
            self._cache = []
            self._token_count = 0

        def __len__(self):
            return max_tokens // block_size if max_tokens else 10000

        def __getitem__(self, idx):
            # For streaming: fetch document and extract block
            for example in self.ds:
                text = example.get("text", "")
                if not isinstance(text, str) or len(text) < self.block_size:
                    continue
                start = random.randint(0, len(text) - self.block_size)
                block = text[start : start + self.block_size]
                return torch.tensor([self.stoi.get(c, unk_id) for c in block], dtype=torch.long)
            # Fallback
            return torch.full((self.block_size,), unk_id, dtype=torch.long)

    train_ds = FineWebDataset(ds, stoi, block_size)
    val_size = max(1, int(len(train_ds) * val_frac))
    train_size = len(train_ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )

    return train_loader, val_loader, stoi, itos, vocab_size, mask_token_id, encode_fn, decode_fn
```

---

## Phase 4: SLURM Job Scripts

### Step 4.1: Dataset Test Job

Create `scripts/test_fineweb.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=fineweb-test
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/fineweb_test_%j.log

# Test FineWeb-Edu streaming
module load anaconda3/2024.06
conda activate microdlm

cd ~/microDLM
python -c "
from data_fineweb import load_fineweb_edu
print('Loading FineWeb-Edu (streaming)...')
train_loader, val_loader, stoi, itos, vocab_size, mask_id, enc, dec = load_fineweb_edu(
    block_size=256, batch_size=64, use_streaming=True
)
print(f'Vocab size: {vocab_size}')
print(f'Train batches: {len(train_loader)}')
print(f'Val batches: {len(val_loader)}')
# Fetch one batch
batch = next(iter(train_loader))
print(f'Batch shape: {batch.shape}')
print('First batch sample:', dec(batch[0].tolist())[:100])
"
```

### Step 4.2: Training Job

Create `scripts/train_fineweb.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=fineweb-train
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/fineweb_train_%j.log
#SBATCH --error=logs/fineweb_train_%j.err

module load anaconda3/2024.06
conda activate microdlm

cd ~/microDLM

echo "=== Training Diffusion LM on FineWeb-Edu ==="
python diffusion.py --data fineweb --train

echo "=== Training GPT on FineWeb-Edu ==="
python gpt.py --data fineweb --train
```

---

## Phase 5: Update Main Files

### Step 5.1: Modify `diffusion.py` and `gpt.py`

Add `--data fineweb` option (parallel existing Shakespeare/OpenWebText):

```python
# In diffusion.py, update DATA_SOURCE parsing:
_data_parser.add_argument(
    "--data",
    default="shakespeare",
    choices=["shakespeare", "openwebtext", "fineweb"]  # Added fineweb
)

# In data loading section:
if DATA_SOURCE == "shakespeare":
    # ... existing code ...
elif DATA_SOURCE == "openwebtext":
    from data_openwebtext import load_openwebtext
    # ... existing code ...
elif DATA_SOURCE == "fineweb":
    from data_fineweb import load_fineweb_edu
    _train_loader, _val_loader, stoi, itos, vocab_size, mask_token_id, encode, decode = load_fineweb_edu(
        block_size=block_size, batch_size=batch_size, num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    _prompt_block = next(iter(_val_loader))[0][0].clone()
```

---

## Execution Checklist

### Pre-Flight (Local)
- [ ] Review scaling_plan_hpc.md
- [ ] Understand current model architecture (diffusion.py, gpt.py)
- [ ] Verify HPC SSH access works

### Phase 1: HPC Setup (Day 1)
- [ ] SSH into Explorer cluster
- [ ] Create conda environment
- [ ] Install PyTorch with CUDA
- [ ] Run test_gpu.sh job
- [ ] Verify GPU access and memory

### Phase 2: Dataset Test (Day 2)
- [ ] Create data_fineweb.py
- [ ] Run test_fineweb.sh (streaming test)
- [ ] Verify vocab size and batch shapes
- [ ] Document any issues in this log

### Phase 3: Training Test (Day 3)
- [ ] Update diffusion.py/gpt.py with fineweb option
- [ ] Submit train_fineweb.sh
- [ ] Monitor training progress
- [ ] Verify loss decreases

### Phase 4: Full Training (Day 4+)
- [ ] Configure checkpoint saving (required for 8h jobs)
- [ ] Submit full training job chain
- [ ] Monitor and log results

---

## Notes & Observations

| Date | Note | Status |
|------|------|--------|
| 2026-03-13 | Initial log creation with corrected dataset size estimates | DONE |
| 2026-03-13 | Clarified streaming vs download strategy | DONE |
| 2026-03-13 | Added execution checklist | DONE |

---

## References

- [Scaling Plan](scaling_plan_hpc.md) - Full scaling roadmap
- [HPC GPU Inventory](hpc_gpu_inventory.md) - Available GPU resources
- [FineWeb-Edu on HuggingFace](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Training pipeline inspiration
