# Scaling microDLM on Northeastern Explorer HPC
## From 10.7M on Shakespeare to 100M+ on OpenWebText

---

## Critical Constraints (read first)

Your biggest constraint is NOT compute — it's wall time.

    gpu partition:          8 hours max
    courses-gpu partition:  24 hours max (needs rc/courses group — check with srun)
    gpu-short:              2 hours max
    gpu-interactive:        2 hours max

Training a 100M+ parameter model on a real dataset takes days, not hours. Every
training run MUST support checkpoint-resume: save state every N steps, restart from
the last checkpoint when the job times out. This is non-negotiable.

Target hardware:
    Primary:   A100 (40/80GB) — nodes d3146, d3203 (8 GPUs each, currently idle)
    Fallback:  V100-SXM2 (32GB) — nodes d3091-d3098 (4 GPUs each, widely available)
    Stretch:   H200 (141GB) — nodes d4052-d4055 (if you can get access)

Strategy: start on 1x A100 for validation, scale to 4x A100 for real training.

---

## Phase 0: Validate Current Code on HPC (Day 1)

Goal: get the existing microDLM training running on HPC with zero code changes.
This proves your environment works before you change anything.

### Step 0a: Set up the environment

```bash
# SSH into the cluster
ssh gupta.yashv@explorer.northeastern.edu

# Create project directory
mkdir -p ~/microDLM
cd ~/microDLM

# Clone your repo
git clone https://github.com/BrutalCaeser/microDLM.git .

# Create conda environment
module load anaconda3
conda create -n microdlm python=3.11 -y
conda activate microdlm

# Install PyTorch with CUDA support
# Check CUDA version first:
module load cuda/12.1  # or whatever version is available
nvidia-smi  # verify

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Step 0b: First SLURM job — test on GPU

Create `scripts/test_gpu.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=microdlm-test
#SBATCH --partition=gpu-short
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

module load anaconda3
module load cuda/12.1
conda activate microdlm

cd ~/microDLM
python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print('CUDA works!')
"
```

```bash
mkdir -p logs
sbatch scripts/test_gpu.sh
squeue -u gupta.yashv  # check status
cat logs/test_*.out     # check output
```

### Step 0c: Run the existing training

Create `scripts/train_shakespeare.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=microdlm-shakespeare
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/shakespeare_%j.out
#SBATCH --error=logs/shakespeare_%j.err

module load anaconda3
module load cuda/12.1
conda activate microdlm

cd ~/microDLM

# Download data if not present
mkdir -p data weights
if [ ! -f data/shakespeare.txt ]; then
    wget -q -O data/shakespeare.txt \
        https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
fi

# Train diffusion model
python diffusion.py --train

# Train GPT baseline
python gpt.py --train

echo "Training complete!"
```

```bash
sbatch scripts/train_shakespeare.sh
```

If this completes and produces weights/diffusion.pt and weights/gpt.pt, Phase 0 is done.
Your code runs on HPC. Now we scale.

---

## Phase 1: BPE Tokenizer (the single highest-impact change)

### Why this matters

Your current model operates on characters. Each "token" is one character. This means:

    "the" = 3 tokens: ['t', 'h', 'e']
    vocab_size = 67 (65 chars + BOS + MASK)

A BPE (Byte-Pair Encoding) tokenizer groups frequent character sequences into single
tokens:

    "the" = 1 token: ['the']
    vocab_size = 4096-32000 (configurable)

This changes three things fundamentally:

1. INFORMATION DENSITY: Each token carries more meaning. A 256-token context window
   covers ~256 characters with char-level but ~1000 characters with BPE. The model
   "sees" 4x more text per forward pass.

2. WHAT MASKING MEANS: Masking one BPE token removes an entire word or subword,
   forcing the model to learn word-level and phrase-level patterns. Masking one
   character only removes one letter — a much easier task.

3. EVALUATION: Perplexity numbers become comparable to published results (MDLM,
   SEDD) which all use BPE tokenization.

### Implementation

Use HuggingFace's `tokenizers` library to train a BPE tokenizer on your data,
or use a pretrained one (GPT-2's tokenizer is the standard for comparison with
MDLM and SEDD).

Option A: Use GPT-2's tokenizer (recommended for comparability):

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Add MASK token
tokenizer.add_special_tokens({'additional_special_tokens': ['[MASK]']})
mask_token_id = tokenizer.convert_tokens_to_ids('[MASK]')
vocab_size = len(tokenizer)  # 50258 (50257 GPT-2 tokens + 1 MASK)

# Encode text
tokens = tokenizer.encode("First Citizen: Before we proceed")
# [5765, 16380, 25, 7413, 356, 5765]  — 6 tokens instead of 35 characters

# Decode back
text = tokenizer.decode(tokens)
```

Option B: Train a smaller BPE tokenizer on your data (good for learning):

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=4096,           # much smaller than GPT-2, good for smaller datasets
    special_tokens=["[MASK]", "[PAD]", "[BOS]", "[EOS]"]
)
tokenizer.train(files=["data/openwebtext_sample.txt"], trainer=trainer)
```

### What changes in the model code

With BPE, the vocabulary jumps from 67 to 4096-50257. This affects:

    Token embedding: nn.Embedding(67, 384)  →  nn.Embedding(50257, 384)
    Output head:     nn.Linear(384, 67)     →  nn.Linear(384, 50257)

The embedding table goes from 67 × 384 = 25K params to 50257 × 384 = 19.3M params.
The output head adds another 19.3M. Together, this adds ~38M parameters just from
the vocabulary change — more than the rest of the model combined.

This is why scaling the architecture (Phase 3) matters: the transformer body needs
to be large enough to justify a 50K-token vocabulary.

### Practical recommendation

Start with a SMALLER BPE vocabulary (4096 or 8192 tokens) trained on your dataset.
This keeps the embedding table manageable (~1.5M-3M params) while still capturing
word-level patterns. Move to GPT-2's 50257 vocabulary only when the model body
is large enough (100M+ params).

---

## Phase 2: Dataset (Shakespeare → OpenWebText)

### Options, ranked by quality and effort

OPTION 1: OpenWebText (recommended)
    Size:       ~38 GB text, ~9.7B tokens (GPT-2 tokenized)
    Quality:    High — filtered web text, similar to GPT-2's training data
    Effort:     Medium — download from HuggingFace, preprocess
    Why:        MDLM and SEDD both train on this. Direct comparison possible.
    
    ```python
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train")
    # ~8M documents, ~9.7B tokens
    ```

OPTION 2: FineWeb-Edu (higher quality, newer)
    Size:       ~1.3T tokens total, but you can use a subset
    Quality:    Very high — filtered for educational content
    Effort:     Medium — same HuggingFace pipeline
    
    ```python
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    # Take first 10B tokens worth
    ```

OPTION 3: A curated 1B-token subset (pragmatic middle ground)
    Take the first 1B tokens of OpenWebText. This is enough for meaningful
    training of a 100M parameter model and fits easily on disk.
    Training a 100M model on 1B tokens takes ~10-20 hours on 4x A100.

### Data preprocessing pipeline

Create `scripts/prepare_data.py`:

```python
"""
Preprocess OpenWebText for microDLM training.
Tokenizes all documents and saves as memory-mapped binary files
for fast random-access during training.

This follows the nanoGPT preprocessing approach.
"""
import os
import numpy as np
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm

# Config
DATA_DIR = "data/openwebtext"
SHARD_SIZE = 100_000_000  # 100M tokens per shard
os.makedirs(DATA_DIR, exist_ok=True)

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load dataset
dataset = load_dataset("Skylion007/openwebtext", split="train", num_proc=8)

# Tokenize
def tokenize(example):
    ids = tokenizer.encode(example['text'])
    return {'ids': ids, 'len': len(ids)}

tokenized = dataset.map(tokenize, remove_columns=['text'], num_proc=8)

# Write to binary shards (memory-mapped for fast training)
all_tokens = []
shard_idx = 0

for example in tqdm(tokenized):
    all_tokens.extend(example['ids'])
    
    if len(all_tokens) >= SHARD_SIZE:
        arr = np.array(all_tokens[:SHARD_SIZE], dtype=np.uint16)
        arr.tofile(os.path.join(DATA_DIR, f'shard_{shard_idx:04d}.bin'))
        all_tokens = all_tokens[SHARD_SIZE:]
        shard_idx += 1
        print(f"Saved shard {shard_idx}, {shard_idx * SHARD_SIZE / 1e9:.1f}B tokens total")

# Save remainder
if all_tokens:
    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(os.path.join(DATA_DIR, f'shard_{shard_idx:04d}.bin'))

print(f"Done. {shard_idx + 1} shards, ~{(shard_idx + 1) * SHARD_SIZE / 1e9:.1f}B tokens")
```

Run this as a CPU job (it doesn't need a GPU):

```bash
#!/bin/bash
#SBATCH --job-name=prepare-data
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/prepare_%j.out

module load anaconda3
conda activate microdlm
pip install datasets transformers

python scripts/prepare_data.py
```

### Data loading for training

Replace the simple Shakespeare loader with a memory-mapped shard loader:

```python
class ShardedDataset:
    """Load tokenized data from binary shards. Memory-mapped for speed."""
    
    def __init__(self, data_dir, block_size):
        self.block_size = block_size
        self.shards = sorted(
            [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.bin')]
        )
        # Memory-map all shards
        self.data = np.concatenate([
            np.memmap(s, dtype=np.uint16, mode='r') for s in self.shards
        ])
        self.n_tokens = len(self.data)
        print(f"Loaded {self.n_tokens / 1e9:.2f}B tokens from {len(self.shards)} shards")
    
    def get_batch(self, batch_size, device):
        idx = torch.randint(self.n_tokens - self.block_size, (batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i:i+self.block_size].astype(np.int64))
            for i in idx
        ])
        return x.to(device)
```

---

## Phase 3: Architecture Scaling

### The scaling targets

    Current:      6L / 6H / 384E    = 10.7M params    (Shakespeare, character-level)
    Medium:      12L / 12H / 768E   = ~124M params     (OWT subset, BPE 4096-50K)
    Large:       24L / 16H / 1024E  = ~350M params     (full OWT, GPT-2 tokenizer)

The "medium" config matches GPT-2 Small (124M). This is the target MDLM and SEDD
use for their main results. Reaching this makes your results directly comparable
to published numbers.

The "large" config matches GPT-2 Medium (350M). This is stretch goal territory
and requires multi-GPU training.

### Architecture changes for scaling

Several things that work fine at 10.7M need to change at 124M:

1. ACTIVATION: ReLU² → SwiGLU

SwiGLU (used in LLaMA, modern transformers) is more expressive:

```python
class SwiGLU_MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # SwiGLU uses 3 projections instead of 2
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)    # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)    # down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)    # up
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

The hidden_dim for SwiGLU is typically (8/3) × embed_dim instead of 4×, to keep
param count similar after accounting for the third projection.

2. NORMALIZATION: RMSNorm stays (it's already what modern models use) ✓

3. POSITIONAL ENCODING: RoPE stays ✓

4. ATTENTION: Add QK-norm for training stability at scale.
   You already have this (norm(q), norm(k) in your code) ✓

5. WEIGHT TYING: Tie the token embedding and output head weights.
   This saves ~19M parameters at vocab_size=50257:

```python
self.lm_head.weight = self.token_emb.weight  # share weights
```

6. GRADIENT CHECKPOINTING: Trades compute for memory at scale.
   Re-computes activations during backward pass instead of storing them:

```python
from torch.utils.checkpoint import checkpoint

for block in self.blocks:
    x = checkpoint(block, x, cos_sin, use_reentrant=False)
```

This roughly halves memory usage at the cost of ~30% slower training.

7. MIXED PRECISION (bfloat16): Halves memory, doubles throughput:

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

for iter in range(max_iters):
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(x, targets, mask)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

On A100, bfloat16 is native and gives nearly 2x speedup with no quality loss.
On V100, use float16 instead (V100 doesn't support bfloat16 natively).

### Config system

Create `configs/` with YAML files for each scale:

```python
# config.py
from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 50257
    dropout: float = 0.0
    bias: bool = False
    use_swiglu: bool = True
    weight_tying: bool = True

@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 6e-4
    max_iters: int = 100_000
    warmup_iters: int = 2000
    lr_decay_iters: int = 100_000
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 500
    checkpoint_interval: int = 1000
    data_dir: str = "data/openwebtext"
    dtype: str = "bfloat16"  # or "float16" for V100

# Presets
SMALL = ModelConfig(n_layer=6, n_head=6, n_embd=384, vocab_size=67)      # current
MEDIUM = ModelConfig(n_layer=12, n_head=12, n_embd=768, vocab_size=50257) # GPT-2 Small
LARGE = ModelConfig(n_layer=24, n_head=16, n_embd=1024, vocab_size=50257) # GPT-2 Medium
```

---

## Phase 4: Checkpoint-Resume (MANDATORY for 8-hour jobs)

This is not optional. Your gpu partition has an 8-hour wall time limit. Training
a 124M model on 1B tokens takes ~20-40 hours on 4x A100. You MUST save checkpoints
and resume from them across multiple SLURM jobs.

### Checkpoint saving

```python
def save_checkpoint(model, optimizer, iter_num, best_val_loss, config, path):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint at iter {iter_num} to {path}")
```

### Checkpoint loading

```python
def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path, map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_iter = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resumed from iter {start_iter}, best val loss {best_val_loss:.4f}")
    return start_iter, best_val_loss
```

### Training loop with resume

```python
# At the start of training:
checkpoint_path = "weights/checkpoint.pt"
start_iter = 0
best_val_loss = float('inf')

if os.path.exists(checkpoint_path):
    start_iter, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer)

for iter_num in range(start_iter, max_iters):
    # ... training step ...
    
    # Save checkpoint periodically
    if iter_num % checkpoint_interval == 0 and iter_num > 0:
        save_checkpoint(model, optimizer, iter_num, best_val_loss, config, checkpoint_path)
    
    # Also save on last iteration (in case job is about to timeout)
    if iter_num == max_iters - 1:
        save_checkpoint(model, optimizer, iter_num, best_val_loss, config, checkpoint_path)
```

### Auto-resubmit SLURM script

Create `scripts/train_chain.sh` that automatically resubmits itself:

```bash
#!/bin/bash
#SBATCH --job-name=microdlm-train
#SBATCH --partition=gpu
#SBATCH --time=07:45:00              # 7h45m — leave 15min buffer before 8h limit
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --signal=B:USR1@300          # Send signal 5 minutes before timeout

module load anaconda3
module load cuda/12.1
conda activate microdlm
cd ~/microDLM

# Trap the timeout signal — save checkpoint and resubmit
handle_timeout() {
    echo "Received timeout signal, saving checkpoint..."
    kill -SIGINT $TRAIN_PID    # send interrupt to Python (triggers graceful save)
    wait $TRAIN_PID
    echo "Resubmitting job..."
    sbatch scripts/train_chain.sh
    exit 0
}
trap handle_timeout USR1

# Run training (will auto-resume from checkpoint if it exists)
python train_scaled.py --config medium --resume &
TRAIN_PID=$!
wait $TRAIN_PID

# If training finished naturally (didn't timeout), don't resubmit
echo "Training completed!"
```

In your Python training script, handle SIGINT gracefully:

```python
import signal

should_stop = False

def handle_interrupt(signum, frame):
    global should_stop
    print("Interrupt received, will save checkpoint after current step...")
    should_stop = True

signal.signal(signal.SIGINT, handle_interrupt)

# In training loop:
for iter_num in range(start_iter, max_iters):
    # ... training step ...
    
    if should_stop:
        save_checkpoint(model, optimizer, iter_num, best_val_loss, config, checkpoint_path)
        print(f"Saved and exiting at iter {iter_num}")
        break
```

This creates a self-chaining job: train for ~7.5 hours, save checkpoint, resubmit,
resume from checkpoint, repeat until done. Fully automatic.

---

## Phase 5: Multi-GPU Distributed Training

Once single-GPU training works with checkpointing, scale to multiple GPUs.

### Why multi-GPU?

A 124M model with batch_size=32 and block_size=1024 on 1x A100:
    ~3,000 tokens/sec → 1B tokens takes ~90 hours → 12 SLURM jobs

Same model on 4x A100 with DDP:
    ~12,000 tokens/sec → 1B tokens takes ~23 hours → 3 SLURM jobs

### PyTorch DDP (Distributed Data Parallel)

DDP is the simplest and most reliable multi-GPU strategy. Each GPU gets the full
model and a fraction of the batch. Gradients are averaged across GPUs.

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

# In main:
local_rank = setup_distributed()
device = f'cuda:{local_rank}'

model = DiffusionModel(config).to(device)
model = DDP(model, device_ids=[local_rank])

# Training loop is nearly identical — DDP handles gradient sync automatically
# Only rank 0 should save checkpoints and log:
if local_rank == 0:
    save_checkpoint(...)
    print(f"step {iter_num}: loss {loss:.4f}")
```

### SLURM script for multi-GPU

```bash
#!/bin/bash
#SBATCH --job-name=microdlm-ddp
#SBATCH --partition=gpu
#SBATCH --time=07:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4           # 4 processes = 4 GPUs
#SBATCH --gres=gpu:a100:4             # request 4 A100s on one node
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/ddp_%j.out
#SBATCH --error=logs/ddp_%j.err
#SBATCH --signal=B:USR1@300

module load anaconda3
module load cuda/12.1
conda activate microdlm
cd ~/microDLM

# Launch with torchrun (handles distributed setup)
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    train_scaled.py \
    --config medium \
    --resume \
    --batch_size 8  # per-GPU batch size; effective = 8 × 4 = 32
```

`torchrun` sets the environment variables (RANK, LOCAL_RANK, WORLD_SIZE) that
DDP needs. You don't need to manage this manually.

---

## Scaling Roadmap (ordered by priority)

### Stage 1: Validate on HPC (1-2 days)
    [ ] Environment setup (conda, PyTorch, CUDA)
    [ ] Run existing Shakespeare training on 1x V100 or A100
    [ ] Verify weights saved and generation works
    [ ] No code changes

### Stage 2: Add checkpoint-resume (1 day)
    [ ] Implement save/load checkpoint
    [ ] Test: start training, kill it, resume, verify loss continues dropping
    [ ] Create auto-resubmit SLURM script
    [ ] This is required for everything after

### Stage 3: BPE tokenizer (2-3 days)
    [ ] Integrate GPT-2 tokenizer (or train custom 4096-vocab BPE)
    [ ] Update model: vocab_size, MASK token handling
    [ ] Update data loading for tokenized binary shards
    [ ] Retrain on Shakespeare with BPE as sanity check
    [ ] Compare loss curves: char-level vs BPE

### Stage 4: Dataset upgrade (2-3 days)
    [ ] Download and preprocess OpenWebText (CPU job, ~12 hours)
    [ ] Or: start with 100M-token subset for fast iteration
    [ ] Update data loader for sharded binary format
    [ ] Train on OWT with current 10.7M architecture as baseline

### Stage 5: Scale architecture to 124M (3-5 days)
    [ ] Implement config system (small/medium/large presets)
    [ ] Add SwiGLU MLP, weight tying, gradient checkpointing
    [ ] Add mixed precision (bfloat16 on A100, float16 on V100)
    [ ] Train on 1x A100: verify loss drops, check memory usage
    [ ] Add learning rate warmup + cosine decay schedule

### Stage 6: Multi-GPU training (2-3 days)
    [ ] Wrap model in DDP
    [ ] Test on 2x GPU, verify gradient sync works
    [ ] Scale to 4x A100
    [ ] Full training run: 124M model on 1B tokens
    [ ] Evaluate: compute perplexity, generate samples, compare to published MDLM numbers

### Stage 7: Evaluation and comparison (2-3 days)
    [ ] Compute perplexity on standard benchmarks (PTB, WikiText, LM1B)
    [ ] Generate samples and compute generative perplexity under GPT-2
    [ ] Compare against:
        - Your GPT baseline (same architecture, same data)
        - Published MDLM numbers (their 124M model on OWT)
        - Published SEDD numbers (their medium model on OWT)
    [ ] Write up results for blog/paper

---

## Compute Budget Estimates

### 124M model, 1B tokens, GPT-2 tokenizer

    1x A100 (bfloat16):   ~3,500 tokens/sec → 80 hours → 11 SLURM jobs
    4x A100 (DDP, bf16):  ~13,000 tokens/sec → 21 hours → 3 SLURM jobs
    1x V100 (float16):    ~1,200 tokens/sec → 230 hours → 30 SLURM jobs (not recommended)

    Note: diffusion training processes ~50% fewer "useful" tokens per step
    (only masked positions contribute to loss), so effective training may
    need ~2x the iterations of AR training to match token exposure.
    Budget accordingly.

### Storage requirements

    OpenWebText tokenized:    ~19 GB (uint16, ~9.7B tokens)
    1B-token subset:          ~2 GB
    Model checkpoints (124M): ~500 MB each
    Total for full pipeline:  ~25 GB

    Check your storage quota: run `squota` on the cluster.

---

## Honest Assessment: What's Realistic

VERY FEASIBLE (2-3 weeks):
    124M parameter diffusion LM trained on a 1B-token subset of OpenWebText
    with BPE tokenization, directly comparable to MDLM's published results.
    This is a strong research artifact.

FEASIBLE WITH EFFORT (4-6 weeks):
    124M model trained on the full 9.7B-token OpenWebText dataset.
    Requires reliable multi-GPU DDP and many chained SLURM jobs.
    Would produce numbers competitive with published MDLM results.

STRETCH (8+ weeks, may not be feasible with 8-hour job limits):
    350M model (GPT-2 Medium scale). Memory-feasible on A100 with gradient
    checkpointing, but training time scales roughly 3x from the 124M model.
    Would need to chain 30+ SLURM jobs reliably.

NOT FEASIBLE on this cluster:
    1B+ parameter models (LLaDA scale). Would require 8x H100/H200 with
    48+ hour job limits. The 8-hour gpu partition cap and mixed H200/H100
    availability make this impractical.

The 124M model on 1B tokens is the sweet spot: achievable in 2-3 weeks,
produces publishable results, directly comparable to MDLM/SEDD papers,
and demonstrates that you can scale a from-scratch implementation to a
real research-grade model.
