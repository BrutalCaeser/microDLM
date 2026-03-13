#!/usr/bin/env python3
"""
diffusion.py — Discrete Diffusion Language Model from Scratch
=============================================================
A single-file implementation of a discrete (masked) diffusion language model.
Train on Tiny Shakespeare (--data shakespeare) or OpenWebText (--data openwebtext).
Everything is here: data loading, model definition, training loop, and generation.

The entire model is ~10.7M parameters with the default 6L/6H/384E architecture.

Train:     python diffusion.py --train
Generate:  python diffusion.py --generate
Both:      python diffusion.py --train --generate

Architecture: Bidirectional Transformer with RoPE, RMSNorm, ReluSquared, QK-Norm.
Training:     Randomly mask tokens with uniform probability, predict original tokens.
Generation:   Iterative parallel unmasking with cosine schedule (SUBS parameterization).

Compare with gpt.py — the 5 differences are marked with: # ← DIFF
"""

import os
import sys
import math
import time
import json
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================================
# Hyperparameters
# ============================================================================

batch_size = 64
block_size = 256        # context window
max_iters = 10000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4

n_embd = 384
n_head = 6
n_layer = 6
head_dim = n_embd // n_head  # 64

device = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# Parse --data before loading so we know which dataset to use
_data_parser = argparse.ArgumentParser()
_data_parser.add_argument("--data", default="shakespeare", choices=["shakespeare", "openwebtext"])
_DATA_ARGS, _ = _data_parser.parse_known_args()
DATA_SOURCE = _DATA_ARGS.data

# ============================================================================
# Data — Shakespeare or OpenWebText
# ============================================================================

train_data = val_data = data = None
_train_loader = _val_loader = None
_train_iter = _val_iter = None
_prompt_block = None  # used by generate() when DATA_SOURCE == "openwebtext"

if DATA_SOURCE == "shakespeare":
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "shakespeare.txt")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    MASK_CHAR = "_"                                            # ← DIFF 1: mask token
    assert MASK_CHAR not in chars, "MASK_CHAR must not appear in data"
    chars_with_mask = [MASK_CHAR] + chars
    vocab_size = len(chars_with_mask)
    stoi = {ch: i for i, ch in enumerate(chars_with_mask)}
    itos = {i: ch for i, ch in enumerate(chars_with_mask)}
    mask_token_id = stoi[MASK_CHAR]
    encode = lambda s: [stoi[ch] for ch in s]
    decode = lambda l: "".join([itos[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
else:
    from data_openwebtext import load_openwebtext
    _train_loader, _val_loader, stoi, itos, vocab_size, mask_token_id, encode, decode = load_openwebtext(
        block_size=block_size, batch_size=batch_size, num_workers=0, pin_memory=(device.type == "cuda")
    )
    # Prompt for generation: first block from first val batch
    _prompt_block = next(iter(_val_loader))[0][0].clone()

# ============================================================================
# Batching — Random Masking                                    # ← DIFF 3/4
# ============================================================================

def _to_device(*tensors):
    """Move tensors to device. Use pinned memory + non_blocking for CUDA so GPU
    is not blocked waiting on transfer (enables overlap with next batch prep)."""
    out = []
    for t in tensors:
        if device.type == "cuda":
            t = t.pin_memory().to(device, non_blocking=True)
        else:
            t = t.to(device)
        out.append(t)
    return tuple(out)


def get_batch(split):
    """Each sample gets a random mask rate from U[0,1], then each token is
    independently masked with that probability. Returns (x, y, mask).
    Built on CPU; transferred to device with pinned+non_blocking when CUDA."""
    global _train_iter, _val_iter
    if DATA_SOURCE == "shakespeare":
        d = train_data if split == "train" else val_data
        idx = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i : i + block_size] for i in idx])
    else:
        loader = _train_loader if split == "train" else _val_loader
        it = _train_iter if split == "train" else _val_iter
        if it is None:
            it = iter(loader)
            if split == "train":
                _train_iter = it
            else:
                _val_iter = it
        try:
            blocks = next(it)
        except StopIteration:
            it = iter(loader)
            if split == "train":
                _train_iter = it
            else:
                _val_iter = it
            blocks = next(it)
        x = blocks  # (B, block_size)
    y = x.clone()
    mask_probs = torch.rand(batch_size, 1)
    mask = torch.rand(batch_size, block_size) < mask_probs
    x[mask] = mask_token_id
    return _to_device(x, y, mask)

# ============================================================================
# Model Components
# ============================================================================

def norm(x):
    """Functional RMSNorm — no learnable parameters."""
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    """Rotary positional embeddings on queries/keys."""
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).to(x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_q = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, n_head, head_dim)
        k = self.c_k(x).view(B, T, n_head, head_dim)
        v = self.c_v(x).view(B, T, n_head, head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)                                # QK-norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v,
                                           is_causal=False)    # ← DIFF 2: bidirectional
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())      # ReluSquared


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.mlp = MLP()

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


class DiffusionLM(nn.Module):
    """Bidirectional transformer trained as a masked diffusion language model."""

    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        cos, sin = self._precompute_rope(block_size * 2)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _precompute_rope(self, seq_len, base=10000):
        dev = self.token_emb.weight.device if hasattr(self, "token_emb") else "cpu"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=dev) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=dev)
        freqs = torch.outer(t, inv_freq)
        return freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]

    def forward(self, idx, targets=None, mask=None):
        B, T = idx.size()
        x = norm(self.token_emb(idx))
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        for block in self.blocks:
            x = block(x, cos_sin)
        logits = self.lm_head(norm(x))

        if targets is None:
            return logits, None

        # Loss only on masked positions                         # ← DIFF 4
        logits_flat = logits.view(B * T, -1)
        targets_flat = targets.view(B * T)
        mask_flat = mask.view(B * T).float()
        loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        loss = (loss * mask_flat).sum() / mask_flat.sum()
        return logits, loss

# ============================================================================
# Training
# ============================================================================

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, M = get_batch(split)
            _, loss = model(X, Y, M)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train():
    print(f"Training Diffusion LM on {device}")
    print(f"Architecture: {n_layer}L / {n_head}H / {n_embd}E ({head_dim}D)")
    data_desc = "OpenWebText" if DATA_SOURCE == "openwebtext" else "Shakespeare"
    print(f"Data: {data_desc} — vocab {vocab_size} chars")
    print(f"Training for {max_iters} iterations")
    print("=" * 60)

    # Phase 1: Build model and optimizer on CPU (no GPU use yet).
    model = DiffusionLM()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Phase 2: Move model to GPU once; sync so we're in a clean state.
    model.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    log = []
    start = time.time()

    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model)
            elapsed = time.time() - start
            log.append(dict(iter=it, train_loss=losses["train"].item(),
                            val_loss=losses["val"].item(), time=elapsed))
            print(f"step {it:5d}: train {losses['train']:.4f}, "
                  f"val {losses['val']:.4f}, time {elapsed:.1f}s")
            if it > 0 and it % 2000 == 0:
                sample = generate(model, max_new_tokens=240)
                print(f"  → {repr(sample[:120])}\n")

        xb, yb, mb = get_batch("train")
        _, loss = model(xb, yb, mb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total = time.time() - start
    print(f"\nDone in {total:.1f}s — train {log[-1]['train_loss']:.4f}, "
          f"val {log[-1]['val_loss']:.4f}")

    # Phase 3: Offload model to CPU before save (free GPU during I/O).
    os.makedirs("weights", exist_ok=True)
    model.cpu()
    torch.save(model.state_dict(), "weights/diffusion.pt")
    print("Saved weights/diffusion.pt")
    model.to(device)  # back on device for any subsequent generate

    with open("weights/training_log_diffusion.json", "w") as f:
        json.dump(log, f, indent=2)

    return model

# ============================================================================
# Generation — Iterative Parallel Unmasking                    # ← DIFF 5
# ============================================================================

@torch.no_grad()
def generate(model, max_new_tokens=500, prompt_len=16, num_steps=40,
             temp=0.8, top_k=3):
    """
    Cosine-scheduled parallel unmasking (SUBS parameterization).

    1. Start from [prompt] + [MASK × N]
    2. At each step, predict all positions, unmask the most-confident batch
    3. Schedule:  n_masked(t) = N × cos(π/2 × t/T)²
    4. SUBS: logits[:, :, MASK] = -∞  (never predict MASK)
    """
    model.eval()
    if DATA_SOURCE == "openwebtext" and _prompt_block is not None:
        prompt = _prompt_block[:prompt_len].tolist()
    else:
        prompt = data[:prompt_len].tolist()
    gen_len = min(max_new_tokens, block_size) - prompt_len

    x = torch.full((1, gen_len + prompt_len), mask_token_id,
                    dtype=torch.long, device=device)
    x[0, :prompt_len] = torch.tensor(prompt, device=device)
    masked = torch.zeros(1, x.size(1), dtype=torch.bool, device=device)
    masked[0, prompt_len:] = True

    # Cosine schedule
    schedule = [int(round(gen_len * math.cos(math.pi / 2 * t / num_steps) ** 2))
                for t in range(num_steps + 1)]
    schedule[0], schedule[-1] = gen_len, 0

    for step in range(1, num_steps + 1):
        n_unmask = schedule[step - 1] - schedule[step]
        if n_unmask <= 0 or not masked.any():
            continue

        logits = model(x)[0]
        logits[:, :, mask_token_id] = -float("inf")            # SUBS
        probs = F.softmax(logits / temp, dim=-1)

        top_probs, top_idx = torch.topk(probs, k=top_k, dim=-1)
        confidence = top_probs[:, :, 0]

        # Pick most-confident masked positions
        conf = torch.where(masked, confidence, torch.tensor(-float("inf"), device=device))
        n_unmask = min(n_unmask, masked.sum().item())
        _, best = torch.topk(conf.view(-1), k=int(n_unmask))
        decode_mask = torch.zeros_like(masked.view(-1))
        decode_mask[best] = True
        decode_mask = decode_mask.view_as(masked).bool()

        # Sample from top-k
        top_probs_norm = top_probs / top_probs.sum(dim=-1, keepdim=True)
        sampled_k = torch.multinomial(top_probs_norm.view(-1, top_k), 1).view(1, x.size(1))
        tokens = torch.gather(top_idx, -1, sampled_k.unsqueeze(-1)).squeeze(-1)

        x = torch.where(decode_mask, tokens, x)
        masked = masked & ~decode_mask

    model.train()
    return decode(x[0].cpu().tolist())

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discrete Diffusion LM")
    parser.add_argument("--data", default="shakespeare", choices=["shakespeare", "openwebtext"],
                        help="Dataset: shakespeare (tiny) or openwebtext (default: shakespeare)")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--generate", action="store_true", help="Generate text from saved weights")
    parser.add_argument("--tokens", type=int, default=500, help="Tokens to generate (default: 500)")
    parser.add_argument("--steps", type=int, default=40, help="Diffusion steps (default: 40)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not args.train and not args.generate:
        parser.print_help()
        print("\nExample:  python diffusion.py --train --generate")
        sys.exit(0)

    torch.manual_seed(args.seed)

    if args.train:
        model = train()
    else:
        model = None

    if args.generate:
        if model is None:
            path = os.path.join("weights", "diffusion.pt")
            if not os.path.exists(path):
                print(f"No weights found at {path}. Train first with --train")
                sys.exit(1)
            model = DiffusionLM()
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            model.to(device)
            print(f"Loaded {path}")

        print("\n" + "=" * 60)
        print("Generating with cosine-scheduled parallel unmasking")
        print(f"({args.steps} diffusion steps, top-k=3, temp=0.8)")
        print("=" * 60 + "\n")
        start = time.time()
        output = generate(model, max_new_tokens=args.tokens, num_steps=args.steps)
        elapsed = time.time() - start
        print(output)
        print(f"\n[{len(output)} chars in {elapsed:.2f}s, {args.steps} steps]")
