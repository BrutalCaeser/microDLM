#!/usr/bin/env python3
"""
gpt.py — Autoregressive GPT Language Model from Scratch
========================================================
A single-file implementation of a character-level GPT trained on Tiny
Shakespeare. Everything is here: data loading, model definition, training
loop, and generation.

The entire model is ~10.7M parameters with the default 6L/6H/384E architecture.

Train:     python gpt.py --train
Generate:  python gpt.py --generate
Both:      python gpt.py --train --generate

Architecture: Causal Transformer with RoPE, RMSNorm, ReluSquared, QK-Norm.
Training:     Standard next-token prediction (cross-entropy over all positions).
Generation:   Sequential left-to-right, one token at a time.

Compare with diffusion.py — the 5 differences are marked with: # ← DIFF
"""

import os
import sys
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
max_iters = 5000        # GPT needs fewer iters (every token contributes to loss)
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

# ============================================================================
# Data — Tiny Shakespeare
# ============================================================================

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "shakespeare.txt")
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
MASK_CHAR = "_"                                                # ← DIFF 1: keep MASK for
chars_with_mask = [MASK_CHAR] + chars                          #   fair param comparison,
vocab_size = len(chars_with_mask)                              #   but it's never used

stoi = {ch: i for i, ch in enumerate(chars_with_mask)}
itos = {i: ch for i, ch in enumerate(chars_with_mask)}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# ============================================================================
# Batching — Next-Token Prediction                             # ← DIFF 3/4
# ============================================================================

def get_batch(split):
    """Standard autoregressive batching: x[i] → y[i] = x[i+1]."""
    d = train_data if split == "train" else val_data
    idx = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in idx])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in idx])
    return x.to(device), y.to(device)

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
                                           is_causal=True)     # ← DIFF 2: causal
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


class GPT(nn.Module):
    """Causal transformer trained with next-token prediction."""

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

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = norm(self.token_emb(idx))
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        for block in self.blocks:
            x = block(x, cos_sin)
        logits = self.lm_head(norm(x))

        if targets is None:
            return logits, None

        # Loss over ALL positions                               # ← DIFF 4
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
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
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train():
    print(f"Training GPT on {device}")
    print(f"Architecture: {n_layer}L / {n_head}H / {n_embd}E ({head_dim}D)")
    print(f"Vocab: {vocab_size} chars (same as diffusion for fair comparison)")
    print(f"Training for {max_iters} iterations")
    print("=" * 60)

    model = GPT().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
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
            if it > 0 and it % 1000 == 0:
                sample = generate(model, max_new_tokens=240)
                print(f"  → {repr(sample[:120])}\n")

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total = time.time() - start
    print(f"\nDone in {total:.1f}s — train {log[-1]['train_loss']:.4f}, "
          f"val {log[-1]['val_loss']:.4f}")

    # Save
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/gpt.pt")
    print("Saved weights/gpt.pt")

    with open("weights/training_log_gpt.json", "w") as f:
        json.dump(log, f, indent=2)

    return model

# ============================================================================
# Generation — Sequential Left-to-Right                        # ← DIFF 5
# ============================================================================

@torch.no_grad()
def generate(model, max_new_tokens=500, prompt_len=16, temp=0.8):
    """Standard autoregressive generation: one token at a time, left to right."""
    model.eval()
    x = data[:prompt_len].unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        ctx = x[:, -block_size:]
        logits = model(ctx)[0]
        logits = logits[:, -1, :]                              # last position only
        probs = F.softmax(logits / temp, dim=-1)
        next_token = (torch.argmax(probs, dim=-1, keepdim=True) if temp == 0
                      else torch.multinomial(probs, num_samples=1))
        x = torch.cat([x, next_token], dim=1)

    model.train()
    return decode(x[0].cpu().tolist())

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Language Model on Tiny Shakespeare")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--generate", action="store_true", help="Generate text from saved weights")
    parser.add_argument("--tokens", type=int, default=500, help="Tokens to generate (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not args.train and not args.generate:
        parser.print_help()
        print("\nExample:  python gpt.py --train --generate")
        sys.exit(0)

    torch.manual_seed(args.seed)

    if args.train:
        model = train()
    else:
        model = None

    if args.generate:
        if model is None:
            path = os.path.join("weights", "gpt.pt")
            if not os.path.exists(path):
                print(f"No weights found at {path}. Train first with --train")
                sys.exit(1)
            model = GPT().to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            print(f"Loaded {path}")

        print("\n" + "=" * 60)
        print("Generating sequentially (left-to-right, one token at a time)")
        print(f"(temp=0.8)")
        print("=" * 60 + "\n")
        start = time.time()
        output = generate(model, max_new_tokens=args.tokens)
        elapsed = time.time() - start
        print(output)
        print(f"\n[{len(output)} chars in {elapsed:.2f}s, sequential]")
