"""
step2_transformer.py — Bidirectional Transformer Denoiser
==========================================================
Replaces the MLP from step1 with a bidirectional transformer.
This is where the quality jump happens — attention allows the model
to use context from ALL positions (both left and right) to predict
masked tokens.

Architecture (inspired by nathan-barry/tiny-diffusion):
- Token embedding + Rotary Positional Embeddings (RoPE)
- Multi-head BIDIRECTIONAL self-attention (no causal mask!)
- MLP block with ReluSquared activation
- RMSNorm (functional, no learnable params)
- Residual connections
- Output projection

Key difference from GPT: is_causal=False in attention.
The diffusion model can see ALL tokens (including future ones)
because it's predicting masked tokens, not next tokens.

Starting hyperparameters (build plan says start small):
- n_layers = 4, n_heads = 4, n_embd = 128, block_size = 256

Expected results:
- Loss drops significantly below MLP ceiling (~3.3) to ~1.5-1.8
- Model learns character-level patterns: common words, spacing, punctuation
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================================
# Hyperparameters
# ============================================================================

batch_size = 64
block_size = 256  # context length
max_iters = 10000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# Model architecture (starting small per build plan)
n_embd = 128
n_head = 4
n_layer = 4
head_dim = n_embd // n_head  # 32

torch.manual_seed(1337)
print(f"Using device: {device}")

# ============================================================================
# Data Loading & Vocabulary
# ============================================================================

data_path = os.path.join(os.path.dirname(__file__), "data", "shakespeare.txt")
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
MASK_CHAR = "_"
assert MASK_CHAR not in chars, f"MASK character '{MASK_CHAR}' already in text!"
chars = [MASK_CHAR] + chars
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
mask_token_id = stoi[MASK_CHAR]


def encode(s):
    return [stoi[ch] for ch in s]


def decode(tokens):
    return "".join([itos[t] for t in tokens])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Vocab size: {vocab_size}, Train: {len(train_data):,}, Val: {len(val_data):,}")
print(f"Architecture: {n_layer} layers, {n_head} heads, {n_embd} embd, {head_dim} head_dim")
print()

# ============================================================================
# Batching with Masking (same as step1)
# ============================================================================


def get_batch(split):
    """Get a batch with random masking. Returns (x_masked, y_clean, mask)."""
    d = train_data if split == "train" else val_data
    idx = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in idx])
    y = x.clone()

    # Random masking: uniform mask probability per sample
    mask_probs = torch.rand(batch_size, 1)
    mask = torch.rand(batch_size, block_size) < mask_probs
    x[mask] = mask_token_id

    x, y, mask = x.to(device), y.to(device), mask.to(device)
    return x, y, mask


# ============================================================================
# Transformer Components
# ============================================================================


def norm(x):
    """Functional RMSNorm — no learnable parameters.
    Following Karpathy/reference repo style."""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    """Apply rotary positional embeddings to queries and keys.

    RoPE encodes relative position information by rotating pairs of
    dimensions in the embedding space. This gives the model position
    awareness without explicit position embeddings.

    Args:
        x: (B, T, H, D) — queries or keys
        cos, sin: (1, T, 1, D/2) — precomputed rotation matrices
    """
    assert x.ndim == 4  # (B, T, H, D) for multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split last dim into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dimensions
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)  # re-assemble
    return out.to(x.dtype)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE.

    CRITICAL DIFFERENCE FROM GPT: is_causal=False
    The diffusion model uses bidirectional attention — it can see
    all positions, including future ones, to predict masked tokens.
    """

    def __init__(self):
        super().__init__()
        self.c_q = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()

        # Project to queries, keys, values
        q = self.c_q(x).view(B, T, n_head, head_dim)
        k = self.c_k(x).view(B, T, n_head, head_dim)
        v = self.c_v(x).view(B, T, n_head, head_dim)

        # Apply Rotary Embeddings for positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        # QK norm for training stability
        q, k = norm(q), norm(k)

        # Transpose for attention: (B, T, H, D) -> (B, H, T, D)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # BIDIRECTIONAL attention (is_causal=False) — the key difference!
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Re-assemble heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """Feed-forward network with ReluSquared activation.
    ReluSquared: ReLU(x)^2 — provides sparsity like ReLU but smoother gradients."""

    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReluSquared activation
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer block: attention + MLP with residual connections and RMSNorm."""

    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.mlp = MLP()

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)  # pre-norm + residual
        x = x + self.mlp(norm(x))  # pre-norm + residual
        return x


# ============================================================================
# Full Model
# ============================================================================


class Model(nn.Module):
    """Bidirectional Transformer for discrete diffusion denoising."""

    def __init__(self):
        super().__init__()

        # Token embeddings (no separate position embeddings — we use RoPE)
        self.token_emb = nn.Embedding(vocab_size, n_embd)

        # Precompute rotary embeddings
        self.rotary_seq_len = block_size * 2
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])

        # Output head
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _precompute_rotary_embeddings(self, seq_len, base=10000, device=None):
        """Precompute cos and sin for rotary positional embeddings."""
        if device is None:
            device = self.token_emb.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        # Add batch and head dims: (seq_len, D/2) -> (1, seq_len, 1, D/2)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def forward(self, idx, targets=None, mask=None):
        """
        Forward pass.

        Args:
            idx: input token indices (B, T) — with MASK tokens at masked positions
            targets: clean token indices for loss (B, T)
            mask: boolean mask, True at masked positions (B, T)

        Returns:
            logits: (B, T, vocab_size)
            loss: cross-entropy at masked positions only
        """
        B, T = idx.size()

        # Token embeddings
        x = self.token_emb(idx)  # (B, T, n_embd)
        x = norm(x)

        # Get rotary embeddings for this sequence length
        assert T <= self.cos.size(1)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])

        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x, cos_sin)
        x = norm(x)

        # Project to vocabulary
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)

            # Only compute loss on masked tokens
            if mask is not None:
                mask_flat = mask.view(B * T).float()
                loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
                loss = (loss * mask_flat).sum() / mask_flat.sum()
            else:
                loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss


# ============================================================================
# Training & Evaluation
# ============================================================================


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, M = get_batch(split)
            _, loss = model(X, Y, M)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


@torch.no_grad()
def generate_sample(model, length=240):
    """Simple greedy denoising: mask everything, predict, take argmax.
    This is NOT the proper iterative sampling (that's step3).
    Just a quick quality check."""
    model.eval()

    # Start from all MASKs
    x = torch.full((1, min(length, block_size)), mask_token_id, dtype=torch.long, device=device)

    # Use first 16 tokens as prompt (unmasked context)
    prompt_len = 16
    x[0, :prompt_len] = data[:prompt_len].to(device)

    # Single-pass prediction (not iterative — that's for step3)
    logits, _ = model(x)
    predictions = logits.argmax(dim=-1)

    # Keep prompt, replace masked positions with predictions
    result = x.clone()
    result[0, prompt_len:] = predictions[0, prompt_len:]

    model.train()
    return decode(result[0].cpu().tolist())


def train():
    model = Model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Transformer parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    log = []
    start = time.time()

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            elapsed = time.time() - start
            log_entry = {
                "iter": iter,
                "train_loss": losses["train"].item(),
                "val_loss": losses["val"].item(),
                "time": elapsed,
            }
            log.append(log_entry)
            print(
                f"step {iter:5d}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, "
                f"time {elapsed:.1f}s"
            )

            # Generate a quick sample every eval
            if iter > 0:
                sample = generate_sample(model)
                print(f"  Sample: {repr(sample[:100])}")
                print()

        # Training step
        xb, yb, mb = get_batch("train")
        _, loss = model(xb, yb, mb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total_time = time.time() - start
    print(f"\nTotal training time: {total_time:.1f}s")

    return model, log


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("MicroDiffusion LM — Step 2: Transformer Denoiser")
    print("=" * 60)
    print()

    model, log = train()

    print("\n" + "=" * 60)
    print("Phase 3 Complete!")
    print("=" * 60)
    print()
    print("Loss comparison:")
    print(f"  Random guessing: {math.log(vocab_size):.4f}")
    print(f"  MLP ceiling:     ~3.31")
    print(f"  Transformer:     {log[-1]['train_loss']:.4f} (train), {log[-1]['val_loss']:.4f} (val)")
    print()
    print("The transformer should be significantly below the MLP ceiling.")
    print("It can use bidirectional context to predict masked tokens.")
    print()

    # Final samples
    print("Final generation samples (single-pass, not iterative):")
    for i in range(3):
        torch.manual_seed(i * 100)
        sample = generate_sample(model)
        print(f"  Sample {i+1}: {repr(sample[:120])}")
    print()
    print("Note: These are single-pass predictions (not proper iterative")
    print("sampling). Quality will improve significantly in step3.")
