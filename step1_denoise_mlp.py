"""
step1_denoise_mlp.py — Train a Simple MLP to Denoise
=====================================================
Trains a simple feedforward network to predict masked tokens.
This will be BAD at generation but proves the training loop works.

The MLP processes each position independently (no attention, no positional
encoding), so it can only learn character frequencies, not context.

What we expect:
- Loss starts ~4.2 (random guessing over 66 chars: -log(1/66) ≈ 4.19)
- Loss drops to ~2.5-3.0 (learns character frequencies but no context)
- Common chars like 'e', 't', ' ' are predicted, but not context-dependent ones

This validates:
1. The data pipeline (tokenization, batching, masking)
2. The loss computation (cross-entropy at masked positions only)
3. Shows WHY attention is needed — the MLP can't use context
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
n_embd = 128  # embedding dimension
n_hidden = 512  # hidden layer size
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

torch.manual_seed(1337)
print(f"Using device: {device}")

# ============================================================================
# Data Loading & Vocabulary (same as step0)
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


# Encode and split data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Vocab size: {vocab_size}")
print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")
print(f"Random guessing loss: {math.log(vocab_size):.4f}")
print()

# ============================================================================
# Batching with Masking
# ============================================================================


def get_batch(split):
    """
    Get a batch of data with random masking.

    Following the reference repo's approach: each sample gets a random
    mask probability drawn uniformly from [0, 1], then each token is
    independently masked with that probability.

    Returns:
        x: masked input tokens (B, T)
        y: original (clean) tokens (B, T)
        mask: boolean mask, True where tokens were masked (B, T)
    """
    d = train_data if split == "train" else val_data
    idx = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in idx])
    y = x.clone()  # original tokens are the targets

    # Random masking: each sample gets a random mask probability
    mask_probs = torch.rand(batch_size, 1)  # (B, 1)
    mask = torch.rand(batch_size, block_size) < mask_probs  # (B, T)
    x[mask] = mask_token_id

    x, y, mask = x.to(device), y.to(device), mask.to(device)
    return x, y, mask


# ============================================================================
# Simple MLP Model
# ============================================================================


class DenoiseMLP(nn.Module):
    """
    Simple MLP denoiser. Processes each position independently.

    Architecture:
        token_embedding → hidden1 (ReLU) → hidden2 (ReLU) → output_projection

    No attention, no positional encoding. Each position sees only its own
    embedding (which is MASK for masked tokens, or the actual char for
    unmasked tokens).

    This means:
    - For masked positions: the MLP only sees MASK embedding → can only
      learn the marginal character distribution (most common chars)
    - For unmasked positions: the MLP sees the char → can learn identity
      (but these positions don't contribute to loss)
    """

    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, vocab_size),
        )

    def forward(self, idx, targets=None, mask=None):
        """
        Args:
            idx: token indices (B, T)
            targets: original tokens for loss computation (B, T)
            mask: boolean mask, True at masked positions (B, T)

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar, cross-entropy at masked positions only
        """
        x = self.token_emb(idx)  # (B, T, n_embd)
        logits = self.net(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)

            if mask is not None:
                # Only compute loss at masked positions
                mask_flat = mask.view(B * T).float()
                loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
                loss = (loss * mask_flat).sum() / mask_flat.sum()
            else:
                loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss


# ============================================================================
# Training
# ============================================================================


@torch.no_grad()
def estimate_loss(model):
    """Estimate loss on train and val splits."""
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


def train():
    """Train the MLP denoiser."""
    model = DenoiseMLP().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLP parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training log
    log = []
    start = time.time()

    for iter in range(max_iters):
        # Evaluate periodically
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

        # Training step
        xb, yb, mb = get_batch("train")
        _, loss = model(xb, yb, mb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total_time = time.time() - start
    print(f"\nTotal training time: {total_time:.1f}s")

    return model, log


def analyze_predictions(model):
    """Analyze what the MLP has learned."""
    print("\n" + "=" * 60)
    print("PREDICTION ANALYSIS")
    print("=" * 60)

    model.eval()

    # Get a batch of fully masked tokens — what does the MLP predict from just MASK?
    x = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(x)

    # The logits should be the same for every position (since all see MASK embedding)
    probs = F.softmax(logits[0, 0], dim=-1)  # probs for first position
    top_k = 10
    top_probs, top_ids = torch.topk(probs, top_k)

    print("\nTop predictions from MASK token (learned character frequencies):")
    for i in range(top_k):
        char = itos[top_ids[i].item()]
        prob = top_probs[i].item()
        char_display = repr(char)
        print(f"  {i+1}. {char_display:6s} prob={prob:.4f}")

    # Compare with actual character frequencies in the dataset
    print("\nActual top character frequencies in dataset:")
    char_counts = {}
    for ch in text:
        char_counts[ch] = char_counts.get(ch, 0) + 1
    sorted_chars = sorted(char_counts.items(), key=lambda x: -x[1])
    for i, (ch, count) in enumerate(sorted_chars[:top_k]):
        freq = count / len(text)
        char_display = repr(ch)
        print(f"  {i+1}. {char_display:6s} freq={freq:.4f}")

    # Show a sample denoising attempt
    print("\nSample denoising attempt:")
    sample = data[:block_size].to(device).unsqueeze(0)
    y_clean = sample.clone()

    # Mask ~50% of tokens
    mask = torch.rand(1, block_size, device=device) < 0.5
    x_masked = sample.clone()
    x_masked[mask] = mask_token_id

    with torch.no_grad():
        logits, _ = model(x_masked)
    predictions = logits.argmax(dim=-1)

    # Show first 80 chars
    display_len = 80
    original = decode(y_clean[0, :display_len].cpu().tolist())
    masked_text = decode(x_masked[0, :display_len].cpu().tolist())
    predicted = decode(predictions[0, :display_len].cpu().tolist())

    print(f"  Original:  {repr(original)}")
    print(f"  Masked:    {repr(masked_text)}")
    print(f"  Predicted: {repr(predicted)}")

    # Calculate accuracy at masked positions
    mask_2d = mask if mask.dim() == 2 else mask.unsqueeze(0)
    mask_flat = mask_2d.view(-1)
    pred_flat = predictions.view(-1)
    true_flat = y_clean.view(-1)
    masked_correct = (pred_flat[mask_flat] == true_flat[mask_flat]).float().mean()
    print(f"\n  Accuracy at masked positions: {masked_correct:.4f} ({masked_correct*100:.1f}%)")
    print(f"  (Random chance: {1/vocab_size:.4f} = {100/vocab_size:.1f}%)")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("MicroDiffusion LM — Step 1: MLP Denoiser")
    print("=" * 60)
    print()

    model, log = train()

    analyze_predictions(model)

    print("\n" + "=" * 60)
    print("Phase 2 Complete!")
    print("=" * 60)
    print()
    print("Key observations:")
    print(f"  • Starting loss ~{log[0]['train_loss']:.2f} (random: {math.log(vocab_size):.2f})")
    print(f"  • Final loss ~{log[-1]['train_loss']:.2f} (learns character frequencies)")
    print("  • MLP hits a ceiling because it processes each position independently")
    print("  • It can predict common chars (space, 'e', 't') but not context")
    print("  • This proves: data pipeline ✓, loss computation ✓, WHY we need attention ✓")
