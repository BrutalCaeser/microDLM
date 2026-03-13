"""
step3_sampling.py — Iterative Sampling for Discrete Diffusion
==============================================================
The generation algorithm for discrete diffusion: how to go from
[MASK MASK MASK ...] back to coherent text.

Step 2 showed that single-pass greedy decoding is bad — the model
predicts every masked position independently in one shot, ignoring
the dependencies between its own predictions. This step fixes that.

The key insight: just like the forward process gradually masks tokens,
the reverse process should gradually UNMASK them. At each step the
model re-evaluates all positions given the tokens revealed so far,
and we unmask the positions where the model is most confident.

This file demonstrates five sampling strategies, from worst to best:

1. SINGLE-PASS GREEDY    — One forward pass, argmax everything (step2 baseline)
2. RANDOM-ORDER UNMASK   — Iterative, but unmask in random order
3. CONFIDENCE-BASED      — Iterative, unmask most-confident first
4. COSINE-SCHEDULED      — Confidence-based + cosine schedule for pacing
5. FULL (SUBS + TOP-K)   — Add SUBS constraint + top-k sampling (= diffusion.py)

Each strategy is visualized step-by-step so you can SEE the quality improve.

Uses the step2 model architecture (4L/4H/128E) to keep training fast.
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================================
# Hyperparameters (same as step2 for consistency)
# ============================================================================

batch_size = 64
block_size = 256
max_iters = 10000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

n_embd = 128
n_head = 4
n_layer = 4
head_dim = n_embd // n_head

torch.manual_seed(1337)
print(f"Using device: {device}")

# ============================================================================
# Data Loading & Vocabulary (same as all steps)
# ============================================================================

data_path = os.path.join(os.path.dirname(__file__), "data", "shakespeare.txt")
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
MASK_CHAR = "_"
assert MASK_CHAR not in chars
chars = [MASK_CHAR] + chars
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
mask_token_id = stoi[MASK_CHAR]

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

print(f"Vocab: {vocab_size}, Train: {len(train_data):,}, Val: {len(val_data):,}")

# ============================================================================
# Model (identical to step2 — just the architecture, no changes)
# ============================================================================


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
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
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, n_head, head_dim)
        k = self.c_k(x).view(B, T, n_head, head_dim)
        v = self.c_v(x).view(B, T, n_head, head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.mlp = MLP()

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


class Model(nn.Module):
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
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _precompute_rope(self, seq_len, base=10000):
        dev = self.token_emb.weight.device
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

        logits_flat = logits.view(B * T, -1)
        targets_flat = targets.view(B * T)
        if mask is not None:
            mask_flat = mask.view(B * T).float()
            loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
            loss = (loss * mask_flat).sum() / mask_flat.sum()
        else:
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss


# ============================================================================
# Batching (same as step2)
# ============================================================================


def get_batch(split):
    d = train_data if split == "train" else val_data
    idx = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in idx])
    y = x.clone()
    mask_probs = torch.rand(batch_size, 1)
    mask = torch.rand(batch_size, block_size) < mask_probs
    x[mask] = mask_token_id
    return x.to(device), y.to(device), mask.to(device)


# ============================================================================
# Training (same as step2 — just need a trained model)
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


def train_model():
    """Train the transformer (same as step2). Returns trained model."""
    model = Model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining transformer: {n_params:,} params ({n_params/1e6:.2f}M)")
    print("(Same architecture as step2 — this step is about SAMPLING, not the model)")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start = time.time()

    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model)
            elapsed = time.time() - start
            print(f"step {it:5d}: train {losses['train']:.4f}, val {losses['val']:.4f}, time {elapsed:.1f}s")

        xb, yb, mb = get_batch("train")
        _, loss = model(xb, yb, mb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"\nTraining done in {time.time()-start:.1f}s")
    return model


# ============================================================================
# SAMPLING STRATEGIES — From Worst to Best
# ============================================================================
#
# All strategies start from: [prompt tokens] + [MASK MASK MASK ...]
# and try to fill in the MASKs to produce coherent text.
#
# The prompt gives the model some context to work with.
# ============================================================================

PROMPT_LEN = 16
GEN_LEN = 120  # shorter for clear visualization


def make_initial_sequence(prompt_len=PROMPT_LEN, gen_len=GEN_LEN):
    """Create the starting point: prompt + all MASKs."""
    total = prompt_len + gen_len
    x = torch.full((1, total), mask_token_id, dtype=torch.long, device=device)
    x[0, :prompt_len] = data[:prompt_len].to(device)
    masked = torch.zeros(1, total, dtype=torch.bool, device=device)
    masked[0, prompt_len:] = True
    return x, masked


# ── Strategy 1: Single-Pass Greedy ──────────────────────────────────────────


@torch.no_grad()
def sample_single_pass(model, prompt_len=PROMPT_LEN, gen_len=GEN_LEN):
    """
    STRATEGY 1: Single-Pass Greedy (the step2 baseline)

    One forward pass through the model, take argmax at every masked position.

    Problems:
    - All masked positions are predicted independently in parallel
    - The model sees MASK at every generation position — no revealed context
    - Predictions at adjacent positions can contradict each other
    - This is like trying to write an entire essay in one breath
    """
    model.eval()
    x, masked = make_initial_sequence(prompt_len, gen_len)

    logits, _ = model(x)
    predictions = logits.argmax(dim=-1)
    x[masked] = predictions[masked]

    return decode(x[0].cpu().tolist())


# ── Strategy 2: Random-Order Iterative Unmasking ────────────────────────────


@torch.no_grad()
def sample_random_order(model, num_steps=20, prompt_len=PROMPT_LEN, gen_len=GEN_LEN,
                        trace=False):
    """
    STRATEGY 2: Random-Order Iterative Unmasking

    Unmask positions in a random order over multiple steps.
    At each step:
      1. Run the model (it sees previously unmasked tokens as context)
      2. Pick a random subset of still-masked positions
      3. Fill them in with argmax predictions

    Better than single-pass because the model sees some revealed tokens
    as context for later predictions. But random order means we might
    unmask uncertain positions early and confident ones late.
    """
    model.eval()
    x, masked = make_initial_sequence(prompt_len, gen_len)

    # How many to unmask per step (evenly divided)
    n_masked_total = masked.sum().item()
    per_step = max(1, n_masked_total // num_steps)

    traces = []
    if trace:
        traces.append(("init", decode(x[0].cpu().tolist())))

    for step in range(num_steps):
        if not masked.any():
            break

        # Run model with current context
        logits, _ = model(x)
        predictions = logits.argmax(dim=-1)

        # Pick random masked positions to unmask
        masked_indices = masked[0].nonzero(as_tuple=False).squeeze(-1)
        n_unmask = min(per_step, len(masked_indices))
        if step == num_steps - 1:
            n_unmask = len(masked_indices)  # unmask all remaining on last step

        perm = torch.randperm(len(masked_indices), device=device)[:n_unmask]
        chosen = masked_indices[perm]

        x[0, chosen] = predictions[0, chosen]
        masked[0, chosen] = False

        if trace:
            traces.append((f"step {step+1:2d}", decode(x[0].cpu().tolist())))

    if trace:
        return decode(x[0].cpu().tolist()), traces
    return decode(x[0].cpu().tolist())


# ── Strategy 3: Confidence-Based Iterative Unmasking ────────────────────────


@torch.no_grad()
def sample_confidence(model, num_steps=20, prompt_len=PROMPT_LEN, gen_len=GEN_LEN,
                      trace=False):
    """
    STRATEGY 3: Confidence-Based Iterative Unmasking

    Same as strategy 2, but instead of random order, we unmask the
    positions where the model is MOST CONFIDENT first.

    Intuition: the model is most confident about predictable positions
    (spaces after words, common letter patterns). Revealing these first
    gives the model better context for the harder predictions later.

    This is like solving a crossword puzzle: fill in the easy clues
    first, then use those letters to help with the hard ones.
    """
    model.eval()
    x, masked = make_initial_sequence(prompt_len, gen_len)

    n_masked_total = masked.sum().item()
    per_step = max(1, n_masked_total // num_steps)

    traces = []
    if trace:
        traces.append(("init", decode(x[0].cpu().tolist())))

    for step in range(num_steps):
        if not masked.any():
            break

        logits, _ = model(x)
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values  # confidence = max probability
        predictions = logits.argmax(dim=-1)

        # Among masked positions, pick the most confident ones
        confidence = torch.where(masked, max_probs, torch.tensor(-float("inf"), device=device))
        n_unmask = min(per_step, masked.sum().item())
        if step == num_steps - 1:
            n_unmask = masked.sum().item()

        _, best = torch.topk(confidence.view(-1), k=int(n_unmask))
        x.view(-1)[best] = predictions.view(-1)[best]
        masked.view(-1)[best] = False

        if trace:
            traces.append((f"step {step+1:2d}", decode(x[0].cpu().tolist())))

    if trace:
        return decode(x[0].cpu().tolist()), traces
    return decode(x[0].cpu().tolist())


# ── Strategy 4: Cosine-Scheduled Confidence Unmasking ───────────────────────


@torch.no_grad()
def sample_cosine_schedule(model, num_steps=20, prompt_len=PROMPT_LEN, gen_len=GEN_LEN,
                           trace=False):
    """
    STRATEGY 4: Cosine-Scheduled Confidence Unmasking

    Strategy 3 unmasks the same number per step (uniform pacing).
    But the cosine schedule is better: unmask slowly at first (when
    context is sparse and predictions are uncertain), then accelerate
    as more context is revealed and predictions become easier.

    Schedule: n_masked(t) = N × cos(π/2 × t/T)²

    This mirrors the forward process: the cosine schedule spends more
    time at intermediate noise levels where learning is most productive.
    """
    model.eval()
    x, masked = make_initial_sequence(prompt_len, gen_len)

    # Precompute cosine schedule: how many tokens should REMAIN masked at each step
    schedule = []
    for t in range(num_steps + 1):
        n_remain = int(round(gen_len * math.cos(math.pi / 2 * t / num_steps) ** 2))
        schedule.append(n_remain)
    schedule[0] = gen_len  # all masked at start
    schedule[-1] = 0  # none masked at end

    traces = []
    if trace:
        traces.append(("init", decode(x[0].cpu().tolist())))

    for step in range(1, num_steps + 1):
        n_unmask = schedule[step - 1] - schedule[step]
        if n_unmask <= 0 or not masked.any():
            continue

        logits, _ = model(x)
        max_probs = F.softmax(logits, dim=-1).max(dim=-1).values
        predictions = logits.argmax(dim=-1)

        confidence = torch.where(masked, max_probs, torch.tensor(-float("inf"), device=device))
        n_unmask = min(n_unmask, masked.sum().item())
        _, best = torch.topk(confidence.view(-1), k=int(n_unmask))

        x.view(-1)[best] = predictions.view(-1)[best]
        masked.view(-1)[best] = False

        if trace and (step <= 5 or step % 4 == 0 or step == num_steps):
            traces.append((f"step {step:2d}", decode(x[0].cpu().tolist())))

    if trace:
        return decode(x[0].cpu().tolist()), traces
    return decode(x[0].cpu().tolist())


# ── Strategy 5: Full Algorithm (SUBS + Top-k Sampling) ──────────────────────


@torch.no_grad()
def sample_full(model, num_steps=20, temp=0.8, top_k=3,
                prompt_len=PROMPT_LEN, gen_len=GEN_LEN, trace=False):
    """
    STRATEGY 5: Full Diffusion Sampling (= the generate() in diffusion.py)

    Adds two final ingredients to strategy 4:

    SUBS parameterization:
      Set logits[:, :, MASK] = -inf so the model NEVER predicts MASK as
      an output token. Without this, the model might "chicken out" and
      predict MASK for hard positions, which defeats the purpose.

    Top-k sampling with temperature:
      Instead of argmax (greedy), sample from the top-k predictions with
      a temperature parameter. This adds diversity — the model doesn't
      always pick the most likely token, allowing for more creative output.
      Temperature < 1.0 sharpens the distribution (more conservative).
      Temperature > 1.0 flattens it (more diverse/random).

    This is the complete algorithm from the discrete diffusion literature
    (MDLM / SUBS parameterization).
    """
    model.eval()
    x, masked = make_initial_sequence(prompt_len, gen_len)

    # Cosine schedule
    schedule = []
    for t in range(num_steps + 1):
        n_remain = int(round(gen_len * math.cos(math.pi / 2 * t / num_steps) ** 2))
        schedule.append(n_remain)
    schedule[0] = gen_len
    schedule[-1] = 0

    traces = []
    if trace:
        traces.append(("init", decode(x[0].cpu().tolist())))

    for step in range(1, num_steps + 1):
        n_unmask = schedule[step - 1] - schedule[step]
        if n_unmask <= 0 or not masked.any():
            continue

        logits, _ = model(x)

        # SUBS: never predict MASK token
        logits[:, :, mask_token_id] = -float("inf")

        # Temperature-scaled probabilities
        probs = F.softmax(logits / temp, dim=-1)

        # Top-k: get the k most likely tokens and their probabilities
        top_probs, top_idx = torch.topk(probs, k=top_k, dim=-1)
        confidence = top_probs[:, :, 0]  # confidence = prob of top-1

        # Pick most-confident masked positions
        conf_masked = torch.where(masked, confidence, torch.tensor(-float("inf"), device=device))
        n_unmask = min(n_unmask, masked.sum().item())
        _, best = torch.topk(conf_masked.view(-1), k=int(n_unmask))

        decode_mask = torch.zeros_like(masked.view(-1))
        decode_mask[best] = True
        decode_mask = decode_mask.view_as(masked).bool()

        # Sample from top-k (not greedy!)
        top_probs_norm = top_probs / top_probs.sum(dim=-1, keepdim=True)
        sampled_k = torch.multinomial(top_probs_norm.view(-1, top_k), 1).view(1, x.size(1))
        tokens = torch.gather(top_idx, -1, sampled_k.unsqueeze(-1)).squeeze(-1)

        x = torch.where(decode_mask, tokens, x)
        masked = masked & ~decode_mask

        if trace and (step <= 5 or step % 4 == 0 or step == num_steps):
            traces.append((f"step {step:2d}", decode(x[0].cpu().tolist())))

    if trace:
        return decode(x[0].cpu().tolist()), traces
    return decode(x[0].cpu().tolist())


# ============================================================================
# Visualization Helpers
# ============================================================================


def print_trace(name, traces, show_len=100):
    """Print a step-by-step trace of the denoising process."""
    print(f"\n{'─'*70}")
    print(f"  {name}")
    print(f"{'─'*70}")
    for label, text in traces:
        # Highlight: show MASKs in a visible way
        display = text[:show_len]
        n_masks = display.count(MASK_CHAR)
        pct = 100 * n_masks / len(display) if len(display) > 0 else 0
        bar_len = 20
        filled = int(bar_len * (1 - n_masks / len(display))) if len(display) > 0 else bar_len
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"  {label:>8s} |{bar}| {display}")
    print()


def compare_strategies(model):
    """Run all strategies and show results side by side."""
    print("\n" + "=" * 70)
    print("  SAMPLING STRATEGY COMPARISON")
    print("=" * 70)

    prompt_text = decode(data[:PROMPT_LEN].tolist())
    print(f"\nPrompt ({PROMPT_LEN} chars): {repr(prompt_text)}")
    print(f"Generating {GEN_LEN} characters after prompt")

    # Strategy 1: Single-pass
    torch.manual_seed(42)
    s1 = sample_single_pass(model)
    print(f"\n1. SINGLE-PASS GREEDY:")
    print(f"   {repr(s1[:100])}")

    # Strategy 2: Random order (with trace)
    torch.manual_seed(42)
    s2, t2 = sample_random_order(model, num_steps=20, trace=True)
    print_trace("2. RANDOM-ORDER ITERATIVE (20 steps)", t2)

    # Strategy 3: Confidence-based (with trace)
    torch.manual_seed(42)
    s3, t3 = sample_confidence(model, num_steps=20, trace=True)
    print_trace("3. CONFIDENCE-BASED (20 steps)", t3)

    # Strategy 4: Cosine-scheduled (with trace)
    torch.manual_seed(42)
    s4, t4 = sample_cosine_schedule(model, num_steps=20, trace=True)
    print_trace("4. COSINE-SCHEDULED (20 steps)", t4)

    # Strategy 5: Full algorithm (with trace)
    torch.manual_seed(42)
    s5, t5 = sample_full(model, num_steps=20, temp=0.8, top_k=3, trace=True)
    print_trace("5. FULL (SUBS + top-k=3, temp=0.8, 20 steps)", t5)


def study_num_steps(model):
    """Show how sample quality changes with number of diffusion steps."""
    print("\n" + "=" * 70)
    print("  EFFECT OF NUMBER OF STEPS")
    print("=" * 70)
    print("\nMore steps → better quality (model gets more chances to refine)")
    print()

    for num_steps in [1, 2, 5, 10, 20, 40]:
        torch.manual_seed(42)
        sample = sample_full(model, num_steps=num_steps, temp=0.8, top_k=3)
        print(f"  {num_steps:2d} steps: {repr(sample[:100])}")

    print()


def study_temperature(model):
    """Show how temperature affects generation diversity."""
    print("\n" + "=" * 70)
    print("  EFFECT OF TEMPERATURE")
    print("=" * 70)
    print("\nLower temp → more conservative. Higher temp → more diverse/random.")
    print()

    for temp in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]:
        torch.manual_seed(42)
        sample = sample_full(model, num_steps=20, temp=temp, top_k=5)
        print(f"  temp={temp:.1f}: {repr(sample[:100])}")

    print()


def study_top_k(model):
    """Show how top-k affects generation."""
    print("\n" + "=" * 70)
    print("  EFFECT OF TOP-K")
    print("=" * 70)
    print("\nk=1 is greedy. Higher k → more variety in token choices.")
    print()

    for k in [1, 2, 3, 5, 10]:
        torch.manual_seed(42)
        sample = sample_full(model, num_steps=20, temp=0.8, top_k=k)
        print(f"  top_k={k:2d}: {repr(sample[:100])}")

    print()


def measure_speed(model):
    """Compare generation speed across strategies."""
    print("\n" + "=" * 70)
    print("  GENERATION SPEED")
    print("=" * 70)
    print()

    strategies = [
        ("Single-pass greedy", lambda: sample_single_pass(model, gen_len=240)),
        ("Random-order (20 steps)", lambda: sample_random_order(model, num_steps=20, gen_len=240)),
        ("Confidence (20 steps)", lambda: sample_confidence(model, num_steps=20, gen_len=240)),
        ("Cosine (20 steps)", lambda: sample_cosine_schedule(model, num_steps=20, gen_len=240)),
        ("Full (20 steps)", lambda: sample_full(model, num_steps=20, gen_len=240)),
        ("Full (40 steps)", lambda: sample_full(model, num_steps=40, gen_len=240)),
    ]

    n_trials = 5
    for name, fn in strategies:
        times = []
        for _ in range(n_trials):
            start = time.time()
            fn()
            times.append(time.time() - start)
        avg = sum(times) / len(times)
        print(f"  {name:30s}: {avg*1000:7.1f} ms avg ({n_trials} trials)")

    print()
    print("  Note: All diffusion strategies generate ALL tokens in parallel")
    print("  at each step (not one-at-a-time like GPT). More steps = more")
    print("  model forward passes, but each step refines the full sequence.")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("MicroDiffusion LM — Step 3: Iterative Sampling")
    print("=" * 70)
    print()
    print("This step demonstrates HOW to generate text from a trained")
    print("diffusion model, progressing from naive to sophisticated.")
    print()

    # Train the model (same as step2)
    model = train_model()

    # Compare all sampling strategies
    compare_strategies(model)

    # Study hyperparameters
    study_num_steps(model)
    study_temperature(model)
    study_top_k(model)

    # Speed comparison
    measure_speed(model)

    print("\n" + "=" * 70)
    print("Step 3 Complete!")
    print("=" * 70)
    print()
    print("Key takeaways:")
    print("  1. Single-pass greedy is bad — no inter-token dependencies")
    print("  2. Iterative unmasking helps — model refines with context")
    print("  3. Confidence-based ordering > random ordering")
    print("  4. Cosine schedule > uniform pacing (mirrors forward process)")
    print("  5. SUBS + top-k sampling adds diversity without losing quality")
    print("  6. More steps = better quality, but diminishing returns past ~20")
    print()
    print("Next: see diffusion.py for the full-size model (6L/6H/384E)")
    print("and evaluate.py for head-to-head comparison with GPT.")
