"""
evaluate.py — Head-to-Head: Diffusion LM vs GPT
=================================================
Loads both trained models and compares them quantitatively.

Metrics computed:
  1. LOSS / PSEUDO-PERPLEXITY  — How well each model predicts held-out text
  2. DISTINCT-N                — Diversity of generated n-grams (higher = more diverse)
  3. N-GRAM NOVELTY            — % of generated n-grams NOT in training data (creativity)
  4. CHAR FREQUENCY KL         — KL divergence from Shakespeare's character distribution
  5. WORD-LEVEL STATS          — Avg word length, % real English words, vocabulary size
  6. REPETITION                — How much the model repeats itself
  7. GENERATION SPEED          — Wall-clock time per sample

Usage:
  First train both models:
    python diffusion.py --train
    python gpt.py --train

  Then evaluate:
    python evaluate.py

  Or train + evaluate in one go:
    python evaluate.py --train
"""

import os
import sys
import time
import math
import json
import argparse
import subprocess
from collections import Counter

import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================================
# Device
# ============================================================================

device = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# ============================================================================
# Shared Data & Vocabulary
# ============================================================================

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "shakespeare.txt")
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
MASK_CHAR = "_"
assert MASK_CHAR not in chars
chars_with_mask = [MASK_CHAR] + chars
vocab_size = len(chars_with_mask)

stoi = {ch: i for i, ch in enumerate(chars_with_mask)}
itos = {i: ch for i, ch in enumerate(chars_with_mask)}
mask_token_id = stoi[MASK_CHAR]

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: "".join([itos[i] for i in l])

data_tensor = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data_tensor))
train_data = data_tensor[:n]
val_data = data_tensor[n:]

train_text = text[:n]
val_text = text[n:]

# ============================================================================
# Hyperparameters (must match diffusion.py and gpt.py)
# ============================================================================

block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
head_dim = n_embd // n_head

# ============================================================================
# Model Definitions (imported inline to keep this file self-contained)
# ============================================================================
# These must exactly match diffusion.py and gpt.py architectures.


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).to(x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(self, is_causal=False):
        super().__init__()
        self.is_causal = is_causal
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
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
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
    def __init__(self, is_causal=False):
        super().__init__()
        self.attn = MultiHeadAttention(is_causal=is_causal)
        self.mlp = MLP()

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


def _precompute_rope(seq_len, base=10000):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]


class DiffusionLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        cos, sin = _precompute_rope(block_size * 2)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.blocks = nn.ModuleList([Block(is_causal=False) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

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
        mask_flat = mask.view(B * T).float()
        loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        loss = (loss * mask_flat).sum() / mask_flat.sum()
        return logits, loss


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        cos, sin = _precompute_rope(block_size * 2)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.blocks = nn.ModuleList([Block(is_causal=True) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = norm(self.token_emb(idx))
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        for block in self.blocks:
            x = block(x, cos_sin)
        logits = self.lm_head(norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ============================================================================
# Generation Functions
# ============================================================================


@torch.no_grad()
def generate_diffusion(model, max_new_tokens=500, prompt_len=16, num_steps=40,
                       temp=0.8, top_k=3):
    """Cosine-scheduled parallel unmasking (from diffusion.py)."""
    model.eval()
    prompt = data_tensor[:prompt_len].tolist()
    gen_len = min(max_new_tokens, block_size) - prompt_len

    x = torch.full((1, gen_len + prompt_len), mask_token_id, dtype=torch.long, device=device)
    x[0, :prompt_len] = torch.tensor(prompt, device=device)
    masked = torch.zeros(1, x.size(1), dtype=torch.bool, device=device)
    masked[0, prompt_len:] = True

    schedule = [int(round(gen_len * math.cos(math.pi / 2 * t / num_steps) ** 2))
                for t in range(num_steps + 1)]
    schedule[0], schedule[-1] = gen_len, 0

    for step in range(1, num_steps + 1):
        n_unmask = schedule[step - 1] - schedule[step]
        if n_unmask <= 0 or not masked.any():
            continue
        logits = model(x)[0]
        logits[:, :, mask_token_id] = -float("inf")
        probs = F.softmax(logits / temp, dim=-1)
        top_probs, top_idx = torch.topk(probs, k=top_k, dim=-1)
        confidence = top_probs[:, :, 0]
        conf = torch.where(masked, confidence, torch.tensor(-float("inf"), device=device))
        n_unmask = min(n_unmask, masked.sum().item())
        _, best = torch.topk(conf.view(-1), k=int(n_unmask))
        decode_mask = torch.zeros_like(masked.view(-1))
        decode_mask[best] = True
        decode_mask = decode_mask.view_as(masked).bool()
        top_probs_norm = top_probs / top_probs.sum(dim=-1, keepdim=True)
        sampled_k = torch.multinomial(top_probs_norm.view(-1, top_k), 1).view(1, x.size(1))
        tokens = torch.gather(top_idx, -1, sampled_k.unsqueeze(-1)).squeeze(-1)
        x = torch.where(decode_mask, tokens, x)
        masked = masked & ~decode_mask

    model.train()
    return decode(x[0].cpu().tolist())


@torch.no_grad()
def generate_gpt(model, max_new_tokens=500, prompt_len=16, temp=0.8):
    """Standard autoregressive generation (from gpt.py)."""
    model.eval()
    x = data_tensor[:prompt_len].unsqueeze(0).to(device)
    for _ in range(max_new_tokens):
        ctx = x[:, -block_size:]
        logits = model(ctx)[0]
        logits = logits[:, -1, :]
        probs = F.softmax(logits / temp, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)
    model.train()
    return decode(x[0].cpu().tolist())


# ============================================================================
# Metric Functions
# ============================================================================


def get_ngrams(text, n):
    """Extract all character-level n-grams from text."""
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def distinct_n(text, n):
    """
    Distinct-N: fraction of unique n-grams among all n-grams.
    Higher = more diverse output. Range: [0, 1].
    A model that repeats itself scores low; a creative model scores high.
    """
    ngrams = get_ngrams(text, n)
    if len(ngrams) == 0:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def ngram_novelty(generated_text, reference_text, n):
    """
    N-gram novelty: fraction of generated n-grams that do NOT appear
    in the reference text. Higher = more creative/novel output.
    0% = pure memorization. 100% = completely novel (possibly gibberish).
    """
    gen_ngrams = set(get_ngrams(generated_text, n))
    ref_ngrams = set(get_ngrams(reference_text, n))
    if len(gen_ngrams) == 0:
        return 0.0
    novel = gen_ngrams - ref_ngrams
    return len(novel) / len(gen_ngrams)


def char_frequency_kl(generated_text, reference_text):
    """
    KL divergence of character frequency distribution.
    KL(generated || reference) — how different the character distribution
    of generated text is from the reference (Shakespeare).
    Lower = more similar to Shakespeare's style.
    """
    # Count characters
    gen_counts = Counter(generated_text)
    ref_counts = Counter(reference_text)

    # Get all characters
    all_chars = set(list(gen_counts.keys()) + list(ref_counts.keys()))

    # Smoothed probability distributions (add-1 smoothing)
    gen_total = sum(gen_counts.values()) + len(all_chars)
    ref_total = sum(ref_counts.values()) + len(all_chars)

    kl = 0.0
    for ch in all_chars:
        p = (gen_counts.get(ch, 0) + 1) / gen_total
        q = (ref_counts.get(ch, 0) + 1) / ref_total
        if p > 0:
            kl += p * math.log(p / q)

    return kl


def word_stats(text):
    """Compute word-level statistics."""
    words = text.split()
    if len(words) == 0:
        return {"n_words": 0, "avg_len": 0, "unique_words": 0, "vocab_size": 0}

    lengths = [len(w) for w in words]
    unique = set(w.lower().strip(".,;:!?'\"()") for w in words)

    return {
        "n_words": len(words),
        "avg_len": sum(lengths) / len(lengths),
        "unique_words": len(unique),
        "type_token_ratio": len(unique) / len(words) if len(words) > 0 else 0,
    }


def repetition_score(text, window=50):
    """
    Measure repetition: what fraction of windows of `window` characters
    contain a repeated substring of length >= 10?
    Lower = less repetitive = better.
    """
    n_windows = max(1, len(text) - window + 1)
    n_repetitive = 0

    for i in range(n_windows):
        chunk = text[i:i+window]
        # Check for any repeated substring of length 10+
        found = False
        for sub_len in range(10, len(chunk) // 2 + 1):
            for j in range(len(chunk) - sub_len * 2 + 1):
                sub = chunk[j:j+sub_len]
                if sub in chunk[j+sub_len:]:
                    found = True
                    break
            if found:
                break
        if found:
            n_repetitive += 1

    return n_repetitive / n_windows


def compute_loss_diffusion(model, n_batches=50):
    """Compute average masked prediction loss on validation data."""
    model.eval()
    batch_size_eval = 32
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            idx = torch.randint(len(val_data) - block_size, (batch_size_eval,))
            x = torch.stack([val_data[i:i+block_size] for i in idx])
            y = x.clone()
            # Use 50% masking for consistent evaluation
            mask = torch.rand(batch_size_eval, block_size) < 0.5
            x[mask] = mask_token_id
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            _, loss = model(x, y, mask)
            losses.append(loss.item())
    return sum(losses) / len(losses)


def compute_loss_gpt(model, n_batches=50):
    """Compute average next-token prediction loss on validation data."""
    model.eval()
    batch_size_eval = 32
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            idx = torch.randint(len(val_data) - block_size, (batch_size_eval,))
            x = torch.stack([val_data[i:i+block_size] for i in idx]).to(device)
            y = torch.stack([val_data[i+1:i+block_size+1] for i in idx]).to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    return sum(losses) / len(losses)


# ============================================================================
# Load Models
# ============================================================================


def load_models(train_if_missing=False):
    """Load both trained models from weights/ directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    diff_path = os.path.join(script_dir, "weights", "diffusion.pt")
    gpt_path = os.path.join(script_dir, "weights", "gpt.pt")

    if train_if_missing:
        if not os.path.exists(diff_path):
            print("Diffusion weights not found. Training...")
            subprocess.run([sys.executable, os.path.join(script_dir, "diffusion.py"), "--train"],
                           check=True)
        if not os.path.exists(gpt_path):
            print("GPT weights not found. Training...")
            subprocess.run([sys.executable, os.path.join(script_dir, "gpt.py"), "--train"],
                           check=True)

    if not os.path.exists(diff_path):
        print(f"ERROR: {diff_path} not found. Run: python diffusion.py --train")
        sys.exit(1)
    if not os.path.exists(gpt_path):
        print(f"ERROR: {gpt_path} not found. Run: python gpt.py --train")
        sys.exit(1)

    diff_model = DiffusionLM().to(device)
    diff_model.load_state_dict(torch.load(diff_path, map_location=device, weights_only=True))
    print(f"Loaded diffusion model from {diff_path}")

    gpt_model = GPT().to(device)
    gpt_model.load_state_dict(torch.load(gpt_path, map_location=device, weights_only=True))
    print(f"Loaded GPT model from {gpt_path}")

    return diff_model, gpt_model


# ============================================================================
# Run Evaluation
# ============================================================================


def generate_samples(diff_model, gpt_model, n_samples=10, tokens=500):
    """Generate samples from both models."""
    print(f"\nGenerating {n_samples} samples from each model ({tokens} tokens each)...")

    diff_samples = []
    gpt_samples = []

    for i in range(n_samples):
        torch.manual_seed(i * 42)
        start = time.time()
        s = generate_diffusion(diff_model, max_new_tokens=tokens, num_steps=40)
        diff_samples.append({"text": s, "time": time.time() - start})

        torch.manual_seed(i * 42)
        start = time.time()
        s = generate_gpt(gpt_model, max_new_tokens=tokens)
        gpt_samples.append({"text": s, "time": time.time() - start})

        print(f"  Sample {i+1}/{n_samples} done")

    return diff_samples, gpt_samples


def evaluate_samples(samples, name, reference_text):
    """Compute all metrics on a list of generated samples."""
    all_text = " ".join([s["text"] for s in samples])
    avg_time = sum(s["time"] for s in samples) / len(samples)

    results = {
        "name": name,
        "n_samples": len(samples),
        "avg_gen_time": avg_time,
        # Diversity
        "distinct_1": distinct_n(all_text, 1),
        "distinct_2": distinct_n(all_text, 2),
        "distinct_3": distinct_n(all_text, 3),
        "distinct_4": distinct_n(all_text, 4),
        # Novelty vs training data
        "novelty_3": ngram_novelty(all_text, reference_text, 3),
        "novelty_5": ngram_novelty(all_text, reference_text, 5),
        "novelty_10": ngram_novelty(all_text, reference_text, 10),
        "novelty_20": ngram_novelty(all_text, reference_text, 20),
        # Character distribution
        "char_kl": char_frequency_kl(all_text, reference_text),
        # Word stats
        **{f"word_{k}": v for k, v in word_stats(all_text).items()},
        # Repetition
        "repetition": repetition_score(all_text),
    }

    return results


def print_comparison(diff_results, gpt_results, diff_loss, gpt_loss):
    """Print a formatted comparison table."""

    print("\n" + "=" * 72)
    print("  HEAD-TO-HEAD COMPARISON: DIFFUSION LM vs GPT")
    print("=" * 72)

    # Parameter counts
    diff_params = sum(p.numel() for p in DiffusionLM().parameters())
    gpt_params = sum(p.numel() for p in GPT().parameters())

    print(f"\n  Architecture: {n_layer}L / {n_head}H / {n_embd}E for both models")
    print(f"  Diffusion params: {diff_params:,}  |  GPT params: {gpt_params:,}")

    def row(label, d_val, g_val, fmt=".4f", better="lower"):
        """Print a comparison row with winner indicator."""
        d_str = f"{d_val:{fmt}}" if isinstance(d_val, float) else str(d_val)
        g_str = f"{g_val:{fmt}}" if isinstance(g_val, float) else str(g_val)
        if better == "lower":
            d_win = " ◀" if d_val < g_val else ""
            g_win = " ◀" if g_val < d_val else ""
        elif better == "higher":
            d_win = " ◀" if d_val > g_val else ""
            g_win = " ◀" if g_val > d_val else ""
        else:
            d_win = g_win = ""
        print(f"  {label:<30s}  {d_str:>12s}{d_win:<3s}  {g_str:>12s}{g_win:<3s}")

    # ── Loss & Perplexity ──
    print(f"\n{'─'*72}")
    print(f"  {'METRIC':<30s}  {'DIFFUSION':>12s}     {'GPT':>12s}")
    print(f"{'─'*72}")

    print(f"\n  LOSS & PERPLEXITY (lower = better prediction)")
    row("Val loss", diff_loss, gpt_loss)
    row("Pseudo-perplexity", math.exp(diff_loss), math.exp(gpt_loss))

    # ── Diversity ──
    print(f"\n  DIVERSITY — distinct-n (higher = more diverse)")
    row("Distinct-1 (chars)", diff_results["distinct_1"], gpt_results["distinct_1"], better="higher")
    row("Distinct-2 (bigrams)", diff_results["distinct_2"], gpt_results["distinct_2"], better="higher")
    row("Distinct-3 (trigrams)", diff_results["distinct_3"], gpt_results["distinct_3"], better="higher")
    row("Distinct-4 (4-grams)", diff_results["distinct_4"], gpt_results["distinct_4"], better="higher")

    # ── Novelty ──
    print(f"\n  NOVELTY vs training data (moderate = best; 0% = memorized, 100% = gibberish)")
    row("3-gram novelty", diff_results["novelty_3"], gpt_results["novelty_3"], ".2%", better="none")
    row("5-gram novelty", diff_results["novelty_5"], gpt_results["novelty_5"], ".2%", better="none")
    row("10-gram novelty", diff_results["novelty_10"], gpt_results["novelty_10"], ".2%", better="none")
    row("20-gram novelty", diff_results["novelty_20"], gpt_results["novelty_20"], ".2%", better="none")

    # ── Distribution match ──
    print(f"\n  STYLE MATCH (lower KL = closer to Shakespeare)")
    row("Char freq KL div", diff_results["char_kl"], gpt_results["char_kl"])

    # ── Word stats ──
    print(f"\n  WORD-LEVEL STATS")
    row("Avg word length", diff_results["word_avg_len"], gpt_results["word_avg_len"], ".2f", better="none")
    row("Type-token ratio", diff_results["word_type_token_ratio"], gpt_results["word_type_token_ratio"], ".4f", better="higher")
    row("Unique words", diff_results["word_unique_words"], gpt_results["word_unique_words"], "d", better="higher")

    # ── Repetition ──
    print(f"\n  REPETITION (lower = less repetitive)")
    row("Repetition score", diff_results["repetition"], gpt_results["repetition"])

    # ── Speed ──
    print(f"\n  GENERATION SPEED")
    row("Avg time/sample (s)", diff_results["avg_gen_time"], gpt_results["avg_gen_time"], ".3f", better="none")

    print(f"\n{'─'*72}")
    print(f"  ◀ = winner for that metric")
    print(f"{'─'*72}")


def show_sample_outputs(diff_samples, gpt_samples, n_show=3):
    """Display side-by-side sample outputs."""
    print(f"\n{'='*72}")
    print(f"  SAMPLE OUTPUTS")
    print(f"{'='*72}")

    for i in range(min(n_show, len(diff_samples))):
        print(f"\n  ── Sample {i+1} ──")
        print(f"\n  DIFFUSION ({diff_samples[i]['time']:.2f}s):")
        # Show first 200 chars, wrapped
        text = diff_samples[i]["text"][:250]
        for line in [text[j:j+72] for j in range(0, len(text), 72)]:
            print(f"    {line}")

        print(f"\n  GPT ({gpt_samples[i]['time']:.2f}s):")
        text = gpt_samples[i]["text"][:250]
        for line in [text[j:j+72] for j in range(0, len(text), 72)]:
            print(f"    {line}")

    print()


def show_training_curves():
    """Load and display training logs if available."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    diff_log_path = os.path.join(script_dir, "weights", "training_log_diffusion.json")
    gpt_log_path = os.path.join(script_dir, "weights", "training_log_gpt.json")

    if not (os.path.exists(diff_log_path) and os.path.exists(gpt_log_path)):
        return

    with open(diff_log_path) as f:
        diff_log = json.load(f)
    with open(gpt_log_path) as f:
        gpt_log = json.load(f)

    print(f"\n{'='*72}")
    print(f"  TRAINING CURVES (val loss)")
    print(f"{'='*72}")
    print(f"\n  {'Step':>6s}  {'Diffusion':>12s}  {'GPT':>12s}  {'Gap':>10s}")
    print(f"  {'─'*46}")

    # Align by iteration (both log at same intervals)
    diff_dict = {e["iter"]: e["val_loss"] for e in diff_log}
    gpt_dict = {e["iter"]: e["val_loss"] for e in gpt_log}
    all_iters = sorted(set(list(diff_dict.keys()) + list(gpt_dict.keys())))

    for it in all_iters:
        d_loss = diff_dict.get(it)
        g_loss = gpt_dict.get(it)
        d_str = f"{d_loss:.4f}" if d_loss is not None else "   —"
        g_str = f"{g_loss:.4f}" if g_loss is not None else "   —"
        if d_loss is not None and g_loss is not None:
            gap = f"{d_loss - g_loss:+.4f}"
        else:
            gap = "   —"
        print(f"  {it:6d}  {d_str:>12s}  {g_str:>12s}  {gap:>10s}")

    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Diffusion LM vs GPT")
    parser.add_argument("--train", action="store_true", help="Train models if weights not found")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to generate (default: 10)")
    parser.add_argument("--tokens", type=int, default=500, help="Tokens per sample (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("Diffusion LM vs GPT — Quantitative Evaluation")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Vocab: {vocab_size} chars | Block size: {block_size}")
    print(f"Samples: {args.samples} × {args.tokens} tokens")
    print()

    # Load models
    diff_model, gpt_model = load_models(train_if_missing=args.train)

    # Show training curves if available
    show_training_curves()

    # Compute validation loss
    print("\nComputing validation loss...")
    diff_loss = compute_loss_diffusion(diff_model)
    gpt_loss = compute_loss_gpt(gpt_model)
    print(f"  Diffusion val loss: {diff_loss:.4f} (pseudo-ppl: {math.exp(diff_loss):.2f})")
    print(f"  GPT val loss:       {gpt_loss:.4f} (perplexity:  {math.exp(gpt_loss):.2f})")

    # Generate samples
    diff_samples, gpt_samples = generate_samples(
        diff_model, gpt_model, n_samples=args.samples, tokens=args.tokens
    )

    # Evaluate
    print("\nComputing metrics...")
    diff_results = evaluate_samples(diff_samples, "Diffusion", train_text)
    gpt_results = evaluate_samples(gpt_samples, "GPT", train_text)

    # Display results
    print_comparison(diff_results, gpt_results, diff_loss, gpt_loss)
    show_sample_outputs(diff_samples, gpt_samples)

    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results = {
        "diffusion": {**diff_results, "val_loss": diff_loss, "pseudo_ppl": math.exp(diff_loss)},
        "gpt": {**gpt_results, "val_loss": gpt_loss, "perplexity": math.exp(gpt_loss)},
    }
    results_path = os.path.join(script_dir, "weights", "eval_results.json")
    os.makedirs(os.path.join(script_dir, "weights"), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    print("\n" + "=" * 72)
    print("  INTERPRETATION GUIDE")
    print("=" * 72)
    print("""
  LOSS: Not directly comparable between models.
    • Diffusion loss: prediction accuracy at masked positions only
    • GPT loss: next-token prediction at ALL positions
    Both use cross-entropy, but over different tasks.
    Pseudo-perplexity gives a rough comparison.

  DIVERSITY (distinct-n): Higher = output uses more varied n-grams.
    A model stuck in loops will score low. Both models should be
    reasonably high for character-level generation.

  NOVELTY: The sweet spot is moderate (30-70% at 10-gram level).
    Too low (< 20%) = memorizing training data.
    Too high (> 90%) = generating gibberish.

  CHAR KL: Lower = character frequencies match Shakespeare more closely.
    A model that over-generates spaces or under-generates rare chars
    will have high KL divergence.

  REPETITION: Lower = less self-repetitive output.
    Character-level models are prone to getting stuck in loops.

  SPEED: Diffusion generates all tokens in parallel across N steps.
    GPT generates one token at a time across T steps.
    For long sequences, diffusion can be faster (N << T).
""")
