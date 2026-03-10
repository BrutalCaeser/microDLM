#!/usr/bin/env python3
"""
export_frames.py — Pre-compute diffusion & GPT race frames → JSON
=================================================================
Loads both trained models, runs generation step-by-step, and exports
every frame as a compact JSON file for the static web demo.

Usage:
    cd microDLM
    python web/export_frames.py

Output: web/frames.json (~80-120 KB)
"""

import os
import sys
import json
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "shakespeare.txt")
DIFF_WEIGHTS = os.path.join(PROJECT_ROOT, "weights", "diffusion.pt")
GPT_WEIGHTS = os.path.join(PROJECT_ROOT, "weights", "gpt.pt")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames.json")

# ---------------------------------------------------------------------------
# Vocab (must match training exactly)
# ---------------------------------------------------------------------------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
MASK_CHAR = "_"
assert MASK_CHAR not in chars
chars_with_mask = [MASK_CHAR] + chars
vocab_size = len(chars_with_mask)  # 66

stoi = {ch: i for i, ch in enumerate(chars_with_mask)}
itos = {i: ch for i, ch in enumerate(chars_with_mask)}
mask_token_id = stoi[MASK_CHAR]  # 0

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# ---------------------------------------------------------------------------
# Hyperparams
# ---------------------------------------------------------------------------
n_embd = 384
n_head = 6
n_layer = 6
head_dim = n_embd // n_head  # 64
block_size = 256

device = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# ---------------------------------------------------------------------------
# Model (same as visualize.py — unified class with is_causal flag)
# ---------------------------------------------------------------------------
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

class Model(nn.Module):
    def __init__(self, is_causal=False):
        super().__init__()
        self.is_causal = is_causal
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        cos, sin = self._precompute_rope(block_size * 2)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.blocks = nn.ModuleList([Block(is_causal=is_causal) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def _precompute_rope(self, seq_len, base=10000):
        dev = self.token_emb.weight.device if hasattr(self, "token_emb") else "cpu"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=dev) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=dev)
        freqs = torch.outer(t, inv_freq)
        return freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]

    def forward(self, idx):
        B, T = idx.size()
        x = norm(self.token_emb(idx))
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        for block in self.blocks:
            x = block(x, cos_sin)
        return self.lm_head(norm(x))

# ---------------------------------------------------------------------------
# Generation (step-by-step, yielding frames)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_diffusion_frames(model, gen_length=240, prompt_len=16,
                               num_steps=40, temp=0.8, top_k=3):
    prompt_tokens = data[:prompt_len].tolist()
    total_gen = min(gen_length, block_size) - prompt_len

    x = torch.full((1, min(gen_length, block_size)), mask_token_id,
                    dtype=torch.long, device=device)
    x[0, :prompt_len] = torch.tensor(prompt_tokens, device=device)
    masked = torch.zeros(1, x.size(1), dtype=torch.bool, device=device)
    masked[0, prompt_len:] = True

    schedule = []
    for t in range(num_steps + 1):
        frac = math.cos(math.pi / 2 * t / num_steps) ** 2
        schedule.append(int(round(total_gen * frac)))
    schedule[0], schedule[-1] = total_gen, 0

    frames = []
    frames.append({"step": 0, "tokens": x[0].cpu().tolist(),
                    "newly": [], "remaining": total_gen})

    for step in range(1, num_steps + 1):
        n_to_unmask = schedule[step - 1] - schedule[step]
        if n_to_unmask <= 0:
            continue
        if not masked.any():
            break

        logits = model(x)
        logits[:, :, mask_token_id] = -float("inf")
        probs = F.softmax(logits / temp, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
        confidences = top_k_probs[:, :, 0]

        masked_conf = torch.where(
            masked, confidences, torch.tensor(-float("inf"), device=device))
        n_to_unmask = min(n_to_unmask, masked.sum().item())
        _, top_indices = torch.topk(masked_conf.view(-1), k=int(n_to_unmask))
        decode_mask = torch.zeros_like(masked.view(-1))
        decode_mask[top_indices] = True
        decode_mask = decode_mask.view_as(masked).bool()

        top_k_probs_norm = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        sampled_k = torch.multinomial(
            top_k_probs_norm.view(-1, top_k), 1).view(1, x.size(1))
        sampled_tokens = torch.gather(
            top_k_indices, -1, sampled_k.unsqueeze(-1)).squeeze(-1)

        x = torch.where(decode_mask, sampled_tokens, x)
        newly = decode_mask[0].nonzero(as_tuple=True)[0].cpu().tolist()
        masked = masked & ~decode_mask
        remaining = masked.sum().item()

        frames.append({"step": step, "tokens": x[0].cpu().tolist(),
                        "newly": newly, "remaining": remaining})

    return frames


@torch.no_grad()
def generate_gpt_frames(model, gen_length=240, prompt_len=16, temp=0.8):
    prompt_tokens = data[:prompt_len].tolist()
    x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    frames = []
    frames.append({"step": 0, "tokens": list(prompt_tokens), "newPos": None})

    for step in range(1, gen_length - prompt_len + 1):
        ctx = x[:, -block_size:]
        logits = model(ctx)
        logits = logits[:, -1, :]
        probs = F.softmax(logits / temp, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)
        tokens = x[0].cpu().tolist()
        frames.append({"step": step, "tokens": tokens,
                        "newPos": len(tokens) - 1})

    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {device}")
    print(f"Vocab: {vocab_size} chars, block_size: {block_size}")

    # Load models
    print("\nLoading diffusion model...")
    diff_model = Model(is_causal=False).to(device)
    diff_model.load_state_dict(
        torch.load(DIFF_WEIGHTS, map_location=device, weights_only=True))
    diff_model.eval()
    n = sum(p.numel() for p in diff_model.parameters())
    print(f"  ✓ {n/1e6:.1f}M params")

    print("Loading GPT model...")
    gpt_model = Model(is_causal=True).to(device)
    gpt_model.load_state_dict(
        torch.load(GPT_WEIGHTS, map_location=device, weights_only=True))
    gpt_model.eval()
    n = sum(p.numel() for p in gpt_model.parameters())
    print(f"  ✓ {n/1e6:.1f}M params")

    # Generate frames (deterministic)
    gen_length = 240
    prompt_len = 16
    seed = 42

    print(f"\nGenerating diffusion frames (seed={seed})...")
    torch.manual_seed(seed)
    diff_frames = generate_diffusion_frames(diff_model, gen_length, prompt_len)
    print(f"  ✓ {len(diff_frames)} frames")

    print(f"Generating GPT frames (seed={seed})...")
    torch.manual_seed(seed)
    gpt_frames = generate_gpt_frames(gpt_model, gen_length, prompt_len)
    print(f"  ✓ {len(gpt_frames)} frames")

    # Build the itos map for the frontend (token ID → character)
    # We send it so JS can decode token arrays without hardcoding
    itos_map = {str(i): ch for i, ch in itos.items()}

    # Prompt text for display
    prompt_text = decode(data[:prompt_len].tolist())

    output = {
        "meta": {
            "genLength": gen_length,
            "promptLen": prompt_len,
            "totalGen": gen_length - prompt_len,
            "seed": seed,
            "vocabSize": vocab_size,
            "maskTokenId": mask_token_id,
            "architecture": f"{n_layer}L/{n_head}H/{n_embd}E",
            "params": "10.7M",
            "promptText": prompt_text,
        },
        "itos": itos_map,
        "diffusion": {
            "frames": diff_frames,
            "totalFrames": len(diff_frames),
        },
        "gpt": {
            "frames": gpt_frames,
            "totalFrames": len(gpt_frames),
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\n✓ Saved {OUTPUT_PATH} ({size_kb:.1f} KB)")
    print(f"  Diffusion: {len(diff_frames)} frames")
    print(f"  GPT: {len(gpt_frames)} frames")


if __name__ == "__main__":
    main()
