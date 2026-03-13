"""
generate_traces.py — Capture Model Inference Traces for Visualization
=====================================================================
Runs both the Diffusion LM and GPT models on a set of prompts and
captures detailed per-token inference data: probabilities, confidence
scores, top-k predictions, and (for diffusion) the step-by-step
unmasking sequence.

Outputs JSON files that the frontend visualization reads.

Usage:
  python generate_traces.py                    # uses saved weights
  python generate_traces.py --train            # trains first if needed
  python generate_traces.py --prompts custom   # use custom prompt set
  python generate_traces.py --tokens 300       # tokens per sample

Output:
  traces/gpt_traces.json
  traces/diffusion_traces.json
"""

import os
import sys
import json
import math
import time
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================================
# Device & Config
# ============================================================================

device = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# Architecture (must match diffusion.py and gpt.py)
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
head_dim = n_embd // n_head

# ============================================================================
# Data & Vocabulary
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data", "shakespeare.txt")
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
train_data, val_data = data_tensor[:n], data_tensor[n:]

# ============================================================================
# Model Definitions (must match diffusion.py / gpt.py exactly)
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

    def forward(self, idx):
        B, T = idx.size()
        x = norm(self.token_emb(idx))
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        for block in self.blocks:
            x = block(x, cos_sin)
        return self.lm_head(norm(x))


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

    def forward(self, idx):
        B, T = idx.size()
        x = norm(self.token_emb(idx))
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        for block in self.blocks:
            x = block(x, cos_sin)
        return self.lm_head(norm(x))


# ============================================================================
# Prompt Sets
# ============================================================================

def get_prompts(prompt_set="default"):
    """Return a list of (name, prompt_text) tuples."""
    if prompt_set == "default":
        return [
            ("opening", text[:64]),
            ("speech", text[1000:1064]),
            ("dialogue", text[5000:5064]),
            ("monologue", text[10000:10064]),
            ("late", text[50000:50064]),
        ]
    elif prompt_set == "short":
        return [
            ("opening", text[:32]),
            ("mid", text[5000:5032]),
        ]
    elif prompt_set == "custom":
        # Read from stdin or a file
        prompts = []
        print("Enter prompts (empty line to finish):")
        while True:
            line = input("> ").strip()
            if not line:
                break
            prompts.append((f"custom_{len(prompts)}", line))
        return prompts
    else:
        raise ValueError(f"Unknown prompt set: {prompt_set}")


# ============================================================================
# GPT Trace Generation
# ============================================================================

@torch.no_grad()
def trace_gpt(model, prompt_text, max_new_tokens=200, temp=0.8, top_k_save=10):
    """
    Generate text with GPT and capture per-token trace data.

    For each generated token, we record:
    - The token string and id
    - The full top-k predictions with probabilities
    - The confidence (probability of the chosen token)
    - Whether it's a prompt token
    """
    model.eval()
    prompt_ids = encode(prompt_text)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    tokens = []
    # Record prompt tokens (confidence = 1.0, they're given)
    for i, tid in enumerate(prompt_ids):
        tokens.append({
            "token": itos[tid],
            "token_id": tid,
            "confidence": 1.0,
            "is_prompt": True,
            "position": i,
            "top_k": [{"token": itos[tid], "token_id": tid, "prob": 1.0}],
        })

    # Generate tokens one at a time
    for step in range(max_new_tokens):
        ctx = x[:, -block_size:]
        logits = model(ctx)
        logits_last = logits[:, -1, :]  # only last position

        probs = F.softmax(logits_last / temp, dim=-1)

        # Get top-k predictions
        top_probs, top_ids = torch.topk(probs[0], k=min(top_k_save, vocab_size))
        top_k_list = []
        for j in range(len(top_ids)):
            top_k_list.append({
                "token": itos[top_ids[j].item()],
                "token_id": top_ids[j].item(),
                "prob": round(top_probs[j].item(), 6),
            })

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)
        chosen_id = next_token[0, 0].item()
        chosen_prob = probs[0, chosen_id].item()

        tokens.append({
            "token": itos[chosen_id],
            "token_id": chosen_id,
            "confidence": round(chosen_prob, 6),
            "is_prompt": False,
            "position": len(prompt_ids) + step,
            "top_k": top_k_list,
        })

        x = torch.cat([x, next_token], dim=1)

    return {
        "model": "gpt",
        "prompt": prompt_text,
        "prompt_length": len(prompt_ids),
        "generated_length": max_new_tokens,
        "temperature": temp,
        "tokens": tokens,
        "full_text": decode(x[0].cpu().tolist()),
    }


# ============================================================================
# Diffusion Trace Generation
# ============================================================================

@torch.no_grad()
def trace_diffusion(model, prompt_text, max_new_tokens=200, num_steps=40,
                    temp=0.8, top_k=3, top_k_save=10):
    """
    Generate text with diffusion and capture the full step-by-step trace.

    Records:
    - The state of every token at every step
    - Which tokens were unmasked at each step
    - Confidence values at each step
    - The final per-token confidence (at moment of unmasking)
    """
    model.eval()
    prompt_ids = encode(prompt_text)
    prompt_len = len(prompt_ids)
    gen_len = min(max_new_tokens, block_size) - prompt_len
    total_len = prompt_len + gen_len

    x = torch.full((1, total_len), mask_token_id, dtype=torch.long, device=device)
    x[0, :prompt_len] = torch.tensor(prompt_ids, device=device)
    masked = torch.zeros(1, total_len, dtype=torch.bool, device=device)
    masked[0, prompt_len:] = True

    # Cosine schedule
    schedule = [int(round(gen_len * math.cos(math.pi / 2 * t / num_steps) ** 2))
                for t in range(num_steps + 1)]
    schedule[0], schedule[-1] = gen_len, 0

    # Track per-token: when it was unmasked and with what confidence
    token_reveal_step = [-1] * total_len  # -1 = prompt token
    token_reveal_confidence = [1.0] * total_len  # 1.0 for prompt

    # Capture step-by-step snapshots
    steps = []

    # Initial state
    step_tokens = []
    for i in range(total_len):
        step_tokens.append({
            "position": i,
            "token": itos[x[0, i].item()],
            "token_id": x[0, i].item(),
            "is_masked": masked[0, i].item(),
            "is_prompt": i < prompt_len,
            "confidence": 1.0 if i < prompt_len else 0.0,
        })
    steps.append({
        "step": 0,
        "n_masked": masked.sum().item(),
        "unmasked_this_step": [],
        "tokens": step_tokens,
    })

    for step in range(1, num_steps + 1):
        n_unmask = schedule[step - 1] - schedule[step]
        if n_unmask <= 0 or not masked.any():
            continue

        logits = model(x)
        logits[:, :, mask_token_id] = -float("inf")  # SUBS
        probs = F.softmax(logits / temp, dim=-1)
        top_probs_all, top_idx_all = torch.topk(probs, k=top_k, dim=-1)
        confidence = top_probs_all[:, :, 0]

        # Pick most-confident masked positions
        conf = torch.where(masked, confidence, torch.tensor(-float("inf"), device=device))
        n_unmask = min(n_unmask, masked.sum().item())
        _, best = torch.topk(conf.view(-1), k=int(n_unmask))

        decode_mask = torch.zeros_like(masked.view(-1))
        decode_mask[best] = True
        decode_mask = decode_mask.view_as(masked).bool()

        # Sample from top-k
        top_probs_norm = top_probs_all / top_probs_all.sum(dim=-1, keepdim=True)
        sampled_k = torch.multinomial(top_probs_norm.view(-1, top_k), 1).view(1, total_len)
        tokens_sampled = torch.gather(top_idx_all, -1, sampled_k.unsqueeze(-1)).squeeze(-1)

        # Apply
        x = torch.where(decode_mask, tokens_sampled, x)
        masked = masked & ~decode_mask

        # Record which positions were unmasked
        unmasked_positions = decode_mask[0].nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
        if isinstance(unmasked_positions, int):
            unmasked_positions = [unmasked_positions]

        for pos in unmasked_positions:
            token_reveal_step[pos] = step
            token_reveal_confidence[pos] = round(confidence[0, pos].item(), 6)

        # Capture step snapshot
        step_tokens = []
        for i in range(total_len):
            top_k_list = []
            if masked[0, i].item() or i in unmasked_positions:
                # Save top-k for positions that are still masked or just unmasked
                for j in range(min(top_k_save, top_k)):
                    top_k_list.append({
                        "token": itos[top_idx_all[0, i, j].item()],
                        "token_id": top_idx_all[0, i, j].item(),
                        "prob": round(top_probs_all[0, i, j].item(), 6),
                    })

            step_tokens.append({
                "position": i,
                "token": itos[x[0, i].item()],
                "token_id": x[0, i].item(),
                "is_masked": masked[0, i].item(),
                "is_prompt": i < prompt_len,
                "confidence": round(confidence[0, i].item(), 6) if i >= prompt_len else 1.0,
                "just_unmasked": i in unmasked_positions,
                "top_k": top_k_list if top_k_list else None,
            })

        steps.append({
            "step": step,
            "n_masked": masked.sum().item(),
            "unmasked_this_step": unmasked_positions,
            "tokens": step_tokens,
        })

    # Build final token list with reveal metadata
    final_tokens = []
    for i in range(total_len):
        final_tokens.append({
            "token": itos[x[0, i].item()],
            "token_id": x[0, i].item(),
            "position": i,
            "is_prompt": i < prompt_len,
            "confidence": token_reveal_confidence[i],
            "reveal_step": token_reveal_step[i],
            "reveal_step_normalized": round(token_reveal_step[i] / num_steps, 4) if token_reveal_step[i] > 0 else 0.0,
        })

    return {
        "model": "diffusion",
        "prompt": prompt_text,
        "prompt_length": prompt_len,
        "generated_length": gen_len,
        "num_steps": num_steps,
        "temperature": temp,
        "top_k": top_k,
        "schedule": schedule,
        "tokens": final_tokens,
        "steps": steps,
        "full_text": decode(x[0].cpu().tolist()),
    }


# ============================================================================
# Load Models
# ============================================================================

def load_models(train_if_missing=False):
    diff_path = os.path.join(script_dir, "weights", "diffusion.pt")
    gpt_path = os.path.join(script_dir, "weights", "gpt.pt")

    if train_if_missing:
        import subprocess
        if not os.path.exists(diff_path):
            print("Training diffusion model...")
            subprocess.run([sys.executable, os.path.join(script_dir, "diffusion.py"), "--train"], check=True)
        if not os.path.exists(gpt_path):
            print("Training GPT model...")
            subprocess.run([sys.executable, os.path.join(script_dir, "gpt.py"), "--train"], check=True)

    if not os.path.exists(diff_path):
        print(f"ERROR: {diff_path} not found. Run: python diffusion.py --train")
        sys.exit(1)
    if not os.path.exists(gpt_path):
        print(f"ERROR: {gpt_path} not found. Run: python gpt.py --train")
        sys.exit(1)

    diff_model = DiffusionLM().to(device)
    diff_model.load_state_dict(torch.load(diff_path, map_location=device, weights_only=True))
    print(f"Loaded diffusion model ({sum(p.numel() for p in diff_model.parameters()):,} params)")

    gpt_model = GPT().to(device)
    gpt_model.load_state_dict(torch.load(gpt_path, map_location=device, weights_only=True))
    print(f"Loaded GPT model ({sum(p.numel() for p in gpt_model.parameters()):,} params)")

    return diff_model, gpt_model


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate inference traces for visualization")
    parser.add_argument("--train", action="store_true", help="Train models if weights not found")
    parser.add_argument("--prompts", type=str, default="default", help="Prompt set: default, short, custom")
    parser.add_argument("--tokens", type=int, default=200, help="Tokens to generate per prompt")
    parser.add_argument("--steps", type=int, default=40, help="Diffusion steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="traces", help="Output directory")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"Device: {device}")
    print(f"Generating {args.tokens} tokens per prompt, {args.steps} diffusion steps")
    print()

    # Load models
    diff_model, gpt_model = load_models(train_if_missing=args.train)

    # Get prompts
    prompts = get_prompts(args.prompts)
    print(f"\nUsing {len(prompts)} prompts:")
    for name, p in prompts:
        print(f"  {name}: {repr(p[:50])}{'...' if len(p) > 50 else ''}")
    print()

    # Output directory
    out_dir = os.path.join(script_dir, args.output)
    os.makedirs(out_dir, exist_ok=True)

    # Generate traces
    gpt_traces = []
    diff_traces = []

    for name, prompt_text in prompts:
        print(f"Tracing '{name}'...")

        # GPT trace
        torch.manual_seed(args.seed)
        start = time.time()
        gpt_trace = trace_gpt(gpt_model, prompt_text,
                              max_new_tokens=args.tokens, temp=args.temp)
        gpt_trace["name"] = name
        gpt_trace["generation_time"] = round(time.time() - start, 3)
        gpt_traces.append(gpt_trace)
        print(f"  GPT:       {gpt_trace['generation_time']:.2f}s — {repr(gpt_trace['full_text'][:80])}")

        # Diffusion trace
        torch.manual_seed(args.seed)
        start = time.time()
        diff_trace = trace_diffusion(diff_model, prompt_text,
                                     max_new_tokens=args.tokens,
                                     num_steps=args.steps, temp=args.temp)
        diff_trace["name"] = name
        diff_trace["generation_time"] = round(time.time() - start, 3)
        diff_traces.append(diff_trace)
        print(f"  Diffusion: {diff_trace['generation_time']:.2f}s — {repr(diff_trace['full_text'][:80])}")
        print()

    # Save traces
    gpt_path = os.path.join(out_dir, "gpt_traces.json")
    with open(gpt_path, "w") as f:
        json.dump({"traces": gpt_traces, "vocab_size": vocab_size, "model": "gpt"}, f)
    print(f"Saved GPT traces to {gpt_path} ({os.path.getsize(gpt_path) / 1024:.0f} KB)")

    diff_path = os.path.join(out_dir, "diffusion_traces.json")
    with open(diff_path, "w") as f:
        json.dump({"traces": diff_traces, "vocab_size": vocab_size, "model": "diffusion"}, f)
    print(f"Saved Diffusion traces to {diff_path} ({os.path.getsize(diff_path) / 1024:.0f} KB)")

    # Also save a combined lightweight version (just tokens + confidence, no steps)
    # This is what the confidence grid visualization needs
    combined = {
        "vocab_size": vocab_size,
        "traces": [],
    }
    for gt in gpt_traces:
        combined["traces"].append({
            "model": "gpt",
            "name": gt["name"],
            "prompt": gt["prompt"],
            "prompt_length": gt["prompt_length"],
            "full_text": gt["full_text"],
            "generation_time": gt["generation_time"],
            "tokens": [{
                "token": t["token"],
                "confidence": t["confidence"],
                "is_prompt": t["is_prompt"],
                "position": t["position"],
            } for t in gt["tokens"]],
        })
    for dt in diff_traces:
        combined["traces"].append({
            "model": "diffusion",
            "name": dt["name"],
            "prompt": dt["prompt"],
            "prompt_length": dt["prompt_length"],
            "full_text": dt["full_text"],
            "generation_time": dt["generation_time"],
            "num_steps": dt["num_steps"],
            "tokens": [{
                "token": t["token"],
                "confidence": t["confidence"],
                "is_prompt": t["is_prompt"],
                "position": t["position"],
                "reveal_step": t.get("reveal_step", -1),
                "reveal_step_normalized": t.get("reveal_step_normalized", 0),
            } for t in dt["tokens"]],
        })

    combined_path = os.path.join(out_dir, "combined_traces.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f)
    print(f"Saved combined traces to {combined_path} ({os.path.getsize(combined_path) / 1024:.0f} KB)")

    print(f"\nDone! Load combined_traces.json in the confidence grid visualization.")


if __name__ == "__main__":
    main()
