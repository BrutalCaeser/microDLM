#!/usr/bin/env python3
"""
visualize.py — Side-by-Side Diffusion vs GPT Generation Animation
===================================================================
Shows HOW the two models generate text differently:

  Diffusion: starts from ALL masks, reveals tokens in parallel
       based on confidence — like developing a photograph.

  GPT: starts from a prompt, generates one token at a time
       left-to-right — like typing on a keyboard.

Run:
    python visualize.py                     # default demo
    python visualize.py --length 500        # longer output
    python visualize.py --no-animate        # instant (no delay)
    python visualize.py --diffusion-only    # show only diffusion
    python visualize.py --gpt-only          # show only GPT

Requires trained weights in weights/diffusion.pt and weights/gpt.pt
"""

import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================================
# ANSI Terminal Colors
# ============================================================================

class C:
    """ANSI color codes for terminal output."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"
    ULINE   = "\033[4m"

    # Foreground colors
    BLACK   = "\033[30m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"

    # Bright foreground
    BRIGHT_BLACK   = "\033[90m"   # grey
    BRIGHT_GREEN   = "\033[92m"
    BRIGHT_YELLOW  = "\033[93m"
    BRIGHT_CYAN    = "\033[96m"
    BRIGHT_WHITE   = "\033[97m"

    # Background
    BG_BLACK   = "\033[40m"
    BG_RED     = "\033[41m"
    BG_GREEN   = "\033[42m"
    BG_YELLOW  = "\033[43m"
    BG_BLUE    = "\033[44m"
    BG_MAGENTA = "\033[45m"

    # Cursor control
    CLEAR_SCREEN = "\033[2J\033[H"
    CLEAR_LINE   = "\033[2K"
    SAVE_CURSOR  = "\033[s"
    RESTORE_CURSOR = "\033[u"
    HIDE_CURSOR  = "\033[?25l"
    SHOW_CURSOR  = "\033[?25h"

    @staticmethod
    def move_to(row, col=1):
        return f"\033[{row};{col}H"


# ============================================================================
# Data & Vocabulary (must match training exactly)
# ============================================================================

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "shakespeare.txt")
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
MASK_CHAR = "_"
assert MASK_CHAR not in chars
chars_with_mask = [MASK_CHAR] + chars
vocab_size = len(chars_with_mask)  # 66

stoi = {ch: i for i, ch in enumerate(chars_with_mask)}
itos = {i: ch for i, ch in enumerate(chars_with_mask)}
mask_token_id = stoi[MASK_CHAR]  # 0

def encode(s):
    return [stoi[ch] for ch in s]

def decode(tokens):
    return "".join([itos[t] for t in tokens])

data = torch.tensor(encode(text), dtype=torch.long)

# ============================================================================
# Model Architecture (must match training — 6L/6H/384E)
# ============================================================================

# Hyperparams for the final trained models
n_embd = 384
n_head = 6
n_layer = 6
head_dim = n_embd // n_head  # 64
block_size = 256

device = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

def norm(x):
    """Functional RMSNorm — no learnable parameters."""
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    """Apply rotary positional embeddings to queries/keys."""
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
        B, T, _C = x.size()
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
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


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
        self.rotary_seq_len = block_size * 2
        cos, sin = self._precompute_rope(self.rotary_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.blocks = nn.ModuleList([Block(is_causal=is_causal) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def _precompute_rope(self, seq_len, base=10000, device=None):
        if device is None:
            device = self.token_emb.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return cos[None, :, None, :], sin[None, :, None, :]

    def forward(self, idx):
        B, T = idx.size()
        x = self.token_emb(idx)
        x = norm(x)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        for block in self.blocks:
            x = block(x, cos_sin)
        x = norm(x)
        return self.lm_head(x)


# ============================================================================
# Load Trained Weights
# ============================================================================

def load_models():
    """Load both trained models."""
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

    diff_path = os.path.join(weights_dir, "diffusion.pt")
    gpt_path = os.path.join(weights_dir, "gpt.pt")

    models = {}

    if os.path.exists(diff_path):
        diffusion_model = Model(is_causal=False).to(device)
        state = torch.load(diff_path, map_location=device, weights_only=True)
        diffusion_model.load_state_dict(state)
        diffusion_model.eval()
        models["diffusion"] = diffusion_model
        n = sum(p.numel() for p in diffusion_model.parameters())
        print(f"  ✓ Diffusion model loaded ({n/1e6:.1f}M params)")
    else:
        print(f"  ✗ {diff_path} not found")

    if os.path.exists(gpt_path):
        gpt_model = Model(is_causal=True).to(device)
        state = torch.load(gpt_path, map_location=device, weights_only=True)
        gpt_model.load_state_dict(state)
        gpt_model.eval()
        models["gpt"] = gpt_model
        n = sum(p.numel() for p in gpt_model.parameters())
        print(f"  ✓ GPT model loaded ({n/1e6:.1f}M params)")
    else:
        print(f"  ✗ {gpt_path} not found")

    return models


# ============================================================================
# Diffusion Generation with Step-by-Step Capture
# ============================================================================

@torch.no_grad()
def generate_diffusion_steps(model, gen_length=240, prompt_len=16,
                              num_steps=40, temp=0.8, top_k=3):
    """
    Generate text with iterative parallel unmasking using a cosine schedule.

    At each step, we predict all positions, then unmask the most-confident
    batch of tokens according to the schedule.  With num_steps=40 for
    224 generated tokens, each step reveals ~5-6 tokens on average.

    Yields (step_num, tokens, newly_unmasked_positions, remaining) at each step.
    """
    prompt_tokens = data[:prompt_len].tolist()
    total_gen = min(gen_length, block_size) - prompt_len

    # Initialize: prompt + MASKs
    x = torch.full((1, min(gen_length, block_size)), mask_token_id,
                    dtype=torch.long, device=device)
    x[0, :prompt_len] = torch.tensor(prompt_tokens, device=device)

    masked = torch.zeros(1, x.size(1), dtype=torch.bool, device=device)
    masked[0, prompt_len:] = True

    # Cosine schedule: how many tokens should remain masked after each step
    # Step 0: all masked.  Step num_steps: 0 masked.
    # n_masked(t) = total_gen * cos(π/2 * t / T)^2
    schedule = []
    for t in range(num_steps + 1):
        frac = math.cos(math.pi / 2 * t / num_steps) ** 2
        schedule.append(int(round(total_gen * frac)))
    schedule[0] = total_gen
    schedule[-1] = 0

    # Yield initial state (all masked)
    yield 0, x[0].cpu().tolist(), set(), total_gen

    for step in range(1, num_steps + 1):
        n_to_unmask = schedule[step - 1] - schedule[step]
        if n_to_unmask <= 0:
            continue
        if not masked.any():
            break

        logits = model(x)
        # SUBS parameterisation: never sample MASK token
        logits[:, :, mask_token_id] = -float("inf")
        probs = F.softmax(logits / temp, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
        confidences = top_k_probs[:, :, 0]  # confidence = prob of top-1

        # Only consider currently masked positions
        masked_conf = torch.where(
            masked, confidences, torch.tensor(-float("inf"), device=device)
        )
        # Pick the n_to_unmask most confident
        n_to_unmask = min(n_to_unmask, masked.sum().item())
        _, top_indices = torch.topk(masked_conf.view(-1), k=int(n_to_unmask))
        decode_mask = torch.zeros_like(masked.view(-1))
        decode_mask[top_indices] = True
        decode_mask = decode_mask.view_as(masked).bool()

        # Sample from top-k for unmasked positions
        top_k_probs_norm = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        sampled_k = torch.multinomial(
            top_k_probs_norm.view(-1, top_k), 1
        ).view(1, x.size(1))
        sampled_tokens = torch.gather(
            top_k_indices, -1, sampled_k.unsqueeze(-1)
        ).squeeze(-1)

        x = torch.where(decode_mask, sampled_tokens, x)
        newly_unmasked = set(decode_mask[0].nonzero(as_tuple=True)[0].cpu().tolist())
        masked = masked & ~decode_mask
        remaining = masked.sum().item()

        yield step, x[0].cpu().tolist(), newly_unmasked, remaining

    return


# ============================================================================
# GPT Generation with Step-by-Step Capture
# ============================================================================

@torch.no_grad()
def generate_gpt_steps(model, gen_length=240, prompt_len=16, temp=0.8):
    """
    Generate text autoregressively (one token at a time).
    Yields (step_num, tokens_so_far, new_position) at each step.
    """
    prompt_tokens = data[:prompt_len].tolist()
    x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    yield 0, prompt_tokens, None  # initial prompt

    for step in range(1, gen_length - prompt_len + 1):
        ctx = x[:, -block_size:]
        logits = model(ctx)
        logits = logits[:, -1, :]  # last position
        probs = F.softmax(logits / temp, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)

        tokens = x[0].cpu().tolist()
        yield step, tokens, len(tokens) - 1


# ============================================================================
# Terminal Rendering
# ============================================================================

def render_diffusion_frame(tokens, prompt_len, newly_unmasked, step, remaining,
                            total, width=80):
    """Render one frame of diffusion generation with colored output."""
    text_str = decode(tokens)
    n_unmasked = total - remaining
    progress = n_unmasked / total if total > 0 else 1.0

    # Progress bar
    bar_width = 40
    filled = int(bar_width * progress)
    bar = f"{'█' * filled}{'░' * (bar_width - filled)}"

    lines = []
    lines.append(f"  {C.CYAN}Step {step:3d}{C.RESET}  "
                 f"{C.DIM}[{bar}]{C.RESET}  "
                 f"{C.BRIGHT_GREEN}{n_unmasked}{C.RESET}/{total} revealed  "
                 f"{C.DIM}({progress:.0%}){C.RESET}")
    lines.append("")

    # Build colored text
    colored = []
    for i, tok_id in enumerate(tokens):
        ch = itos[tok_id]
        if i < prompt_len:
            # Prompt: bold white
            colored.append(f"{C.BOLD}{C.WHITE}{ch}{C.RESET}")
        elif tok_id == mask_token_id:
            # Still masked: dim grey dot
            colored.append(f"{C.BRIGHT_BLACK}·{C.RESET}")
        elif i in newly_unmasked:
            # Just unmasked this step: bright green with underline
            colored.append(f"{C.BOLD}{C.BRIGHT_GREEN}{ch}{C.RESET}")
        else:
            # Previously unmasked: normal yellow
            colored.append(f"{C.YELLOW}{ch}{C.RESET}")

    # Word-wrap the colored text
    raw_text = decode(tokens).replace(itos[mask_token_id], "·")
    # We'll print line by line, wrapping at width
    line_buf = "  "
    color_idx = 0
    col = 2
    for c_str in colored:
        # Get the raw character for width calculation
        raw_ch = raw_text[color_idx] if color_idx < len(raw_text) else "?"
        if raw_ch == "\n":
            lines.append(line_buf)
            line_buf = "  "
            col = 2
        else:
            line_buf += c_str
            col += 1
            if col >= width:
                lines.append(line_buf)
                line_buf = "  "
                col = 2
        color_idx += 1

    if line_buf.strip():
        lines.append(line_buf)

    return "\n".join(lines)


def render_gpt_frame(tokens, prompt_len, new_pos, step, total, width=80):
    """Render one frame of GPT generation with colored output."""
    n_generated = len(tokens) - prompt_len
    progress = n_generated / total if total > 0 else 0

    bar_width = 40
    filled = int(bar_width * progress)
    bar = f"{'█' * filled}{'░' * (bar_width - filled)}"

    lines = []
    lines.append(f"  {C.MAGENTA}Token {step:3d}{C.RESET}  "
                 f"{C.DIM}[{bar}]{C.RESET}  "
                 f"{C.BRIGHT_GREEN}{n_generated}{C.RESET}/{total} generated  "
                 f"{C.DIM}({progress:.0%}){C.RESET}")
    lines.append("")

    colored = []
    for i, tok_id in enumerate(tokens):
        ch = itos[tok_id]
        if i < prompt_len:
            colored.append(f"{C.BOLD}{C.WHITE}{ch}{C.RESET}")
        elif i == new_pos:
            colored.append(f"{C.BOLD}{C.BRIGHT_GREEN}{ch}{C.RESET}")
        else:
            colored.append(f"{C.CYAN}{ch}{C.RESET}")

    raw_text = decode(tokens)
    line_buf = "  "
    col = 2
    color_idx = 0
    for c_str in colored:
        raw_ch = raw_text[color_idx] if color_idx < len(raw_text) else "?"
        if raw_ch == "\n":
            lines.append(line_buf)
            line_buf = "  "
            col = 2
        else:
            line_buf += c_str
            col += 1
            if col >= width:
                lines.append(line_buf)
                line_buf = "  "
                col = 2
        color_idx += 1

    if line_buf.strip():
        lines.append(line_buf)

    return "\n".join(lines)


def print_header(title, subtitle="", color=C.CYAN):
    """Print a styled section header."""
    width = 72
    print()
    print(f"  {color}{'━' * width}{C.RESET}")
    print(f"  {color}{C.BOLD}{title}{C.RESET}")
    if subtitle:
        print(f"  {C.DIM}{subtitle}{C.RESET}")
    print(f"  {color}{'━' * width}{C.RESET}")
    print()


def print_box(title, content, color=C.CYAN):
    """Print content in a simple box."""
    width = 72
    print(f"  {color}┌{'─' * (width - 2)}┐{C.RESET}")
    print(f"  {color}│{C.RESET} {C.BOLD}{title}{C.RESET}{' ' * (width - 3 - len(title))}{color}│{C.RESET}")
    print(f"  {color}├{'─' * (width - 2)}┤{C.RESET}")
    for line in content.split("\n"):
        # Truncate if needed
        display = line[:width - 4]
        padding = width - 3 - len(display)
        print(f"  {color}│{C.RESET} {display}{' ' * max(0, padding)}{color}│{C.RESET}")
    print(f"  {color}└{'─' * (width - 2)}┘{C.RESET}")


# ============================================================================
# Pre-computation (generate all frames before animating)
# ============================================================================

def precompute_diffusion_frames(model, gen_length=240, prompt_len=16):
    """Pre-compute all diffusion generation frames."""
    frames = []  # list of (step, tokens, newly_unmasked, remaining)
    for step, tokens, newly_unmasked, remaining in \
            generate_diffusion_steps(model, gen_length, prompt_len):
        frames.append((step, list(tokens), set(newly_unmasked), remaining))
    return frames


def precompute_gpt_frames(model, gen_length=240, prompt_len=16):
    """Pre-compute all GPT generation frames."""
    frames = []  # list of (step, tokens, new_pos)
    for step, tokens, new_pos in generate_gpt_steps(model, gen_length, prompt_len):
        frames.append((step, list(tokens), new_pos))
    return frames


# ============================================================================
# Animation Runners
# ============================================================================

def animate_race(diff_frames, gpt_frames, gen_length, prompt_len, delay=0.02):
    """
    Animate both models on the SAME timeline.

    Both advance one frame per tick. Since diffusion has fewer frames
    (~40 for 240 tokens) vs GPT (~240 frames), diffusion finishes first.
    This correctly shows diffusion's parallelism advantage.

    After a model finishes, it shows ✓ DONE while the other continues.
    """
    total_gen = gen_length - prompt_len
    max_frames = max(len(diff_frames), len(gpt_frames))

    diff_done_at = None
    gpt_done_at = None

    for frame_idx in range(max_frames):
        # Get current state for each model
        # Diffusion
        if frame_idx < len(diff_frames):
            d_step, d_tokens, d_newly, d_remaining = diff_frames[frame_idx]
            diff_done = False
        else:
            if diff_done_at is None:
                diff_done_at = frame_idx
            d_step, d_tokens, d_newly, d_remaining = diff_frames[-1]
            diff_done = True

        # GPT
        if frame_idx < len(gpt_frames):
            g_step, g_tokens, g_new_pos = gpt_frames[frame_idx]
            gpt_done = False
        else:
            if gpt_done_at is None:
                gpt_done_at = frame_idx
            g_step, g_tokens, g_new_pos = gpt_frames[-1]
            gpt_done = True

        # Render
        sys.stdout.write("\033[2J\033[H")  # clear screen

        # Title bar
        print(f"\n  {C.BOLD}{C.BRIGHT_WHITE}🔬 MicroDiffusion LM — RACE{C.RESET}  "
              f"{C.DIM}(frame {frame_idx + 1}/{max_frames}){C.RESET}")
        print()

        # --- Diffusion section ---
        status = f"{C.BOLD}{C.BRIGHT_GREEN}✓ DONE in {len(diff_frames)} steps{C.RESET}" \
                 if diff_done else f"{C.CYAN}Step {d_step}{C.RESET}"
        print(f"  {C.CYAN}{'━' * 74}{C.RESET}")
        print(f"  {C.CYAN}{C.BOLD}🎨 DIFFUSION{C.RESET}  {status}  "
              f"{C.DIM}(parallel unmasking){C.RESET}")
        print(f"  {C.CYAN}{'━' * 74}{C.RESET}")

        n_unmasked = total_gen - d_remaining
        progress = n_unmasked / total_gen if total_gen > 0 else 1.0
        bar_w = 30
        filled = int(bar_w * progress)
        bar = f"{'█' * filled}{'░' * (bar_w - filled)}"
        print(f"  {C.DIM}[{bar}]{C.RESET} {n_unmasked}/{total_gen} "
              f"({progress:.0%})")

        # Render diffusion text (compact)
        text_line = "  "
        for i, tok_id in enumerate(d_tokens):
            ch = itos[tok_id]
            if i < prompt_len:
                text_line += f"{C.BOLD}{C.WHITE}{ch}{C.RESET}"
            elif tok_id == mask_token_id:
                text_line += f"{C.BRIGHT_BLACK}·{C.RESET}"
            elif not diff_done and i in d_newly:
                text_line += f"{C.BOLD}{C.BRIGHT_GREEN}{ch}{C.RESET}"
            else:
                text_line += f"{C.YELLOW}{ch}{C.RESET}"
        print(text_line)
        print()

        # --- GPT section ---
        g_n_gen = len(g_tokens) - prompt_len
        status = f"{C.BOLD}{C.BRIGHT_GREEN}✓ DONE in {len(gpt_frames)} steps{C.RESET}" \
                 if gpt_done else f"{C.MAGENTA}Token {g_step}{C.RESET}"
        print(f"  {C.MAGENTA}{'━' * 74}{C.RESET}")
        print(f"  {C.MAGENTA}{C.BOLD}⌨️  GPT{C.RESET}  {status}  "
              f"{C.DIM}(sequential left-to-right){C.RESET}")
        print(f"  {C.MAGENTA}{'━' * 74}{C.RESET}")

        progress_g = g_n_gen / total_gen if total_gen > 0 else 0
        filled_g = int(bar_w * progress_g)
        bar_g = f"{'█' * filled_g}{'░' * (bar_w - filled_g)}"
        print(f"  {C.DIM}[{bar_g}]{C.RESET} {g_n_gen}/{total_gen} "
              f"({progress_g:.0%})")

        # Render GPT text (compact)
        text_line = "  "
        for i, tok_id in enumerate(g_tokens):
            ch = itos[tok_id]
            if i < prompt_len:
                text_line += f"{C.BOLD}{C.WHITE}{ch}{C.RESET}"
            elif not gpt_done and i == g_new_pos:
                text_line += f"{C.BOLD}{C.BRIGHT_GREEN}{ch}{C.RESET}"
            else:
                text_line += f"{C.CYAN}{ch}{C.RESET}"
        print(text_line)
        print()

        # Speed comparison footer
        if diff_done and not gpt_done:
            speedup = f"~{len(gpt_frames) / max(len(diff_frames), 1):.0f}×"
            print(f"  {C.BRIGHT_GREEN}{C.BOLD}⚡ Diffusion finished! "
                  f"{speedup} fewer steps than GPT.{C.RESET} "
                  f"{C.DIM}GPT still generating...{C.RESET}")

        sys.stdout.flush()
        time.sleep(delay)

    return (
        decode(diff_frames[-1][1]), len(diff_frames),
        decode(gpt_frames[-1][1]), len(gpt_frames)
    )


def animate_single_diffusion(model, gen_length=240, prompt_len=16, delay=0.05):
    """Animate diffusion generation alone."""
    torch.manual_seed(42)
    total_gen = gen_length - prompt_len

    frames = precompute_diffusion_frames(model, gen_length, prompt_len)
    for step, tokens, newly_unmasked, remaining in frames:
        frame = render_diffusion_frame(
            tokens, prompt_len, newly_unmasked, step, remaining,
            total_gen, width=76
        )
        sys.stdout.write("\033[2J\033[H")
        print_header(
            "🎨 DIFFUSION: Parallel Unmasking",
            "Reveals tokens by confidence — most confident first",
            C.CYAN
        )
        print(frame)
        sys.stdout.flush()
        time.sleep(delay)

    print()
    print(f"  {C.DIM}Completed in {len(frames)} steps{C.RESET}")
    print(f"  {C.DIM}Avg tokens decoded per step: "
          f"{total_gen / max(len(frames), 1):.1f}{C.RESET}")
    return decode(frames[-1][1])


def animate_single_gpt(model, gen_length=240, prompt_len=16, delay=0.01):
    """Animate GPT generation alone."""
    torch.manual_seed(42)
    total_gen = gen_length - prompt_len

    frames = precompute_gpt_frames(model, gen_length, prompt_len)
    for step, tokens, new_pos in frames:
        frame = render_gpt_frame(
            tokens, prompt_len, new_pos, step, total_gen, width=76
        )
        sys.stdout.write("\033[2J\033[H")
        print_header(
            "⌨️  GPT: Sequential Left-to-Right",
            "Generates one token at a time, left to right",
            C.MAGENTA
        )
        print(frame)
        sys.stdout.flush()
        time.sleep(delay)

    print()
    print(f"  {C.DIM}Completed in {len(frames)} steps{C.RESET}")
    return decode(frames[-1][1])


# ============================================================================
# Comparison Summary
# ============================================================================

def show_comparison(diff_text, diff_steps, gpt_text, gpt_steps,
                    gen_length, prompt_len):
    """Print a beautiful side-by-side comparison."""

    total_gen = gen_length - prompt_len
    speedup = gpt_steps / max(diff_steps, 1)

    print_header(
        "📊 COMPARISON: Diffusion vs GPT",
        "Same architecture (6L/6H/384E, ~10.7M params), same data, same prompt",
        C.YELLOW
    )

    # Stats table
    print(f"  {C.BOLD}{'':22s} {'Diffusion':>16s}  {'GPT':>16s}{C.RESET}")
    print(f"  {'─' * 58}")
    print(f"  {'Attention':22s} {C.CYAN}{'Bidirectional':>16s}{C.RESET}  {C.MAGENTA}{'Causal (L→R)':>16s}{C.RESET}")
    print(f"  {'Forward passes':22s} {C.CYAN}{diff_steps:>16d}{C.RESET}  {C.MAGENTA}{gpt_steps:>16d}{C.RESET}")
    print(f"  {'Tokens per step':22s} {C.CYAN}{total_gen/max(diff_steps,1):>16.1f}{C.RESET}  {C.MAGENTA}{'1.0':>16s}{C.RESET}")
    print(f"  {'Step speedup':22s} {C.CYAN}{speedup:>15.1f}×{C.RESET}  {C.MAGENTA}{'baseline':>16s}{C.RESET}")
    print(f"  {'Parallelism':22s} {C.CYAN}{'✓ Parallel':>16s}{C.RESET}  {C.MAGENTA}{'✗ Sequential':>16s}{C.RESET}")
    print()

    # The 5 key differences
    print(f"  {C.BOLD}{C.YELLOW}5 Changes from GPT → Diffusion:{C.RESET}")
    diffs = [
        ("Add MASK token",      "vocab_size + 1",          "standard vocab"),
        ("Attention direction",  "Bidirectional (sees all)", "Causal (sees left)"),
        ("Training objective",   "Denoise masked tokens",   "Predict next token"),
        ("Loss scope",           "Masked positions only",   "All positions"),
        ("Generation",           "Parallel by confidence",  "Sequential L→R"),
    ]
    for i, (name, diff_val, gpt_val) in enumerate(diffs, 1):
        print(f"    {C.YELLOW}{i}.{C.RESET} {C.BOLD}{name}{C.RESET}")
        print(f"       Diffusion: {C.CYAN}{diff_val}{C.RESET}")
        print(f"       GPT:       {C.MAGENTA}{gpt_val}{C.RESET}")

    print()

    # Show outputs
    preview = 300
    print_box("🎨 Diffusion Output", diff_text[:preview], C.CYAN)
    print()
    print_box("⌨️  GPT Output", gpt_text[:preview], C.MAGENTA)
    print()


# ============================================================================
# Static Demo (no animation)
# ============================================================================

def static_demo(models, gen_length=240, prompt_len=16):
    """Show snapshots of generation at key moments (no animation)."""

    if "diffusion" in models:
        print_header(
            "🎨 DIFFUSION: Generation Snapshots",
            "Showing key frames of the parallel unmasking process",
            C.CYAN
        )

        torch.manual_seed(42)
        total_gen = gen_length - prompt_len
        snapshots = []
        target_pcts = {0, 10, 25, 50, 75, 90, 100}
        captured_pcts = set()

        for step, tokens, newly_unmasked, remaining in \
                generate_diffusion_steps(models["diffusion"], gen_length, prompt_len):
            pct = int(100 * (1 - remaining / total_gen)) if total_gen > 0 else 100
            # Capture the first time we cross each target percentage
            for target in sorted(target_pcts):
                if target not in captured_pcts and pct >= target:
                    snapshots.append((step, tokens, newly_unmasked, remaining, pct))
                    captured_pcts.add(target)

        # Display snapshots
        for step, tokens, newly_unmasked, remaining, pct in snapshots:
            n_revealed = total_gen - remaining
            label = f"Step {step} — {pct}% revealed ({n_revealed}/{total_gen})"
            print(f"  {C.CYAN}{C.BOLD}{label}{C.RESET}")

            text_display = ""
            for i, tok_id in enumerate(tokens):
                ch = itos[tok_id]
                if i < prompt_len:
                    text_display += f"{C.BOLD}{C.WHITE}{ch}{C.RESET}"
                elif tok_id == mask_token_id:
                    text_display += f"{C.BRIGHT_BLACK}·{C.RESET}"
                elif i in newly_unmasked:
                    text_display += f"{C.BRIGHT_GREEN}{ch}{C.RESET}"
                else:
                    text_display += f"{C.YELLOW}{ch}{C.RESET}"

            # Print with wrapping (simple approach: print raw then colored)
            raw_text = ""
            for i, tok_id in enumerate(tokens):
                if tok_id == mask_token_id:
                    raw_text += "·"
                else:
                    raw_text += itos[tok_id]

            # Print colored version
            print(f"  {text_display}")
            print()

        final_text = decode(snapshots[-1][1]) if snapshots else ""

    if "gpt" in models:
        print_header(
            "⌨️  GPT: Generation Preview",
            "Showing progressive left-to-right generation",
            C.MAGENTA
        )

        torch.manual_seed(42)
        total_gen = gen_length - prompt_len
        snapshots = []
        target_pcts = {0, 10, 25, 50, 75, 100}
        captured_pcts = set()

        for step, tokens, new_pos in generate_gpt_steps(models["gpt"], gen_length, prompt_len):
            pct = int(100 * step / total_gen) if total_gen > 0 else 100
            for target in sorted(target_pcts):
                if target not in captured_pcts and pct >= target:
                    snapshots.append((step, tokens, new_pos, pct))
                    captured_pcts.add(target)

        for step, tokens, new_pos, pct in snapshots:
            n_gen = len(tokens) - prompt_len
            label = f"Token {step} — {pct}% generated ({n_gen}/{total_gen})"
            print(f"  {C.MAGENTA}{C.BOLD}{label}{C.RESET}")

            text_display = ""
            for i, tok_id in enumerate(tokens):
                ch = itos[tok_id]
                if i < prompt_len:
                    text_display += f"{C.BOLD}{C.WHITE}{ch}{C.RESET}"
                elif i == new_pos:
                    text_display += f"{C.BRIGHT_GREEN}{ch}{C.RESET}"
                else:
                    text_display += f"{C.CYAN}{ch}{C.RESET}"

            print(f"  {text_display}")
            print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MicroDiffusion LM — Diffusion vs GPT generation visualization"
    )
    parser.add_argument("--length", type=int, default=240,
                        help="Total sequence length to generate (default: 240)")
    parser.add_argument("--prompt-len", type=int, default=16,
                        help="Number of prompt tokens from Shakespeare (default: 16)")
    parser.add_argument("--no-animate", action="store_true",
                        help="Show static snapshots instead of animation")
    parser.add_argument("--diffusion-only", action="store_true",
                        help="Only show diffusion generation")
    parser.add_argument("--gpt-only", action="store_true",
                        help="Only show GPT generation")
    parser.add_argument("--delay", type=float, default=0.02,
                        help="Animation delay per frame in seconds (default: 0.02)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for generation (default: 42)")
    args = parser.parse_args()

    # Banner
    print(f"\n{C.BOLD}{C.BRIGHT_WHITE}")
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║          🔬  MicroDiffusion LM — Visualizer  🔬            ║")
    print("  ║     Discrete Diffusion vs GPT on Tiny Shakespeare          ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print(f"{C.RESET}")

    # Info
    print(f"  {C.DIM}Device: {device} | Vocab: {vocab_size} chars | "
          f"Context: {block_size} | Architecture: {n_layer}L/{n_head}H/{n_embd}E{C.RESET}")
    print(f"  {C.DIM}Generating {args.length} chars with {args.prompt_len} prompt tokens{C.RESET}")
    print()

    # Load models
    print(f"  {C.BOLD}Loading models...{C.RESET}")
    models = load_models()
    print()

    if not models:
        print(f"  {C.RED}No trained weights found in weights/ directory.{C.RESET}")
        print(f"  {C.DIM}Run training first (see colab_training.ipynb){C.RESET}")
        sys.exit(1)

    # Filter models based on flags
    if args.diffusion_only:
        models = {k: v for k, v in models.items() if k == "diffusion"}
    elif args.gpt_only:
        models = {k: v for k, v in models.items() if k == "gpt"}

    torch.manual_seed(args.seed)

    if args.no_animate:
        static_demo(models, args.length, args.prompt_len)
    else:
        try:
            print(C.HIDE_CURSOR, end="")

            # === RACE MODE (both models, same timeline) ===
            if "diffusion" in models and "gpt" in models:
                print(f"  {C.DIM}Pre-computing diffusion frames...{C.RESET}", end="", flush=True)
                torch.manual_seed(args.seed)
                diff_frames = precompute_diffusion_frames(
                    models["diffusion"], args.length, args.prompt_len
                )
                print(f" {C.BRIGHT_GREEN}✓{C.RESET} ({len(diff_frames)} frames)")

                print(f"  {C.DIM}Pre-computing GPT frames...{C.RESET}", end="", flush=True)
                torch.manual_seed(args.seed)
                gpt_frames = precompute_gpt_frames(
                    models["gpt"], args.length, args.prompt_len
                )
                print(f" {C.BRIGHT_GREEN}✓{C.RESET} ({len(gpt_frames)} frames)")
                print()
                print(f"  {C.BOLD}Starting race — same speed per frame,{C.RESET} "
                      f"{C.CYAN}diffusion{C.RESET} has {C.BOLD}{len(diff_frames)}{C.RESET} frames, "
                      f"{C.MAGENTA}GPT{C.RESET} has {C.BOLD}{len(gpt_frames)}{C.RESET}")
                print(f"  {C.DIM}Press Enter to start...{C.RESET}", end="")
                sys.stdout.flush()
                input()

                diff_text, diff_steps, gpt_text, gpt_steps = animate_race(
                    diff_frames, gpt_frames,
                    args.length, args.prompt_len,
                    delay=args.delay
                )

                # Show comparison summary
                sys.stdout.write("\033[2J\033[H")
                show_comparison(
                    diff_text, diff_steps,
                    gpt_text, gpt_steps,
                    args.length, args.prompt_len
                )

            # === Single model modes ===
            elif "diffusion" in models:
                animate_single_diffusion(
                    models["diffusion"], args.length, args.prompt_len,
                    delay=args.delay
                )
            elif "gpt" in models:
                animate_single_gpt(
                    models["gpt"], args.length, args.prompt_len,
                    delay=args.delay
                )

        finally:
            print(C.SHOW_CURSOR, end="")

    prompt_text = decode(data[:args.prompt_len].tolist())
    print(f"  {C.DIM}Prompt: {repr(prompt_text)}{C.RESET}")
    print()


if __name__ == "__main__":
    main()
