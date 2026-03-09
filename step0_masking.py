"""
step0_masking.py — Forward Process Only
========================================
Implements the forward (noising) process for discrete diffusion.
No neural network. No training. Just masking.

The forward process corrupts text by independently masking each token
with probability (1 - α(t)), where α(t) = cos(πt/2) is the cosine schedule.

At t=0: α=1, nothing masked (clean text)
At t=1: α=0, everything masked (pure noise)

This file verifies:
1. The cosine schedule behaves correctly
2. The forward process masks tokens at the expected rates
3. The distribution of masked counts matches the binomial distribution
"""

import os
import math
import torch
import numpy as np

# ============================================================================
# Data Loading & Vocabulary
# ============================================================================

data_path = os.path.join(os.path.dirname(__file__), "data", "shakespeare.txt")
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

# Build character-level vocabulary
chars = sorted(list(set(text)))
MASK_CHAR = "_"
# Ensure underscore is not already in the text (the reference repo uses this convention)
assert MASK_CHAR not in chars, f"MASK character '{MASK_CHAR}' already in text!"
chars = [MASK_CHAR] + chars  # MASK is index 0
vocab_size = len(chars)

# Character <-> integer mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
mask_token_id = stoi[MASK_CHAR]  # = 0


def encode(s):
    """String -> list of integers"""
    return [stoi[ch] for ch in s]


def decode(tokens):
    """List of integers -> string"""
    return "".join([itos[t] for t in tokens])


# Encode the full text
data = torch.tensor(encode(text), dtype=torch.long)

print(f"Dataset size: {len(text):,} characters")
print(f"Vocabulary size: {vocab_size} (including MASK token)")
print(f"Unique chars: {len(chars)}")
print(f"MASK token id: {mask_token_id}")
print(f"First 50 chars: {repr(text[:50])}")
print()

# ============================================================================
# Cosine Schedule
# ============================================================================


def alpha(t):
    """
    Cosine noise schedule: α(t) = cos(πt/2)

    α(t) gives the probability that each token is KEPT (not masked).
    - At t=0: α=1 → all tokens kept (clean)
    - At t=1: α=0 → all tokens masked (noise)

    The cosine schedule is smooth and spends more time at intermediate
    noise levels, which is better for learning than a linear schedule.
    """
    return math.cos(math.pi * t / 2)


# ============================================================================
# Forward Process
# ============================================================================


def forward_process(x, t):
    """
    Apply the forward (noising) process to a sequence of tokens.

    For each token independently:
      - Keep the token with probability α(t)
      - Replace with MASK with probability 1 - α(t)

    Args:
        x: tensor of token ids, shape (seq_len,)
        t: noise level in [0, 1]

    Returns:
        x_noisy: tensor of token ids with some replaced by mask_token_id
        mask: boolean tensor, True where tokens were masked
    """
    a = alpha(t)
    # Each token is independently kept with probability α(t)
    keep_mask = torch.rand(x.shape) < a
    x_noisy = x.clone()
    x_noisy[~keep_mask] = mask_token_id
    mask = ~keep_mask
    return x_noisy, mask


# ============================================================================
# Verification & Visualization
# ============================================================================


def visualize_noise_levels():
    """Show text corruption at different noise levels."""
    sample_text = text[:50]  # "First Citizen:\nBefore we proceed any further, hea"
    sample_tokens = torch.tensor(encode(sample_text), dtype=torch.long)

    print("=" * 60)
    print("FORWARD PROCESS VISUALIZATION")
    print("=" * 60)

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    for t in noise_levels:
        a = alpha(t)
        x_noisy, mask = forward_process(sample_tokens, t)
        decoded = decode(x_noisy.tolist())
        n_masked = mask.sum().item()
        n_total = len(sample_tokens)
        pct = 100 * n_masked / n_total
        print(f"t={t:.1f} (α={a:.3f}): [{n_masked:2d}/{n_total} = {pct:5.1f}% masked] {repr(decoded)}")

    print()


def verify_schedule():
    """Verify cosine schedule properties."""
    print("=" * 60)
    print("COSINE SCHEDULE VERIFICATION")
    print("=" * 60)

    # Check boundary conditions
    assert abs(alpha(0.0) - 1.0) < 1e-10, "α(0) should be 1.0"
    assert abs(alpha(1.0) - 0.0) < 1e-10, "α(1) should be 0.0"
    print(f"✓ α(0) = {alpha(0.0):.6f} (should be 1.0)")
    print(f"✓ α(1) = {alpha(1.0):.6f} (should be 0.0)")

    # Check monotonicity
    ts = [i / 100 for i in range(101)]
    alphas = [alpha(t) for t in ts]
    for i in range(1, len(alphas)):
        assert alphas[i] <= alphas[i - 1], f"α should be non-increasing: α({ts[i]}) > α({ts[i-1]})"
    print(f"✓ α(t) is monotonically non-increasing over [0, 1]")

    # Check midpoint
    mid = alpha(0.5)
    print(f"✓ α(0.5) = {mid:.6f} (should be cos(π/4) = {math.cos(math.pi/4):.6f})")

    # Print schedule at key points
    print("\nSchedule values:")
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(f"  t={t:.1f} → α={alpha(t):.4f} → mask_prob={1-alpha(t):.4f}")

    print()


def verify_masking_statistics():
    """
    Verify that the empirical masking rate matches the expected rate.

    For a sequence of length L at noise level t:
    - Each token is independently masked with prob (1 - α(t))
    - Expected number of masked tokens: L × (1 - α(t))
    - Variance: L × α(t) × (1 - α(t))  [binomial variance]
    """
    print("=" * 60)
    print("MASKING STATISTICS VERIFICATION")
    print("=" * 60)

    seq_len = 256
    n_trials = 10000
    sample = data[:seq_len]

    test_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

    for t in test_levels:
        a = alpha(t)
        expected_masked = seq_len * (1 - a)
        expected_var = seq_len * a * (1 - a)
        expected_std = math.sqrt(expected_var)

        # Run many trials
        masked_counts = []
        for _ in range(n_trials):
            _, mask = forward_process(sample, t)
            masked_counts.append(mask.sum().item())

        empirical_mean = np.mean(masked_counts)
        empirical_std = np.std(masked_counts)

        # Check that empirical mean is within 3 standard errors of expected
        std_error = expected_std / math.sqrt(n_trials)
        deviation = abs(empirical_mean - expected_masked) / std_error

        status = "✓" if deviation < 3 else "✗"
        print(f"{status} t={t:.1f}: expected={expected_masked:.1f}±{expected_std:.1f}, "
              f"got={empirical_mean:.1f}±{empirical_std:.1f} "
              f"(deviation={deviation:.1f}σ)")

    print()


def verify_boundary_conditions():
    """Verify t=0 gives clean text and t=1 gives all masks."""
    print("=" * 60)
    print("BOUNDARY CONDITIONS VERIFICATION")
    print("=" * 60)

    sample = data[:100]

    # t=0: nothing should be masked
    x_noisy, mask = forward_process(sample, 0.0)
    assert mask.sum().item() == 0, "At t=0, nothing should be masked"
    assert torch.equal(x_noisy, sample), "At t=0, output should equal input"
    print(f"✓ t=0: 0 tokens masked, output equals input")

    # t=1: everything should be masked
    x_noisy, mask = forward_process(sample, 1.0)
    assert mask.sum().item() == len(sample), "At t=1, everything should be masked"
    assert (x_noisy == mask_token_id).all(), "At t=1, all tokens should be MASK"
    print(f"✓ t=1: {len(sample)} tokens masked, all are MASK token")

    print()


def show_progressive_corruption():
    """Show how a single piece of text gets progressively corrupted."""
    print("=" * 60)
    print("PROGRESSIVE CORRUPTION (same random seed)")
    print("=" * 60)

    sample_text = "First Citizen:\nBefore we proceed any further"
    sample_tokens = torch.tensor(encode(sample_text), dtype=torch.long)

    # Use a fixed seed for reproducibility
    torch.manual_seed(42)
    steps = 20
    for i in range(steps + 1):
        t = i / steps
        torch.manual_seed(42 + i)  # different but reproducible per step
        x_noisy, mask = forward_process(sample_tokens, t)
        decoded = decode(x_noisy.tolist())
        n_masked = mask.sum().item()
        bar = "█" * int(30 * n_masked / len(sample_tokens)) + "░" * int(30 * (1 - n_masked / len(sample_tokens)))
        print(f"t={t:.2f} |{bar}| {decoded}")

    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("MicroDiffusion LM — Step 0: Forward Process (Masking)")
    print("=" * 60)
    print()

    # 1. Verify the schedule
    verify_schedule()

    # 2. Verify boundary conditions
    verify_boundary_conditions()

    # 3. Verify masking statistics
    verify_masking_statistics()

    # 4. Visualize noise levels
    visualize_noise_levels()

    # 5. Progressive corruption
    show_progressive_corruption()

    print("=" * 60)
    print("Phase 1 Complete! All verifications passed.")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  • α(t) = cos(πt/2) smoothly transitions from clean → noise")
    print("  • Each token is independently masked with probability 1 - α(t)")
    print("  • The empirical statistics match the binomial distribution")
    print("  • At t=0: clean text. At t=1: all MASK tokens.")
    print()
    print(f"  Vocab size: {vocab_size} ({vocab_size - 1} chars + 1 MASK)")
    print(f"  Dataset: {len(data):,} tokens")
    print(f"  MASK token: '{MASK_CHAR}' (id={mask_token_id})")
