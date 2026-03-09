# MicroDiffusion LM — Build Plan
## A from-scratch discrete diffusion language model on Tiny Shakespeare

---

## Repository Structure (what you'll build)

```
microdiffusion/
├── README.md                  ← project overview, math explanation, results
├── data/
│   └── shakespeare.txt        ← Tiny Shakespeare dataset (~1MB)
├── step0_masking.py           ← Forward process only. No neural net.
├── step1_denoise_mlp.py       ← Train a simple MLP to denoise. Proves the loss works.
├── step2_transformer.py       ← Replace MLP with bidirectional transformer. Quality jump.
├── step3_sampling.py          ← Add iterative unmasking. Generate text from noise.
├── diffusion.py               ← Final clean single-file implementation (train + generate)
├── gpt.py                     ← Equivalent autoregressive model for comparison
├── visualize.py               ← Side-by-side generation animation
├── weights/
│   ├── diffusion.pt           ← Trained diffusion model weights
│   └── gpt.pt                 ← Trained GPT model weights
└── blog/
    └── post.md                ← Blog post walking through the build
```

---

## Phase 1: step0_masking.py (Day 1)
**Goal:** Implement the forward process and verify it matches the math.
**No neural network. No training. Just masking.**

What to build:
- Load Shakespeare text, build character-level vocabulary (unique chars + MASK)
- Implement the cosine schedule: α(t) = cos(πt/2)
- Implement forward_process(x, t) that masks tokens
- Verify: at t=0, nothing masked. At t=1, everything masked.
- Verify: the distribution of masked counts matches α^k × (1-α)^m
- Print examples at different noise levels so you can see the corruption

What you should see:
```
t=0.0: "First Citizen:\nBefore we proceed any further"
t=0.2: "First C_tize_:\nBefore w_ proceed _ny furth_r"
t=0.5: "F_rs_ _i__z__:\__ef_r_ __ _r_ce__ _ny f_rth__"
t=0.8: "____t _____e_:_______e __ _______ ___ ______r"
t=1.0: "________________________________________________"
```

Milestone: You can explain every line and verify the statistics by hand.

---

## Phase 2: step1_denoise_mlp.py (Day 2)
**Goal:** Train a simple feedforward network to predict masked tokens.
**This will be BAD at generation but proves the training loop works.**

What to build:
- Simple MLP: embedding → 2 hidden layers with ReLU → output projection
- No attention, no positional encoding (the MLP processes each position independently)
- Training loop: sample batch, sample t, mask tokens, compute cross-entropy at masked positions
- Log the loss curve

What you should see:
- Loss starts around -log(1/vocab_size) ≈ 4.2 (random guessing over ~67 chars)
- Loss drops to around 2.5-3.0 (learns character frequencies but no context)
- The MLP can predict common characters ('e', 't', ' ') but not context-dependent ones

Why this step matters:
- Validates the entire data pipeline (tokenization, batching, masking)
- Validates the loss computation (cross-entropy only at masked positions)
- Shows WHY attention is needed — the MLP can't use context

Milestone: Loss curve drops. You understand why the MLP hits a ceiling.

---

## Phase 3: step2_transformer.py (Day 3-4)
**Goal:** Replace MLP with a single-layer bidirectional transformer. See the quality jump.**

What to build:
- Token embedding + positional embedding (learned)
- Multi-head self-attention (BIDIRECTIONAL — no causal mask)
- MLP block (feedforward with ReLU/GELU)
- RMSNorm (simpler than LayerNorm, following Karpathy)
- Residual connections
- Output projection (tied with embedding weights)
- SUBS zero-masking: clamp MASK logit to -inf

Architecture hyperparameters (start small):
- n_layers = 4
- n_heads = 4
- n_embd = 128
- block_size = 256 (context length)
- vocab_size = len(chars) + 1 (for MASK)

Training:
- AdamW optimizer, learning rate 3e-4 with cosine decay
- Batch size 64, train for 5000-10000 steps
- Log loss every 100 steps

What you should see:
- Loss drops significantly below the MLP ceiling (to ~1.5-1.8)
- The model learns character-level patterns: common words, spacing, punctuation

Milestone: Loss clearly better than MLP. You can explain every component of the transformer.

---

## Phase 4: step3_sampling.py (Day 4-5)
**Goal:** Implement iterative parallel unmasking. Generate actual text.**

What to build:
- Start from all-MASK sequence
- Cosine unmasking schedule
- Confidence-based selection (unmask most confident positions first)
- Temperature-controlled sampling (multinomial, not argmax)
- Semi-autoregressive block generation for arbitrary length:
  generate one block of 256 tokens, then use it as context for the next block

What you should see:
- Generated text that looks vaguely Shakespearean
- Character-level patterns: proper spacing, common words, name-like strings
- NOT perfect prose — this is a small model on character-level data

Example output (what to realistically expect from a ~10M param model):
```
KING HENRY:
The sorrow of the people shand the beart,
That we should in the proth of his servent,
And so the weathers that shall be so stander.
```

Milestone: Generated text is recognizably English/Shakespearean. Not gibberish.

---

## Phase 5: diffusion.py + gpt.py (Day 5-6)
**Goal:** Clean, final single-file implementations for both models.**

diffusion.py should contain:
- Everything from steps 0-3, cleaned up into one file
- Train mode (python diffusion.py --train)
- Generate mode (python diffusion.py --generate)
- ~350-400 lines total

gpt.py should contain:
- Equivalent autoregressive model with identical architecture EXCEPT:
  - Causal attention mask (not bidirectional)
  - Next-token prediction loss (not masked denoising)
  - Sequential left-to-right generation (not parallel unmasking)
- Same parameter count, same data, same training budget
- ~300-350 lines total

This direct comparison is the strongest part of your blog post.

---

## Phase 6: visualize.py + README + blog (Day 6-7)
**Goal:** Make the results presentable and publishable.**

visualize.py:
- Show diffusion generation step by step (all MASK → partial → complete)
- Show GPT generation token by token for comparison
- Terminal-based visualization (colored text, step counter)

README.md:
- What this project is and why it exists
- The 5 key changes from GPT to diffusion
- How to run training and generation
- Sample outputs
- Links to your blog post

blog/post.md:
- Walk through the math (your Week 1 understanding)
- Walk through the progressive build (step0 → step3)
- Show the comparison between diffusion and GPT outputs
- Discuss what works, what doesn't, and why
- Link to the broader dLLM landscape

---

## Training Setup

**Local (MacBook Air M4):** Good for step0 and step1. MPS acceleration works for small models.

**Google Colab Pro (T4/A100):** Use for step2 onwards. The transformer training
benefits significantly from GPU acceleration.

Training time estimates on A100:
- step1 (MLP): ~2 minutes for 5000 steps
- step2 (Transformer): ~10-20 minutes for 10000 steps
- Final diffusion.py: ~20 minutes for 10000 steps
- Final gpt.py: ~10 minutes for 5000 steps (half the iterations since every token contributes to loss)

---

## Ready to Start

Phase 1 (step0_masking.py) requires:
1. Download Tiny Shakespeare: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
2. Save as data/shakespeare.txt
3. Start coding — the forward process is ~50 lines

Start with Phase 1 now?
