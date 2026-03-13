# Training Pipeline Verification & GPU Efficiency

## Current pipeline (before optimization)

### Flow today

1. **Module load**
   - Read data file → encode → `train_data` / `val_data` (CPU). ✓ Good: no GPU yet.
   - `device = cuda if available else mps else cpu` — no GPU memory yet.

2. **`get_batch(split)`**
   - Build batch on CPU (indexing, stack, mask). ✓ Good.
   - **Gap:** `.to(device)` is **synchronous**: CPU→GPU copy blocks until done. GPU is idle during this.

3. **`train()`**
   - `model = Model().to(device)` — model on GPU. ✓
   - Loop: `get_batch("train")` → `model(xb, yb, mb)` → `backward` → `step`.
   - **Gap:** Every iteration the GPU waits for `get_batch` (CPU work + blocking transfer). No overlap.

4. **`estimate_loss(model)`**
   - Runs `eval_iters` × 2 (train + val) = 400 batches: each `get_batch` + `.to(device)` + forward.
   - **Gap:** All transfers blocking; GPU idle during batch construction and transfer.

5. **Checkpoint load (generate path)**
   - `torch.load(path, map_location=device)` — can allocate on GPU during load.
   - **Gap:** Prefer load to CPU then `.to(device)` so we don’t hold two copies and control when GPU is used.

6. **Save**
   - `torch.save(model.state_dict(), ...)` while model is still on GPU.
   - **Gap:** Optional: move model to CPU before save to “offload” and free GPU during I/O.

---

## Target pipeline (GPU‑efficient)

### Phase 1 — CPU only (before any GPU use)

- Load and cache all data, build vocab, create tensors on CPU.
- Create **model on CPU** (no `.to(device)` yet).
- Create optimizer (parameters still on CPU).
- Optionally pre-allocate pinned buffers for batches (for async transfer).

### Phase 2 — Move to GPU once

- `model.to(device)`.
- If CUDA: `torch.cuda.synchronize()` so timing and “GPU phase” are clear.

### Phase 3 — Training loop (minimize GPU idle time)

- **Batches:** Build on CPU, put in **pinned memory**, then `.to(device, non_blocking=True)` so transfer is asynchronous. GPU can overlap with next batch prep.
- **No** blocking `.to(device)` in the hot path.
- Forward/backward/step as today.

### Phase 4 — Offload after training

- Optionally `model.cpu()` before `torch.save` to free GPU during I/O and match “offload” requirement.

### Checkpoint load

- `torch.load(path, map_location="cpu", weights_only=True)` then `model.load_state_dict(...)` then `model.to(device)` so GPU is used only when we’re ready.

---

## Gaps addressed in code

| Gap | Change |
|-----|--------|
| Blocking batch transfer | Pinned memory + `non_blocking=True` in `get_batch`. |
| Model on GPU during “setup” | Create model on CPU; call `model.to(device)` only at start of training phase. |
| No explicit CPU→GPU phase split | Document and enforce: data/optimizer on CPU first, then single `model.to(device)`, then loop. |
| Eval doing many blocking transfers | Same pinned + `non_blocking=True` in `get_batch` used by `estimate_loss`. |
| Load checkpoint straight to GPU | Load with `map_location="cpu"`, then `model.to(device)`. |
| GPU held during save | Optional `model.cpu()` before `torch.save`. |

---

## Files updated

- **diffusion.py**: `get_batch` uses pinned + `non_blocking`; `train()` builds model on CPU, moves to GPU once, then loop; save offloads to CPU then moves back to device; load uses `map_location="cpu"` then `to(device)`.
- **gpt.py**: Same pattern.

The same pattern can be applied to **steps/step1_denoise_mlp.py** and **steps/step2_transformer.py** if you run them for training on HPC.

---

## What we are *not* changing (for now)

- **DataLoader / prefetch:** For small in-memory data (e.g. Tiny Shakespeare), a single-threaded `get_batch` with pinned + non_blocking is enough. For large datasets (e.g. OpenWebText), a later step is to use `DataLoader` with `num_workers` and `pin_memory=True`.
- **Double-buffering:** Pre-building the next batch on CPU while GPU runs would require a separate thread or async iterator; we can add later if needed.
- **RoPE/cos_sin on GPU:** Model still creates cos/sin on `model.device`; that’s one-off and small. Moving that to a cached, device-side buffer is a possible micro-optimization later.

---

## How to verify

1. Run `python diffusion.py --train` and `python gpt.py --train` and confirm loss curves and final loss match previous behavior (or improve slightly due to less blocking).
2. On a GPU machine, compare training time per step before/after; you should see a small improvement and more stable step time with async transfer.
3. Confirm no new CUDA errors (e.g. use of freed memory); `non_blocking=True` is safe as long as we don’t reuse the same buffer before the transfer completes (we don’t—each `get_batch` returns new tensors).
