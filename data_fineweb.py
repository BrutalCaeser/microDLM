#!/usr/bin/env python3
"""
FineWeb-Edu data loader for microDLM.
Uses streaming from HuggingFace or preprocessed binary shards.
Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

FineWeb-Edu is a high-quality subset of FineWeb, filtered for educational content.
License: CC-BY-4.0

Usage:
    from data_fineweb import load_fineweb_edu
    train_loader, val_loader, ... = load_fineweb_edu(block_size=256, batch_size=64)
"""
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader


def _default_cache_dir():
    """Default cache directory for FineWeb-Edu data."""
    return os.path.expanduser("~/microDLM/data/fineweb")


def _build_char_vocab(ds, sample_docs=5000, mask_char="_", unk_char="<UNK>"):
    """
    Build character-level vocabulary from a sample of documents.

    Args:
        ds: Dataset or iterable of examples
        sample_docs: Number of documents to sample for vocab building
        mask_char: Special character for masking (diffusion)
        unk_char: Unknown token character

    Returns:
        stoi, itos, vocab_size, mask_token_id, unk_id
    """
    chars = set()
    n_sampled = 0

    # Sample documents for vocabulary building
    for example in ds:
        text = example.get("text", "")
        if isinstance(text, str):
            chars.update(text)
        n_sampled += 1
        if n_sampled >= sample_docs:
            break

    # Remove special characters if they exist in text
    if mask_char in chars:
        chars.remove(mask_char)
    if unk_char in chars:
        chars.remove(unk_char)

    # Build vocabulary: [MASK, UNK, ...chars]
    itos = [mask_char, unk_char] + sorted(chars)
    stoi = {ch: i for i, ch in enumerate(itos)}

    return stoi, itos, len(itos), stoi[mask_char], stoi[unk_char]


class FineWebDataset(Dataset):
    """
    FineWeb-Edu dataset wrapper for character-level tokenization.

    Supports both streaming (network) and cached (local) modes.
    Randomly samples blocks from documents for training.
    """

    def __init__(self, ds, stoi, block_size, unk_id, max_samples=100000):
        self.ds = ds
        self.stoi = stoi
        self.block_size = block_size
        self.unk_id = unk_id
        self.max_samples = max_samples
        self._doc_cache = []
        self._doc_index = 0

    def __len__(self):
        return self.max_samples

    def _encode(self, text):
        """Encode string to token IDs."""
        return [self.stoi.get(c, self.unk_id) for c in text]

    def __getitem__(self, idx):
        """
        Get a random block from the dataset.

        For streaming: iterates through documents until finding one with sufficient length.
        For cached: can use pre-indexed documents.
        """
        # Try to get a valid block from streaming
        attempts = 0
        max_attempts = 100  # Prevent infinite loops

        for example in self.ds:
            if attempts >= max_attempts:
                break
            attempts += 1

            text = example.get("text", "")
            if not isinstance(text, str) or len(text) < self.block_size:
                continue

            # Random crop within document
            if len(text) > self.block_size:
                start = random.randint(0, len(text) - self.block_size)
                block = text[start: start + self.block_size]
            else:
                block = text

            return torch.tensor(self._encode(block), dtype=torch.long)

        # Fallback: return padding
        return torch.full((self.block_size,), self.unk_id, dtype=torch.long)


def load_fineweb_edu(
    block_size=256,
    batch_size=64,
    val_frac=0.01,
    num_workers=0,
    pin_memory=True,
    use_streaming=True,
    sample_vocab_docs=5000,
    max_samples=100000,
    cache_dir=None,
):
    """
    Load FineWeb-Edu dataset for training microDLM.

    Args:
        block_size: Context window size (tokens per sample)
        batch_size: Batch size for training
        val_frac: Fraction of data for validation
        num_workers: Number of DataLoader workers
        pin_memory: Pin memory for faster GPU transfer
        use_streaming: If True, stream from HuggingFace (no download required)
        sample_vocab_docs: Number of documents to sample for vocab building
        max_samples: Maximum number of samples to generate
        cache_dir: Directory for caching (default: ~/microDLM/data/fineweb)

    Returns:
        train_loader, val_loader, stoi, itos, vocab_size, mask_token_id, encode_fn, decode_fn

    Example:
        >>> train_loader, val_loader, stoi, itos, vocab_size, mask_id, enc, dec = load_fineweb_edu()
        >>> batch = next(iter(train_loader))  # (batch_size, block_size)
        >>> print(batch.shape)  # torch.Size([64, 256])
    """
    from datasets import load_dataset

    cache_dir = cache_dir or _default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)

    # Load dataset
    if use_streaming:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
            cache_dir=cache_dir
        )
    else:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            cache_dir=cache_dir
        )

    # Build vocabulary
    stoi, itos, vocab_size, mask_id, unk_id = _build_char_vocab(ds, sample_vocab_docs)

    print(f"FineWeb-Edu vocab size: {vocab_size:,}")
    print(f"Cache directory: {cache_dir}")

    # Create dataset and split
    full_dataset = FineWebDataset(ds, stoi, block_size, unk_id, max_samples)

    # Split into train/val
    n_val = max(1, int(len(full_dataset) * val_frac))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator()
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    encode_fn = lambda s: [stoi.get(c, unk_id) for c in s]
    decode_fn = lambda ids: "".join([itos[i] if i < len(itos) else "?" for i in ids])

    return train_loader, val_loader, stoi, itos, vocab_size, mask_id, encode_fn, decode_fn


def test_loader():
    """Test the FineWeb-Edu loader."""
    print("=" * 60)
    print("Testing FineWeb-Edu Data Loader")
    print("=" * 60)

    train_loader, val_loader, stoi, itos, vocab_size, mask_id, enc, dec = load_fineweb_edu(
        block_size=256,
        batch_size=4,  # Small batch for testing
        use_streaming=True,
        sample_vocab_docs=1000,
        max_samples=100,  # Small sample for testing
    )

    print(f"\nVocab size: {vocab_size:,}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test batch shape
    batch = next(iter(train_loader))
    print(f"\nBatch shape: {batch.shape}")
    assert batch.shape == (4, 256), f"Expected (4, 256), got {batch.shape}"

    # Test decoding
    sample_ids = batch[0].tolist()
    decoded = dec(sample_ids)
    print(f"Decoded sample (first 100 chars): {repr(decoded[:100])}")

    print("\n✓ FineWeb-Edu loader test passed!")
    return True


if __name__ == "__main__":
    import sys

    try:
        test_loader()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
