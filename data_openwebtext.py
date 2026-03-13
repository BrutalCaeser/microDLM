#!/usr/bin/env python3
"""
OpenWebText data loader for diffusion.py and gpt.py.
Uses Hugging Face datasets; cache should be at HF_DATASETS_CACHE or
~/microDLM/data/openwebtext_cache (e.g. after running download_openwebtext.py on HPC).
"""
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader


def _default_cache_dir():
    return os.path.expanduser(
        os.environ.get("HF_DATASETS_CACHE", "~/microDLM/data/openwebtext_cache")
    )


def _load_hf_dataset(cache_dir=None):
    cache_dir = cache_dir or _default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = os.path.join(cache_dir, "hf_home")
    from datasets import load_dataset
    return load_dataset("Skylion007/openwebtext", "plain_text", split="train")


def build_vocab(ds, sample_docs=50000):
    """Build character vocab from a sample of the dataset. Returns stoi, itos, vocab_size."""
    MASK_CHAR = "_"
    UNK_CHAR = "<UNK>"
    n = min(sample_docs, len(ds))
    chars = set()
    for i in range(n):
        text = ds[i]["text"]
        if isinstance(text, str):
            chars.update(text)
    chars = sorted(chars)
    if MASK_CHAR in chars:
        chars.remove(MASK_CHAR)
    if UNK_CHAR in chars:
        chars.remove(UNK_CHAR)
    # vocab: MASK, UNK, then all seen chars (so mask_id=0, unk_id=1)
    itos = [MASK_CHAR, UNK_CHAR] + chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos, len(itos)


class OpenWebTextDataset(Dataset):
    """Samples random (doc, start) and returns a block of block_size token ids."""

    def __init__(self, ds, stoi, block_size, indices):
        self.ds = ds
        self.stoi = stoi
        self.unk_id = stoi.get("<UNK>", 1)
        self.block_size = block_size
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def _encode(self, s):
        return [self.stoi.get(c, self.unk_id) for c in s]

    def __getitem__(self, i):
        doc_idx = self.indices[i]
        text = self.ds[doc_idx]["text"]
        if not isinstance(text, str):
            text = str(text) if text is not None else " "
        if len(text) < self.block_size:
            text = text + " " * (self.block_size - len(text))
        start = random.randint(0, len(text) - self.block_size)
        block = text[start : start + self.block_size]
        return torch.tensor(self._encode(block), dtype=torch.long)


def load_openwebtext(
    cache_dir=None,
    block_size=256,
    batch_size=64,
    val_frac=0.01,
    sample_vocab_docs=50000,
    num_workers=0,
    pin_memory=True,
):
    """
    Load OpenWebText from HuggingFace cache, build vocab, create train/val dataloaders.
    Returns: train_loader, val_loader, stoi, itos, vocab_size, mask_token_id, encode, decode
    """
    cache_dir = cache_dir or _default_cache_dir()
    ds = _load_hf_dataset(cache_dir)
    stoi, itos, vocab_size = build_vocab(ds, sample_docs=sample_vocab_docs)
    mask_token_id = stoi["_"]

    encode_fn = lambda s: [stoi.get(c, stoi["<UNK>"]) for c in s]
    decode_fn = lambda ids: "".join([itos[i] if i < len(itos) else "?" for i in ids])

    n = len(ds)
    val_size = max(1, int(n * val_frac))
    train_indices = list(range(n - val_size))
    val_indices = list(range(n - val_size, n))

    train_ds = OpenWebTextDataset(ds, stoi, block_size, train_indices)
    val_ds = OpenWebTextDataset(ds, stoi, block_size, val_indices)

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

    return (
        train_loader,
        val_loader,
        stoi,
        itos,
        vocab_size,
        mask_token_id,
        encode_fn,
        decode_fn,
    )
