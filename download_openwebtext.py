#!/usr/bin/env python3
"""
Download OpenWebText dataset on HPC. Uses Hugging Face datasets; cache is stored
in ~/microDLM/data/openwebtext_cache (or set HF_DATASETS_CACHE).
Do not run on login node - use Slurm job download_dataset_job.sh
"""
import os
import sys

# Keep cache under project dir so it's in one place
cache_dir = os.path.expanduser("~/microDLM/data/openwebtext_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HF_HOME"] = os.path.join(cache_dir, "hf_home")

print("Cache directory:", cache_dir)
print("Loading dataset Skylion007/openwebtext (this may take a while, ~24GB)...")
sys.stdout.flush()

from datasets import load_dataset

ds = load_dataset("Skylion007/openwebtext", "plain_text")
print("Download complete.")
print("Dataset:", ds)
print("Train split size:", len(ds["train"]))
if len(ds["train"]) > 0:
    print("Sample keys:", ds["train"].column_names)
    print("First row (text length):", len(ds["train"][0]["text"]))
print("Cache location:", cache_dir)
