"""
Download and cache Qwen/Qwen2-VL-2B-Instruct weights with resumable progress.
"""
import sys
import os
from huggingface_hub import snapshot_download

print("Starting resumable download for Qwen/Qwen2-VL-2B-Instruct...")
model_id = "Qwen/Qwen2-VL-2B-Instruct"

try:
    path = snapshot_download(
        repo_id=model_id,
        resume_download=True,
    )
    print(f"\n[SUCCESS] Model fully downloaded and cached at: {path}")
except Exception as e:
    print(f"\n[ERROR] Download error: {e}")
    sys.exit(1)
