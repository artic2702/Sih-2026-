"""
scripts/download_vrsbench.py

Download and manage the official VRSBench (Visual Remote Sensing Benchmark) dataset
from Hugging Face (xiang709/VRSBench).

Features:
- Download official evaluation annotations: VQA (37k questions), Captioning, and Grounding.
- Download curated high-resolution sample scenes or bulk download image batches.
- Option to download full training/validation archives.

Usage:
    # Download annotations + 10 curated high-resolution test scenes (fast, < 30 MB)
    python scripts/download_vrsbench.py --samples 10

    # Download full validation annotations
    python scripts/download_vrsbench.py --annotations-only

    # Download the full evaluation zip (3.79 GB, ~9,350 high-res images)
    python scripts/download_vrsbench.py --download-val-zip
"""

import argparse
import io
import json
import os
import sys
import urllib.request
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "vrsbench_samples"

HF_BASE_URL = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main"

ANNOTATION_FILES = {
    "VRSBench_EVAL_vqa.json": f"{HF_BASE_URL}/VRSBench_EVAL_vqa.json",
    "VRSBench_EVAL_Cap.json": f"{HF_BASE_URL}/VRSBench_EVAL_Cap.json",
    "VRSBench_EVAL_referring.json": f"{HF_BASE_URL}/VRSBench_EVAL_referring.json",
}

CURATED_SCENES = {
    "P0003_0002.png": "Airport apron with yellow transport buses",
    "P0161_0007.png": "Airport tarmac with airplane and taxiway intersections",
    "P0019_0003.png": "Coastal harbor with ships docked in water",
    "P1179_0094.png": "Industrial plant with circular storage tanks",
    "P1410_0067.png": "Sports arena and stadium with adjacent parking lot",
    "P0837_0000.png": "Dense urban residential neighborhood and road network",
    "P1023_0001.png": "Highway interchange and traffic overpass",
    "P1647_0013.png": "Container port and shipping freight terminal",
    "P0998_0016.png": "Solar farm array and agricultural farmland",
    "P1022_0015.png": "Railway junction and train depot",
}


def download_annotations(dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    print("\n[1/2] Checking official VRSBench annotation files...")
    for filename, url in ANNOTATION_FILES.items():
        filepath = dest_dir / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  -> Saved {filename} ({size_mb:.2f} MB)")
        else:
            print(f"  -> {filename} already present.")


def download_samples(dest_dir: Path, count: int = 5):
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[2/2] Fetching {count} curated high-resolution benchmark images from VRSBench...")
    import fsspec

    val_zip_url = f"{HF_BASE_URL}/Images_val.zip"
    items = list(CURATED_SCENES.items())[:count]

    for img_id, description in items:
        target_path = dest_dir / img_id
        if target_path.exists():
            print(f"  -> {img_id} already exists ({description})")
            continue

        print(f"  Fetching {img_id} ({description})...")
        try:
            url = f"zip://Images_val/{img_id}::{val_zip_url}"
            fs, path = fsspec.core.url_to_fs(url)
            with fs.open(path) as f:
                img = Image.open(io.BytesIO(f.read()))
                img.save(target_path)
                print(f"     Saved {img_id} ({img.size[0]}x{img.size[1]} RGB)")
        except Exception as e:
            print(f"     Warning: could not fetch {img_id}: {e}")

    print("\nSamples ready in:", dest_dir)


def download_full_archive(dest_dir: Path, split: str = "val"):
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = f"Images_{split}.zip"
    url = f"{HF_BASE_URL}/{filename}"
    target = dest_dir / filename
    print(f"\nDownloading full archive {filename} from Hugging Face...")
    print("WARNING: This is a large file (~3.8 GB for val, ~8.0 GB for train).")
    urllib.request.urlretrieve(url, target)
    print(f"Saved archive to {target}")


def main():
    parser = argparse.ArgumentParser(description="VRSBench remote sensing dataset downloader.")
    parser.add_argument("--samples", type=int, default=5, help="Number of curated high-res scenes to download (default: 5)")
    parser.add_argument("--annotations-only", action="store_true", help="Download only annotation JSON files")
    parser.add_argument("--download-val-zip", action="store_true", help="Download full Images_val.zip (~3.8 GB)")
    parser.add_argument("--download-train-zip", action="store_true", help="Download full Images_train.zip (~8.0 GB)")

    args = parser.parse_args()

    download_annotations(DATA_DIR)

    if args.download_val_zip:
        download_full_archive(DATA_DIR, split="val")
    elif args.download_train_zip:
        download_full_archive(DATA_DIR, split="train")
    elif not args.annotations_only:
        download_samples(DATA_DIR, count=args.samples)

    print("\nDataset setup complete!")
    print("Open the web app (http://localhost:8501) to test them immediately!")


if __name__ == "__main__":
    main()
