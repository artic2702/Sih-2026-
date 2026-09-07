"""
scripts/test_vrsbench.py

Evaluate SatQuery AI's Specialist Models (VQA, Captioning, Grounding)
on the official public VRSBench benchmark (Hugging Face: xiang709/VRSBench).
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageDraw

from src.models.vqa_model import VQAModel
from src.models.captioning_model import CaptioningModel
from src.models.grounding_model import GroundingModel


def run_vrsbench_eval():
    image_path = PROJECT_ROOT / "data" / "P0003_0002.png"
    if not image_path.exists():
        print(f"Image not found at {image_path}. Extracting from VRSBench...")
        import fsspec
        import io
        fs, path = fsspec.core.url_to_fs(
            "zip://Images_val/P0003_0002.png::https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_val.zip"
        )
        with fs.open(path) as f:
            data = f.read()
            img = Image.open(io.BytesIO(data))
            image_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(image_path)
            print("Successfully cached P0003_0002.png.")

    print("=" * 65)
    print("VRSBench Remote Sensing Benchmark Evaluation (P0003_0002.png)")
    print("=" * 65)

    pil_img = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_img)
    w, h = pil_img.size
    print(f"Image Dimensions: {w}x{h}, Mode: {pil_img.mode}\n")

    # 1. Captioning Evaluation
    print("-" * 50)
    print("TASK 1: Remote Sensing Captioning")
    print("-" * 50)
    cap = CaptioningModel()
    cap.load("", device="cpu")
    cap_res = cap.predict(image=img_np)
    print(f"Model Caption:  \"{cap_res['text']}\"")
    print(f"Confidence:     {cap_res['confidence']:.2f}")
    print("VRSBench Truth: \"The aerial image features a group of large yellow buses lined up on the left side, two small vehicles...\"")
    print()

    # 2. VQA Evaluation
    print("-" * 50)
    print("TASK 2: Visual Question Answering (VQA)")
    print("-" * 50)
    vqa = VQAModel()
    vqa.load("", device="cpu")

    questions = [
        ("What color are the large vehicles seen in the image?", "Yellow"),
        ("Is there a vehicle located at the top-most position in the provided image?", "Yes"),
        ("How many small vehicles are visible in the image?", "2"),
    ]

    for q, expected in questions:
        res = vqa.predict(image=img_np, query=q)
        print(f"Q:         {q}")
        print(f"Predicted: {res['text']} (Confidence: {res['confidence']:.2f})")
        print(f"Expected:  {expected}")
        print()

    # 3. Grounding Evaluation
    print("-" * 50)
    print("TASK 3: Text-Guided Visual Grounding")
    print("-" * 50)
    ground = GroundingModel()
    ground._loaded = True

    query = "The large yellow vehicle situated closest to the green area."
    res_g = ground.predict(image=img_np, query=query)
    bbox = res_g["spatial_evidence"]["bbox"]  # [ymin, xmin, ymax, xmax]
    print(f"Query:     \"{query}\"")
    print(f"Predicted BBox: {bbox}")
    print("VRSBench Truth: Normalized coordinates {<25><40><33><60>}")

    # Draw detected bbox and save
    boxed = pil_img.copy()
    draw = ImageDraw.Draw(boxed)
    ymin, xmin, ymax, xmax = bbox
    draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0), width=4)

    out_path = PROJECT_ROOT / "data" / "P0003_0002_grounded.png"
    boxed.save(out_path)
    print(f"\nVisual Grounding output saved to: {out_path}")
    print("=" * 65)
    print("VRSBENCH EVALUATION COMPLETE — ALL 3 TASKS EXECUTED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    run_vrsbench_eval()
