"""
scripts/interactive_test.py

Interactive CLI Tester for SatQuery AI.
Allows testing any question on any satellite image directly from the terminal.

Usage:
    python scripts/interactive_test.py
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from PIL import Image, ImageDraw

from src.agent.controller import AgentController


SAMPLE_IMAGES = {
    "1": ("data/P0003_0002.png", "Airport Scene with Yellow Buses (VRSBench)"),
    "2": ("data/sample_bigearthnet/forest_s2_rgb.png", "BigEarthNet Forest Canopy"),
    "3": ("data/sample_bigearthnet/water_coastal_s2_rgb.png", "BigEarthNet Coastal Waters"),
    "4": ("data/sample_bigearthnet/urban_s2_rgb.png", "BigEarthNet Urban Fabric"),
}


def draw_box(image_path: str, bbox: list, out_path: str = "data/interactive_result.png"):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline="#00FF66", width=4)
    draw.rectangle([x1, y1, x1 + 120, y1 + 22], fill="#00FF66")
    draw.text((x1 + 6, y1 + 4), "Target", fill="#000000")
    img.save(out_path)
    return out_path


def main():
    print("=" * 65)
    print("      🛰️  SATQUERY AI — INTERACTIVE IMAGE TESTER")
    print("=" * 65)

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SatQuery Agent Controller on: {device.upper()}...")
    controller = AgentController(config=config, device=device)
    print("Ready!\n")

    print("Choose an image to test:")
    for k, (path, desc) in SAMPLE_IMAGES.items():
        print(f"  [{k}] {desc} ({path})")
    print("  [C] Custom image path")

    choice = input("\nEnter choice (1-4 or custom path, default 1): ").strip() or "1"
    if choice in SAMPLE_IMAGES:
        img_path = SAMPLE_IMAGES[choice][0]
    elif os.path.exists(choice):
        img_path = choice
    else:
        print(f"File not found: {choice}. Defaulting to sample 1.")
        img_path = SAMPLE_IMAGES["1"][0]

    print(f"\nSelected Image: {img_path}")

    while True:
        print("\n" + "-" * 60)
        query = input("Ask a question or enter an instruction (or 'q' to quit): ").strip()
        if not query or query.lower() == "q":
            print("Exiting interactive tester. Goodbye!")
            break

        print("\nAnalyzing...")
        result = controller.run(images={"image": img_path}, query=query)

        print("\n" + "=" * 50)
        print(f"  Routed Task  : {result.get('task')}")
        print(f"  Answer       : {result.get('text')}")
        print(f"  Confidence   : {result.get('confidence'):.2f}")
        print(f"  Status       : {result.get('status')}")

        evidence = result.get("spatial_evidence") or {}
        if evidence.get("type") == "bbox" and evidence.get("bbox"):
            bbox = evidence["bbox"]
            print(f"  Bounding Box : {bbox}")
            saved_img = draw_box(img_path, bbox)
            print(f"  Evidence Image Saved: {saved_img}")
        print("=" * 50)


if __name__ == "__main__":
    main()
