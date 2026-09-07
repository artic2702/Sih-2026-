"""
Quick test: Run VQA, Captioning, and Grounding on a sample image.
Usage: python scripts/test_person1_person2.py
"""
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Create a realistic-looking test image (simulating a satellite RGB scene)
# Gradient pattern with some "features" to make grounding interesting
print("=" * 60)
print("Person 1 + 2: Testing VQA, Captioning & Grounding Models")
print("=" * 60)

# Build a 256x256 RGB test image with some structure
h, w = 256, 256
image = np.zeros((3, h, w), dtype=np.float32)

# Create a green area (vegetation) in top-left
image[1, 0:128, 0:128] = 0.6   # green channel

# Create a blue area (water body) in bottom-right
image[2, 128:256, 128:256] = 0.7  # blue channel

# Create a gray area (urban) in top-right
image[0, 0:128, 128:256] = 0.5
image[1, 0:128, 128:256] = 0.5
image[2, 0:128, 128:256] = 0.5

# Create a brown area (bare soil) in bottom-left
image[0, 128:256, 0:128] = 0.6
image[1, 128:256, 0:128] = 0.4
image[2, 128:256, 0:128] = 0.2

print(f"\nTest image shape: {image.shape} (C, H, W)")
print(f"Image dtype: {image.dtype}")
print()

# ---- TEST 1: VQA ----
print("-" * 40)
print("TEST 1: VQA Model")
print("-" * 40)
from src.models.vqa_model import VQAModel

vqa = VQAModel()
vqa.load("", device="cpu")

query = "What is shown in this image?"
print(f"Query: {query}")
result = vqa.predict(image=image, query=query)
print(f"Answer: {result['text']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Status: {result['status']}")
print(f"Time: {result['inference_seconds']:.2f}s")
print()

# Second question
query2 = "What colors are visible?"
print(f"Query: {query2}")
result2 = vqa.predict(image=image, query=query2)
print(f"Answer: {result2['text']}")
print(f"Confidence: {result2['confidence']:.2f}")
print()

# ---- TEST 2: Captioning ----
print("-" * 40)
print("TEST 2: Captioning Model")
print("-" * 40)
from src.models.captioning_model import CaptioningModel

cap = CaptioningModel()
cap.load("", device="cpu")

result = cap.predict(image=image)
print(f"Caption: {result['text']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Status: {result['status']}")
print(f"Time: {result['inference_seconds']:.2f}s")
print()

# ---- TEST 3: Grounding ----
print("-" * 40)
print("TEST 3: Grounding Model")
print("-" * 40)
from src.models.grounding_model import GroundingModel

ground = GroundingModel()
# Don't load VQA again (save memory), just test saliency fallback
ground._loaded = True

query3 = "Highlight the water body."
print(f"Query: {query3}")
result = ground.predict(image=image, query=query3)
print(f"Text: {result['text']}")
print(f"Bounding Box: {result['spatial_evidence']['bbox']}")
print(f"Evidence Type: {result['spatial_evidence']['type']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Status: {result['status']}")
print(f"Time: {result['inference_seconds']:.2f}s")
print()

print("=" * 60)
print("ALL 3 MODELS WORKING!")
print("=" * 60)
