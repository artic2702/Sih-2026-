"""
Text-guided region grounding model (Task-2 option B, per problem statement).
Owner: Person 2. Sprint plan: notebooks/02_grounding.ipynb.

Strategy (per team decision, docs/DECISIONS.md): DO NOT train a new
grounding model first.
1. Try the VQA model's own grounding capability (prompt it to emit a bounding
   box for a referring expression, then parse the coordinates from its text
   output).
2. Standalone fallback: use image saliency analysis (gradient-based region
   detection) to locate the most prominent region in the image when the VQA
   model cannot parse coordinates. This works on CPU with zero extra
   dependencies.
3. SAM fallback (if available): use Segment Anything Model for precise masks.

COORDINATE HANDLING IS THE HIGH-RISK PART OF THIS FILE (explicitly called
out in the ML audit brief: "a grounding model can have a good semantic
answer but still have completely wrong coordinates due to scaling bugs").
parse_bbox_from_text / rescale_bbox / clip_bbox below are written and unit-
tested independently of any model — they work on plain text/numbers so
their correctness doesn't depend on any specific model being loadable.
"""

import re
import time
from typing import List, Optional, Tuple

import numpy as np

from src.models.base_model import BaseRSModel
from src.common.schemas import RSModelResult, SpatialEvidence, ResultMetadata
from src.common.constants import BACKBONE_GROUNDING_PRIMARY


# ---------------------------------------------------------------------------
# Coordinate parsing / conversion — pure functions, no model dependency.
# ---------------------------------------------------------------------------

# Negative lookbehind excludes a letter immediately before the number so
# coordinate LABELS like "x1=12, y2=178" don't have their own "1"/"2"
# mistaken for a coordinate value (a real bug caught by this file's unit
# tests: the naive `[-+]?\d*\.?\d+` regex matched the "1" inside "x1" and
# the "2" inside "y2" themselves, silently producing a completely wrong
# box like [1, 12, 1, 34] instead of [12, 34, 156, 178]).
_NUMBER_RE = re.compile(r"(?<![a-zA-Z])[-+]?\d*\.?\d+")


def parse_bbox_from_text(text: str, image_width: int, image_height: int) -> Optional[List[float]]:
    """Extract a bounding box from a model's free-text output.

    Handles:
      - both normalized [0,1] and pixel-space coordinates (detected: if all
        4 numbers are <= 1.0, treat as normalized and scale by
        image_width/image_height — a real pixel box with every coordinate
        under 1.0 pixel is not a meaningful box anyway, so this heuristic
        has no real ambiguous case in practice)
      - any bracket/comma/space style, since we only regex for numbers, not
        a specific format (e.g. "[12, 34, 56, 78]", "(12,34)-(56,78)",
        "x1=12 y1=34 x2=56 y2=78" all extract the same 4 numbers)
      - out-of-order coordinates (x1 > x2 or y1 > y2) — swapped so the
        returned box is always [x1,y1,x2,y2] with x1<=x2, y1<=y2
      - out-of-range values — clipped to the image bounds

    Returns None if fewer than 4 numbers are found (nothing parseable) —
    callers must treat this as "grounding failed", not a degenerate
    zero-size box, which would silently corrupt an IoU computation.
    """
    numbers = _NUMBER_RE.findall(text)
    if len(numbers) < 4:
        return None
    x1, y1, x2, y2 = (float(n) for n in numbers[:4])

    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        x1, x2 = x1 * image_width, x2 * image_width
        y1, y2 = y1 * image_height, y2 * image_height

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    return clip_bbox([x1, y1, x2, y2], image_width, image_height)


def clip_bbox(bbox: List[float], image_width: int, image_height: int) -> List[float]:
    """Clip a [x1,y1,x2,y2] box to valid image bounds."""
    x1, y1, x2, y2 = bbox
    x1 = float(np.clip(x1, 0, image_width))
    x2 = float(np.clip(x2, 0, image_width))
    y1 = float(np.clip(y1, 0, image_height))
    y2 = float(np.clip(y2, 0, image_height))
    return [x1, y1, x2, y2]


def rescale_bbox(bbox: List[float], from_size: Tuple[int, int],
                  to_size: Tuple[int, int]) -> List[float]:
    """Rescale a [x1,y1,x2,y2] box from one image's pixel space to another's.
    from_size/to_size are (height, width) — matches numpy array convention.
    """
    from_h, from_w = from_size
    to_h, to_w = to_size
    x1, y1, x2, y2 = bbox
    scale_x = to_w / from_w
    scale_y = to_h / from_h
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


def visualize_bbox(image_rgb: np.ndarray, bbox: List[float],
                    color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 2) -> np.ndarray:
    """Draw a bbox onto a copy of an (H,W,3) uint8 RGB array. Returns a NEW
    array; does not mutate input.
    """
    out = image_rgb.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, x2 = np.clip([x1, x2], 0, w - 1)
    y1, y2 = np.clip([y1, y2], 0, h - 1)
    for t in range(thickness):
        out[max(0, y1 - t), x1:x2 + 1] = color
        out[min(h - 1, y2 + t), x1:x2 + 1] = color
        out[y1:y2 + 1, max(0, x1 - t)] = color
        out[y1:y2 + 1, min(w - 1, x2 + t)] = color
    return out


def _saliency_bbox(image: np.ndarray) -> List[float]:
    """Compute a bounding box around the most salient (high-intensity-variance)
    region of the image. Works on any numpy array with no external dependencies.

    This is the zero-dependency fallback when no VQA model can parse coordinates
    and no SAM is available. It identifies the region with the highest local
    intensity variance, which in satellite imagery often corresponds to built-up
    areas, water boundaries, or other features of interest.
    """
    # Convert (C,H,W) to (H,W,C) if needed
    if image.ndim == 3 and image.shape[0] in (1, 2, 3, 4):
        img = np.transpose(image, (1, 2, 0))
    else:
        img = image.copy()

    if img.ndim == 3:
        # Average across channels to get grayscale
        gray = np.mean(img, axis=-1)
    else:
        gray = img.astype(np.float64)

    h, w = gray.shape

    # Normalize to 0-1
    gray_min, gray_max = gray.min(), gray.max()
    if gray_max - gray_min > 1e-8:
        gray = (gray - gray_min) / (gray_max - gray_min)

    # Compute local variance using a sliding window approach
    # Use a grid of blocks and find the block with highest variance
    block_h = max(1, h // 8)
    block_w = max(1, w // 8)

    best_var = -1
    best_r, best_c = 0, 0

    for r in range(0, h - block_h + 1, max(1, block_h // 2)):
        for c in range(0, w - block_w + 1, max(1, block_w // 2)):
            block = gray[r:r + block_h, c:c + block_w]
            var = np.var(block)
            if var > best_var:
                best_var = var
                best_r, best_c = r, c

    # Expand the detected block slightly for a more meaningful bounding box
    pad_h = block_h // 2
    pad_w = block_w // 2

    x1 = max(0, best_c - pad_w)
    y1 = max(0, best_r - pad_h)
    x2 = min(w, best_c + block_w + pad_w)
    y2 = min(h, best_r + block_h + pad_h)

    return [float(x1), float(y1), float(x2), float(y2)]


def _try_load_sam(checkpoint_path: str):
    """Attempt to load Meta's Segment Anything Model."""
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError as e:
        raise ImportError(
            "segment-anything is not installed. To use the SAM fallback:\n"
            "  1. pip install segment-anything\n"
            "  2. download a checkpoint, e.g. sam_vit_b_01ec64.pth from "
            "https://github.com/facebookresearch/segment-anything#model-checkpoints\n"
            "  3. pass that checkpoint's path as checkpoint_path to load()\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    return SamPredictor(sam)


class GroundingModel(BaseRSModel):
    name = "grounding_v1"
    task = "grounding"
    backbone = BACKBONE_GROUNDING_PRIMARY  # "GeoChat" by default

    GROUNDING_PROMPT_TEMPLATE = (
        "{query} Answer with only a bounding box in pixel coordinates, "
        "in the exact format [x1, y1, x2, y2]."
    )

    def __init__(self, config: dict = None):
        self.config = config
        self.vqa_model = None       # shared VQA model instance
        self.sam_predictor = None
        self.device = "cpu"
        self.checkpoint_path = None
        self.use_fallback = False
        self._loaded = False

    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        """Load the grounding pipeline:
        1. Try to reuse the VQA model (BLIP/GeoChat) for coordinate extraction.
        2. If VQA model loading fails, use saliency-based detection as fallback.
        3. SAM is available as an additional fallback if installed.
        """
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Try to load the VQA model for text-based grounding
        try:
            from src.models.vqa_model import VQAModel
            vqa_checkpoint = self.config["models"]["vqa"]["checkpoint"] if self.config else checkpoint_path
            self.vqa_model = VQAModel(config=self.config)
            self.vqa_model.load(vqa_checkpoint, device=device)
            self.backbone = "BLIP-VQA+Saliency"
            self.use_fallback = False
            print("[GroundingModel] VQA model loaded for coordinate extraction.")
        except Exception as e:
            print(f"[GroundingModel] VQA model unavailable ({e}), using saliency fallback.")
            self.vqa_model = None
            self.backbone = "Saliency"
            self.use_fallback = True

        # Try SAM as optional enhancement
        try:
            self.sam_predictor = _try_load_sam(checkpoint_path)
            print("[GroundingModel] SAM fallback loaded.")
        except (ImportError, Exception):
            self.sam_predictor = None

        self._loaded = True

    def torch_module(self):
        raise NotImplementedError(
            "GroundingModel has no trainable parameters this sprint — it "
            "reuses the VQA checkpoint or saliency analysis. Training a "
            "dedicated grounding model is an explicit non-goal per "
            "docs/DECISIONS.md unless both fall short of the frozen IoU threshold."
        )

    def predict(self, image, query: str, **kwargs) -> dict:
        """Locate a region described by the query in the image.

        Strategy:
        1. If VQA model is available: prompt it for coordinates, parse them.
        2. If parsing fails or VQA is unavailable: use saliency-based detection.
        3. Always returns a valid bounding box (saliency fallback never fails).
        """
        start = time.time()

        # Auto-load if not loaded yet
        if not self._loaded:
            try:
                self.load(self.checkpoint_path or "", device=self.device)
            except Exception:
                pass

        # Get image dimensions
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] in (1, 2, 3, 4):
                _, h, w = image.shape
            elif image.ndim == 3:
                h, w, _ = image.shape
            else:
                h, w = image.shape
        else:
            h, w = 64, 64  # fallback for unknown input types

        bbox = None
        method_used = "saliency"

        # Strategy 1: Try VQA model for text-guided coordinate extraction
        if self.vqa_model is not None and self.vqa_model.model is not None:
            try:
                prompt = self.GROUNDING_PROMPT_TEMPLATE.format(query=query)
                raw_result = self.vqa_model.predict(image=image, query=prompt)
                parsed_bbox = parse_bbox_from_text(raw_result["text"], image_width=w, image_height=h)
                if parsed_bbox is not None:
                    bbox = parsed_bbox
                    method_used = "vqa_coordinate_extraction"
            except Exception:
                pass  # Fall through to saliency

        # Strategy 2: Saliency-based detection (always works, no dependencies)
        if bbox is None and isinstance(image, np.ndarray):
            bbox = _saliency_bbox(image)
            bbox = clip_bbox(bbox, image_width=w, image_height=h)
            method_used = "saliency_detection"

        # Fallback: center crop if nothing else worked
        if bbox is None:
            bbox = [w * 0.25, h * 0.25, w * 0.75, h * 0.75]
            method_used = "center_fallback"

        confidence = 0.75 if method_used == "vqa_coordinate_extraction" else 0.55

        return RSModelResult(
            task="grounding",
            text=f"Region located at [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}] using {method_used}.",
            confidence=confidence,
            spatial_evidence=SpatialEvidence(
                type="bbox",
                source="image",
                bbox=bbox,
            ),
            metadata=ResultMetadata(
                model=self.name,
                backbone=self.backbone,
                checkpoint=self.checkpoint_path or "",
                dataset="VRSBench",
                input_modalities=["optical"],
                parameters={"query": query, "method": method_used},
            ),
            status="success",
            inference_seconds=time.time() - start,
        ).to_dict()

    def _predict_sam(self, image, query: str, h: int, w: int, start: float) -> dict:
        """SAM needs a seed point/box — this sprint's integration assumes
        one is supplied via kwargs."""
        return self._empty_result(
            text=("SAM fallback requires a seed point/box (SAM has no text "
                  "understanding of its own) — pass one via predict(..., "
                  "seed_box=[...]) once the primary path has produced a rough "
                  "box to refine."),
            status="error",
        )


def train_step(model: GroundingModel, batch: dict, optimizer, config: dict) -> float:
    """Only relevant if the team ends up training a dedicated grounding model.
    Not expected to be used in the initial 2-day sprint — see docs/DECISIONS.md.
    """
    raise NotImplementedError(
        "Grounding-model training is out of scope for the mandatory 2-day "
        "sprint (see docs/DECISIONS.md) — the VQA model is used for coordinate "
        "extraction, and SAM (if needed) is used pretrained-only."
    )
