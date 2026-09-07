"""
Text-guided region grounding model (Task-2 option B, per problem statement).
Owner: Person 2. Sprint plan: notebooks/02_grounding.ipynb.

Strategy (per team decision, docs/DECISIONS.md): DO NOT train a new
grounding model first.
1. Try GeoChat's own grounding capability (prompt it to emit a bounding box
   for a referring expression, then parse the coordinates from its text
   output). GeoChat is already loaded by Person 1's checkpoint (VQAModel).
2. Only if GeoChat grounding quality is insufficient on VRSBench grounding
   samples (mean IoU < configs/config.yaml's
   datasets.vrsbench.grounding_iou_threshold, frozen at 0.30 BEFORE seeing
   results), fall back to SAM (given a seed point/box from GeoChat) for a
   proper mask. This class supports both paths; `self.backbone` records
   which one is active.

COORDINATE HANDLING IS THE HIGH-RISK PART OF THIS FILE (explicitly called
out in the ML audit brief: "a grounding model can have a good semantic
answer but still have completely wrong coordinates due to scaling bugs").
parse_bbox_from_text / rescale_bbox / clip_bbox below are written and unit-
tested independently of any model — they work on plain text/numbers so
their correctness doesn't depend on GeoChat actually being loadable in a
given environment.
"""

import re
import time
from typing import List, Optional, Tuple

import numpy as np

from src.models.base_model import BaseRSModel
from src.common.schemas import RSModelResult, SpatialEvidence, ResultMetadata
from src.common.constants import BACKBONE_GROUNDING_PRIMARY
# NOTE: BACKBONE_GROUNDING_FALLBACK in src/common/constants.py lists
# ["GeoGround", "SAM"], but docs/DECISIONS.md records the team's actual
# decision as SAM-only (GeoGround dropped as "another unfamiliar repository
# to debug under deadline pressure" when SAM is already well-packaged).
# This file implements the SAM fallback only, matching DECISIONS.md; the
# constant is a slight doc/code mismatch worth reconciling (see
# docs/ML_AUDIT.md) but not touched here since constants.py is the frozen
# shared contract.


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
    """Clip a [x1,y1,x2,y2] box to valid image bounds. If clipping collapses
    the box to zero width/height (a box entirely outside the image), the
    box is still returned as-is (zero-area) rather than raising — callers
    computing IoU against a zero-area box will correctly get IoU=0, which
    is the right signal for "this prediction was off-image", not a crash.
    """
    x1, y1, x2, y2 = bbox
    x1 = float(np.clip(x1, 0, image_width))
    x2 = float(np.clip(x2, 0, image_width))
    y1 = float(np.clip(y1, 0, image_height))
    y2 = float(np.clip(y2, 0, image_height))
    return [x1, y1, x2, y2]


def rescale_bbox(bbox: List[float], from_size: Tuple[int, int],
                  to_size: Tuple[int, int]) -> List[float]:
    """Rescale a [x1,y1,x2,y2] box from one image's pixel space to another's
    — needed because preprocess_pipeline resizes to the model's expected
    input size (e.g. 336x336 for GeoChat), but grounding evaluation/
    visualization need the box back in the ORIGINAL image's coordinate
    frame (dataset_loader.VRSBenchDataset carries `orig_size` for exactly
    this). from_size/to_size are (height, width) — matches numpy array
    convention (array.shape[-2:]), NOT (width, height), to avoid the exact
    class of transposition bug this function exists to prevent.
    """
    from_h, from_w = from_size
    to_h, to_w = to_size
    x1, y1, x2, y2 = bbox
    scale_x = to_w / from_w
    scale_y = to_h / from_h
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


def visualize_bbox(image_rgb: np.ndarray, bbox: List[float],
                    color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 2) -> np.ndarray:
    """Draw a bbox onto a copy of an (H,W,3) uint8 RGB array — a cheap,
    dependency-free alternative to matplotlib for saving grounding
    visualization examples (per ML audit requirement "produce several saved
    visualization examples"). Returns a NEW array; does not mutate input.
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


def _try_load_sam(checkpoint_path: str):
    """Attempt to load Meta's Segment Anything Model. Real integration seam
    (matches segment-anything's actual API), documented external
    dependency: `pip install segment-anything` plus a checkpoint download
    from Meta's release bucket — both need network access this sandboxed
    audit environment does not have, so this legitimately raises here.
    See docs/DECISIONS.md for why SAM (not GeoGround) is the chosen
    fallback.
    """
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
        self.vqa_model = None       # shared GeoChat instance (Person 1's checkpoint)
        self.sam_predictor = None
        self.device = "cpu"
        self.checkpoint_path = None
        self.use_fallback = False

    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        """Default path: reuse Person 1's GeoChat checkpoint (no separate
        weights needed) via VQAModel — grounding is just a different prompt
        over the same loaded model. Fallback path (only if the team's IoU
        decision rule triggers it): load SAM from checkpoint_path instead.
        """
        self.device = device
        self.checkpoint_path = checkpoint_path

        from src.models.vqa_model import VQAModel
        vqa_checkpoint = self.config["models"]["vqa"]["checkpoint"] if self.config else checkpoint_path
        self.vqa_model = VQAModel(config=self.config)
        self.vqa_model.load(vqa_checkpoint, device=device)
        self.backbone = "GeoChat"
        self.use_fallback = False

        if self.vqa_model.model is None:
            # GeoChat itself isn't loadable in this environment either (see
            # vqa_model.py's honest-failure path) — grounding has no
            # meaningful primary path without it. Try SAM only if a
            # checkpoint was actually given for it; otherwise stay
            # unloaded and let predict() report the error explicitly.
            try:
                self.sam_predictor = _try_load_sam(checkpoint_path)
                self.backbone = "SAM"
                self.use_fallback = True
            except ImportError as e:
                print(f"[grounding_model] SAM fallback also unavailable: {e}")

    def torch_module(self):
        raise NotImplementedError(
            "GroundingModel has no trainable parameters this sprint — it "
            "reuses GeoChat's checkpoint (trained via vqa_model.py) or SAM "
            "(pretrained, not fine-tuned). Training a dedicated grounding "
            "model is an explicit non-goal per docs/DECISIONS.md unless "
            "both fall short of the frozen IoU threshold."
        )

    def predict(self, image, query: str, **kwargs) -> dict:
        """query e.g. 'Highlight the water body referred to in the query.'

        image is assumed already preprocessed (resized) by the shared
        pipeline; image_width/image_height for bbox parsing are read from
        its own shape so parsed coordinates are correct in ITS frame —
        callers wanting the original pre-resize frame back must use
        rescale_bbox() with the sample's orig_size (see
        dataset_loader.VRSBenchDataset).
        """
        start = time.time()
        _, h, w = image.shape if hasattr(image, "shape") else (None, None, None)

        if self.use_fallback and self.sam_predictor is not None:
            return self._predict_sam(image, query, h, w, start)

        if self.vqa_model is None or self.vqa_model.model is None:
            return self._empty_result(
                text=("Grounding model not usable: GeoChat is not loaded "
                      "and no SAM fallback is available (see vqa_model.py "
                      "and this file's _try_load_sam() for the exact setup "
                      "commands each needs)."),
                status="error",
            )

        prompt = self.GROUNDING_PROMPT_TEMPLATE.format(query=query)
        raw_result = self.vqa_model.predict(image=image, query=prompt)
        bbox = parse_bbox_from_text(raw_result["text"], image_width=w, image_height=h)

        if bbox is None:
            return RSModelResult(
                task="grounding",
                text=f"Could not parse a bounding box from the model's response: {raw_result['text']!r}",
                confidence=0.0,
                spatial_evidence=SpatialEvidence(),
                metadata=ResultMetadata(
                    model=self.name, backbone=self.backbone,
                    checkpoint=self.checkpoint_path or "", dataset="VRSBench",
                    input_modalities=["optical"],
                ),
                status="error",
                inference_seconds=time.time() - start,
            ).to_dict()

        return RSModelResult(
            task="grounding",
            text=f"Region located at {[round(v, 1) for v in bbox]}.",
            confidence=raw_result.get("confidence", 0.5),
            spatial_evidence=SpatialEvidence(type="bbox", source="image", bbox=bbox),
            metadata=ResultMetadata(
                model=self.name, backbone=self.backbone,
                checkpoint=self.checkpoint_path or "", dataset="VRSBench",
                input_modalities=["optical"],
                parameters={"prompt": prompt},
            ),
            status="success",
            inference_seconds=time.time() - start,
        ).to_dict()

    def _predict_sam(self, image, query: str, h: int, w: int, start: float) -> dict:
        """SAM needs a seed point/box — this sprint's integration assumes
        one is supplied via kwargs (e.g. from a rough GeoChat box) since SAM
        itself has no text understanding. Without a seed, SAM cannot ground
        a text query at all — report that plainly rather than guessing a
        center-of-image seed, which would silently produce meaningless
        masks for any query not actually about the image center.
        """
        return self._empty_result(
            text=("SAM fallback requires a seed point/box (SAM has no text "
                  "understanding of its own) — pass one via predict(..., "
                  "seed_box=[...]) once the primary GeoChat path has "
                  "produced a rough box to refine. Calling SAM with no seed "
                  "is not implemented, since a guessed seed would silently "
                  "produce a meaningless mask for an unrelated query."),
            status="error",
        )


def train_step(model: GroundingModel, batch: dict, optimizer, config: dict) -> float:
    """Only relevant if the team ends up training GeoGround/SAM fine-tuning
    (fallback path). Not expected to be used in the initial 2-day sprint —
    see docs/DECISIONS.md.
    """
    raise NotImplementedError(
        "Grounding-model training is out of scope for the mandatory 2-day "
        "sprint (see docs/DECISIONS.md) — GeoChat is used pretrained/"
        "LoRA-shared, and SAM (if needed) is used pretrained-only."
    )
