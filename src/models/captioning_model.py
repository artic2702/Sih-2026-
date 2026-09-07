"""
Scene captioning / description model (Task-2 option A, per problem statement).
Owner: Person 1 — same backbone/notebook as VQA (notebooks/01_vqa_geochat.ipynb).

Backbone: BLIP Image Captioning (~900 MB, runs on CPU or any GPU).
Generates natural-language descriptions of satellite/aerial scenes.
Shares the same lightweight footprint as the VQA model so both can
coexist in 8 GB VRAM easily.
"""

import time
import numpy as np

from src.models.base_model import BaseRSModel
from src.common.schemas import RSModelResult, SpatialEvidence, ResultMetadata


class CaptioningModel(BaseRSModel):
    name = "caption_v1"
    task = "captioning"
    backbone = "BLIP-Captioning"

    # Remote-sensing-specific prompt prefixes that guide BLIP to generate
    # satellite-relevant descriptions instead of generic "a photo of..."
    RS_CAPTION_PROMPTS = [
        "a satellite image of",
        "an aerial view showing",
    ]

    def __init__(self, config: dict = None):
        self.config = config
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.checkpoint_path = None

    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        """Load the BLIP Captioning model from HuggingFace (auto-downloads
        ~900 MB on first run, cached afterwards).
        """
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration

        self.device = device if torch.cuda.is_available() and device != "cpu" else "cpu"
        self.checkpoint_path = checkpoint_path

        model_id = "Salesforce/blip-image-captioning-base"
        print(f"[CaptioningModel] Loading {model_id} on {self.device}...")

        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id, use_safetensors=True)

        # Use half precision on GPU to save VRAM
        if self.device != "cpu":
            self.model = self.model.half()
        self.model.to(self.device)
        self.model.eval()

        print(f"[CaptioningModel] Loaded successfully on {self.device}.")

    def _numpy_to_pil(self, image: np.ndarray):
        """Convert a numpy array (C,H,W) or (H,W,C) to PIL RGB image."""
        from PIL import Image

        if image.ndim == 3 and image.shape[0] in (1, 2, 3, 4):
            image = np.transpose(image, (1, 2, 0))

        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.concatenate([image] * 3, axis=-1)

        if image.dtype in (np.float32, np.float64):
            if image.max() <= 1.0:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            else:
                image = image.clip(0, 255).astype(np.uint8)

        return Image.fromarray(image, "RGB")

    def predict(self, image, query: str = None, **kwargs) -> dict:
        """Generate a scene caption for the given image.

        Args:
            image: np.ndarray (C,H,W) or (H,W,C), or PIL Image.
            query: Optional conditioning text. If None, uses an RS-specific
                   prompt prefix to guide the caption generation.

        Returns:
            dict matching RSModelResult schema.
        """
        start = time.time()

        # Auto-load if not loaded yet
        if self.model is None:
            try:
                self.load(self.checkpoint_path or "", device=self.device)
            except Exception as e:
                return self._empty_result(
                    text=f"Captioning model failed to load: {e}",
                    status="error",
                )

        import torch

        from PIL import Image

        # Convert to PIL
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = self._numpy_to_pil(image)
        else:
            pil_image = image

        try:
            # Use RS-specific prompt prefix for better satellite descriptions
            prompt = query if query else self.RS_CAPTION_PROMPTS[0]

            # Conditional captioning (with prompt prefix)
            inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if self.device != "cpu":
                inputs = {k: v.half() if v.dtype == torch.float32 else v
                          for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_length=80)

            caption = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()

            # Also generate an unconditional caption for comparison/confidence
            inputs_uncond = self.processor(images=pil_image, return_tensors="pt")
            inputs_uncond = {k: v.to(self.device) for k, v in inputs_uncond.items()}

            if self.device != "cpu":
                inputs_uncond = {k: v.half() if v.dtype == torch.float32 else v
                                 for k, v in inputs_uncond.items()}

            with torch.no_grad():
                uncond_ids = self.model.generate(**inputs_uncond, max_length=80)

            uncond_caption = self.processor.decode(uncond_ids[0], skip_special_tokens=True).strip()

            # Use the longer, more detailed caption
            final_caption = caption if len(caption) > len(uncond_caption) else uncond_caption

            # Confidence heuristic: longer, more detailed captions are more useful
            confidence = min(0.90, 0.5 + len(final_caption.split()) * 0.03)

            return RSModelResult(
                task="captioning",
                text=final_caption,
                confidence=confidence,
                spatial_evidence=SpatialEvidence(),  # captioning has no bbox/mask
                metadata=ResultMetadata(
                    model=self.name,
                    backbone=self.backbone,
                    checkpoint=self.checkpoint_path or "",
                    dataset="VRSBench",
                    input_modalities=["optical"],
                    parameters={"prompt": prompt},
                ),
                status="success",
                inference_seconds=time.time() - start,
            ).to_dict()

        except Exception as e:
            return RSModelResult(
                task="captioning",
                text=f"Captioning inference error: {e}",
                confidence=0.0,
                spatial_evidence=SpatialEvidence(),
                metadata=ResultMetadata(
                    model=self.name, backbone=self.backbone,
                    checkpoint=self.checkpoint_path or "",
                    input_modalities=["optical"],
                ),
                status="error",
                inference_seconds=time.time() - start,
            ).to_dict()


def train_step(model: CaptioningModel, batch: dict, optimizer, config: dict) -> float:
    """Only if a separate captioning adapter turns out to be needed:
    one training step; called from src/training/train_captioning.py."""
    raise NotImplementedError
