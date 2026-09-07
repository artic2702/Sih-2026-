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
    backbone = "Qwen2-VL-2B"

    RS_CAPTION_PROMPTS = [
        "Describe this aerial or satellite remote-sensing image in detail, noting visible land cover, buildings, vehicles, or natural terrain.",
        "an aerial view showing",
    ]

    def __init__(self, config: dict = None, vqa_model = None):
        self.config = config or {}
        self.vqa_model = vqa_model
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.checkpoint_path = None
        self.is_qwen = False

    def load(self, checkpoint_path: str = "", device: str = "cpu") -> None:
        """Load Qwen2-VL-2B (or fallback to BLIP-Captioning)."""
        import torch

        self.device = device if torch.cuda.is_available() and device != "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path

        # If a pre-loaded VQA model is available, reuse its loaded weights directly
        if self.vqa_model is not None and self.vqa_model.model is not None:
            self.model = self.vqa_model.model
            self.processor = self.vqa_model.processor
            self.device = self.vqa_model.device
            self.is_qwen = getattr(self.vqa_model, "is_qwen", False)
            self.backbone = self.vqa_model.backbone
            print(f"[CaptioningModel] Reusing shared {self.backbone} instance on {self.device}.")
            return

        cap_cfg = self.config.get("models", {}).get("captioning", {}) if isinstance(self.config, dict) else {}
        preferred_backbone = cap_cfg.get("backbone", "Qwen2-VL-2B")
        model_id = cap_cfg.get("model_id", "Qwen/Qwen2-VL-2B-Instruct")

        if preferred_backbone == "Qwen2-VL-2B":
            try:
                print(f"[CaptioningModel] Loading {model_id} on {self.device}...")
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

                dtype = torch.float16 if self.device == "cuda" else torch.float32
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map="auto" if self.device == "cuda" else None,
                )
                if self.device != "cuda":
                    self.model.to(self.device)

                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model.eval()
                self.is_qwen = True
                self.backbone = "Qwen2-VL-2B"
                print(f"[CaptioningModel] Qwen2-VL-2B loaded successfully on {self.device}.")
                return
            except Exception as e:
                print(f"[CaptioningModel] Warning: Could not load Qwen2-VL ({e}). Falling back to BLIP-Captioning...")

        # Fallback to BLIP Captioning
        from transformers import BlipProcessor, BlipForConditionalGeneration
        blip_id = "Salesforce/blip-image-captioning-base"
        print(f"[CaptioningModel] Loading fallback {blip_id} on {self.device}...")
        self.processor = BlipProcessor.from_pretrained(blip_id)
        self.model = BlipForConditionalGeneration.from_pretrained(blip_id, use_safetensors=True)
        if self.device != "cpu":
            self.model = self.model.half()
        self.model.to(self.device)
        self.model.eval()
        self.is_qwen = False
        self.backbone = "BLIP-Captioning"
        print(f"[CaptioningModel] BLIP-Captioning fallback loaded successfully on {self.device}.")

    def _numpy_to_pil(self, image):
        """Convert any satellite image representation to a PIL RGB Image."""
        from src.preprocessing.geotiff_utils import to_pil_rgb
        return to_pil_rgb(image)

    def predict(self, image, query: str = None, **kwargs) -> dict:
        """Generate a detailed scene caption for the satellite image."""
        start = time.time()

        if self.model is None:
            try:
                self.load(self.checkpoint_path or "", device=self.device)
            except Exception as e:
                return self._empty_result(
                    text=f"Captioning model failed to load: {e}",
                    status="error",
                )

        import torch
        from src.preprocessing.geotiff_utils import to_pil_rgb

        pil_image = to_pil_rgb(image)

        try:
            if self.is_qwen:
                try:
                    from qwen_vl_utils import process_vision_info
                    has_qwen_utils = True
                except ImportError:
                    has_qwen_utils = False

                prompt = query if query else self.RS_CAPTION_PROMPTS[0]
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                text_prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                if has_qwen_utils:
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.processor(
                        text=[text_prompt],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                else:
                    inputs = self.processor(
                        text=[text_prompt],
                        images=[pil_image],
                        padding=True,
                        return_tensors="pt",
                    )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
                ]
                final_caption = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()

                confidence = 0.90

            else:
                prompt = query if query else self.RS_CAPTION_PROMPTS[1]
                inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                if self.device != "cpu":
                    inputs = {k: v.half() if v.dtype == torch.float32 else v
                              for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, max_length=80)

                final_caption = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
                confidence = min(0.90, 0.5 + len(final_caption.split()) * 0.03)

            return RSModelResult(
                task="captioning",
                text=final_caption,
                confidence=confidence,
                spatial_evidence=SpatialEvidence(),
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
