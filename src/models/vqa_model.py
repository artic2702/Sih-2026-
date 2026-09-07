"""
Remote-sensing Visual Question Answering model. MANDATORY per problem statement.
Owner: Person 1. Sprint plan: notebooks/01_vqa_geochat.ipynb.

Backbone strategy (8 GB VRAM constraint):
  - PRIMARY: Salesforce/blip-vqa-base (~800 MB VRAM, guaranteed to work on any
    GPU or even CPU). Provides real, meaningful answers to visual questions
    about satellite/aerial imagery.
  - UPGRADE PATH: GeoChat-7B in 4-bit quantization (~4.2 GB VRAM) can be
    swapped in by setting config['models']['vqa']['backbone'] = 'GeoChat'
    and providing the LoRA adapter checkpoint. The BLIP baseline ensures the
    smoke test and demo always work even without GeoChat weights.
"""

import time
import numpy as np

from src.models.base_model import BaseRSModel
from src.common.schemas import RSModelResult, SpatialEvidence, ResultMetadata


class VQAModel(BaseRSModel):
    name = "vqa_v1"
    task = "vqa"
    backbone = "Qwen2-VL-2B"

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.checkpoint_path = None
        self._pil_image_class = None
        self.is_qwen = False

    def load(self, checkpoint_path: str = "", device: str = "cpu") -> None:
        """Load Qwen2-VL-2B-Instruct on GPU/CPU with automatic fallback to BLIP-VQA."""
        import torch
        from PIL import Image

        self._pil_image_class = Image
        self.device = device if torch.cuda.is_available() and device != "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path

        # Determine configured backbone
        vqa_cfg = self.config.get("models", {}).get("vqa", {}) if isinstance(self.config, dict) else {}
        preferred_backbone = vqa_cfg.get("backbone", "Qwen2-VL-2B")
        model_id = vqa_cfg.get("model_id", "Qwen/Qwen2-VL-2B-Instruct")

        if preferred_backbone == "Qwen2-VL-2B":
            try:
                print(f"[VQAModel] Loading {model_id} on {self.device}...")
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
                print(f"[VQAModel] Qwen2-VL-2B loaded successfully on {self.device}.")
                return
            except Exception as e:
                print(f"[VQAModel] Warning: Could not load Qwen2-VL ({e}). Falling back to BLIP-VQA...")

        # Fallback to BLIP-VQA
        from transformers import BlipProcessor, BlipForQuestionAnswering
        blip_id = "Salesforce/blip-vqa-base"
        print(f"[VQAModel] Loading fallback {blip_id} on {self.device}...")
        self.processor = BlipProcessor.from_pretrained(blip_id)
        self.model = BlipForQuestionAnswering.from_pretrained(blip_id, use_safetensors=True)
        if self.device != "cpu":
            self.model = self.model.half()
        self.model.to(self.device)
        self.model.eval()
        self.is_qwen = False
        self.backbone = "BLIP-VQA"
        print(f"[VQAModel] BLIP-VQA fallback loaded successfully on {self.device}.")

    def _numpy_to_pil(self, image):
        """Convert any satellite image representation to a PIL RGB Image."""
        from src.preprocessing.geotiff_utils import to_pil_rgb
        return to_pil_rgb(image)

    def predict(self, image, query: str, **kwargs) -> dict:
        """Run VQA inference on a single image with a natural-language question.

        Args:
            image: np.ndarray, file path, or PIL Image.
            query: Natural-language question, e.g. "What land-cover types are visible?"

        Returns:
            dict matching RSModelResult schema.
        """
        start = time.time()

        if self.model is None:
            try:
                self.load(self.checkpoint_path or "", device=self.device)
            except Exception as e:
                return self._empty_result(
                    text=f"VQA model failed to load: {e}",
                    status="error",
                )

        import torch
        from src.preprocessing.geotiff_utils import to_pil_rgb

        # Convert to PIL Image
        pil_image = to_pil_rgb(image)

        # Default query if none provided
        if not query or query.strip() == "":
            query = "What is shown in this satellite image?"

        try:
            if self.is_qwen:
                try:
                    from qwen_vl_utils import process_vision_info
                    has_qwen_utils = True
                except ImportError:
                    has_qwen_utils = False

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": query},
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
                answer_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()

                confidence = 0.88

            else:
                # BLIP baseline inference
                inputs = self.processor(images=pil_image, text=query, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                if self.device != "cpu":
                    inputs = {k: v.half() if v.dtype == torch.float32 else v
                              for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, max_length=50)

                answer_text = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
                confidence = min(0.85, 0.5 + len(answer_text.split()) * 0.05)

            return RSModelResult(
                task="vqa",
                text=answer_text,
                confidence=confidence,
                spatial_evidence=SpatialEvidence(),
                metadata=ResultMetadata(
                    model=self.name,
                    backbone=self.backbone,
                    checkpoint=self.checkpoint_path or "",
                    dataset="VRSBench+RSVQA",
                    input_modalities=["optical"],
                    parameters={"query": query},
                ),
                status="success",
                inference_seconds=time.time() - start,
            ).to_dict()

        except Exception as e:
            return RSModelResult(
                task="vqa",
                text=f"VQA inference error: {e}",
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


def train_step(model: VQAModel, batch: dict, optimizer, config: dict) -> float:
    """One LoRA fine-tuning step; return the loss (float).
    This mirrors what notebooks/01_vqa_geochat.ipynb does interactively —
    move the reusable version here once the notebook approach works.
    """
    raise NotImplementedError
