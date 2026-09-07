"""
Remote-sensing Visual Question Answering model. MANDATORY per problem statement.
Owner: Person 1. Sprint plan: notebooks/01_vqa_geochat.ipynb.

Backbone: GeoChat (already remote-sensing-adapted). Fine-tune further with
LoRA on a BigEarthNet.txt subset to make the "remote-sensing adaptation"
requirement defensible, then evaluate on RSVQA/VRSBench VQA splits.
Do NOT swap this for a generic BLIP-2/LLaVA backbone — GeoChat is the
team's chosen model for this slot.
"""

from src.models.base_model import BaseRSModel
from src.common.schemas import RSModelResult, SpatialEvidence, ResultMetadata


class VQAModel(BaseRSModel):
    name = "vqa_v1"
    task = "vqa"
    backbone = "GeoChat"

    def __init__(self, config: dict = None):
        self.config = config
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.checkpoint_path = None

    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        """TODO(Person 1):
        - Load GeoChat base weights + the LoRA adapter from checkpoint_path
          (built in notebooks/01_vqa_geochat.ipynb).
        - Load the matching image/text processor.
        - Move model to `device` and set eval mode.
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        raise NotImplementedError

    def predict(self, image, query: str, **kwargs) -> dict:
        """image: preprocessed np.ndarray or path; query: natural-language question.

        TODO(Person 1):
        - Run processor -> model.generate -> decode answer text.
        - Derive a confidence score (e.g. token-probability based, or a
          calibrated head).
        - Return RSModelResult(...).to_dict() — do not hand-roll the dict.

        Example of the expected return shape once implemented:
            return RSModelResult(
                task="vqa", text=answer_text, confidence=confidence,
                spatial_evidence=SpatialEvidence(),  # VQA has no bbox/mask
                metadata=ResultMetadata(model="GeoChat",
                                         checkpoint=self.checkpoint_path or "",
                                         dataset="BigEarthNet.txt+RSVQA",
                                         backbone=self.backbone,
                                         input_modalities=["optical"]),
                status="success",
            ).to_dict()
        """
        raise NotImplementedError


def train_step(model: VQAModel, batch: dict, optimizer, config: dict) -> float:
    """TODO(Person 1): one LoRA fine-tuning step; return the loss (float).
    This mirrors what notebooks/01_vqa_geochat.ipynb does interactively —
    move the reusable version here once the notebook approach works.
    """
    raise NotImplementedError
