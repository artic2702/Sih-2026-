"""
Scene captioning / description model (Task-2 option A, per problem statement).
Owner: Person 1 — same backbone/notebook as VQA (notebooks/01_vqa_geochat.ipynb).

Backbone: GeoChat, prompted for captioning rather than fine-tuned separately.
Only build a dedicated captioning adapter if the generic caption prompt on
the VQA-adapted checkpoint is not good enough.
"""

from src.models.base_model import BaseRSModel
from src.common.schemas import RSModelResult, SpatialEvidence, ResultMetadata


class CaptioningModel(BaseRSModel):
    name = "caption_v1"
    task = "captioning"
    backbone = "GeoChat"

    def __init__(self, config: dict = None):
        self.config = config
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.checkpoint_path = None

    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        """TODO(Person 1): load the same GeoChat (+ LoRA) checkpoint used for
        VQA — captioning can reuse it with a different prompt template."""
        self.device = device
        self.checkpoint_path = checkpoint_path
        raise NotImplementedError

    def predict(self, image, query: str = None, **kwargs) -> dict:
        """query is optional; default to a generic describe-the-scene prompt
        if None (e.g. 'Describe the land-cover and major objects visible in
        this image.').

        TODO(Person 1): implement generation + confidence scoring, return
        RSModelResult(...).to_dict() with task="captioning", text=<caption>,
        spatial_evidence=SpatialEvidence() (none for captioning),
        metadata.input_modalities=["optical"] (or ["sar"] if applicable).
        """
        raise NotImplementedError


def train_step(model: CaptioningModel, batch: dict, optimizer, config: dict) -> float:
    """TODO(Person 1, only if a separate captioning adapter turns out to be
    needed): one training step; called from src/training/train_captioning.py."""
    raise NotImplementedError
