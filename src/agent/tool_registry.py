"""
Tool registry — maps task labels to loaded specialist model instances.
Owner: Person 4 (integration phase, after the 4 ML notebooks are stable).

Registry now; intelligent LLM-based routing can stay simple (task_router.py)
until the ML backend is proven. This file just wires task -> model instance.
"""

from src.models.vqa_model import VQAModel
from src.models.captioning_model import CaptioningModel
from src.models.grounding_model import GroundingModel
from src.models.change_model import ChangeModel
from src.models.fusion_model import FusionModel


class ToolRegistry:
    """Loads and holds one instance per task. The controller calls
    registry.get(task) to fetch the right model without knowing its
    implementation details (per BaseRSModel contract).
    """

    _MODEL_CLASSES = {
        "vqa": VQAModel,
        "captioning": CaptioningModel,
        "grounding": GroundingModel,
        "change": ChangeModel,
        "fusion": FusionModel,
    }

    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = device
        self._tools = {}
        self._build_registry()

    def _build_registry(self):
        model_cfg = self.config["models"]

        # VQA is always required (Person 1 — GeoChat).
        self.register(VQAModel(config=self.config), model_cfg["vqa"]["checkpoint"])

        # Captioning is GeoChat-shared by default; grounding (Person 2) can
        # be GeoChat-based or fall back to a dedicated model — both are
        # registered under the same "grounding" task key regardless of which
        # backbone ends up being used internally.
        if model_cfg.get("captioning", {}).get("enabled", True):
            self.register(CaptioningModel(config=self.config), model_cfg["captioning"]["checkpoint"])
        if model_cfg.get("grounding", {}).get("enabled", True):
            self.register(GroundingModel(config=self.config), model_cfg["grounding"]["checkpoint"])

        self.register(ChangeModel(config=self.config), model_cfg["change"]["checkpoint"])
        self.register(FusionModel(config=self.config), model_cfg["fusion"]["checkpoint"])

    def register(self, model, checkpoint_path: str):
        """Register specialist model and initialize its weights."""
        try:
            if hasattr(model, "load"):
                model.load(checkpoint_path, device=self.device)
        except Exception:
            # Model will use its built-in fallback on predict if checkpoint is uninitialized
            pass
        self._tools[model.task] = model

    def get(self, task: str):
        if task not in self._tools:
            raise KeyError(f"No tool registered for task '{task}'. "
                            f"Available: {list(self._tools.keys())}")
        return self._tools[task]
