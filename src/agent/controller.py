"""
Agent controller — the core orchestration entry point.
Pipeline: validate input -> route task -> select tool(s) -> run inference ->
integrate outputs -> log execution trace -> return response.
Owner: Person 4 — this is INTEGRATION-PHASE work, after the 4 ML notebooks
(GeoChat/VQA+captioning, GeoChat-or-GeoGround/SAM grounding, VisTA/change,
optical-SAR fusion) each produce a stable predict() following the
src.common.schemas.RSModelResult contract. Do not block ML work on this file.
"""

from src.agent.input_validator import validate_input
from src.agent.task_router import route_task
from src.agent.tool_registry import ToolRegistry
from src.agent.execution_trace import ExecutionTrace


class AgentController:
    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.registry = ToolRegistry(config, device=device)
        self.confidence_threshold = config["agent"]["confidence_threshold"]

    def run(self, images: dict, query: str) -> dict:
        """images: see input_validator.validate_input for expected keys.
        query: natural-language question/instruction from the user.

        Returns:
            {
              "success": bool,
              "task": str | None,
              "text": str | None,
              "confidence": float | None,
              "spatial_evidence": {"type":.., "source":.., "bbox": list|None, "mask": np.ndarray|None} | None,
              "metadata": {"model":.., "backbone":.., "checkpoint":.., "dataset":..,
                           "input_modalities": [...], "parameters": {...}} | None,
              "status": "success" | "error" | "low_confidence" | None,
              "errors": list[str],
              "trace": dict,
            }
        """
        # 1. Validate input
        validation = validate_input(images, self.config)
        trace = ExecutionTrace(query=query, mode=validation.get("mode"))
        trace.set_input_metadata(validation.get("metadata", {}))

        if not validation["valid"]:
            return self._error_response(validation["errors"], trace, task=None)

        mode = validation["mode"]

        # 2. Route task
        task = route_task(query, mode)
        trace.set_task(task)

        # 3. Select + execute tool
        try:
            tool = self.registry.get(task)
        except KeyError as e:
            trace.add_warning(str(e))
            return self._error_response([str(e)], trace, task=task)

        params = self._build_predict_kwargs(images, query, mode)
        trace.add_tool_call(name=tool.name, task=task, params={"query": query, "mode": mode})

        # TODO(Person 4, once a person's model is wired up):
        # result = tool.predict(**params)   # returns RSModelResult.to_dict()
        # For now, return a stub so the rest of the pipeline (GUI, trace,
        # reporting) can be built and tested before ML notebooks finish.
        result = tool._empty_result(
            text="[stub] Model not yet implemented — this is a placeholder response.",
            confidence=0.0,
        )

        # 4. Confidence flag
        if result["confidence"] < self.confidence_threshold:
            trace.add_warning(
                f"Low confidence ({result['confidence']:.2f} < {self.confidence_threshold})"
            )
        trace.set_confidence(result["confidence"])

        return {
            "success": True,
            "task": result["task"],
            "text": result["text"],
            "confidence": result["confidence"],
            "spatial_evidence": result.get("spatial_evidence"),
            "metadata": result.get("metadata"),
            "status": result.get("status"),
            "errors": [],
            "trace": trace.finalize(),
        }

    def _error_response(self, errors: list, trace: "ExecutionTrace", task: str = None) -> dict:
        return {
            "success": False,
            "task": task,
            "text": None,
            "confidence": None,
            "spatial_evidence": None,
            "metadata": None,
            "status": "error",
            "errors": errors,
            "trace": trace.finalize(),
        }

    def _build_predict_kwargs(self, images: dict, query: str, mode: str) -> dict:
        """Map validated image paths -> the kwargs each model's predict()
        expects (see docs/api_contracts.md table).

        TODO(Person 4): once preprocessing is wired in, load + preprocess
        images here (not just pass raw paths) before calling tool.predict().
        """
        if mode == "single":
            return {"image": images["image"], "query": query}
        if mode == "cross_modal":
            return {"image_optical": images["image_optical"],
                    "image_sar": images["image_sar"], "query": query}
        if mode == "bi_temporal":
            return {"image_t1": images["image_t1"],
                    "image_t2": images["image_t2"], "query": query}
        raise ValueError(f"Unknown mode: {mode}")
