"""
Health and System Metadata API endpoints.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from src.api.schemas import HealthResponse, ModelsListResponse, ModelInfoSchema
from src.api.deps import get_controller, get_device
from src.agent.controller import AgentController

router = APIRouter(tags=["System & Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health Check",
    description="Check the operational readiness of the SatQuery AI service, GPU availability, and registered models.",
)
@router.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="API v1 Health Check",
    include_in_schema=False,
)
def health_check(controller: AgentController = Depends(get_controller)) -> HealthResponse:
    device = get_device()
    models = list(controller.registry._tools.keys())
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        device=device,
        models_available=models,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/api/v1/models",
    response_model=ModelsListResponse,
    summary="List Registered Specialist Models",
    description="Retrieve details on all available vision-language, change detection, and multi-modal fusion specialists.",
)
def list_models(controller: AgentController = Depends(get_controller)) -> ModelsListResponse:
    info_list = []
    for task_name, model in controller.registry._tools.items():
        info_list.append(
            ModelInfoSchema(
                task=task_name,
                name=getattr(model, "name", task_name),
                backbone=getattr(model, "backbone", "Unknown"),
                loaded=True,
            )
        )
    return ModelsListResponse(models=info_list)
