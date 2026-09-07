"""
Bi-Temporal Change Detection specialist API endpoint.
Analyzes differences, deforestation, construction, or disaster impacts between two timestamps.
"""

from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form
from src.api.schemas import ChangeDetectionRequest, RSModelResultSchema
from src.api.deps import get_controller, resolve_image_input
from src.agent.controller import AgentController
from src.preprocessing.geotiff_utils import read_image

router = APIRouter(prefix="/api/v1/models/change", tags=["Specialist: Change Detection"])


@router.post(
    "",
    response_model=RSModelResultSchema,
    summary="Detect Bi-Temporal Change (JSON / Paths / Base64)",
    description="Compare Time T1 (before) and Time T2 (after) satellite images to detect construction, land clearing, or structural changes.",
)
def predict_change(
    request: ChangeDetectionRequest,
    controller: AgentController = Depends(get_controller),
) -> RSModelResultSchema:
    t1_path = resolve_image_input(path=request.image_t1_path, b64=request.image_t1_base64)
    t2_path = resolve_image_input(path=request.image_t2_path, b64=request.image_t2_base64)
    model = controller.registry.get("change")
    rs_t1 = read_image(t1_path)
    rs_t2 = read_image(t2_path)
    result = model.predict(
        image_t1=rs_t1.array,
        image_t2=rs_t2.array,
        query=request.query or "What changed between time T1 and time T2?",
    )
    return RSModelResultSchema(**result)


@router.post(
    "/upload",
    response_model=RSModelResultSchema,
    summary="Detect Bi-Temporal Change (Multipart File Upload)",
    description="Upload Time T1 and Time T2 satellite images to extract change percentages, bounding boxes, and difference masks.",
)
def upload_change(
    image_t1: UploadFile = File(..., description="Time T1 (before) satellite image"),
    image_t2: UploadFile = File(..., description="Time T2 (after) satellite image"),
    query: Optional[str] = Form("What changed between time T1 and time T2?", description="Change query instruction"),
    controller: AgentController = Depends(get_controller),
) -> RSModelResultSchema:
    t1_path = resolve_image_input(upload=image_t1)
    t2_path = resolve_image_input(upload=image_t2)
    model = controller.registry.get("change")
    rs_t1 = read_image(t1_path)
    rs_t2 = read_image(t2_path)
    result = model.predict(
        image_t1=rs_t1.array,
        image_t2=rs_t2.array,
        query=query,
    )
    return RSModelResultSchema(**result)
