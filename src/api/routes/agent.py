"""
Autonomous Agent Orchestration Endpoints.
Routes queries to specialists and runs multi-criteria Evidence Verification.
"""

from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException

from src.api.schemas import AgentQueryRequest, AgentResponse
from src.api.deps import get_controller, resolve_image_input
from src.agent.controller import AgentController

router = APIRouter(prefix="/api/v1/agent", tags=["Autonomous Agent"])


@router.post(
    "/query",
    response_model=AgentResponse,
    summary="Execute Agentic Satellite Query (JSON / Paths / Base64)",
    description="Pass a natural language query and satellite image reference (single image, optical+SAR pair, or bi-temporal pair). The agent automatically validates inputs, routes to the specialist model, and enforces verification.",
)
def agent_query(
    request: AgentQueryRequest,
    controller: AgentController = Depends(get_controller),
) -> AgentResponse:
    images = {}

    # Check for Bi-Temporal pair
    if (request.image_t1_path or request.image_t1_base64) and (request.image_t2_path or request.image_t2_base64):
        images["image_t1"] = resolve_image_input(path=request.image_t1_path, b64=request.image_t1_base64)
        images["image_t2"] = resolve_image_input(path=request.image_t2_path, b64=request.image_t2_base64)

    # Check for Cross-Modal Optical + SAR pair
    elif (request.image_optical_path or request.image_optical_base64) and (request.image_sar_path or request.image_sar_base64):
        images["image_optical"] = resolve_image_input(path=request.image_optical_path, b64=request.image_optical_base64)
        images["image_sar"] = resolve_image_input(path=request.image_sar_path, b64=request.image_sar_base64)

    # Single Image
    elif request.image_path or request.image_base64:
        images["image"] = resolve_image_input(path=request.image_path, b64=request.image_base64)
    else:
        raise HTTPException(
            status_code=400,
            detail="No valid image inputs provided. Supply 'image_path'/'image_base64', optical+SAR pair, or T1+T2 pair.",
        )

    result = controller.run(images=images, query=request.query)
    return AgentResponse(**result)


@router.post(
    "/upload-query",
    response_model=AgentResponse,
    summary="Execute Agentic Satellite Query (Multipart File Upload)",
    description="Upload raw image binary files (PNG, JPEG, GeoTIFF) with a query instruction. Supports single image, optical+SAR, or bi-temporal uploads.",
)
def agent_upload_query(
    query: str = Form(..., description="Query or instruction", examples=["Describe this satellite scene."]),
    image: Optional[UploadFile] = File(None, description="Single satellite image (optical or SAR)"),
    image_optical: Optional[UploadFile] = File(None, description="Optical image (for fusion)"),
    image_sar: Optional[UploadFile] = File(None, description="SAR radar image (for fusion)"),
    image_t1: Optional[UploadFile] = File(None, description="Time T1 before image (for change detection)"),
    image_t2: Optional[UploadFile] = File(None, description="Time T2 after image (for change detection)"),
    controller: AgentController = Depends(get_controller),
) -> AgentResponse:
    images = {}

    if image_t1 and image_t2:
        images["image_t1"] = resolve_image_input(upload=image_t1)
        images["image_t2"] = resolve_image_input(upload=image_t2)
    elif image_optical and image_sar:
        images["image_optical"] = resolve_image_input(upload=image_optical)
        images["image_sar"] = resolve_image_input(upload=image_sar)
    elif image:
        images["image"] = resolve_image_input(upload=image)
    else:
        raise HTTPException(
            status_code=400,
            detail="Must upload at least one image file ('image', optical+sar pair, or t1+t2 pair).",
        )

    result = controller.run(images=images, query=query)
    return AgentResponse(**result)
