"""
Scene Captioning specialist API endpoint.
Generates comprehensive descriptions of Earth observation scenes.
"""

from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form
from src.api.schemas import CaptionRequest, RSModelResultSchema
from src.api.deps import get_controller, resolve_image_input
from src.agent.controller import AgentController
from src.preprocessing.geotiff_utils import read_image

router = APIRouter(prefix="/api/v1/models/captioning", tags=["Specialist: Captioning"])


@router.post(
    "",
    response_model=RSModelResultSchema,
    summary="Generate Scene Caption (JSON / Path / Base64)",
    description="Generate detailed descriptive remote sensing captions of an optical or SAR satellite scene.",
)
def predict_caption(
    request: CaptionRequest,
    controller: AgentController = Depends(get_controller),
) -> RSModelResultSchema:
    img_path = resolve_image_input(path=request.image_path, b64=request.image_base64)
    model = controller.registry.get("captioning")
    rs_img = read_image(img_path)
    result = model.predict(image=rs_img.array, query=request.query or "Describe this satellite image in detail.")
    return RSModelResultSchema(**result)


@router.post(
    "/upload",
    response_model=RSModelResultSchema,
    summary="Generate Scene Caption (Multipart File Upload)",
    description="Upload a satellite image and generate an automated remote sensing scene caption.",
)
def upload_caption(
    image: UploadFile = File(..., description="Satellite image file (PNG/JPEG/GeoTIFF)"),
    query: Optional[str] = Form("Describe this satellite image in detail.", description="Captioning prompt instruction"),
    controller: AgentController = Depends(get_controller),
) -> RSModelResultSchema:
    img_path = resolve_image_input(upload=image)
    model = controller.registry.get("captioning")
    rs_img = read_image(img_path)
    result = model.predict(image=rs_img.array, query=query)
    return RSModelResultSchema(**result)
