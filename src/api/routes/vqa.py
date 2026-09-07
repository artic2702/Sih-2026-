"""
Visual Question Answering (VQA) specialist API endpoint.
Powered by Qwen2-VL-2B backbone.
"""

from fastapi import APIRouter, Depends, UploadFile, File, Form
from src.api.schemas import VQARequest, RSModelResultSchema
from src.api.deps import get_controller, resolve_image_input
from src.agent.controller import AgentController
from src.preprocessing.geotiff_utils import read_image

router = APIRouter(prefix="/api/v1/models/vqa", tags=["Specialist: VQA"])


@router.post(
    "",
    response_model=RSModelResultSchema,
    summary="Query VQA Model (JSON / Path / Base64)",
    description="Ask specific questions about objects, counts, colors, and features in satellite imagery.",
)
def predict_vqa(
    request: VQARequest,
    controller: AgentController = Depends(get_controller),
) -> RSModelResultSchema:
    img_path = resolve_image_input(path=request.image_path, b64=request.image_base64)
    model = controller.registry.get("vqa")
    rs_img = read_image(img_path)
    result = model.predict(image=rs_img.array, query=request.query)
    return RSModelResultSchema(**result)


@router.post(
    "/upload",
    response_model=RSModelResultSchema,
    summary="Query VQA Model (Multipart File Upload)",
    description="Upload a satellite image and ask a question directly to the VQA model.",
)
def upload_vqa(
    query: str = Form(..., description="Question regarding the satellite scene", examples=["What color are the buses in the image?"]),
    image: UploadFile = File(..., description="Satellite image file (PNG/JPEG/GeoTIFF)"),
    controller: AgentController = Depends(get_controller),
) -> RSModelResultSchema:
    img_path = resolve_image_input(upload=image)
    model = controller.registry.get("vqa")
    rs_img = read_image(img_path)
    result = model.predict(image=rs_img.array, query=query)
    return RSModelResultSchema(**result)
