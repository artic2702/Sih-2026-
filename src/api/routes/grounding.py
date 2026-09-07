"""
Spatial Referring Grounding & Localization specialist API endpoint.
Detects object/feature bounding boxes and multi-region land-cover extents.
"""

from fastapi import APIRouter, Depends, UploadFile, File, Form
from src.api.schemas import GroundingRequest, RSModelResultSchema
from src.api.deps import get_controller, resolve_image_input
from src.agent.controller import AgentController
from src.preprocessing.geotiff_utils import read_image

router = APIRouter(prefix="/api/v1/models/grounding", tags=["Specialist: Grounding"])


@router.post(
    "",
    response_model=RSModelResultSchema,
    summary="Ground & Localize Objects (JSON / Path / Base64)",
    description="Localize targets, vehicles, buildings, or land cover in satellite imagery. Returns normalized and pixel coordinates [x1, y1, x2, y2].",
)
def predict_grounding(
    request: GroundingRequest,
    controller: AgentController = Depends(get_controller),
) -> RSModelResultSchema:
    img_path = resolve_image_input(path=request.image_path, b64=request.image_base64)
    model = controller.registry.get("grounding")
    rs_img = read_image(img_path)
    result = model.predict(image=rs_img.array, query=request.query)
    return RSModelResultSchema(**result)


@router.post(
    "/upload",
    response_model=RSModelResultSchema,
    summary="Ground & Localize Objects (Multipart File Upload)",
    description="Upload a satellite image and specify an entity to localize in bounding box coordinates.",
)
def upload_grounding(
    query: str = Form(..., description="Target phrase to locate", examples=["Localize the yellow bus in the scene."]),
    image: UploadFile = File(..., description="Satellite image file (PNG/JPEG/GeoTIFF)"),
    controller: AgentController = Depends(get_controller),
) -> RSModelResultSchema:
    img_path = resolve_image_input(upload=image)
    model = controller.registry.get("grounding")
    rs_img = read_image(img_path)
    result = model.predict(image=rs_img.array, query=query)
    return RSModelResultSchema(**result)
