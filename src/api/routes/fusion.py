"""
Optical + SAR Multi-Modal Fusion specialist API endpoint.
Processes paired Sentinel-2 (optical) + Sentinel-1 (radar) imagery for land-cover classification.
"""

from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form
from src.api.schemas import FusionRequest, RSModelResultSchema
from src.api.deps import get_controller, resolve_image_input
from src.agent.controller import AgentController
from src.preprocessing.geotiff_utils import read_image

router = APIRouter(prefix="/api/v1/models/fusion", tags=["Specialist: Optical-SAR Fusion"])


@router.post(
    "",
    response_model=RSModelResultSchema,
    summary="Fused Land Cover Extraction (JSON / Paths / Base64)",
    description="Feed co-registered Optical (Sentinel-2) and SAR Radar (Sentinel-1) images to extract 19-class BigEarthNet/CORINE land cover.",
)
def predict_fusion(
    request: FusionRequest,
    controller: AgentController = Depends(get_controller),
) -> RSModelResultSchema:
    opt_path = resolve_image_input(path=request.image_optical_path, b64=request.image_optical_base64)
    sar_path = resolve_image_input(path=request.image_sar_path, b64=request.image_sar_base64)
    model = controller.registry.get("fusion")
    rs_opt = read_image(opt_path)
    rs_sar = read_image(sar_path)
    result = model.predict(
        image_optical=rs_opt.array,
        image_sar=rs_sar.array,
        query=request.query or "Analyze this multi-modal pair and classify the land cover.",
    )
    return RSModelResultSchema(**result)


@router.post(
    "/upload",
    response_model=RSModelResultSchema,
    summary="Fused Land Cover Extraction (Multipart File Upload)",
    description="Upload an Optical image and a matching SAR radar image to perform dual-encoder joint feature fusion.",
)
def upload_fusion(
    image_optical: UploadFile = File(..., description="Optical satellite image (Sentinel-2 GeoTIFF or RGB PNG)"),
    image_sar: UploadFile = File(..., description="SAR radar image (Sentinel-1 GeoTIFF or radar PNG)"),
    query: Optional[str] = Form("Analyze this multi-modal pair and classify the land cover.", description="Fusion query instruction"),
    controller: AgentController = Depends(get_controller),
) -> RSModelResultSchema:
    opt_path = resolve_image_input(upload=image_optical)
    sar_path = resolve_image_input(upload=image_sar)
    model = controller.registry.get("fusion")
    rs_opt = read_image(opt_path)
    rs_sar = read_image(sar_path)
    result = model.predict(
        image_optical=rs_opt.array,
        image_sar=rs_sar.array,
        query=query,
    )
    return RSModelResultSchema(**result)
