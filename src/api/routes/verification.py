"""
Evidence Verification Gate API endpoint.
Audits model outputs for hallucination, semantic alignment, and GEE routing.
"""

from fastapi import APIRouter, Depends
from src.api.schemas import VerificationRequest, VerificationReportSchema
from src.api.deps import get_controller, resolve_image_input
from src.agent.controller import AgentController

router = APIRouter(prefix="/api/v1/verify", tags=["Evidence Verification Gate"])


@router.post(
    "",
    response_model=VerificationReportSchema,
    summary="Audit Model Prediction with Verification Gate",
    description="Subject any model prediction to multi-criteria verification: task compliance, target alignment, output sanity, and geospatial context.",
)
def verify_prediction(
    request: VerificationRequest,
    controller: AgentController = Depends(get_controller),
) -> VerificationReportSchema:
    images = {}
    if request.image_path:
        images["image"] = resolve_image_input(path=request.image_path)

    report = controller.verification_gate.verify(
        query=request.query,
        task=request.task,
        images=images,
        result=request.result,
    )
    return VerificationReportSchema(**report.to_dict())
