"""
Pydantic API schemas for SatQuery AI.
Defines typed request and response contracts for all ML specialists and the Agent.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field, ConfigDict, field_validator


# ---------------------------------------------------------------------------
# Common Evidence & Metadata Schemas
# ---------------------------------------------------------------------------

class SpatialEvidenceSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: str = Field("none", description="Evidence type: 'bbox', 'mask', or 'none'")
    source: str = Field("image", description="Image coordinate frame: 'image', 'optical', 'sar', 'image_t1', 'image_t2', or 'fused'")
    bbox: Optional[List[float]] = Field(None, description="Bounding box [x1, y1, x2, y2] in pixel coordinates")
    mask: Optional[Any] = Field(None, description="Spatial mask metadata or summary")

    @field_validator("mask", mode="before")
    @classmethod
    def serialize_mask(cls, v):
        if isinstance(v, np.ndarray):
            return {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "nonzero_pixels": int(np.count_nonzero(v)),
                "changed_ratio": float(np.count_nonzero(v) / max(v.size, 1)),
            }
        return v


class ResultMetadataSchema(BaseModel):
    model: str = Field("", description="Specialist model name, e.g. 'vqa_v1', 'fusion_v1'")
    backbone: str = Field("", description="Backbone architecture, e.g. 'Qwen2-VL-2B', 'OpticalSAR-FeatureFusion'")
    checkpoint: str = Field("", description="Loaded checkpoint identifier or path")
    dataset: str = Field("", description="Benchmark or evaluation dataset, e.g. 'VRSBench', 'BigEarthNet-MM'")
    input_modalities: List[str] = Field(default_factory=list, description="List of input modalities utilized, e.g. ['optical', 'sar']")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model-specific output parameters, class probabilities, or coordinates")


class VerificationReportSchema(BaseModel):
    passed: bool = Field(..., description="Whether the prediction passed all verification checks")
    decision: str = Field(..., description="Decision state: 'SUPPORTED', 'FALLBACK_GEE', or 'UNVERIFIED'")
    reasons: List[str] = Field(default_factory=list, description="Audit warnings or failure reasons")
    checks: Dict[str, Any] = Field(default_factory=dict, description="Multi-criteria check breakdown (task_compliance, target_alignment, output_sanity, geospatial_context)")
    gee_evidence: Optional[Dict[str, Any]] = Field(None, description="Google Earth Engine catalog fallback package if activated")


class RSModelResultSchema(BaseModel):
    task: str = Field(..., description="Remote sensing task: 'vqa', 'captioning', 'grounding', 'change', or 'fusion'")
    text: str = Field(..., description="Generated natural language response or classification summary")
    confidence: float = Field(..., description="Confidence score [0.0, 1.0]")
    spatial_evidence: SpatialEvidenceSchema = Field(default_factory=SpatialEvidenceSchema, description="Spatial evidence coordinates or mask")
    metadata: ResultMetadataSchema = Field(default_factory=ResultMetadataSchema, description="Execution and provenance metadata")
    status: str = Field("success", description="Prediction status: 'success', 'low_confidence', or 'error'")
    inference_seconds: float = Field(0.0, description="Latency of model inference in seconds")


# ---------------------------------------------------------------------------
# Health & Model Metadata Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field("healthy", description="API operational health status")
    version: str = Field("1.0.0", description="SatQuery AI version")
    device: str = Field("cpu", description="Computing device utilized ('cuda' or 'cpu')")
    models_available: List[str] = Field(default_factory=list, description="List of registered model tasks")
    timestamp: str = Field(..., description="UTC ISO timestamp of health check")


class ModelInfoSchema(BaseModel):
    task: str = Field(..., description="Task category: 'vqa', 'captioning', 'grounding', 'change', 'fusion'")
    name: str = Field(..., description="Model identifier")
    backbone: str = Field(..., description="Underlying neural network backbone")
    loaded: bool = Field(..., description="Whether model weights are loaded and ready for inference")


class ModelsListResponse(BaseModel):
    models: List[ModelInfoSchema] = Field(default_factory=list, description="List of available specialist models")


# ---------------------------------------------------------------------------
# Agent Request & Response Schemas
# ---------------------------------------------------------------------------

class AgentQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question or instruction for satellite imagery", examples=["What color are the buses in the airport?"])
    image_path: Optional[str] = Field(None, description="Local path to single satellite image (optical or SAR)", examples=["data/P0003_0002.png"])
    image_base64: Optional[str] = Field(None, description="Base64 encoded string of single satellite image")
    image_optical_path: Optional[str] = Field(None, description="Path to optical image (for fusion mode)", examples=["data/sample_bigearthnet/urban_s2_rgb.png"])
    image_optical_base64: Optional[str] = Field(None, description="Base64 encoded optical image (for fusion mode)")
    image_sar_path: Optional[str] = Field(None, description="Path to SAR radar image (for fusion mode)", examples=["data/sample_bigearthnet/urban_s1_sar.png"])
    image_sar_base64: Optional[str] = Field(None, description="Base64 encoded SAR radar image (for fusion mode)")
    image_t1_path: Optional[str] = Field(None, description="Path to Time T1 before image (for change detection)", examples=["data/sample_bitemporal/bitemporal_t1_before.png"])
    image_t1_base64: Optional[str] = Field(None, description="Base64 encoded Time T1 before image")
    image_t2_path: Optional[str] = Field(None, description="Path to Time T2 after image (for change detection)", examples=["data/sample_bitemporal/bitemporal_t2_after.png"])
    image_t2_base64: Optional[str] = Field(None, description="Base64 encoded Time T2 after image")


class AgentResponse(BaseModel):
    success: bool = Field(..., description="Whether the agentic pipeline completed successfully")
    task: str = Field(..., description="Task classified and routed by AgentController")
    text: str = Field(..., description="Natural language answer or classification text")
    confidence: float = Field(..., description="Model prediction confidence score")
    spatial_evidence: Optional[SpatialEvidenceSchema] = Field(None, description="Visual grounding bounding boxes or change mask")
    metadata: Optional[ResultMetadataSchema] = Field(None, description="Model provenance and configuration metadata")
    verification: VerificationReportSchema = Field(..., description="Evidence Verification Gate multi-criteria audit report")
    status: str = Field(..., description="Overall gate status: 'verified', 'unverified', 'gee_fallback', or 'error'")
    errors: List[str] = Field(default_factory=list, description="List of errors encountered during execution")
    trace: Dict[str, Any] = Field(default_factory=dict, description="Audit execution trace log")


# ---------------------------------------------------------------------------
# Specialist Model Request Schemas
# ---------------------------------------------------------------------------

class VQARequest(BaseModel):
    query: str = Field(..., description="Natural language question regarding the satellite scene", examples=["What color are the buses in the image?"])
    image_path: Optional[str] = Field(None, description="Path to optical/radar satellite image", examples=["data/P0003_0002.png"])
    image_base64: Optional[str] = Field(None, description="Base64 encoded satellite image")


class CaptionRequest(BaseModel):
    query: Optional[str] = Field("Describe this satellite image in detail.", description="Captioning prompt instruction")
    image_path: Optional[str] = Field(None, description="Path to satellite image", examples=["data/vrsbench_samples/harbor_ships.png"])
    image_base64: Optional[str] = Field(None, description="Base64 encoded satellite image")


class GroundingRequest(BaseModel):
    query: str = Field(..., description="Referring grounding phrase or object to localize", examples=["Localize the yellow bus in the scene."])
    image_path: Optional[str] = Field(None, description="Path to satellite image", examples=["data/P0003_0002.png"])
    image_base64: Optional[str] = Field(None, description="Base64 encoded satellite image")


class ChangeDetectionRequest(BaseModel):
    query: Optional[str] = Field("What changed between time T1 and time T2?", description="Change detection query")
    image_t1_path: Optional[str] = Field(None, description="Path to Time T1 before satellite image", examples=["data/sample_bitemporal/bitemporal_t1_before.png"])
    image_t1_base64: Optional[str] = Field(None, description="Base64 encoded Time T1 image")
    image_t2_path: Optional[str] = Field(None, description="Path to Time T2 after satellite image", examples=["data/sample_bitemporal/bitemporal_t2_after.png"])
    image_t2_base64: Optional[str] = Field(None, description="Base64 encoded Time T2 image")


class FusionRequest(BaseModel):
    query: Optional[str] = Field("Analyze this multi-modal pair and classify the land cover.", description="Fusion query")
    image_optical_path: Optional[str] = Field(None, description="Path to Sentinel-2 optical image (GeoTIFF/PNG)", examples=["data/sample_bigearthnet/urban_s2_rgb.png"])
    image_optical_base64: Optional[str] = Field(None, description="Base64 encoded Sentinel-2 optical image")
    image_sar_path: Optional[str] = Field(None, description="Path to Sentinel-1 SAR image (GeoTIFF/PNG)", examples=["data/sample_bigearthnet/urban_s1_sar.png"])
    image_sar_base64: Optional[str] = Field(None, description="Base64 encoded Sentinel-1 SAR image")


class VerificationRequest(BaseModel):
    query: str = Field(..., description="User query submitted to the model", examples=["Localize the airplane"])
    task: str = Field(..., description="Specialist task: 'vqa', 'captioning', 'grounding', 'change', or 'fusion'")
    result: Dict[str, Any] = Field(..., description="Dictionary conforming to RSModelResult")
    image_path: Optional[str] = Field(None, description="Path to the primary image for spatial/spectral verification")
