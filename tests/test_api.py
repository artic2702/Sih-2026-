"""
Automated unit and integration tests for SatQuery AI FastAPI Endpoints.
Uses FastAPI TestClient to test health, routing, specialists, and file uploads.
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.server import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "models_available" in data
    assert "vqa" in data["models_available"]
    assert "fusion" in data["models_available"]


def test_models_list_endpoint(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    tasks = [m["task"] for m in data["models"]]
    assert "vqa" in tasks
    assert "captioning" in tasks
    assert "grounding" in tasks
    assert "change" in tasks
    assert "fusion" in tasks


def test_agent_query_single_image(client):
    payload = {
        "query": "What color are the buses in the image?",
        "image_path": "data/P0003_0002.png",
    }
    response = client.post("/api/v1/agent/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["task"] in ("vqa", "captioning", "grounding")
    assert "text" in data
    assert "verification" in data
    assert data["verification"]["decision"] in ("SUPPORTED", "FALLBACK_GEE", "UNVERIFIED")


def test_agent_query_cross_modal(client):
    payload = {
        "query": "Analyze this multi-modal pair and classify the land cover.",
        "image_optical_path": "data/sample_bigearthnet/urban_s2_rgb.png",
        "image_sar_path": "data/sample_bigearthnet/urban_s1_sar.png",
    }
    response = client.post("/api/v1/agent/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["task"] == "fusion"
    assert "text" in data


def test_agent_query_bi_temporal(client):
    payload = {
        "query": "What changed between time T1 and time T2?",
        "image_t1_path": "data/sample_bitemporal/bitemporal_t1_before.png",
        "image_t2_path": "data/sample_bitemporal/bitemporal_t2_after.png",
    }
    response = client.post("/api/v1/agent/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["task"] == "change"
    assert "text" in data


def test_specialist_vqa(client):
    payload = {
        "query": "What objects are visible on the ground?",
        "image_path": "data/P0003_0002.png",
    }
    response = client.post("/api/v1/models/vqa", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "vqa"
    assert len(data["text"]) > 0


def test_specialist_captioning(client):
    payload = {
        "query": "Describe this satellite scene.",
        "image_path": "data/P0003_0002.png",
    }
    response = client.post("/api/v1/models/captioning", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "captioning"
    assert len(data["text"]) > 0


def test_specialist_grounding(client):
    payload = {
        "query": "Localize the yellow bus in the scene.",
        "image_path": "data/P0003_0002.png",
    }
    response = client.post("/api/v1/models/grounding", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "grounding"
    assert data["spatial_evidence"]["type"] == "bbox"
    assert data["spatial_evidence"]["bbox"] is not None


def test_specialist_change(client):
    payload = {
        "query": "Detect changes between T1 and T2.",
        "image_t1_path": "data/sample_bitemporal/bitemporal_t1_before.png",
        "image_t2_path": "data/sample_bitemporal/bitemporal_t2_after.png",
    }
    response = client.post("/api/v1/models/change", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "change"
    assert "parameters" in data["metadata"]
    assert "percent_changed" in data["metadata"]["parameters"]


def test_specialist_fusion(client):
    payload = {
        "query": "Classify the multi-modal land cover.",
        "image_optical_path": "data/sample_bigearthnet/forest_s2_rgb.png",
        "image_sar_path": "data/sample_bigearthnet/forest_s1_sar.png",
    }
    response = client.post("/api/v1/models/fusion", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "fusion"
    assert "class_probabilities" in data["metadata"]["parameters"]


def test_verification_endpoint(client):
    payload = {
        "query": "Localize the yellow bus",
        "task": "grounding",
        "result": {
            "task": "grounding",
            "text": "yellow bus",
            "confidence": 0.85,
            "spatial_evidence": {"type": "bbox", "bbox": [10.0, 20.0, 50.0, 60.0]},
            "metadata": {},
        },
        "image_path": "data/P0003_0002.png",
    }
    response = client.post("/api/v1/verify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["passed"] is True
    assert data["decision"] == "SUPPORTED"


def test_multipart_file_upload_vqa(client):
    with open("data/P0003_0002.png", "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/models/vqa/upload",
        data={"query": "Are there buses in the scene?"},
        files={"image": ("sample.png", file_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "vqa"
    assert len(data["text"]) > 0
