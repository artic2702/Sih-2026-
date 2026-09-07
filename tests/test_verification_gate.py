"""
Tests for Evidence Verification Gate & GEE Fallback Logic.
Validates multi-criteria verification:
1. Task-Output Compatibility
2. Target Entity Semantic Alignment
3. Output Sanity & Degeneration
4. Geospatial Context & GEE Routing
"""

import sys
import os
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.verification_gate import VerificationGate, VerificationReport


@pytest.fixture
def gate():
    return VerificationGate()


def test_extract_target_concept(gate):
    assert gate._extract_target_concept("Mark the grass in this scene") == "grass"
    assert gate._extract_target_concept("Where is the airplane parked?") == "airplane"
    assert gate._extract_target_concept("Count the number of buildings") == "buildings"
    assert gate._extract_target_concept("Describe the land cover in detail") is None


def test_task_compliance_localization_success(gate):
    passed, detail = gate._check_task_compliance(
        query="Locate the yellow bus",
        task="grounding",
        result={"spatial_evidence": {"bbox": [10.0, 20.0, 50.0, 60.0]}, "text": "yellow bus"}
    )
    assert passed is True
    assert "Spatial localization provided" in detail


def test_task_compliance_localization_missing_bbox(gate):
    passed, detail = gate._check_task_compliance(
        query="Mark the aircraft on the runway",
        task="grounding",
        result={"spatial_evidence": {}, "text": "no"}
    )
    assert passed is False
    assert "returned text without a valid bounding box" in detail


def test_task_compliance_count_success(gate):
    passed, _ = gate._check_task_compliance(
        query="How many storage tanks are visible?",
        task="vqa",
        result={"text": "There are 4 storage tanks."}
    )
    assert passed is True

    passed_word, _ = gate._check_task_compliance(
        query="How many storage tanks are visible?",
        task="vqa",
        result={"text": "three"}
    )
    assert passed_word is True


def test_task_compliance_count_missing_number(gate):
    passed, detail = gate._check_task_compliance(
        query="How many vehicles are parked?",
        task="vqa",
        result={"text": "Yes, vehicles are parked near the entrance."}
    )
    assert passed is False
    assert "contains no numeric quantity" in detail


def test_output_sanity_repetition_loop(gate):
    degenerate_text = "aerial, aerial, aerial, aerial, aerial, aerial, aerial"
    passed, detail = gate._check_output_sanity(
        task="captioning",
        result={"text": degenerate_text}
    )
    assert passed is False
    assert "Degenerate repetition detected" in detail


def test_output_sanity_normal_text(gate):
    normal_text = "A high-resolution aerial view of an active airport runway with taxiways."
    passed, detail = gate._check_output_sanity(
        task="captioning",
        result={"text": normal_text}
    )
    assert passed is True


def test_output_sanity_degenerate_bbox(gate):
    # Inverted bbox (x1 >= x2)
    passed, detail = gate._check_output_sanity(
        task="grounding",
        result={"spatial_evidence": {"bbox": [100.0, 50.0, 20.0, 80.0]}}
    )
    assert passed is False
    assert "Degenerate bounding box" in detail
    assert "inverted" in detail


def test_target_alignment_vegetation_mismatch(gate, tmp_path):
    # Create a non-vegetated (gray/asphalt) test image
    gray_arr = np.full((100, 100, 3), 128, dtype=np.uint8)
    img_path = str(tmp_path / "gray_scene.png")
    Image.fromarray(gray_arr).save(img_path)

    report = gate.verify(
        query="Mark the grass on the runway",
        task="grounding",
        images={"image": img_path},
        result={
            "task": "grounding",
            "text": "grass",
            "confidence": 0.55,
            "spatial_evidence": {"bbox": [10, 10, 80, 80]},
        }
    )
    # Target alignment should fail because gray area lacks green vegetation signature
    assert report.passed is False
    assert any("vegetation" in r.lower() or "grass" in r.lower() for r in report.reasons)
    assert report.decision == "UNVERIFIED"


def test_geospatial_context_triggers_gee_fallback(gate, tmp_path):
    # Image path containing Sentinel-2 MGRS tile code
    s2_mgrs_path = str(tmp_path / "S2A_MSIL2A_20200615T103031_N0214_R108_T33UUP_20200615T141852.tif")
    # Empty file just for path check
    with open(s2_mgrs_path, "w") as f:
        f.write("dummy")

    report = gate.verify(
        query="Mark the grass",
        task="grounding",
        images={"image": s2_mgrs_path},
        result={
            "task": "grounding",
            "text": "no",  # Fails task compliance
            "confidence": 0.3,
            "spatial_evidence": {},
        }
    )
    assert report.passed is False
    assert report.decision == "FALLBACK_GEE"
    assert report.fallback_source == "Google Earth Engine"
    assert report.gee_evidence is not None
    assert report.gee_evidence["engine"] == "Google Earth Engine (GEE)"


def test_fully_supported_vqa(gate, tmp_path):
    # Valid VQA prediction
    dummy_img = str(tmp_path / "dummy.png")
    Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8)).save(dummy_img)

    report = gate.verify(
        query="What is the color of the roof?",
        task="vqa",
        images={"image": dummy_img},
        result={
            "task": "vqa",
            "text": "The roof is blue.",
            "confidence": 0.85,
            "spatial_evidence": None,
        }
    )
    assert report.passed is True
    assert report.decision == "SUPPORTED"
    assert len(report.reasons) == 0
