"""
Basic tests for the agentic pipeline. Owner: Person 4, extend as models land.
These tests check wiring/contracts, not model quality (no trained weights yet).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.agent.input_validator import validate_input
from src.agent.task_router import route_task


def test_validate_input_rejects_unknown_keys():
    result = validate_input({"foo": "bar.tif"}, config={})
    assert result["valid"] is False


def test_validate_input_rejects_bad_format():
    result = validate_input({"image": "scene.bmp"}, config={})
    assert result["valid"] is False
    assert any("unsupported format" in e for e in result["errors"])


def test_route_task_bi_temporal_is_change():
    assert route_task("what happened here?", "bi_temporal") == "change"


def test_route_task_cross_modal_is_fusion():
    assert route_task("combine these", "cross_modal") == "fusion"


def test_route_task_single_defaults_to_vqa():
    assert route_task("how many buildings are there?", "single") == "vqa"


def test_route_task_single_grounding_keyword():
    assert route_task("Highlight the water body referred to in the query.", "single") == "grounding"


def test_route_task_single_captioning_keyword():
    assert route_task("Describe the land-cover in this image.", "single") == "captioning"


def test_base_model_empty_result_contract():
    from src.models.vqa_model import VQAModel
    model = VQAModel(config={})
    result = model._empty_result(text="stub answer", confidence=0.1)
    assert result["task"] == "vqa"
    assert result["text"] == "stub answer"
    assert result["metadata"]["model"] == "vqa_v1"
