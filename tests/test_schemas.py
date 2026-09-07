"""
Contract tests for the shared output schema. Owner: shared — run this before
integrating anyone's model to make sure their predict() output still
matches the frozen contract.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.common.schemas import RSModelResult, SpatialEvidence, ResultMetadata, empty_result


def test_rsmodelresult_to_dict_shape():
    result = RSModelResult(
        task="vqa",
        text="This is a cropland area.",
        confidence=0.9,
        spatial_evidence=SpatialEvidence(type="bbox", source="image", bbox=[1, 2, 3, 4]),
        metadata=ResultMetadata(model="vqa_v1", backbone="GeoChat", checkpoint="ckpt.pt",
                                 dataset="RSVQA", input_modalities=["optical"]),
        status="success",
    ).to_dict()

    assert set(result.keys()) == {
        "task", "text", "confidence", "spatial_evidence", "metadata",
        "status", "inference_seconds",
    }
    assert set(result["spatial_evidence"].keys()) == {"type", "source", "bbox", "mask"}
    assert set(result["metadata"].keys()) == {
        "model", "backbone", "checkpoint", "dataset", "input_modalities", "parameters",
    }
    assert result["spatial_evidence"]["bbox"] == [1, 2, 3, 4]
    assert result["spatial_evidence"]["source"] == "image"
    assert result["metadata"]["backbone"] == "GeoChat"


def test_rsmodelresult_preserves_numpy_mask():
    mask = np.zeros((4, 4), dtype=np.uint8)
    result = RSModelResult(
        task="change", text="Built-up area increased.", confidence=0.7,
        spatial_evidence=SpatialEvidence(type="mask", source="image_t2", mask=mask),
    ).to_dict()
    assert isinstance(result["spatial_evidence"]["mask"], np.ndarray)
    assert result["spatial_evidence"]["mask"].shape == (4, 4)
    assert result["spatial_evidence"]["source"] == "image_t2"


def test_empty_result_matches_contract():
    result = empty_result(task="fusion", text="stub", confidence=0.0,
                           model="fusion_v1", checkpoint="", dataset="",
                           status="error")
    assert result["task"] == "fusion"
    assert result["text"] == "stub"
    assert result["confidence"] == 0.0
    assert result["status"] == "error"
    assert result["spatial_evidence"]["bbox"] is None
    assert result["spatial_evidence"]["mask"] is None


def test_dataset_roles_guardrail():
    from src.common.constants import assert_usable_for_training

    # eval-only dataset must raise
    try:
        assert_usable_for_training("vrsbench")
        assert False, "expected ValueError for eval-only dataset"
    except ValueError:
        pass

    # training dataset must not raise
    assert_usable_for_training("bigearthnet")
