"""
scripts/smoke_test.py — the single objective "is ML done" status board.

Write this on day 1, before any model works. It imports all four/five
predict() functions and calls them with dummy/placeholder inputs. Initially
every line prints [FAIL] with a NotImplementedError. As each person
finishes, their line(s) flip to [PASS]. This replaces four people
separately claiming "mine works" with one command anyone can run.

"ML backend complete" (per docs/api_contracts.md) means every line below
says [PASS] AND every result validates against the frozen RSModelResult
schema (checked by _validate_result, not just "didn't crash").

Usage:
    python scripts/smoke_test.py
"""

import sys
import traceback

import numpy as np
import yaml

from src.agent.tool_registry import ToolRegistry
from src.common.constants import ALL_TASKS

REQUIRED_TOP_LEVEL_KEYS = {
    "task", "text", "confidence", "spatial_evidence", "metadata",
    "status", "inference_seconds",
}
REQUIRED_EVIDENCE_KEYS = {"type", "source", "bbox", "mask"}
REQUIRED_METADATA_KEYS = {
    "model", "backbone", "checkpoint", "dataset", "input_modalities", "parameters",
}


def _dummy_image(size=64):
    return np.zeros((4, size, size), dtype=np.float32)


def _validate_result(result: dict) -> list:
    """Returns a list of schema violations (empty = valid)."""
    problems = []
    missing = REQUIRED_TOP_LEVEL_KEYS - set(result.keys())
    if missing:
        problems.append(f"missing top-level keys: {missing}")
        return problems  # can't check nested structure if top-level is wrong

    se = result.get("spatial_evidence") or {}
    if REQUIRED_EVIDENCE_KEYS - set(se.keys()):
        problems.append(f"spatial_evidence missing keys: {REQUIRED_EVIDENCE_KEYS - set(se.keys())}")

    md = result.get("metadata") or {}
    if REQUIRED_METADATA_KEYS - set(md.keys()):
        problems.append(f"metadata missing keys: {REQUIRED_METADATA_KEYS - set(md.keys())}")

    if result.get("status") not in ("success", "error", "low_confidence"):
        problems.append(f"invalid status: {result.get('status')!r}")

    if not (0.0 <= float(result.get("confidence", -1)) <= 1.0):
        problems.append(f"confidence out of [0,1]: {result.get('confidence')}")

    return problems


CHECKS = [
    ("GeoChat VQA",        "vqa",        dict(image=_dummy_image(), query="What land-cover types are visible?")),
    ("GeoChat Captioning", "captioning", dict(image=_dummy_image())),
    ("Grounding",          "grounding",  dict(image=_dummy_image(), query="Highlight the water body.")),
    ("Change (VisTA)",     "change",     dict(image_t1=_dummy_image(), image_t2=_dummy_image(),
                                               query="What changed between these two dates?")),
    ("Fusion",             "fusion",     dict(image_optical=_dummy_image(), image_sar=_dummy_image(),
                                               query="Identify built-up and water-covered regions.")),
]


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    registry = ToolRegistry(config)
    results = []

    for label, task, kwargs in CHECKS:
        try:
            tool = registry.get(task)
            out = tool.predict(**kwargs)
            problems = _validate_result(out)
            if problems:
                results.append((label, "FAIL", "; ".join(problems)))
            else:
                results.append((label, "PASS", f"status={out['status']}"))
        except NotImplementedError:
            results.append((label, "TODO", "predict() not implemented yet"))
        except Exception as e:  # noqa: BLE001 — smoke test wants to report, not crash
            results.append((label, "FAIL", f"{type(e).__name__}: {e}"))

    print()
    for label, status, note in results:
        print(f"[{status:4s}] {label:<22s} — {note}")
    print()

    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    print(f"{n_pass}/{len(results)} passing.")
    if n_pass < len(results):
        print("ML backend is NOT complete yet — see docs/api_contracts.md for the definition.")
        sys.exit(1)
    print("All specialist tools return schema-valid RSModelResult — ML backend complete.")


if __name__ == "__main__":
    main()
