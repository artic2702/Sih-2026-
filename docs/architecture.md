# Architecture Notes

## ⚡ Current phase: 2-day ML sprint

The team has 4 people, ML-only work first, and ~2 days. The repository is
organized as **one shared foundation + four independent notebooks**, not
four separate projects and not one shared notebook. See
`docs/team_roles.md` for the full sprint plan; this file explains *why* the
repo looks the way it does.

```text
                    COMMON FOUNDATION (frozen contract)
        configs/config.yaml · src/common/schemas.py · src/common/constants.py
        src/preprocessing/* · src/models/base_model.py
                              │
        ┌──────────────────────┼──────────────────────┬──────────────────┐
        ▼                      ▼                      ▼                  ▼
  01_vqa_geochat.ipynb   02_grounding.ipynb   03_change_vista.ipynb  04_optical_sar_fusion.ipynb
     Person 1                Person 2              Person 3              Person 4
     GeoChat (LoRA)      GeoChat→GeoGround/SAM        VisTA          Optical+SAR fusion head
                              │
                    COMMON OUTPUT (RSModelResult)
                              │
                    ───────── LATER ─────────
                              │
                      src/agent/* (controller, router, registry, trace)
                              │
                          app/app.py (GUI)
```

**Notebooks are workspaces, not the final architecture.** Once a person's
`predict()` works and is evaluated, its stable version moves into the
matching file in `src/models/` (which already exists, pre-wired with
`TODO`s for exactly this).

## Design principles

1. **Separation of orchestration from models.** The controller
   (`src/agent/controller.py`) never contains model-specific code. It only
   knows about `BaseRSModel` / `RSModelResult`. Person 1-4 can change model
   internals (GeoChat prompts, VisTA version, fusion architecture) freely
   without touching agent code, as long as `predict()` still returns a
   contract-compliant dict.

2. **Config-driven, not code-driven.** Checkpoint paths, dataset roots, and
   thresholds all live in `configs/config.yaml`. Critically, preprocessing
   is **per-dataset** (band indices/normalization — Sentinel-2 ≠ Cartosat-2S,
   Sentinel-1 ≠ RISAT) and **per-model** (`image_size` — GeoChat, VisTA, and
   the fusion encoders each expect their own input size), not one global
   assumption. See `src/preprocessing/normalize.py::preprocess_pipeline`.

3. **One frozen output contract.** `src/common/schemas.py::RSModelResult`
   is the single shape every specialist model returns:
   `{task, text, confidence, spatial_evidence: {type, source, bbox, mask},
   metadata: {model, backbone, checkpoint, dataset, input_modalities,
   parameters}, status, inference_seconds}`. `source` on the evidence
   disambiguates which image/coordinate frame it belongs to (T1 vs T2,
   optical vs SAR, etc.) — without it a bbox from change detection is
   ambiguous. This is what lets four independently built notebooks merge
   into one agent without rewrites — the agent only ever needs to know
   this one shape, never GeoChat/VisTA/fusion internals.

3b. **Sensor domain-gap seam.** Development data is Sentinel-1/2; the
   hidden ISRO/SAC judging set is Cartosat-2S/RISAT — a different sensor
   pair with a different radiometric range. `src/preprocessing/sensor_registry.py`
   dispatches to per-sensor adapters (`sensor_adapters/`); the Cartosat/RISAT
   adapters are deliberate `NotImplementedError` stubs rather than guessed
   constants, so the gap fails loudly instead of silently. See `docs/DECISIONS.md`.

3c. **Dataset roles are enforced, not just documented.** `src/common/constants.py::DATASET_ROLES`
   marks VRSBench/RSVQA/CDVQA/ISRO-SAC as eval-only;
   `dataset_loader.get_dataloader(..., split="train", ...)` raises if asked
   to train on one of them. See `docs/DATASETS.md`.

4. **Get inference working before training.** For every workstream: load
   the pretrained backbone and get a baseline prediction out *before*
   attempting any fine-tuning. Only Person 1 (GeoChat LoRA on
   BigEarthNet.txt, mandatory RS adaptation) and Person 4 (fusion head on
   BigEarthNet-MM) actually need to train something this sprint. Person 2
   (grounding) and Person 3 (change) should ship pretrained-backbone
   inference first and only add training as a stretch goal.

5. **Everything is testable without trained weights.** `AgentController.run()`
   currently returns a stub (`tool._empty_result(...)`) instead of calling
   `tool.predict()`. This lets the GUI, tracing, and integration tests be
   built and demoed before any model finishes training. Swapping in the
   real `tool.predict(**params)` call is a one-line change flagged with a
   `TODO` in `controller.py`.

6. **Auditable by construction.** `ExecutionTrace` is populated at every
   pipeline stage (input mode, task chosen, tool used, parameters,
   confidence, warnings) because the problem statement explicitly states
   this trace is what gets evaluated, not internal reasoning text.

## Why this task decomposition (task -> model mapping)

| Input mode | Possible tasks | Model | Backbone |
|---|---|---|---|
| single | vqa (always) | `vqa_model.py` | GeoChat |
| single | captioning OR grounding | `captioning_model.py` / `grounding_model.py` | GeoChat (shared) / GeoChat→GeoGround/SAM |
| bi_temporal | change | `change_model.py` | VisTA |
| cross_modal | fusion | `fusion_model.py` | Optical+SAR feature fusion |

The router (`task_router.py`) uses input mode as a hard constraint (a
bi-temporal pair can only trigger "change", a cross-modal pair only
"fusion") and keyword/LLM classification only to disambiguate among the
single-image tasks. This keeps the agentic behavior simple and auditable
rather than a black box.

## Extending the system

- **Adding a new specialist model**: subclass `BaseRSModel`, have
  `predict()` return `RSModelResult(...).to_dict()`, register it in
  `ToolRegistry._MODEL_CLASSES` and `_build_registry()`, add its checkpoint
  path to `config.yaml`, and (if it's a genuinely new task) add the task
  name to `src/common/constants.py::ALL_TASKS` and teach
  `task_router.route_task` to select it.
- **Adding a new dataset**: add a `datasets.<name>` block to `config.yaml`
  (with its own band indices/normalization — never assume another
  dataset's sensor characteristics apply) and a loader branch in
  `RSDataset._load_index()` / `__getitem__` keyed by `dataset_name`.
  Training/inference code doesn't change.
- **Swapping a backbone**: change `models.<task>.backbone` in config and
  the corresponding `load()`/`predict()` implementation — the agent layer
  and every other person's model are unaffected.

## Known gaps to resolve as a team early in the sprint

- Exact BigEarthNet.txt schema (confirm before writing the parser in
  `src/preprocessing/bigearthnet_adapter.py`).
- GeoChat's actual expected input size / coordinate convention for
  grounding output (normalized `[0,1]` vs. pixel space) — confirm
  empirically in notebook 01/02 and update `configs/config.yaml`.
- VisTA's actual expected input size and checkpoint source.
- Cartosat-2S / RISAT band layout for the `isro_sac` dataset block in
  `config.yaml` — currently placeholder values marked `TODO`.
- Whether GeoChat grounding is good enough on its own, or the GeoGround/SAM
  fallback is needed (Person 2's decision point, notebook 02 Section 8).
