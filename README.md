# SatQuery AI — Agentic Vision-Language Assistant for Remote Sensing

Team scaffold for the ISRO/SAC problem statement: an agentic VLM system that answers
natural-language queries over single images, optical–SAR pairs, and bi-temporal pairs.

This repo is a **starting skeleton**, not a finished product. Interfaces (function
signatures, config schema, folder boundaries) are fixed so 4 people can build in
parallel without stepping on each other. Fill in the `TODO` blocks.

> **⚡ Running a 2-day ML-only sprint right now?** Read
> [`docs/team_roles.md`](docs/team_roles.md) first — it has the actual
> current plan: 4 notebooks (`notebooks/01_vqa_geochat.ipynb` … `04_optical_sar_fusion.ipynb`),
> one person and one backbone each (GeoChat / GeoChat→GeoGround-SAM / VisTA /
> Optical-SAR fusion), and a frozen input/output contract
> (`src/common/schemas.py`) so the four notebooks merge cleanly later. Skip
> `app/` and `src/agent/` entirely until the ML backend is proven — they
> already work end-to-end with stub responses, so there's nothing urgent
> there. The team split further down this README (Data / Single-image /
> Multi-image / Agent+GUI) is the **longer-project** version for after the
> sprint — see `docs/team_roles.md` for which one applies to you right now.

---

## 1. Architecture Overview

```
                         ┌─────────────────────────────┐
                         │        Frontend / GUI        │
                         │   (app/app.py — Streamlit)   │
                         │  upload images + text query  │
                         └───────────────┬──────────────┘
                                         │
                                         ▼
                         ┌─────────────────────────────┐
                         │      Input Validator         │
                         │ src/agent/input_validator.py │
                         │  - modality / format check   │
                         │  - co-registration check     │
                         │  - single / pair / bitemporal│
                         └───────────────┬──────────────┘
                                         ▼
                         ┌─────────────────────────────┐
                         │      Task Router (LLM)       │
                         │  src/agent/task_router.py    │
                         │  query → task label          │
                         │  (vqa/caption/ground/change/ │
                         │   fusion)                    │
                         └───────────────┬──────────────┘
                                         ▼
                         ┌─────────────────────────────┐
                         │   Tool Registry + Controller │
                         │  src/agent/tool_registry.py  │
                         │  src/agent/controller.py     │
                         │  selects & sequences models  │
                         └──────┬───────┬───────┬───────┘
                                │       │       │
                 ┌──────────────┘       │       └──────────────┐
                 ▼                      ▼                      ▼
        ┌────────────────┐    ┌─────────────────┐    ┌──────────────────┐
        │  VQA / Caption  │    │  Change model    │    │ Optical–SAR      │
        │  / Grounding    │    │  (bi-temporal)   │    │ fusion model     │
        │ src/models/     │    │ src/models/      │    │ src/models/      │
        │ vqa_model.py    │    │ change_model.py  │    │ fusion_model.py  │
        │ captioning_*.py │    │                  │    │                  │
        │ grounding_*.py  │    │                  │    │                  │
        └────────┬────────┘    └────────┬─────────┘    └────────┬─────────┘
                  │                      │                       │
                  └──────────────┬───────┴───────────┬───────────┘
                                 ▼                    ▼
                     ┌────────────────────┐  ┌─────────────────────┐
                     │ Output Integrator   │  │ Execution Trace Log │
                     │ (controller.py)     │  │ execution_trace.py  │
                     │ text + bbox/mask +  │  │ auditable summary:  │
                     │ confidence          │  │ task, model, params │
                     └──────────┬──────────┘  └──────────┬──────────┘
                                 └────────────┬───────────┘
                                              ▼
                                ┌───────────────────────────┐
                                │   Response to Frontend     │
                                │  text + overlay image +    │
                                │  confidence + report (PDF) │
                                └───────────────────────────┘
```

All models are trained/fine-tuned offline (`src/training/`) and loaded by
`src/agent/tool_registry.py` at inference time. The controller never trains
anything — it only orchestrates already-trained tools.

---

## 2. Data Flow (preprocessing)

```
raw imagery (GeoTIFF/TIFF/PNG/JPEG)
   │
   ▼
src/preprocessing/geotiff_utils.py   → reads bands, CRS, resolution, georef
   │
   ▼
src/preprocessing/normalize.py       → band selection, resize, normalize, tiling
   │
   ▼
src/preprocessing/dataset_loader.py  → PyTorch Dataset/DataLoader per task
   │
   ▼
src/preprocessing/bigearthnet_adapter.py → BigEarthNet.txt → pretraining pairs
```

Benchmark-specific loaders (VRSBench, RSVQA, CDVQA) plug into
`dataset_loader.py` behind a common interface so the training scripts don't
care which dataset they're pointed at.

---

## 3. Team Split (4 people)

| # | Role | Owns | Key files |
|---|------|------|-----------|
| 1 | **Data & Preprocessing** | GeoTIFF I/O, band handling, normalization, BigEarthNet adaptation, dataset loaders for VRSBench/RSVQA/CDVQA | `src/preprocessing/*` |
| 2 | **Single-image Models** | VQA fine-tuning (mandatory) + captioning OR grounding (pick one) | `src/models/vqa_model.py`, `captioning_model.py` or `grounding_model.py`, `src/training/train_vqa.py`, `train_captioning.py` |
| 3 | **Multi-image Models** | Bi-temporal change detection/VQA, optical–SAR fusion | `src/models/change_model.py`, `fusion_model.py`, `src/training/train_change.py`, `train_fusion.py` |
| 4 | **Agentic Orchestration + GUI** | Task router, tool registry, controller, execution trace, evaluation harness, Streamlit app | `src/agent/*`, `app/app.py`, `src/evaluation/*` |

Everyone shares: `configs/config.yaml`, `src/utils/*`, and the base
interfaces in `src/models/base_model.py`. **Do not change these interfaces
without telling the team** — everyone else's code depends on them.

See `docs/team_roles.md` for a week-by-week breakdown and
`docs/api_contracts.md` for the exact function signatures each person must
implement so integration doesn't break.

---

## 4. Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Put raw imagery in `data/raw/`, run preprocessing, then:

```bash
python -m src.training.train_vqa --config configs/config.yaml
python -m src.evaluation.evaluate --task vqa --config configs/config.yaml
streamlit run app/app.py
```

## 5. Repo Map

```
satquery-ai/
├── README.md
├── requirements.txt
├── configs/config.yaml          # single source of truth: per-dataset + per-model params
├── data/raw/ , data/processed/  # not committed, .gitkeep only
├── models/checkpoints/          # trained weights land here (not committed)
├── notebooks/
│   ├── 00_data_exploration.ipynb
│   ├── 01_vqa_geochat.ipynb          # Person 1 — GeoChat VQA + captioning
│   ├── 02_grounding.ipynb            # Person 2 — GeoChat grounding → GeoGround/SAM fallback
│   ├── 03_change_vista.ipynb         # Person 3 — VisTA bi-temporal change
│   └── 04_optical_sar_fusion.ipynb   # Person 4 — optical-SAR feature fusion
├── src/
│   ├── common/                   # FROZEN CONTRACT: schemas.py, constants.py
│   ├── preprocessing/            # shared GeoTIFF I/O, per-dataset normalization, dataset loaders
│   ├── models/                   # base_model.py + one file per specialist (Person 1-4)
│   ├── training/                 # reusable training loops (move code here from notebooks once stable)
│   ├── evaluation/               # metrics + benchmark eval harness
│   ├── agent/                    # orchestration — build AFTER the 4 notebooks are stable
│   └── utils/                    # shared logging/config helpers
├── app/                          # GUI — build LAST, already wired to run with stub responses today
├── tests/                        # unit tests, one file per module
└── docs/                         # architecture.md, team_roles.md, api_contracts.md
```

## 6. Ground Rules:

- **Config-driven**: no hardcoded paths/hyperparameters in code — read from `configs/config.yaml`. Preprocessing is per-dataset and per-model, not global (see `docs/architecture.md`).
- **Interfaces first**: every model class inherits `BaseRSModel` (`src/models/base_model.py`) and returns `src.common.schemas.RSModelResult(...).to_dict()` so the controller can call any model the same way. This output shape is frozen — don't hand-roll a different one.
- **Branch per person**: `feature/preprocessing`, `feature/single-image`, `feature/multi-image`, `feature/agent-gui`. Merge to `main` via PR.
- **Execution trace is graded**: the controller must log task, model/tool name, and parameters for every query — see `src/agent/execution_trace.py`.
