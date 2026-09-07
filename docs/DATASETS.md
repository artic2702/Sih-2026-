# Datasets

## Roles — frozen, enforced in code

This table is the human-readable mirror of
`src/common/constants.py::DATASET_ROLES`, which is the actual source of
truth and is checked at runtime by `src/preprocessing/dataset_loader.get_dataloader`
(requesting `split="train"` on a `usable_for_training: False` dataset raises
immediately). Don't let a teammate arbitrarily repurpose an eval-only
dataset for training under deadline pressure — if a stretch fine-tune is
genuinely needed (e.g. CDVQA), flip the flag deliberately in
`constants.py` and record why in `docs/DECISIONS.md`.

| Dataset | Purpose | `usable_for_training` | Owner |
|---|---|---|---|
| BigEarthNet.txt / reBEN | RS domain adaptation (image↔text) for GeoChat | ✅ | Person 1 |
| BigEarthNet-MM | Optical-SAR fusion development + training | ✅ | Person 4 |
| VRSBench | VQA / captioning / grounding **evaluation** | ❌ eval-only | Person 1 & 2 |
| RSVQA | VQA **evaluation** | ❌ eval-only | Person 1 |
| CDVQA | Change-VQA **evaluation** (+ optional stretch fine-tune) | ⚠️ stretch-only, deliberate flip | Person 3 |
| ISRO/SAC | Final hidden evaluation (Cartosat-2S optical + RISAT SAR) | ❌ never seen during dev | — |

## Sensors

Development/training data is Sentinel-1 (SAR) / Sentinel-2 (optical) based.
The hidden ISRO/SAC judging set uses Cartosat-2S (optical) + RISAT (SAR) —
a different sensor pair. See `docs/DECISIONS.md` for the full rationale and
`src/preprocessing/sensor_registry.py` / `sensor_adapters/` for the seam
that keeps this from becoming a silent failure at judging time.

| Dataset | Optical sensor | SAR sensor |
|---|---|---|
| BigEarthNet | Sentinel-2 | — |
| BigEarthNet-MM | Sentinel-2 | Sentinel-1 |
| VRSBench / RSVQA / CDVQA | unknown (benchmark PNG/JPEG, no georeferencing) | — |
| ISRO/SAC | Cartosat-2S | RISAT |

## `00_data_validation.ipynb` — run before the four specialist notebooks

Before any person starts their notebook, one shared pass confirms the
pipeline can actually read every required dataset:

- **BigEarthNet**: image, labels, text, dimensions
- **BigEarthNet-MM**: optical, SAR, pairing/co-registration
- **VRSBench**: image, question, answer, grounding annotation, caption
- **RSVQA**: image, question, answer
- **CDVQA**: T1, T2, question, answer, change annotation

This notebook answers exactly one question: *can our pipeline actually
read every required dataset?* — cheap insurance against discovering a
format-parsing bug on day 2.

## Evaluation JSON schema

Each person's notebook writes exactly one file in its last cell:
`src/evaluation/results/<person>.json`

```json
{
  "model": "GeoChat+LoRA",
  "dataset": "RSVQA",
  "metric": "accuracy",
  "score": 0.71,
  "n_samples": 40,
  "errors": [
    {"query": "...", "prediction": "...", "ground_truth": "...", "confidence": 0.4}
  ]
}
```

`errors` should hold enough examples for error analysis — don't only report
an aggregate score. `src/evaluation/generate_report.py` globs all four
files into one markdown table.

For the fusion ablation specifically, write three such JSON files (one per
leg: optical-only, SAR-only, optical+SAR) so `generate_report.py` renders
them side by side.
