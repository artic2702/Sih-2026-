# SatQuery AI — Partial Audit/Implementation Status

**This is an in-progress pass, stopped early at your request.** Below is exactly
what was reviewed, what was changed, what was verified by running it, and —
importantly — what is still untouched. Treat the "Not yet done" section as
the real TODO list, not a minor footnote.

The updated repo is in `satquery-ai-updated.zip`. Unzip it over (or in place
of) your original `satquery-ai/` folder.

---

## 1. What was audited

Read in full: `README.md`, all five `docs/*.md` files, `configs/config.yaml`,
every file under `src/common`, `src/models`, `src/preprocessing` (incl. all
four sensor adapters), `src/training`, `src/evaluation`, `src/agent`,
`src/utils`, all three test files, `requirements.txt`, `scripts/smoke_test.py`,
and the full contents of all five notebooks (cell by cell).

**Baseline before any changes:** 14/14 existing pytest tests passed. The
architecture (frozen `RSModelResult` contract, `DATASET_ROLES` guardrail,
sensor-adapter seam, execution trace, tool registry) was already
well-designed — that part was preserved exactly as instructed, not rewritten.

## 2. Bugs found

1. **`trainer_utils.py` called `model.parameters()` / `model.state_dict()`
   directly on a `BaseRSModel` instance.** `BaseRSModel` is a plain wrapper,
   not an `nn.Module` — every training script would have crashed with
   `AttributeError` the moment a model had real internals.
2. **All four notebooks' "export" cells import `Evidence` and use an
   `answer=` field** — the actual frozen schema is `SpatialEvidence` and
   `text=`. Commented-out today, so nothing crashes yet, but it's a live
   landmine for whoever uncomments it first. **Not yet fixed** (see below).
3. **Notebook 04 hardcodes `assert fusion_score >= optical_only_score`**,
   directly contradicting the explicit "never force fusion to beat optical,
   report honestly" rule stated in `fusion_model.py`'s own docstring and
   `docs/DECISIONS.md`. **Not yet fixed in the notebook** (the rule is
   correctly implemented in the new `metrics.fusion_ablation()`, which
   contains no such assertion).
4. **`dataset_loader.py`'s `get_dataloader()` always instantiated one
   generic `RSDataset` class** regardless of which dataset was requested —
   there was no actual per-benchmark reading logic despite the config and
   every training script assuming one existed.
5. **`evaluate.py`'s `TASK_METRIC_FN` had no `"fusion"` entry** even though
   `--task fusion` was a valid CLI choice — running it raised `ValueError`
   immediately. Its eval loop body was also literally `pass` (never
   collected any predictions).
6. **`metrics.py` was missing `fusion_ablation()`**, which `docs/
   api_contracts.md` §6 documents as existing.
7. **Coordinate-parsing bug caught by testing, not inspection**: the
   original bbox-parsing regex idea (`[-+]?\d*\.?\d+`) matches the "1" and
   "2" inside labels like `x1=12, y2=178`, silently producing a completely
   wrong box. Fixed with a negative lookbehind in the new
   `grounding_model.py` and confirmed fixed by re-running the test.
8. Minor: `docs/DATASETS.md` refers to a notebook named
   `00_data_validation.ipynb`; the actual file is
   `00_data_exploration.ipynb`. **Not yet fixed.**

## 3. What was actually implemented and tested this pass

Everything below was not just written but **executed in a real Python/
PyTorch environment** (torch 2.14 CPU, torchvision 0.29) with targeted test
scripts — not just read through.

| File | What changed |
|---|---|
| `src/models/base_model.py` | Added `torch_module()` hook so training code can get the real `nn.Module` out of any `BaseRSModel` subclass. |
| `src/training/trainer_utils.py` | Rewritten: fixes bug #1 above; adds AMP context/scaler helpers, LR scheduler, checkpoint resume, structured checkpoint metadata (model/dataset/split/epoch/seed/git-commit/config snapshot), best-checkpoint tracking, `EarlyStopper`. |
| `src/preprocessing/normalize.py` | `select_bands`, `normalize` (5 real methods: per-band minmax, percentile 2–98, dB-scale SAR, ImageNet, z-score), `resize`, `tile_image` — all previously `NotImplementedError`. |
| `src/preprocessing/geotiff_utils.py` | Added real PNG/JPEG reading and a real `check_coregistration()` (CRS/resolution/bounds-overlap check, with a documented shape-only fallback for non-georeferenced benchmark pairs). |
| `src/preprocessing/bigearthnet_adapter.py` | Real `BigEarthNet.txt` parser (tab- or comma-delimited, explicitly documented as an *assumed* format since the real file wasn't available to confirm against) + varied-template caption generation. |
| `src/preprocessing/dataset_loader.py` | Added the missing per-dataset dispatch: `BigEarthNetDataset`, `BigEarthNetMMDataset`, `VRSBenchDataset`, `RSVQADataset`, `CDVQADataset`, each with its assumed annotation schema documented in its own docstring, plus a real collate function. |
| `src/evaluation/metrics.py` | Real `caption_scores` (from-scratch BLEU-4/ROUGE-L, no nltk download needed), `change_f1` (pixel P/R/F1/kappa/IoU), `multilabel_f1`, and the previously-missing `fusion_ablation`. |
| `src/evaluation/evaluate.py` | Real prediction-collection loop per task (was `pass`); added the missing `fusion` route. |
| `src/models/fusion_model.py` | Full dual-ResNet-encoder + trainable fusion head implementation, `predict`/`predict_optical_only`/`predict_sar_only` for the mandatory 3-way ablation, templated text generation, `train_step`. **Verified end-to-end**, including confirming the sandbox's blocked ImageNet-weight download correctly triggers the documented random-init fallback rather than failing silently. |
| `src/models/change_model.py` | Real, documented VisTA-load attempt (honest about why it can't resolve here — VisTA is a git-clone research repo, not pip-installable) + a fully implemented, tested siamese-difference fallback (Otsu auto-thresholding, region localization, magnitude-based confidence). |
| `src/models/grounding_model.py` | Real, tested coordinate parsing/conversion (`parse_bbox_from_text`, `clip_bbox`, `rescale_bbox`, `visualize_bbox`) + GeoChat-reuse primary path + documented SAM-fallback dependency. |

## 4. Not yet done

This is the important part — the pass was stopped partway through the plan:

- **`src/models/vqa_model.py` — untouched.** This is the mandatory P1 piece
  (GeoChat + LoRA, the "remote-sensing adaptation" requirement) and it's
  still the original stub. Nothing else in this list matters for the demo
  until this exists, since `grounding_model.py` and `captioning_model.py`
  both depend on it.
- **`src/models/captioning_model.py` — untouched** (still a stub).
- **All five `src/training/train_*.py` scripts — untouched.** They're still
  the identical copy-paste templates with no `dataset=` argument passed to
  `get_dataloader`, no validation loop, no use of the new AMP/resume/
  early-stopping helpers in `trainer_utils.py`.
- **The three schema-drift/assert bugs in the notebooks (items 2 and 3
  above) — not fixed.** The notebooks still have the stale `Evidence`/
  `answer=` example code and the hardcoded `assert fusion >= optical`.
- **No `docs/ML_AUDIT.md` or `docs/MODEL_STATUS.md`** — the gap-analysis
  and model-readiness tables you asked for were never written as files.
- **No `scripts/check_environment.py` or `docs/COLAB.md`.**
- **No new tests added** for any of the newly-implemented code (fusion,
  change, grounding, metrics, dataset_loader, normalize, geotiff_utils) —
  everything above was verified with ad-hoc scripts in this session, not
  committed as `pytest` tests the team can rerun.
- **`requirements.txt`, `docs/DECISIONS.md`, `docs/api_contracts.md`,
  `docs/architecture.md`, `docs/team_roles.md` — untouched** (the
  grounding-fallback wording mismatch noted in the code comments was not
  reconciled in the docs themselves).
- The final integration smoke test (`scripts/smoke_test.py`) was not rerun
  against the updated model files.

## 5. Suggested next step

If you resume this, the highest-leverage next piece is `vqa_model.py`
(GeoChat + LoRA via `transformers`/`peft`) — everything else this pass built
(dataset loader, metrics, fusion, change, grounding's GeoChat-reuse path)
is ready to plug into it, but VQA itself is the one mandatory piece still
completely unstarted.
