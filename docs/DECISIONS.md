# Design Decisions

Cheap to write (10 minutes), and exactly what a judge asks about — having
this written down before the demo beats improvising on the spot. Update it
as decisions actually get made; this is a living log, not a one-time form.

## Why GeoChat for VQA/captioning/grounding?

It is already remote-sensing-adapted (not a generic VLM), and the problem
statement explicitly disallows a generic LLM/VLM without RS adaptation.
Using one backbone for three of the five capabilities (VQA, captioning,
grounding) keeps the sprint to two real training workloads instead of five,
and avoids burning time integrating a second VLM unless GeoChat
underperforms on a specific task.

## Why VisTA for change understanding?

VisTA is chosen specifically because it can produce both a change
description/answer *and* a spatial change mask in one pass, which gives the
strongest visual evidence for the agent's evidence-grounded response
requirement, without needing a second model bolted on for the mask.
**Risk flag:** VisTA is less mainstream than GeoChat/SAM — obtainability
(pretrained weights, working repo/environment) must be confirmed on day 1,
before anything else is built around the assumption it works. Fallback if
unobtainable: a simple siamese-difference approach, or prompting GeoChat on
T1 and T2 separately plus a third "what's different" prompt.

## Why Path A (dual-encoder feature fusion) over Path B (CROMA/DOFA/SkySense) for optical-SAR?

Path A — separate pretrained optical/SAR encoders + a small trainable
fusion head — is the practical choice for a short deadline: it doesn't
require integrating a new large multimodal foundation model, and it's the
only optical-SAR slot with a mandatory training component (the fusion
head), which the problem statement's RS-adaptation requirement expects.
CROMA/DOFA/SkySense remain a stronger stretch upgrade (Path B) only if time
remains and Path A proves fragile — not attempted this sprint.

## Why fusion outputs a classification (not open-ended text) with a templated language wrapper?

The fusion slot has no pretrained VLM backbone to lean on, unlike the other
three slots. Asking it to generate open-ended natural language directly
from fused features is needlessly ambitious and hard to evaluate credibly
in the time available. Multi-label land-cover classification (matching
BigEarthNet-MM's taxonomy) is a well-defined, trainable, evaluable task; a
thin template (`"The area shows significant {top_label}..."`) turns that
into the evidence-grounded natural-language response the problem statement
asks for, without requiring fusion itself to be a language model.

## Why LoRA, not full fine-tuning, for GeoChat adaptation?

Full fine-tuning of a VLM in a short sprint is both compute- and
time-infeasible, and risks catastrophic forgetting of GeoChat's existing
RS-adapted capabilities. LoRA gives a reproducible, fast, low-VRAM
adaptation path that satisfies the "at least one component fine-tuned or
adapted" mandatory requirement while leaving the base model's general
competence intact.

## Why GeoChat-first for grounding, with SAM (not GeoGround) as the fallback?

Training a dedicated grounding model by default is unnecessary work if
GeoChat's own grounding capability is good enough — GeoChat is already
loaded for VQA/captioning, so trying it first is nearly free. SAM is
preferred over GeoGround as the fallback because it's a well-packaged,
reliable, widely-used off-the-shelf segmentation tool given a seed
point/box, whereas GeoGround is another unfamiliar repository to debug
under deadline pressure. The escalation threshold (mean IoU ≥ 0.30 on a
VRSBench subset) is fixed *before* seeing results specifically to avoid
rationalizing a borderline number as "probably fine" under time pressure.

## Why is BigEarthNet.txt training-only and VRSBench/RSVQA/CDVQA eval-only?

This matches the problem statement's explicit dataset roles: BigEarthNet.txt
is the primary adaptation dataset; VRSBench/RSVQA/CDVQA are prescribed
benchmark evaluation sets. Training on an evaluation-only set would
invalidate the reported metrics as a measure of generalization. This is
enforced in code (`src/common/constants.py::DATASET_ROLES`,
checked by `dataset_loader.get_dataloader`), not just documented, because a
documentation-only norm is exactly the kind of thing that gets silently
violated under deadline pressure.

## Why does a Cartosat-2S/RISAT sensor-adapter seam exist with no real data yet?

Development/training data is entirely Sentinel-1/2 based. The hidden
ISRO/SAC judging set is Cartosat-2S (optical) + RISAT (SAR) — a different
sensor pair, almost certainly with a different radiometric range, band
count/order, and resolution. If Sentinel assumptions are hardcoded into
shared preprocessing, Cartosat/RISAT imagery gets silently mis-normalized
and every downstream model receives garbage input with no error thrown —
the worst failure mode, because it looks like it's working right up until
judging. Building the seam (`src/preprocessing/sensor_registry.py` +
`sensor_adapters/cartosat2s.py` / `risat.py`, both deliberate
`NotImplementedError` stubs) costs about 30 minutes and means that if/when
sample data appears, one file gets filled in instead of four notebooks
getting audited for hardcoded Sentinel-2 constants.

## Known, deliberately out-of-scope gaps (for this sprint)

- **SAR speckle filtering** — not applied; the seam exists in
  `sensor_adapters/sentinel1.py::normalize(speckle_filter=...)` but raises
  if requested. Guessing filter constants without ground truth wastes time.
- **Cartosat-2S / RISAT radiometric calibration constants** — unconfirmed,
  deliberately left as `NotImplementedError` rather than guessed.
- **VisTA fine-tuning on CDVQA** — stretch goal only, attempted only if all
  four mandatory deliverables are otherwise complete.
- **Fusion segmentation head (spatial masks)** — stretch goal (P1.5), only
  after the classification path works end-to-end.
- **Agent/GUI/FastAPI/deployment** — Layer C, explicitly deferred until all
  four Layer B specialists pass `scripts/smoke_test.py`.
