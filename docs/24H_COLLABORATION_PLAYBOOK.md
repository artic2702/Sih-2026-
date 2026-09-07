# SatQuery AI — 24-Hour Team Collaboration Playbook

> **Target**: Final Project Submission & Live Demo within 24 Hours  
> **Hardware Constraint**: 8 GB VRAM per machine  
> **Golden Rule**: Zero Crashes + Working Baseline First. Everyone sticks to their assigned files and adheres strictly to the frozen `RSModelResult` schema.

---

## 1. Executive Summary & Strategy

With 24 hours remaining and 8 GB VRAM:
1. **Do not attempt heavy training from scratch** on large vision-language models.
2. **Use Pretrained RS Models / Inference-First** for VQA, Grounding, and Change Detection.
3. **Only train the Optical–SAR Fusion head** (Dual ResNet-18 is lightweight and trains in 15–20 minutes on an 8 GB GPU).
4. **All 4 workstreams run in parallel**. Nobody blocks anyone else as long as everyone satisfies the `predict()` return contract.

---

## 2. Team Division & File Ownership

| Member | Workstream / Task | Backbone Architecture | Files Owned (Only Edit These!) | Compute / VRAM Footprint |
| :--- | :--- | :--- | :--- | :--- |
| **Person 1** | **Single-Image VQA & Captioning** | **Qwen2-VL-2B-Instruct** (fp16/bf16)<br>*(Fallback: BLIP-VQA / Captioning)* | • `src/models/vqa_model.py`<br>• `src/models/captioning_model.py`<br>• `notebooks/01_vqa_geochat.ipynb` | ~4.5 GB VRAM (RTX 4060 GPU)<br>~1.2 GB VRAM (BLIP fallback) |
| **Person 2** | **Text-Guided Region Grounding** | **Qwen2-VL Coordinate Grounding**<br>*(Fallback: Saliency / SAM)* | • `src/models/grounding_model.py`<br>• `notebooks/02_grounding.ipynb` | ~1.5 GB VRAM<br>(Zero training required) |
| **Person 3** | **Bi-Temporal Change Detection** | **Siamese Difference + Otsu**<br>*(Optional: Pretrained VisTA)* | • `src/models/change_model.py`<br>• `notebooks/03_change_vista.ipynb` | < 500 MB VRAM / CPU<br>(Zero training required) |
| **Person 4** | **Optical–SAR Fusion & UI Integration** | **Dual ResNet-18** (4-band Optical + 2-band SAR) + MLP Head | • `src/models/fusion_model.py`<br>• `notebooks/04_optical_sar_fusion.ipynb`<br>• `train_fusion.py`<br>• `src/agent/tool_registry.py`<br>• `app/app.py` | ~1.8 GB VRAM<br>(Train locally for 15-20 min) |

---

## 3. Individual Workstream Playbooks

### 👤 Person 1: Single-Image VQA & Captioning
* **Mission**: Make `predict(image, query)` for VQA and `predict(image)` for captioning return real text without raising `NotImplementedError`.
* **Approach**:
  1. **Primary**: Load **`Qwen/Qwen2-VL-2B-Instruct`** in fp16 on GPU (`Qwen2VLForConditionalGeneration` + `AutoProcessor`). Generates articulate multi-sentence answers, complex spatial reasoning, and detailed remote-sensing descriptions.
  2. **Emergency Fallback (Fastest & Safest)**: Automatic fallback to `Salesforce/blip-vqa-base` and `Salesforce/blip-image-captioning-base` if offline or downloading.
* **Return Contract**:
  ```python
  from src.common.schemas import RSModelResult

  return RSModelResult(
      task="vqa",  # or "captioning"
      text=answer_string,
      confidence=0.88,
      status="success",
      metadata={"model": "vqa_v1", "backbone": "Qwen2-VL-2B"},
      inference_seconds=elapsed_time
  ).to_dict()
  ```

---

### 👤 Person 2: Visual Grounding
* **Mission**: Given an image and a text prompt (e.g. *"localise the airstrip"* or *"water body"*), return spatial bounding-box coordinates.
* **Approach**:
  1. `src/models/grounding_model.py` **already contains coordinate parsing logic** for `[ymin, xmin, ymax, xmax]`!
  2. Either prompt GeoChat for normalized coordinates or use a lightweight zero-shot detector like `google/owlvit-base-patch32` (~1.2 GB VRAM).
  3. Scale the bounding box to image dimensions `[ymin, xmin, ymax, xmax]`.
* **Return Contract**:
  ```python
  from src.common.schemas import RSModelResult, SpatialEvidence

  return RSModelResult(
      task="grounding",
      text=f"Detected target at coordinates {scaled_box}.",
      confidence=0.85,
      spatial_evidence=SpatialEvidence(
          type="bbox",
          source="image",
          bbox=scaled_box,  # [ymin, xmin, ymax, xmax]
          mask=None
      ),
      status="success",
      inference_seconds=elapsed_time
  ).to_dict()
  ```

---

### 👤 Person 3: Bi-Temporal Change Detection
* **Mission**: Compare `image_t1` (before) and `image_t2` (after) with a query and output a change explanation plus a binary change mask.
* **Approach**:
  1. **You already have a working solution!** In `src/models/change_model.py`, the `siamese_difference` fallback is **fully written**:
     - Computes normalized Euclidean difference between T1 and T2.
     - Uses Otsu auto-thresholding to isolate change pixels.
     - Filters tiny noise using morphological operations.
  2. Simply ensure `backend: "siamese_difference"` is used in your test/evaluation.
  3. Evaluates on CDVQA pairs; runs in < 50ms with virtually 0 VRAM!
* **Return Contract**:
  ```python
  from src.common.schemas import RSModelResult, SpatialEvidence

  return RSModelResult(
      task="change",
      text="Significant urban expansion and vegetation reduction detected between T1 and T2.",
      confidence=0.91,
      spatial_evidence=SpatialEvidence(
          type="mask",
          source="image_t2",
          bbox=None,
          mask=change_mask_2d  # np.ndarray of shape (H, W), dtype uint8 or bool
      ),
      status="success",
      inference_seconds=elapsed_time
  ).to_dict()
  ```

---

### 👤 Person 4: Optical–SAR Fusion & Agent/App Integration
* **Mission**: Train the dual-encoder fusion head, generate the 3-way ablation table, and wire all specialist models to the Streamlit app.
* **Approach**:
  1. `src/models/fusion_model.py` is **fully coded** (Dual ResNet-18 for Optical 4-band + SAR 2-band).
  2. Run `python src/training/train_fusion.py` locally:
     - `batch_size = 32` (uses ~1.8 GB VRAM)
     - Train for 3–5 epochs (takes only 15–20 minutes on an RTX 3050/3060/4060).
     - Saves checkpoint to `models/checkpoints/fusion/fusion_head/model.pt`.
  3. Produce the **3-Way Ablation Table** (Optical-only vs. SAR-only vs. Optical+SAR Fused). This is a top-priority evaluation requirement!
  4. In `src/agent/tool_registry.py`: Connect each model's real class instance so the Streamlit UI runs real inferences.

---

## 4. The Frozen Contract: Do Not Break This

To merge all code cleanly at Hour 18 without conflicts:
1. **NEVER edit `src/common/schemas.py` or `src/common/constants.py`**.
2. **Always return `RSModelResult(...).to_dict()`** or `self._empty_result(...)`.
   - Top-level keys: `task`, `text`, `confidence`, `spatial_evidence`, `metadata`, `status`, `inference_seconds`.
   - Evidence keys: `type`, `source`, `bbox`, `mask`.
3. Test your model continuously against the smoke test:
   ```powershell
   python -m scripts.smoke_test
   ```

---

## 5. 8 GB VRAM Memory Optimization Rules

To prevent CUDA Out-of-Memory (OOM) crashes:
1. **Lazy Loading**: Do not load all 4 models into GPU memory at once. Load the active model on demand, or use CPU for lightweight preprocessing.
2. **Clear CUDA Cache**: Between model runs or notebook cells:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```
3. **Inference Precision**: Always run inference in half precision (`torch.float16`) or 4-bit (`load_in_4bit=True`).
4. **Batch Size**: Keep `batch_size = 1` for all inference and interactive demo runs.

---

## 6. Git Workflow (No Conflicts)

1. Everyone branches off `main`:
   ```bash
   git checkout -b feature/person1-vqa
   git checkout -b feature/person2-grounding
   git checkout -b feature/person3-change
   git checkout -b feature/person4-fusion-app
   ```
2. Each member only commits changes to their assigned files.
3. At Hour 16, Person 4 creates a PR/merge into `main`.
4. Everyone runs `python -m scripts.smoke_test` to verify 5/5 `[PASS]`.

---

## 7. Hour-by-Hour 24-Hour Timeline

```
Hour 00 - 02: Environment setup & package installation (torch, torchvision, transformers, streamlit).
Hour 02 - 08: Each member implements their model predict() method in their assigned file.
Hour 08 - 12: Person 4 trains Dual ResNet-18 locally; Person 1 verifies VQA output; Person 3 runs change mask.
Hour 12 - 16: Fill in notebook outputs (01, 02, 03, 04) and generate the 3-way ablation table.
Hour 16 - 18: Merge all branches into main.
Hour 18 - 20: Run smoke test (python -m scripts.smoke_test) and verify Streamlit UI (streamlit run app/app.py).
Hour 20 - 22: Prepare 4-5 test satellite images/pairs for the live demonstration.
Hour 22 - 24: Final rehearsal, zip codebase/documentation, ready for submission!
```
