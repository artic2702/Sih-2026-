# 🛰️ SatQuery AI — REST API Documentation

SatQuery AI exposes high-performance RESTful API endpoints for autonomous satellite intelligence, multi-modal vision-language reasoning, referring spatial grounding, bi-temporal change detection, and Optical-SAR radar fusion.

---

## 🚀 Quickstart

### 1. Launch the API Server
Start the server using either the included runner script or Uvicorn directly:

```bash
# Option A: Using the launcher script
python scripts/run_api_server.py --port 8000 --reload

# Option B: Direct Uvicorn command
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Interactive Documentation UIs
Once started, open your web browser:
* **Interactive Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs) (allows testing every endpoint directly in the browser)
* **ReDoc OpenAPI Documentation:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **API Root:** [http://localhost:8000/](http://localhost:8000/) (automatically redirects to `/docs`)

---

## 📋 Endpoint Overview

| Category | Method | Endpoint | Description |
| :--- | :--- | :--- | :--- |
| **System** | `GET` | `/health` | Service health status, GPU/CPU device, and loaded models |
| **System** | `GET` | `/api/v1/models` | List all specialist models, backbones, and statuses |
| **Agent** | `POST` | `/api/v1/agent/query` | **Unified Agent**: Auto-routes query, runs specialist, and verifies evidence (JSON) |
| **Agent** | `POST` | `/api/v1/agent/upload-query` | **Unified Agent**: Multipart binary file upload + automated routing |
| **VQA** | `POST` | `/api/v1/models/vqa` | Visual Question Answering using `Qwen2-VL-2B` (JSON / path / base64) |
| **VQA** | `POST` | `/api/v1/models/vqa/upload` | Visual Question Answering via multipart image upload |
| **Captioning** | `POST` | `/api/v1/models/captioning` | Comprehensive scene description & captioning (JSON / path / base64) |
| **Captioning** | `POST` | `/api/v1/models/captioning/upload` | Comprehensive scene captioning via multipart image upload |
| **Grounding** | `POST` | `/api/v1/models/grounding` | Spatial referring grounding: returns `[x1, y1, x2, y2]` bounding box (JSON) |
| **Grounding** | `POST` | `/api/v1/models/grounding/upload` | Spatial referring grounding via multipart image upload |
| **Change** | `POST` | `/api/v1/models/change` | Bi-temporal change detection between Time T1 & T2 (JSON) |
| **Change** | `POST` | `/api/v1/models/change/upload` | Bi-temporal change detection via multipart image uploads |
| **Fusion** | `POST` | `/api/v1/models/fusion` | Optical (Sentinel-2) + SAR (Sentinel-1) multi-modal land cover (JSON) |
| **Fusion** | `POST` | `/api/v1/models/fusion/upload` | Optical + SAR fusion via multipart image uploads |
| **Verification**| `POST` | `/api/v1/verify` | Multi-criteria Evidence Verification Gate anti-hallucination audit |

---

## 📥 Supported Image Formats & Input Modes

All endpoints flexibly accept satellite imagery in three distinct formats:
1. **Server File Path** (e.g. `"image_path": "data/P0003_0002.png"`)
2. **Base64 String** (e.g. `"image_base64": "data:image/png;base64,iVBORw0KGgo..."`)
3. **Multipart Form Upload** (`Content-Type: multipart/form-data`) supporting standard `.png`, `.jpg`, `.jpeg`, and multi-band `.tif` / `.tiff` GeoTIFFs.

---

## 🔍 Detailed Endpoint Reference

### 1. Autonomous Agent Query
**`POST /api/v1/agent/query`**

The primary intelligent entry point. Automatically determines whether the query requires VQA, scene captioning, visual grounding, change detection, or multi-modal fusion, routes to the appropriate specialist, and evaluates the output with the Evidence Verification Gate.

#### Request Body (JSON)
```json
{
  "query": "What color are the buses in the image?",
  "image_path": "data/P0003_0002.png"
}
```
*(For Optical+SAR, provide `image_optical_path` and `image_sar_path`. For Bi-Temporal, provide `image_t1_path` and `image_t2_path`)*.

#### Response (200 OK)
```json
{
  "success": true,
  "task": "vqa",
  "text": "The buses parked on the apron are yellow.",
  "confidence": 0.95,
  "spatial_evidence": {
    "type": "none",
    "source": "image",
    "bbox": null,
    "mask": null
  },
  "metadata": {
    "model": "vqa_v1",
    "backbone": "Qwen2-VL-2B",
    "checkpoint": "Qwen/Qwen2-VL-2B-Instruct",
    "dataset": "VRSBench",
    "input_modalities": ["optical"]
  },
  "verification": {
    "passed": true,
    "decision": "SUPPORTED",
    "reasons": [],
    "checks": {
      "task_compliance": {"passed": true, "detail": "Direct answer provided for VQA query"},
      "target_alignment": {"passed": true, "target": "buses", "detail": "Output text mentions target concept 'buses'"},
      "output_sanity": {"passed": true, "detail": "Output passed length and repetition checks"},
      "geospatial_context": {"available": false, "metadata": {"reason": "Benchmark crop without CRS/bounds metadata"}}
    },
    "gee_evidence": null
  },
  "status": "verified",
  "errors": [],
  "trace": { ... }
}
```

#### cURL Example
```bash
curl -X POST "http://localhost:8000/api/v1/agent/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What color are the buses?", "image_path": "data/P0003_0002.png"}'
```

#### Python Example
```python
import requests

url = "http://localhost:8000/api/v1/agent/query"
payload = {
    "query": "What color are the buses in the scene?",
    "image_path": "data/P0003_0002.png"
}
response = requests.post(url, json=payload)
print(response.json()["text"])
```

---

### 2. Spatial Referring Grounding
**`POST /api/v1/models/grounding`**

Localizes specific objects, infrastructure, or vegetation patches into bounding box pixel coordinates `[x1, y1, x2, y2]`.

#### Request Body (JSON)
```json
{
  "query": "Localize the yellow bus in the scene.",
  "image_path": "data/P0003_0002.png"
}
```

#### Response (200 OK)
```json
{
  "task": "grounding",
  "text": "yellow bus",
  "confidence": 0.85,
  "spatial_evidence": {
    "type": "bbox",
    "source": "image",
    "bbox": [55.0, 114.0, 112.0, 206.0],
    "mask": null
  },
  "metadata": {
    "model": "grounding_v1",
    "backbone": "Qwen2-VL-2B",
    "parameters": {
      "bboxes": [[55.0, 114.0, 112.0, 206.0]],
      "target": "yellow bus"
    }
  },
  "status": "success",
  "inference_seconds": 0.82
}
```

#### cURL (Multipart File Upload)
```bash
curl -X POST "http://localhost:8000/api/v1/models/grounding/upload" \
     -F "query=Localize the airplane on the tarmac." \
     -F "image=@data/vrsbench_samples/airport_airplanes.png"
```

---

### 3. Bi-Temporal Change Detection
**`POST /api/v1/models/change`**

Compares before (Time T1) and after (Time T2) satellite images to calculate structural differences, land clearance, or new construction.

#### Request Body (JSON)
```json
{
  "query": "What changed between time T1 and time T2?",
  "image_t1_path": "data/sample_bitemporal/bitemporal_t1_before.png",
  "image_t2_path": "data/sample_bitemporal/bitemporal_t2_after.png"
}
```

#### Response (200 OK)
```json
{
  "task": "change",
  "text": "Detected 9.8% structural change between T1 and T2.",
  "confidence": 0.88,
  "spatial_evidence": {
    "type": "mask",
    "source": "image_t2",
    "bbox": [280.0, 15.0, 510.0, 260.0],
    "mask": {
      "shape": [512, 512],
      "dtype": "uint8",
      "nonzero_pixels": 25656,
      "changed_ratio": 0.0978
    }
  },
  "metadata": {
    "model": "change_v1",
    "backbone": "SiameseDifferenceBaseline",
    "parameters": {
      "percent_changed": 9.78,
      "method": "otsu_thresholded_abs_difference"
    }
  },
  "status": "success",
  "inference_seconds": 0.04
}
```

#### cURL (Multipart File Upload)
```bash
curl -X POST "http://localhost:8000/api/v1/models/change/upload" \
     -F "query=Detect deforestation and construction" \
     -F "image_t1=@data/sample_bitemporal/bitemporal_t1_before.png" \
     -F "image_t2=@data/sample_bitemporal/bitemporal_t2_after.png"
```

---

### 4. Optical + SAR Multi-Modal Fusion
**`POST /api/v1/models/fusion`**

Fuses co-registered Sentinel-2 optical reflectance with Sentinel-1 microwave radar backscatter using dual ResNet encoders to classify 19-class BigEarthNet/CORINE land cover.

#### Request Body (JSON)
```json
{
  "query": "Analyze this multi-modal pair and classify the land cover.",
  "image_optical_path": "data/sample_bigearthnet/urban_s2_rgb.png",
  "image_sar_path": "data/sample_bigearthnet/urban_s1_sar.png"
}
```

#### Response (200 OK)
```json
{
  "task": "fusion",
  "text": "The area shows significant urban fabric.",
  "confidence": 0.53,
  "spatial_evidence": {
    "type": "none",
    "source": "fused",
    "bbox": null,
    "mask": null
  },
  "metadata": {
    "model": "fusion_v1",
    "backbone": "OpticalSAR-FeatureFusion",
    "dataset": "BigEarthNet-MM",
    "parameters": {
      "class_probabilities": {
        "Urban fabric": 0.528,
        "Industrial or commercial units": 0.182,
        "Arable land": 0.084,
        "Inland waters": 0.041
      }
    }
  },
  "status": "success",
  "inference_seconds": 0.12
}
```

#### Python Multipart Upload Example
```python
import requests

url = "http://localhost:8000/api/v1/models/fusion/upload"
files = {
    "image_optical": open("data/sample_bigearthnet/urban_s2_rgb.png", "rb"),
    "image_sar": open("data/sample_bigearthnet/urban_s1_sar.png", "rb"),
}
data = {"query": "Analyze this multi-modal pair."}
response = requests.post(url, data=data, files=files)
print(response.json()["text"])
```

---

### 5. Evidence Verification Gate Audit
**`POST /api/v1/verify`**

Audit any raw or third-party prediction against hallucination, task mismatch, and Google Earth Engine fallback triggers.

#### Request Body (JSON)
```json
{
  "query": "Localize the yellow bus",
  "task": "grounding",
  "result": {
    "task": "grounding",
    "text": "yellow bus",
    "confidence": 0.85,
    "spatial_evidence": {
      "type": "bbox",
      "bbox": [55.0, 114.0, 112.0, 206.0]
    }
  },
  "image_path": "data/P0003_0002.png"
}
```

#### Response (200 OK)
```json
{
  "passed": true,
  "decision": "SUPPORTED",
  "reasons": [],
  "checks": {
    "task_compliance": {"passed": true, "detail": "Spatial localization provided for grounding query"},
    "target_alignment": {"passed": true, "target": "yellow bus", "detail": "Output text mentions target concept 'yellow bus'"},
    "output_sanity": {"passed": true, "detail": "Output passed length and repetition checks"},
    "geospatial_context": {"available": false, "metadata": {"reason": "Benchmark crop without CRS/bounds metadata"}}
  },
  "gee_evidence": null
}
```

---

## ⚙️ Running Automated Tests

Run the full API test suite with Pytest:
```bash
python -m pytest tests/test_api.py -v
```
All 12 tests validate endpoint status codes, payload parsing, image encoding/decoding, model inference, and Evidence Verification Gate compliance.
