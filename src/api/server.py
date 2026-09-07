"""
Main FastAPI Application for SatQuery AI.
Configures CORS, OpenAPI documentation, routing, and error handling.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.api.deps import get_controller
from src.api.routes import health, agent, vqa, captioning, grounding, change, fusion, verification


tags_metadata = [
    {
        "name": "System & Health",
        "description": "API status, hardware accelerator info, and registered ML specialist models.",
    },
    {
        "name": "Autonomous Agent",
        "description": "Unified natural language satellite intelligence agent with automated intent routing and verification.",
    },
    {
        "name": "Specialist: VQA",
        "description": "Visual Question Answering powered by fine-tuned Qwen2-VL-2B backbone.",
    },
    {
        "name": "Specialist: Captioning",
        "description": "Natural language scene captioning and descriptive feature summarization.",
    },
    {
        "name": "Specialist: Grounding",
        "description": "Referring spatial grounding: coordinates [x1, y1, x2, y2] and multi-region patch localization.",
    },
    {
        "name": "Specialist: Change Detection",
        "description": "Bi-temporal change detection, structural expansion, and deforestation analysis between Time T1 and T2.",
    },
    {
        "name": "Specialist: Optical-SAR Fusion",
        "description": "Multi-modal feature fusion combining Sentinel-2 (optical) and Sentinel-1 (radar) for all-weather land cover.",
    },
    {
        "name": "Evidence Verification Gate",
        "description": "Anti-hallucination semantic gate, sanity auditing, and Google Earth Engine (GEE) fallback routing.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly initialize controller on startup so models are preloaded in memory
    get_controller()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="🛰️ SatQuery AI — Remote Sensing Intelligence API",
        description="""
## Autonomous Vision-Language & Multi-Modal Satellite Intelligence Agent

SatQuery AI provides production-ready REST API endpoints for:
* **Natural Language VQA & Captioning**: Inquire about satellite imagery or generate detailed descriptions using `Qwen2-VL-2B`.
* **Spatial Referring Grounding**: Localize vehicles, buildings, airplanes, harbors, and vegetation patches into exact coordinates.
* **Bi-Temporal Change Detection**: Compare pre- and post-event imagery to detect construction, land clearing, or disasters.
* **Optical + SAR Multi-Modal Fusion**: Combine Sentinel-2 optical reflectance with Sentinel-1 microwave radar backscatter.
* **Evidence Verification Gate**: Multi-criteria anti-hallucination audit checking semantic alignment and GEE routing.

All endpoints support both **JSON payloads** (server file paths or Base64 strings) and **Multipart File Uploads** (`multipart/form-data`).
        """,
        version="1.0.0",
        openapi_tags=tags_metadata,
        lifespan=lifespan,
    )

    # Enable CORS for web and client integrations
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Redirect root to interactive Swagger UI
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    # Include all modular routers
    app.include_router(health.router)
    app.include_router(agent.router)
    app.include_router(vqa.router)
    app.include_router(captioning.router)
    app.include_router(grounding.router)
    app.include_router(change.router)
    app.include_router(fusion.router)
    app.include_router(verification.router)

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
