"""
Dependency injection and input resolution utilities for SatQuery AI API.
Provides lazy-singleton AgentController access and robust image decoders.
"""

import os
import io
import uuid
import base64
from typing import Optional, Union, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi import HTTPException, UploadFile
import torch

from src.utils.io_utils import load_config
from src.preprocessing.geotiff_utils import read_image
from src.agent.controller import AgentController


# Global controller instance (lazy singleton)
_controller_instance: Optional[AgentController] = None
_device_used: str = "cpu"


def get_controller() -> AgentController:
    """Returns the cached AgentController instance, initializing on first call."""
    global _controller_instance, _device_used
    if _controller_instance is None:
        config = load_config("configs/config.yaml")
        _device_used = "cuda" if torch.cuda.is_available() else "cpu"
        _controller_instance = AgentController(config, device=_device_used)
    return _controller_instance


def get_device() -> str:
    """Returns 'cuda' or 'cpu' depending on active compute device."""
    global _device_used
    if _controller_instance is None:
        _device_used = "cuda" if torch.cuda.is_available() else "cpu"
    return _device_used


def resolve_image_input(
    path: Optional[str] = None,
    b64: Optional[str] = None,
    upload: Optional[UploadFile] = None,
    target_dir: str = "data/raw/uploads",
) -> str:
    """
    Resolves image input from a file path, base64 string, or FastAPI UploadFile.
    Returns the absolute or relative file path on disk suitable for read_image.
    """
    os.makedirs(target_dir, exist_ok=True)

    # 1. From FastAPI UploadFile
    if upload is not None and upload.filename:
        safe_name = f"upload_{uuid.uuid4().hex[:8]}_{Path(upload.filename).name}"
        save_path = os.path.join(target_dir, safe_name)
        with open(save_path, "wb") as f:
            f.write(upload.file.read())
        return save_path

    # 2. From Base64 string
    if b64 is not None and len(b64.strip()) > 0:
        cleaned_b64 = b64.strip()
        ext = "png"
        if "," in cleaned_b64 and cleaned_b64.startswith("data:"):
            header, cleaned_b64 = cleaned_b64.split(",", 1)
            if "jpeg" in header or "jpg" in header:
                ext = "jpg"
            elif "tiff" in header or "tif" in header:
                ext = "tif"

        try:
            image_bytes = base64.b64decode(cleaned_b64)
            safe_name = f"b64_{uuid.uuid4().hex[:8]}.{ext}"
            save_path = os.path.join(target_dir, safe_name)
            with open(save_path, "wb") as f:
                f.write(image_bytes)
            return save_path
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")

    # 3. From local filesystem path
    if path is not None and len(path.strip()) > 0:
        clean_path = path.strip()
        if not os.path.isfile(clean_path):
            raise HTTPException(status_code=404, detail=f"Image file not found on server at: '{clean_path}'")
        return clean_path

    raise HTTPException(
        status_code=400,
        detail="Missing image input. Provide one of: file upload, file path, or base64 string.",
    )
