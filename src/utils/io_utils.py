"""Small shared I/O helpers (config loading, path helpers). Owner: shared."""

import os
import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_uploaded_file(uploaded_file, target_dir: str) -> str:
    """Save a file-like object (e.g. from Streamlit's file_uploader) to
    target_dir and return the saved path. Used by app/app.py.
    """
    ensure_dir(target_dir)
    target_path = os.path.join(target_dir, uploaded_file.name)
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target_path
