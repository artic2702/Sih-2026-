"""
SatQuery AI REST API package.
Exposes ML specialist endpoints, agent orchestration, and verification gate.
"""

from src.api.server import create_app, app

__all__ = ["create_app", "app"]
