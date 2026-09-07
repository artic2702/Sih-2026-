"""
CLI launcher for the SatQuery AI FastAPI Server.

Usage:
    python scripts/run_api_server.py
    python scripts/run_api_server.py --port 8000 --reload
"""

import os
import sys
from pathlib import Path
import argparse
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="SatQuery AI API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reloading on code changes")
    args = parser.parse_args()

    print(f"[*] Starting SatQuery AI API Server on http://{args.host}:{args.port}")
    print(f"[*] Interactive Swagger UI: http://localhost:{args.port}/docs")
    print(f"[*] ReDoc Documentation:    http://localhost:{args.port}/redoc")
    uvicorn.run("src.api.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
