"""DataPilot — Root Application Launcher.

Run the FastAPI backend directly from the workspace root:
    python run.py
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

if __name__ == "__main__":
    import uvicorn
    from app.config import get_settings

    settings = get_settings()
    print(f"Starting DataPilot API on http://{settings.host}:{settings.port} ...")
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        app_dir=str(backend_dir),
    )
