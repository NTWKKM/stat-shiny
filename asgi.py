"""ASGI wrapper for serving static files with Shiny app.

This file is used for production deployment with gunicorn/uvicorn.
It mounts the Shiny app and serves static files from ./static/ directory.

Usage:
  gunicorn -w 4 -k uvicorn.workers.UvicornWorker asgi:app
"""

from pathlib import Path
from starlette.staticfiles import StaticFiles
from app import app as shiny_app

# Get the directory where this file is located
BASE_DIR = Path(__file__).parent

# Mount static files
# Serves files from ./static/ at /static/ URL path
app = shiny_app

try:
    static_dir = BASE_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        print(f"✅ Static files mounted from {static_dir}")
    else:
        print(f"⚠️  Static directory not found: {static_dir}")
except Exception as e:
    print(f"❌ Error mounting static files: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
