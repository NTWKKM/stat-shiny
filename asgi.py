from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

from app import app as shiny_app

# 1. Define the path for static files
static_dir = Path(__file__).parent / "static"

# 2. Create routes
# Order matters: /static must be checked first, then Shiny handles the rest (/)
routes = [
    # Mount static files at /static (to match href="/static/styles.css")
    Mount("/static", app=StaticFiles(directory=str(static_dir)), name="static"),
    # Mount Shiny app at root (/)
    Mount("/", app=shiny_app, name="shiny"),
]

# 3. Add GZip middleware
middleware = [Middleware(GZipMiddleware, minimum_size=1000)]

# 4. Create the combined app (use this to run Gunicorn)
app = Starlette(routes=routes, middleware=middleware)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
