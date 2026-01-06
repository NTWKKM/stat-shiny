from pathlib import Path
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from app import app as shiny_app

# 1. กำหนด Path ของ Static Files
static_dir = Path(__file__).parent / "static"

# 2. สร้าง Routes
# ลำดับสำคัญ: ต้องเช็ค /static ก่อน แล้วค่อยให้ Shiny จัดการส่วนที่เหลือ (/)
routes = [
    # Mount Static Files ที่ /static (เพื่อให้ตรงกับ href="/static/styles.css")
    Mount("/static", app=StaticFiles(directory=str(static_dir)), name="static"),
    # Mount Shiny App ที่ Root (/)
    Mount("/", app=shiny_app, name="shiny"),
]

# 3. ใส่ Gzip Middleware
middleware = [
    Middleware(GZipMiddleware, minimum_size=1000)
]

# 4. สร้าง App รวม (ใช้ตัวนี้รัน Gunicorn)
app = Starlette(routes=routes, middleware=middleware)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
