"""ASGI wrapper for serving static files with Shiny app (Optimized).

Features:
- Gzip Compression enabled (faster load times)
- Correct mounting on underlying Starlette app
- Explicit static file handling
"""

from pathlib import Path
from starlette.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware
from app import app as shiny_app

# 1. เข้าถึงตัว Starlette App ที่อยู่ข้างใน Shiny App
# (เพื่อความชัวร์ในการเรียกใช้ฟังก์ชัน .mount และ .add_middleware)
asgi_app = shiny_app.app

# 2. 🚀 OPTIMIZATION: เพิ่ม Gzip Compression
# ช่วยลดขนาดไฟล์ HTML, CSS, JS และ JSON ที่ส่งกลับไปหา User
# minimum_size=1000 แปลว่าไฟล์เล็กกว่า 1KB ไม่ต้องบีบอัด (เพื่อไม่ให้เปลือง CPU)
asgi_app.add_middleware(GZipMiddleware, minimum_size=1000)

# 3. กำหนด Path ของ Static Files
BASE_DIR = Path(__file__).parent
static_dir = BASE_DIR / "static"

# 4. Mount Static Files
if static_dir.exists():
    # Mount ไปที่ path "/static" เพื่อให้ตรงกับ HTML href="/static/styles.css"
    asgi_app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    print(f"✅ Static files mounted from {static_dir} (with Gzip)")
else:
    print(f"⚠️  Static directory not found: {static_dir}")

# Expose 'app' object for Gunicorn/Uvicorn to find
app = asgi_app

if __name__ == "__main__":
    import uvicorn
    # รันด้วย production configuration เบื้องต้น
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        # workers=4 # ใช้ flag นี้ผ่าน command line เท่านั้น (uvicorn main:app --workers 4)
    )
