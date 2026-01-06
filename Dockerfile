# ใช้ Python 3.12-slim Image เป็นฐาน
# ✅ Security support and latest features
# ✅ Smaller container vs full Python image
FROM python:3.12-slim

# ตั้งค่า Working Directory
WORKDIR /code

# Copy ไฟล์ requirements และติดตั้ง
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy โค้ดทั้งหมดเข้า Container
COPY . .

# สร้าง non-root user สำหรับ security best practices
# ✅ ลดความเสี่ยง: ไม่ run as root
# ✅ ปลอดภัย: non-root user มี limited permissions
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /code
USER appuser

# เปิด Port 7860 (Port มาตรฐานของ Hugging Face Spaces)
EXPOSE 7860

# เพิ่ม Health Check สำหรับ container orchestration
# ✅ Kubernetes/Docker จะตรวจสอบ container health
# ✅ Auto-restart unhealthy containers
# ✅ Interval: 30 seconds, Timeout: 3 seconds, Retries: 3
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:7860', timeout=2)" || exit 1

# ✅ UPDATED: Use ASGI wrapper for proper static file serving
# This ensures /static/styles.css is properly served alongside the Shiny app
CMD ["python", "-m", "uvicorn", "asgi:app", "--host", "0.0.0.0", "--port", "7860"]
