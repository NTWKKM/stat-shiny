# ใช้ Python 3.12-slim
FROM python:3.12-slim

# ตั้งค่า Working Directory
WORKDIR /code

# Copy requirements และ Install
# (ขั้นตอนนี้จะลง gunicorn ให้ด้วย เพราะมันอยู่ในไฟล์ requirements.txt แล้ว)
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy โค้ดทั้งหมด
COPY . .

# สร้าง User (Security)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /code
USER appuser

# เปิด Port
EXPOSE 7860

# Health Check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:7860', timeout=2)" || exit 1

# คำสั่งรัน (ใช้ gunicorn เรียก asgi:app ตามที่เราทำกันไว้)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "--timeout", "120", "--bind", "0.0.0.0:7860", "asgi:app"]
