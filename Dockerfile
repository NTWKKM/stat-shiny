# ใช้ Python Image เป็นฐาน
FROM python:3.9

# ตั้งค่า Working Directory
WORKDIR /code

# Copy ไฟล์ requirements และติดตั้ง
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy โค้ดทั้งหมดเข้า Container
COPY . .

# เปิด Port 7860 (Port มาตรฐานของ Hugging Face Spaces)
EXPOSE 7860

# คำสั่งรัน App
CMD ["shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]
