"""
🧪 Pytest Configuration for stat-shiny E2E Tests

This conftest.py sets up fixtures for the entire test suite:
- Automatically starts Shiny server before E2E tests
- Handles cleanup after tests
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import requests

# ============================================================================
# 🚀 Session-Scoped Fixture: Start Shiny Server
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def start_shiny_server(request):
    """
    Start a Shiny app server for the test session.
    Improved with Error Log capturing for better debugging.
    """
    
    # ────────────────────────────────────────────────────────────────────
    # Step 0: Check if we need the server
    # ────────────────────────────────────────────────────────────────────
    collected_items = getattr(request.session, 'items', [])
    
    # ข้ามการเปิด Server ถ้าเป็น unit test ทั้งหมด หรือรันในโฟลเดอร์ unit
    has_e2e_tests = any(
        not any(mark.name == 'unit' for mark in getattr(item, 'iter_markers', lambda: iter([]))())
        for item in collected_items
    )
    is_unit_dir = all("tests/unit" in str(getattr(item, 'fspath', '')) for item in collected_items)

    if not has_e2e_tests or is_unit_dir:
        print("\n⏭️  Unit tests detected - skipping server startup")
        yield
        return

    # ────────────────────────────────────────────────────────────────────
    # Step 1: Find app.py
    # ────────────────────────────────────────────────────────────────────
    project_root = Path(__file__).parent.parent
    app_path = project_root / "app.py"
    
    if not app_path.exists():
        raise FileNotFoundError(f"❌ app.py not found at {app_path}")

    print(f"\n{'='*70}")
    print(f"🚀 Starting Shiny Server for E2E Tests")
    print(f"{'='*70}")
    
    # ────────────────────────────────────────────────────────────────────
    # Step 2: Start Shiny server (Redirect output to a Temp File)
    # ────────────────────────────────────────────────────────────────────
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    # สร้างไฟล์ชั่วคราวเพื่อเก็บ Log ของแอป
    log_file = tempfile.NamedTemporaryFile(delete=False, mode='w+')
    
    try:
        process = subprocess.Popen(
            [
                sys.executable, "-m", "shiny", "run",
                "--host", "127.0.0.1",
                "--port", "8000",
                str(app_path)
            ],
            stdout=log_file, # พ่น Log ลงไฟล์เพื่อกัน Buffer เต็ม
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(project_root),
            text=True
        )
        print(f"✅ Subprocess started (PID: {process.pid})")
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError(f"❌ Failed to start Shiny server: {e}") from e

    # ────────────────────────────────────────────────────────────────────
    # Step 3: Wait for server and Capture Error if crashed
    # ────────────────────────────────────────────────────────────────────
    server_ready = False
    start_time = time.time()
    
    print("⏳ Waiting for server (max 60s)...", end="", flush=True)
    
    try:
        while time.time() - start_time < 60:
            # ตรวจสอบว่าแอปพังกลางคันหรือไม่
            if process.poll() is not None:
                log_file.close()
                with open(log_file.name) as f:
                    error_log = f.read()
                raise RuntimeError(f"\n❌ Server crashed on startup!\n--- ERROR LOG ---\n{error_log}\n-----------------")
                
            try:
                response = requests.get("http://127.0.0.1:8000", timeout=2)
                if response.status_code in [200, 304]:
                    server_ready = True
                    print(f" ✅ Ready!")
                    break
            except (requests.ConnectionError, requests.Timeout):
                print(".", end="", flush=True)
                time.sleep(1)
        
        if not server_ready:
            process.terminate()
            raise RuntimeError("❌ Timeout: Shiny server failed to start within 60s")

        yield

    finally:
        # ────────────────────────────────────────────────────────────────────
        # Step 5: Cleanup
        # ────────────────────────────────────────────────────────────────────
        if process.poll() is None:
            print(f"\n🛑 Stopping Shiny Server (PID: {process.pid})...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            print("✅ Server stopped")
        
        log_file.close()
        if os.path.exists(log_file.name):
            os.remove(log_file.name)

# ============================================================================
# 🎨 Pytest Configuration & Markers
# ============================================================================

def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: marks tests as E2E tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")

def pytest_sessionstart(session): # แก้จาก _session เป็น session
    print("\n" + "="*70)
    print("📊 Starting Test Session")
    print("="*70)

def pytest_sessionfinish(session, exitstatus): # แก้จาก (_session, _exitstatus) เป็น (session, exitstatus)
    print("\n" + "="*70)
    print("✅ Test Session Complete")
    print("="*70)