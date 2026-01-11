"""
🧪 Pytest Configuration for stat-shiny E2E Tests

This conftest.py sets up fixtures for the entire test suite:
- Automatically starts Shiny server before E2E tests
- Provides Playwright page fixture (from shiny.pytest)
- Handles cleanup after tests
"""

import os
import subprocess
import sys
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
    Skips startup if only unit tests are being run.
    """
    
    # ────────────────────────────────────────────────────────────────────
    # Step 0: Check if we need the server
    # ────────────────────────────────────────────────────────────────────
    collected_items = getattr(request.session, 'items', [])
    
    # Check if there's any test that is NOT a unit test
    # (i.e., we have e2e or integration tests)
    has_e2e_tests = any(
        not any(mark.name == 'unit' for mark in getattr(item, 'iter_markers', list)())
        for item in collected_items
    )
    
    # Also check if we are explicitly running tests in the unit directory
    is_unit_dir = all("tests/unit" in str(getattr(item, 'fspath', '')) for item in collected_items)

    if not has_e2e_tests or is_unit_dir:
        print("\n⏭️  Unit tests detected - skipping server startup to save time")
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
    # Step 2: Start Shiny server in subprocess
    # ────────────────────────────────────────────────────────────────────
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    try:
        # Fixed: Redirect output to DEVNULL or stdout to prevent buffer-fill deadlock
        process = subprocess.Popen(
            [
                sys.executable, "-m", "shiny", "run",
                "--host", "127.0.0.1",
                "--port", "8000",
                str(app_path)
            ],
            stdout=subprocess.DEVNULL, # หรือใช้ sys.stdout ถ้าต้องการดู log
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(project_root),
            text=True
        )
        print(f"✅ Subprocess started (PID: {process.pid})")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to start Shiny server: {e}")

    # ────────────────────────────────────────────────────────────────────
    # Step 3: Wait for server to be ready
    # ────────────────────────────────────────────────────────────────────
    server_ready = False
    start_time = time.time()
    
    print("⏳ Waiting for server (max 60s)...", end="", flush=True)
    
    while time.time() - start_time < 60:
        if process.poll() is not None:
            # Server crashed early
            stdout, _ = process.communicate()
            raise RuntimeError(f"❌ Server crashed on startup. Error: {stdout}")
            
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

    # ────────────────────────────────────────────────────────────────────
    # Step 5: Cleanup
    # ────────────────────────────────────────────────────────────────────
    print(f"\n🛑 Stopping Shiny Server (PID: {process.pid})...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    print("✅ Server stopped")

# ============================================================================
# 🎨 Pytest Configuration & Markers
# ============================================================================

def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: marks tests as E2E tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")

def pytest_sessionstart(session):
    print("\n" + "="*70)
    print("📊 Starting Test Session")
    print("="*70)

def pytest_sessionfinish(session, exitstatus):
    print("\n" + "="*70)
    print("✅ Test Session Complete")
    print("="*70)