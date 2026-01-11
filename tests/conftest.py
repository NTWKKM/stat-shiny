"""
🧪 Pytest Configuration for stat-shiny E2E Tests

This conftest.py sets up fixtures for the entire test suite:
- Automatically starts Shiny server before E2E tests
- Provides Playwright page fixture (from shiny.pytest)
- Handles cleanup after tests

Fixtures:
- start_shiny_server: Session-scoped fixture that starts the app
- page: Playwright page object (auto-provided by shiny.pytest)
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
    🚀 Start Shiny server before running E2E tests
    
    This fixture:
    - Starts the Shiny app in a subprocess (port 8000)
    - Waits for the server to be ready (max 60 seconds)
    - Stops the server after all tests complete
    
    Args:
        request: Pytest request object
    
    Yields:
        None (tests run between start and stop)
    
    Raises:
        RuntimeError: If server fails to start within 60 seconds
    
    Notes:
        - scope="session" → Server runs ONCE for entire test session
        - Only runs when explicitly requested (e.g., for E2E tests)
        - Unit tests marked with @pytest.mark.unit skip this fixture
    """
    
    # Check if this is a unit test run - skip server startup
    # This is triggered when running with -m unit or when all tests are unit tests
    collected_items = getattr(request.session, 'items', [])
    if collected_items and all(
        any(mark.name == 'unit' for mark in getattr(item, 'iter_markers', list)())
        for item in collected_items
    ):
        print("\n⏭️  Unit tests only - skipping server startup")
        yield
        return
    
    # ────────────────────────────────────────────────────────────────────
    # Step 1: Find app.py
    # ────────────────────────────────────────────────────────────────────
    
    project_root = Path(__file__).parent.parent
    app_path = project_root / "app.py"
    
    # Validate app.py exists
    if not app_path.exists():
        raise FileNotFoundError(
            f"❌ app.py not found at {app_path}\n"
            f"   Expected path: {app_path}\n"
            f"   Project root: {project_root}"
        )
    
    print(f"\n{'='*70}")
    print(f"🚀 Starting Shiny Server for E2E Tests")
    print(f"{'='*70}")
    print(f"📂 Project root: {project_root}")
    print(f"📂 App path:     {app_path}")
    print(f"🌐 Server URL:   http://localhost:8000")
    print(f"{'='*70}")
    
    # ────────────────────────────────────────────────────────────────────
    # Step 2: Start Shiny server in subprocess
    # ────────────────────────────────────────────────────────────────────
    
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'  # Real-time output (no buffering)
    
    try:
        # Use 'python -m shiny run' instead of 'shiny run'
        # This is more reliable across different Python installations
        process = subprocess.Popen(
            [
                sys.executable, "-m", "shiny", "run",
                "--host", "0.0.0.0",
                "--port", "8000",
                str(app_path)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(project_root),
            text=True
        )
        print(f"✅ Subprocess started (PID: {process.pid})")
    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to start Shiny server subprocess\n"
            f"   Error: {e}\n"
            f"   Command: python -m shiny run --host 0.0.0.0 --port 8000 {app_path}"
        ) from e
    
    # ────────────────────────────────────────────────────────────────────
    # Step 3: Wait for server to be ready (max 60 seconds)
    # ────────────────────────────────────────────────────────────────────
    
    server_ready = False
    start_time = time.time()
    elapsed = 0
    
    print("\n⏳ Waiting for server to start...", end="", flush=True)
    
    while elapsed < 60:
        try:
            # Try to connect to server
            response = requests.get(
                "http://localhost:8000",
                timeout=2
            )
            # Accept any response code - just need to confirm server is running
            if response.status_code in [200, 304]:
                server_ready = True
                elapsed = time.time() - start_time
                print(f"\r✅ Server ready after {elapsed:.1f}s")
                break
        except (requests.ConnectionError, requests.Timeout):
            # Server not ready yet, wait a bit more
            elapsed = time.time() - start_time
            print(".", end="", flush=True)
            time.sleep(0.5)
    
    if not server_ready:
        process.terminate()
        raise RuntimeError(
            f"❌ Shiny server failed to start within 60 seconds\n"
            f"   Check that:\n"
            f"   1. Port 8000 is available\n"
            f"   2. All dependencies are installed\n"
            f"   3. app.py has no syntax errors"
        )
    
    print(f"🌐 Server running at http://localhost:8000")
    print(f"{'='*70}\n")
    
    # ────────────────────────────────────────────────────────────────────
    # Step 4: Yield control to tests
    # ────────────────────────────────────────────────────────────────────
    
    yield  # ← TESTS RUN HERE
    
    # ────────────────────────────────────────────────────────────────────
    # Step 5: Cleanup - Stop server
    # ────────────────────────────────────────────────────────────────────
    
    print(f"\n{'='*70}")
    print(f"🛑 Stopping Shiny Server")
    print(f"{'='*70}")
    
    # Try graceful termination first
    process.terminate()
    try:
        process.wait(timeout=5)
        print("✅ Server stopped gracefully")
    except subprocess.TimeoutExpired:
        # If graceful stop fails, force kill
        print("⚠️  Server didn't stop gracefully, force killing...")
        process.kill()
        process.wait()
        print("✅ Server force killed")
    
    print(f"{'='*70}\n")


# ============================================================================
# 🎨 Optional: Add markers for test organization
# ============================================================================

def pytest_configure(config):
    """
    Register custom pytest markers
    
    This allows you to:
    - Mark tests: @pytest.mark.e2e
    - Run specific tests: pytest -m e2e
    """
    config.addinivalue_line(
        "markers",
        "e2e: marks tests as E2E tests (require running server)"
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests (fast, no server needed)"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )


# ============================================================================
# 🔧 Optional: Pytest hooks for better logging
# ============================================================================

def pytest_sessionstart(session):
    """Called at the very start of test session"""
    print("\n" + "="*70)
    print("📊 Starting Test Session")
    print("="*70)


def pytest_sessionfinish(session, exitstatus):
    """Called at the very end of test session"""
    print("\n" + "="*70)
    print("✅ Test Session Complete")
    print("="*70)
