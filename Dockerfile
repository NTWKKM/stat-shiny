# ==============================================================================
# Medical Stat Tool (stat-shiny) - Dockerfile
# ==============================================================================
# Optimized for HuggingFace Spaces deployment
# Build: docker build -t stat-shiny .
# Run:   docker run -p 7860:7860 stat-shiny
# ==============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies
# -----------------------------------------------------------------------------
FROM python:3.14-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies (if needed for C extensions)
RUN set -o pipefail && apt-get update && apt-get install -y --no-install-recommends \
  gcc="$(apt-cache show gcc | grep -m1 Version | cut -d' ' -f2)" \
  && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
# Copy only requirements first to leverage Docker cache
COPY requirements-prod.txt ./
RUN pip install --target=/build/deps --no-cache-dir -r requirements-prod.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Lean production image
# -----------------------------------------------------------------------------
FROM python:3.14-slim AS runtime

# OCI Labels
LABEL org.opencontainers.image.title="Medical Stat Tool" \
  org.opencontainers.image.description="Medical statistical analysis web app" \
  org.opencontainers.image.source="https://github.com/NTWKKM/stat-shiny" \
  org.opencontainers.image.licenses="MIT"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=/app/deps \
  HOME=/home/appuser

WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /build/deps /app/deps

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
  chown -R appuser:appuser /app

# Copy application code (respecting .dockerignore)
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port (HuggingFace Spaces uses 7860)
EXPOSE 7860

# Health check (Ensure python is available)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860', timeout=3)" || exit 1

# Run with Gunicorn + Uvicorn worker
# - workers: 2 (Increased to 2 for better concurrency if memory allows, or keep 1)
# - timeout: 120s (for long-running statistical computations)
CMD ["python", "-m", "gunicorn", \
  "-k", "uvicorn.workers.UvicornWorker", \
  "-w", "2", \
  "--timeout", "120", \
  "--graceful-timeout", "30", \
  "--bind", "0.0.0.0:7860", \
  "--preload", \
  "asgi:app"]
