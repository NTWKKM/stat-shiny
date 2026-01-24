# ==============================================================================
# Medical Stat Tool (stat-shiny) - Dockerfile (Secured & Patched)
# ==============================================================================
# Optimized for HuggingFace Spaces deployment
# Build: docker build -t stat-shiny .
# Run:   docker run -p 7860:7860 stat-shiny
# ==============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Update OS packages to fix system vulnerabilities
RUN apt-get update && \
  apt-get upgrade -y && \
  apt-cache policy gcc && \
  apt-get install -y --no-install-recommends gcc=4:12.2.0-3 && \
  rm -rf /var/lib/apt/lists/*

# Pre-install latest security tools into the target directory (ensuring fresh install)
# This fixes CVE-2026-24049 (wheel/setuptools) for the app dependencies
RUN rm -rf /build/deps && mkdir -p /build/deps && \
  pip install --target=/build/deps --no-cache-dir --upgrade pip "setuptools>=80.10.1" "wheel>=0.46.3"

# Copy and install requirements
COPY requirements-prod.txt ./

# Install remaining dependencies
RUN pip install --target=/build/deps --no-cache-dir -r requirements-prod.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Lean production image
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

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

# -----------------------------------------------------------------------------
# SECURITY FIXES (Runtime)
# -----------------------------------------------------------------------------
# 1. Update OS packages (fixes system-level CVEs like glibc, openssl)
# 2. Update System PIP (fixes CVE-2025-8869: pip <= 25.2)
RUN apt-get update && \
  apt-get upgrade -y && \
  pip install --no-cache-dir --upgrade "pip>=25.3" && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*
# -----------------------------------------------------------------------------

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
CMD ["python", "-m", "gunicorn", \
  "-k", "uvicorn.workers.UvicornWorker", \
  "-w", "2", \
  "--timeout", "120", \
  "--graceful-timeout", "30", \
  "--bind", "0.0.0.0:7860", \
  "--preload", \
  "asgi:app"]