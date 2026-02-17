# --- STAGE 1: Builder ---
FROM python:3.10-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install setuptools wheel

RUN uv pip install "numpy==1.26.4"

# 1. Install Torch, Vision, and Audio 
RUN uv pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0

# 2. Install PyG Core and Extensions
RUN uv pip install \
    --index-strategy unsafe-best-match \
    --find-links https://data.pyg.org/whl/torch-2.1.0+cu121.html \
    pyg_lib==0.3.1+pt21cu121 \
    torch-scatter==2.1.2+pt21cu121 \
    torch-sparse==0.6.18+pt21cu121 \
    torch-cluster==1.6.3+pt21cu121 \
    torch-spline-conv==1.2.2+pt21cu121

# 3. Install High-level PyG (from PyPI)
RUN uv pip install torch-geometric==2.3.1

# ---- DeepST ----
RUN uv pip install deepstkit

# ---- cleanup ----
RUN set -e; \
    find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true; \
    find /opt/venv -name "*.a"                 -delete 2>/dev/null || true; \
    find /opt/venv -name "*.pyc"               -delete 2>/dev/null || true


# --- STAGE 2: Runtime ---
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# System dependencies:
# libgl1 and libglib2.0-0 are REQUIRED for torchvision (OpenCV/PIL backends)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the lean, pre-built virtual environment
COPY --from=builder /opt/venv /opt/venv

# Copy entrypoint
COPY entrypoint.py .

ENTRYPOINT ["python", "entrypoint.py"]