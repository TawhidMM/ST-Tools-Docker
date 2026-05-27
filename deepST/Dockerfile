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

# Create venv
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install base deps
RUN uv pip install wheel "numpy==1.26.4"

# Torch (CUDA)
RUN uv pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

# PyG
RUN uv pip install \
    --index-strategy unsafe-best-match \
    --find-links https://data.pyg.org/whl/torch-2.1.0+cu121.html \
    pyg_lib==0.3.1+pt21cu121 \
    torch-scatter==2.1.2+pt21cu121 \
    torch-sparse==0.6.18+pt21cu121 \
    torch-cluster==1.6.3+pt21cu121 \
    torch-spline-conv==1.2.2+pt21cu121

# App deps
RUN uv pip install torch-geometric==2.3.1 "setuptools==65.5.0" deepstkit

# Pre-download model weights
RUN python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.DEFAULT)"

# Cleanup (safe)
RUN find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true


# --- STAGE 2: Runtime ---
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOME="/home/appuser" \
    MPLCONFIGDIR="/home/appuser/.config/matplotlib" \
    NUMBA_CACHE_DIR="/home/appuser/.numba" \
    TORCH_HOME="/home/appuser/.cache/torch"

WORKDIR /app

# Runtime libs only (keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -u 1000 -s /bin/bash appuser \
    && mkdir -p /workspace /home/appuser/.cache/torch \
    && chown -R appuser:appuser /home/appuser /workspace

COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /root/.cache/torch /home/appuser/.cache/torch

# App entry
COPY --chown=appuser:appuser entrypoint.py .

USER appuser
ENTRYPOINT ["python", "entrypoint.py"]