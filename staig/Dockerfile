# --- STAGE 1: Builder ---
FROM python:3.10-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    DEBIAN_FRONTEND=noninteractive \
    SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    r-base-dev libxml2-dev libssl-dev libcurl4-openssl-dev \
    libblas-dev liblapack-dev gfortran \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Venv
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Torch (install only what you need)
RUN uv pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

# PyG
RUN uv pip install \
    torch-geometric==2.5.2 \
    pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# Python deps
COPY requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

# R deps
RUN mkdir -p /usr/local/lib/R/site-library && \
    Rscript -e "install.packages('remotes', repos='https://cloud.r-project.org/')" && \
    Rscript -e "remotes::install_version('mclust', version='5.4.10', repos='https://cloud.r-project.org/', lib='/usr/local/lib/R/site-library')"

# Tool code
RUN git clone --depth 1 https://github.com/y-itao/STAIG.git /opt/STAIG \
    && rm -rf /opt/STAIG/Dataset /opt/STAIG/example /opt/STAIG/.git


# --- STAGE 2: Runtime ---
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    USER="appuser" \
    HOME="/home/appuser" \
    MPLCONFIGDIR="/home/appuser/.config/matplotlib" \
    NUMBA_CACHE_DIR="/home/appuser/.numba" \
    R_LIBS_USER="/usr/local/lib/R/site-library"

WORKDIR /opt/STAIG

# Runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base-core \
    libgl1 \
    libglib2.0-0 \
    libxml2 \
    libcurl4 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -u 1000 -s /bin/bash appuser \
    && mkdir -p /workspace /home/appuser/.config \
    && chown -R appuser:appuser /workspace /home/appuser

# Copy with ownership (CRITICAL FIX)
COPY --chown=appuser:appuser --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/local/lib/R/site-library /usr/local/lib/R/site-library
COPY --chown=appuser:appuser --from=builder /opt/STAIG /opt/STAIG

COPY --chown=appuser:appuser entrypoint.py .

USER appuser

ENTRYPOINT ["python", "entrypoint.py"]