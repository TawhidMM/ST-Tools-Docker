# --- STAGE 1: Builder ---
FROM python:3.10-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    DEBIAN_FRONTEND=noninteractive \
    SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install build-time dependencies (including R-dev for compiling mclust)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl r-base-dev libxml2-dev libssl-dev \
    libcurl4-openssl-dev libblas-dev liblapack-dev gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Setup venv
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 1. Install Torch Stack (cu121 is the stable match for Torch 2.2.2)
RUN uv pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

# 2. Install PyG Stack
RUN uv pip install torch-geometric==2.5.2 \
    pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# 3. Install remaining Python deps from requirements.txt
COPY requirements.txt .
RUN uv pip install --no-cache -r requirements.txt


# 4. Compile R dependencies (mclust 5.4.10)
RUN mkdir -p /usr/local/lib/R/site-library && \
    Rscript -e "install.packages('remotes', repos='https://cloud.r-project.org/')" && \
    Rscript -e "remotes::install_version('mclust', version='5.4.10', repos='https://cloud.r-project.org/', lib='/usr/local/lib/R/site-library')"

# 5. Clone the tool
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

# Install ONLY runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base-core \
    libgl1 \
    libglib2.0-0 \
    libxml2 \
    libcurl4 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser \
    && mkdir -p /workspace /home/appuser/.config \
    && chown -R appuser:appuser /workspace /home/appuser

# Copy Python Venv
COPY --from=builder /opt/venv /opt/venv
RUN chown -R appuser:appuser /opt/venv

# Copy R Libraries
COPY --from=builder /usr/local/lib/R/site-library /usr/local/lib/R/site-library

# Copy Tool Code
COPY --from=builder /opt/STAIG /opt/STAIG
RUN chown -R appuser:appuser /opt/STAIG

COPY entrypoint.py .
RUN chown appuser:appuser entrypoint.py

USER appuser

ENTRYPOINT ["python", "entrypoint.py"]