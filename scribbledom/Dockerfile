# ============================================================
# Stage 1 – Builder
# ============================================================
FROM python:3.10-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    DEBIAN_FRONTEND=noninteractive

# Build-time dependencies (Standard R + System tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl gfortran \
    r-base-dev libcurl4-openssl-dev libssl-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Setup Python Environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 2. Install Python Deps (via requirements.txt)
COPY requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

# 3. Clone and Install R Libraries
RUN git clone --depth 1 https://github.com/1alnoman/ScribbleDom.git /opt/ScribbleDom \
    && mkdir -p /usr/local/lib/R/site-library \
    && Rscript -e ".libPaths('/usr/local/lib/R/site-library'); \
                 source('/opt/ScribbleDom/code_utils/read_data/installation.R'); \
                 install_libraries()"

# ============================================================
# Stage 2 – Runtime
# ============================================================
FROM python:3.10-slim

ENV PATH="/opt/venv/bin:$PATH" \
    R_LIBS_USER="/usr/local/lib/R/site-library" \
    PYTHONUNBUFFERED=1

# ---- Install Runtime Tools ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base-core \
    jq \
    libcurl4 \
    libxml2 \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup Non-Root User
RUN useradd -m -u 1000 -s /bin/bash appuser \
    && mkdir -p /workspace /home/appuser/.cache \
    && chown -R appuser:appuser /home/appuser /workspace

# 3. Copy Artifacts from Builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/local/lib/R/site-library /usr/local/lib/R/site-library
COPY --from=builder --chown=appuser:appuser /opt/ScribbleDom /opt/ScribbleDom

WORKDIR /opt/ScribbleDom
COPY --chown=appuser:appuser run_scribbledom.sh /usr/local/bin/run_scribbledom.sh
RUN chmod +x /usr/local/bin/run_scribbledom.sh

USER appuser
ENTRYPOINT ["/usr/local/bin/run_scribbledom.sh"]
