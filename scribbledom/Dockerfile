# ============================================================
# Stage 1 – Builder
# ============================================================
FROM python:3.10-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    DEBIAN_FRONTEND=noninteractive

# 1. Install Build-time dependencies
# These headers are required for R packages like scater/scran to compile/link correctly
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl gfortran pkg-config \
    r-base-dev \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libcairo2-dev libharfbuzz-dev libfribidi-dev \
    libpng-dev libjpeg-dev libtiff-dev \
    libfontconfig1-dev libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Setup Python Environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 3. Install Python Dependencies
COPY requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

# 4. Clone and Install R Libraries
# We set the CRAN mirror EXPLICITLY here to fix the mirror error.
# We use the Posit Package Manager to get BINARIES for Debian Bookworm (fast).
RUN git clone --depth 1 https://github.com/1alnoman/ScribbleDom.git /opt/ScribbleDom \
    && mkdir -p /usr/local/lib/R/site-library \
    && Rscript -e ' \
        options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/bookworm/latest")); \
        .libPaths("/usr/local/lib/R/site-library"); \
        source("/opt/ScribbleDom/code_utils/read_data/installation.R"); \
        install_libraries();' \
    && rm -rf /tmp/* /root/.cache/R

# ============================================================
# Stage 2 – Runtime (Lean)
# ============================================================
FROM python:3.10-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    R_LIBS_SITE="/usr/local/lib/R/site-library:/usr/lib/R/site-library:/usr/lib/R/library"

# 1. Install Runtime Dependencies (Shared libs only, no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base-core \
    jq \
    libcurl4 libxml2 libssl3 libgomp1 \
    libcairo2 libharfbuzz0b libfribidi0 \
    libpng16-16 libjpeg62-turbo libtiff6 \
    libfontconfig1 libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# 2. Persistently configure R for runtime
RUN mkdir -p /etc/R \
    && echo '.libPaths(c("/usr/local/lib/R/site-library", .libPaths()))' > /etc/R/Rprofile.site \
    && echo 'options(repos = c(CRAN = "https://cloud.r-project.org"))' >> /etc/R/Rprofile.site \
    && echo 'R_LIBS_SITE="/usr/local/lib/R/site-library:/usr/lib/R/site-library:/usr/lib/R/library"' >> /etc/R/Renviron.site

# 3. Setup Non-Root User
RUN useradd -m -u 1000 -s /bin/bash appuser \
    && mkdir -p /workspace /home/appuser/.cache \
    && chown -R appuser:appuser /home/appuser /workspace /usr/local/lib/R/site-library

# 4. Copy optimized artifacts from Stage 1
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/local/lib/R/site-library /usr/local/lib/R/site-library
COPY --from=builder --chown=appuser:appuser /opt/ScribbleDom /opt/ScribbleDom

WORKDIR /opt/ScribbleDom

COPY --chown=appuser:appuser run_scribbledom.sh /usr/local/bin/run_scribbledom.sh
RUN chmod +x /usr/local/bin/run_scribbledom.sh

USER appuser
ENTRYPOINT ["/usr/local/bin/run_scribbledom.sh"]
