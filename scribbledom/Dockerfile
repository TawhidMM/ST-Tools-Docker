# ============================================================
# Stage 1 – Builder
# ============================================================
FROM python:3.10-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    DEBIAN_FRONTEND=noninteractive
    
# Build-time dependencies (CRITICAL for scater + scran)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl gfortran \
    r-base-dev \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    # === Added for scater / Bioconductor graphics & C++ packages ===
    libcairo2-dev libharfbuzz-dev libfribidi-dev \
    libpng-dev libjpeg-dev libtiff-dev \
    libfontconfig1-dev libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /app

# 1. Setup Python Environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 2. Install Python Deps
COPY requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

# 3. Clone and Install R Libraries (with proper error handling)
RUN git clone --depth 1 https://github.com/1alnoman/ScribbleDom.git /opt/ScribbleDom \
    && mkdir -p /usr/local/lib/R/site-library \
    && NCPUS=$(nproc) \
    && Rscript -e ' \
	    .libPaths("/usr/local/lib/R/site-library");  \
	    options(repos = c(CRAN = "https://cloud.r-project.org")); \
	    source("/opt/ScribbleDom/code_utils/read_data/installation.R"); \
	    install_libraries();'  \
    && rm -rf /tmp/* /root/.cache/R /root/.cache/RStudio
    
# ============================================================
# Stage 2 – Runtime
# ============================================================
FROM python:3.10-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    R_LIBS_SITE="/usr/local/lib/R/site-library:/usr/lib/R/site-library:/usr/lib/R/library"
    
# Runtime dependencies (non-dev versions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base-core \
    jq \
    libcurl4 libxml2 libssl3 \
    # === Runtime libs needed for scater to actually LOAD ===
    libcairo2 libharfbuzz0b libfribidi0 \
    libpng16-16 libjpeg62-turbo libtiff6 \
    libfontconfig1 libfreetype6 \
    && rm -rf /var/lib/apt/lists/*
    
# Force correct .libPaths for everyone
RUN mkdir -p /etc/R \
    && echo '.libPaths(c("/usr/local/lib/R/site-library", .libPaths()))' > /etc/R/Rprofile.site \
    && echo 'R_LIBS_SITE="/usr/local/lib/R/site-library:/usr/lib/R/site-library:/usr/lib/R/library"' >> /etc/R/Renviron.site
    
# Non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser \
    && mkdir -p /workspace /home/appuser/.cache \
    && chown -R appuser:appuser /home/appuser /workspace /usr/local/lib/R/site-library
    
# Copy artifacts
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/local/lib/R/site-library /usr/local/lib/R/site-library
COPY --from=builder --chown=appuser:appuser /opt/ScribbleDom /opt/ScribbleDom

WORKDIR /opt/ScribbleDom

COPY --chown=appuser:appuser run_scribbledom.sh /usr/local/bin/run_scribbledom.sh
RUN chmod +x /usr/local/bin/run_scribbledom.sh

USER appuser
ENTRYPOINT ["/usr/local/bin/run_scribbledom.sh"]
