# ============================================================
# Stage 1 – Builder
# ============================================================
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install Python 3.10 and Build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common wget curl ca-certificates gnupg git build-essential \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y python3.10 python3.10-dev python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# 2. Install 'uv'
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 3. Setup Python Virtual Environment and Install Requirements
COPY requirements.txt .
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN uv pip install --no-cache -r requirements.txt

# 4. Clone the repository
RUN git clone --depth 1 https://github.com/TawhidMM/ScribbleDom.git /opt/ScribbleDom \
    && cd /opt/ScribbleDom \
    && rm -rf .git preprocessed_data configs


# ============================================================
# Stage 2 – Runtime (Lean)
# ============================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# 1. Install Runtime Prerequisites & Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common wget curl ca-certificates gnupg dirmngr \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y python3.10 python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# 2. Add Keys and Repo for r2u
RUN wget -qO- https://eddelbuettel.github.io/r2u/assets/dirk_eddelbuettel_key.asc \
    | tee /etc/apt/trusted.gpg.d/cranapt_key.asc \
    && gpg --keyserver keyserver.ubuntu.com --recv-keys 51716619E084DAB9 \
    && gpg --export 51716619E084DAB9 > /etc/apt/trusted.gpg.d/cran_ubuntu_key.gpg \
    && echo "deb [arch=amd64] https://r2u.stat.illinois.edu/ubuntu jammy main" \
    > /etc/apt/sources.list.d/r2u.list \
    && echo "deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" \
    > /etc/apt/sources.list.d/cran.list

# 3. Pin r2u priority so apt prefers these over standard Ubuntu repos
RUN echo "Package: *\nPin: release o=CRAN-Apt Repository\nPin-Priority: 1000" \
    > /etc/apt/preferences.d/r2u


# 4. Install R Packages & Vital Python Runtime Shared Libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base-core \
    r-cran-biocmanager \
    r-bioc-singlecellexperiment \
    r-bioc-scater \
    r-bioc-scran \
    r-bioc-rhdf5 \
    r-cran-mclust \
    r-cran-matrix \
    r-cran-tidyverse \
    r-cran-rjson \
    r-cran-rcurl \
    r-cran-assertthat \
    r-bioc-biocfilecache \
    libgomp1 \
    libgl1-mesa-glx \
    libxml2 \
    jq \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy optimized artifacts from Stage 1
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/ScribbleDom /opt/ScribbleDom

# 6. Setup User
RUN useradd -m -u 1000 -s /bin/bash appuser \
    && chown -R appuser:appuser /opt/ScribbleDom /home/appuser

WORKDIR /opt/ScribbleDom

COPY --chown=appuser:appuser ./runner /runner/
RUN chmod +x /runner/run_pipeline.py

USER appuser

ENTRYPOINT ["/runner/run_pipeline.py"]