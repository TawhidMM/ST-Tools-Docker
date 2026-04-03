# ============================================================
# Stage 1 – Builder (Compilers + Dev Headers)
# ============================================================
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    R_LIBS=/opt/r-library

# ---- System build deps + R repo setup ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git gnupg software-properties-common dirmngr \
    build-essential gfortran cmake \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libcairo2-dev libfontconfig1-dev \
    libharfbuzz-dev libfribidi-dev \
    libpng-dev libtiff5-dev libjpeg-dev libfreetype6-dev \
    libhdf5-dev zlib1g-dev libbz2-dev liblzma-dev libxt-dev \
    && wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
    | gpg --dearmor -o /usr/share/keyrings/r-project.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/r-project.gpg] https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/" \
    > /etc/apt/sources.list.d/r-project.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    r-base r-base-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- uv + Python 3.10 venv ----
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv venv /opt/venv --python 3.10
ENV PATH="/opt/venv/bin:$PATH" VIRTUAL_ENV=/opt/venv

# ---- PyTorch & Science Stack (Pinning NumPy to 1.x is CRITICAL) ----
# ---- Master Python Stack (Torch + Science) ----
# We combine these to prevent "os error 22" conflicts and speed up resolution.
RUN uv pip install \
    --extra-index-url https://download.pytorch.org/whl/cu116 \
    --index-strategy unsafe-best-match \
    "torch==1.13.1+cu116" \
    "torchvision==0.14.1+cu116" \
    "torchaudio==0.13.1" \
    "numpy<2.0.0" \
    "scipy==1.8.1" \
    "pandas==1.5.3" \
    "scikit-learn==1.1.1" \
    "scanpy==1.9.2" \
    "anndata==0.8.0" \
    "umap-learn==0.5.3" \
    "leidenalg>=0.10.1" \
    "igraph>=0.10.0" \
    tqdm joblib seaborn matplotlib \
    "pyqt5-sip==12.11.0"
    
# ---- Clone project ----
RUN git clone --depth 1 https://github.com/1alnoman/ScribbleDom.git /opt/ScribbleDom \
    && rm -rf /opt/ScribbleDom/preprocessed_data

# ---- R packages (into isolated directory) ----
RUN mkdir -p /opt/r-library \
    && Rscript -e \
      ".libPaths('/opt/r-library'); \
       source('/opt/ScribbleDom/code_utils/read_data/installation.R'); \
       install_libraries()"

# ---- Optimization: Strip binaries & Clean cache ----
RUN find /opt/r-library -name "*.so" -exec strip --strip-unneeded {} + 2>/dev/null || true \
    && find /opt/venv -name "*.so" -exec strip --strip-unneeded {} + 2>/dev/null || true \
    && find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true


# ============================================================
# Stage 2 – Runtime (Lean, No Compilers)
# ============================================================
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv \
    R_LIBS=/opt/r-library \
    TORCH_HOME="/home/appuser/.cache/torch" \
    MPLCONFIGDIR="/home/appuser/.config/matplotlib" \
    NUMBA_CACHE_DIR="/home/appuser/.cache/numba"

# ---- System runtime libs ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4 libssl1.1 libxml2 \
    libcairo2 libfontconfig1 libharfbuzz0b libfribidi0 \
    libpng16-16 libtiff5 libjpeg-turbo8 libfreetype6 \
    libhdf5-103 zlib1g libbz2-1.0 liblzma5 libxt6 \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- R 4.x Runtime (Copy repo config from builder) ----
COPY --from=builder /usr/share/keyrings/r-project.gpg /usr/share/keyrings/r-project.gpg
COPY --from=builder /etc/apt/sources.list.d/r-project.list /etc/apt/sources.list.d/r-project.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    && rm -rf /var/lib/apt/lists/*

# ---- Copy artifacts ----
COPY --from=builder /opt/venv        /opt/venv
COPY --from=builder /opt/r-library   /opt/r-library
COPY --from=builder /opt/ScribbleDom /opt/ScribbleDom

# ---- Non-root user setup ----
RUN useradd -m -u 1000 -s /bin/bash appuser \
    && mkdir -p /home/appuser/.cache/torch /home/appuser/.cache/numba /home/appuser/.config/matplotlib /workspace \
    && chown -R appuser:appuser /home/appuser /opt/ScribbleDom /workspace

WORKDIR /opt/ScribbleDom
COPY run_scribbledom.sh /usr/local/bin/run_scribbledom.sh
RUN chmod +x /usr/local/bin/run_scribbledom.sh

USER appuser
ENTRYPOINT ["/usr/local/bin/run_scribbledom.sh"]