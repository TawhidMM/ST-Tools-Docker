# ==================================================
# ScribbleDom – CUDA + Micromamba + R (ROBUST)
# ==================================================
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ----------------
# System deps (minimal)
# ----------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    jq \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# ----------------
# Micromamba (single binary, no base env)
# ----------------
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=/opt/micromamba/bin:$PATH

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj bin/micromamba && \
    mv bin/micromamba /usr/local/bin/micromamba && \
    chmod +x /usr/local/bin/micromamba

# ----------------
# Create Conda environment (Python + R)
# ----------------
WORKDIR /opt
COPY environment.yml .

RUN micromamba create -y -n scribble_dom -f environment.yml && \
    micromamba clean --all --yes

ENV CONDA_DEFAULT_ENV=scribble_dom
ENV PATH=/opt/micromamba/envs/scribble_dom/bin:$PATH

# ----------------
# Clone project
# ----------------
RUN git clone --depth 1 https://github.com/1alnoman/ScribbleDom.git /opt/ScribbleDom && \
    rm -rf /opt/ScribbleDom/preprocessed_data
WORKDIR /opt/ScribbleDom

# ----------------
# Install R libraries
# ----------------
RUN micromamba run -n scribble_dom \
    Rscript -e "source('code_utils/read_data/installation.R'); install_libraries()"


# ----------------
# Entrypoint
# ----------------
COPY run_scribbledom.sh /usr/local/bin/run_scribbledom.sh
RUN chmod +x /usr/local/bin/run_scribbledom.sh
ENTRYPOINT ["/usr/local/bin/run_scribbledom.sh"]
