FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV R_HOME=/usr/lib/R

# 1. Install System R + System Dependencies (Moved from environment.yml)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    pkg-config \
    r-base r-base-dev \
    libxml2-dev libssl-dev libcurl4-openssl-dev \
    libblas-dev liblapack-dev gfortran \
    && rm -rf /var/lib/apt/lists/*

# 2. Install specific mclust 5.4.10
# RUN Rscript -e "install.packages('remotes', repos='https://cloud.r-project.org/')" \
#     && Rscript -e "remotes::install_version('mclust', version='5.4.10', repos='https://cloud.r-project.org/')"

# 3. Micromamba Setup
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# 4. Create Env (This should work now!)
COPY environment.yml /tmp/environment.yml
COPY requirements.txt /tmp/requirements.txt
RUN micromamba create -y -n staig-env -f /tmp/environment.yml && micromamba clean --all --yes
ENV PATH=$MAMBA_ROOT_PREFIX/envs/staig-env/bin:$PATH

# 5. Install PyTorch + PyG (Using the high-speed wheels for CUDA 12.6)
# FIX: We use --force-reinstall and --ignore-installed to bypass the "licenses" folder error
# RUN pip install --no-cache-dir --ignore-installed \
#     torch==2.5.1 \
#     torchvision==0.20.1 \
#     torchaudio==2.5.1 \
#     --index-url https://download.pytorch.org/whl/cu124

# # Install PyG wheels separately to ensure they link correctly to the torch we just forced in
# RUN pip install --no-cache-dir \
#     pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
#     -f https://data.pyg.org/whl/torch-2.5.1+cu124.html \
#     && pip install torch-geometric

WORKDIR /opt/STAIG
RUN git clone --depth 1 https://github.com/y-itao/STAIG.git . \
    && rm -rf Dataset example .git

# 7. Environment Activation Logic
RUN micromamba shell init --shell bash --root-prefix=$MAMBA_ROOT_PREFIX
RUN echo "micromamba activate staig-env" >> ~/.bashrc

# Ensure the environment is active for the ENTRYPOINT
ENV MAMBA_DEFAULT_ENV=staig-env
ENV PATH=$MAMBA_ROOT_PREFIX/envs/staig-env/bin:$PATH

RUN pip install --no-deps GeneClust

# 8. Set the Entrypoint
COPY entrypoint.py /opt/STAIG/entrypoint.py

# Using the "exec" form of ENTRYPOINT
ENTRYPOINT ["python", "/opt/STAIG/entrypoint.py"]