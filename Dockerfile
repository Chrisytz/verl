# syntax=docker/dockerfile:experimental
# Start from the official TPU development image.
# It already has the correct Python, torch, and torch_xla.
FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/development:tpu

# Install only essential system packages that are missing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    fuse \
    libopenblas-base \
    libopenmpi-dev \
    libomp-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces


# --- vLLM Installation ---
# First, install vLLM and its TPU dependencies.
# The `requirements/tpu.txt` will use the torch/torch-xla from the base image.
RUN git clone https://github.com/vllm-project/vllm.git
WORKDIR /workspaces/vllm
RUN pip install -r requirements/tpu.txt
RUN apt-get install libopenblas-base libopenmpi-dev libomp-dev
RUN VLLM_TARGET_DEVICE="tpu" python setup.py develop

# --- VERL Installation ---
# Now, install your application and its dependencies.
WORKDIR /workspaces
RUN git clone --depth 1 https://github.com/Chrisytz/verl.git
WORKDIR /workspaces/verl
# Install VERL's dependencies first
# RUN pip install --no-cache-dir --no-deps -e . -r requirements.txt
# Now install VERL itself
RUN pip install --no-cache-dir -e .

RUN pip uninstall -y tensordict transformers
RUN pip install tensordict transformers==4.53.0

WORKDIR /workspaces/
