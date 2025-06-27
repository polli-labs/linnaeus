# Stage 1: Base image for compilation and dependencies
ARG NVIDIA_CUDA_TAG="12.8.0-cudnn-devel-ubuntu22.04"
FROM nvidia/cuda:${NVIDIA_CUDA_TAG} AS base

# Build arguments for PyTorch, Flash Attention, and CUDA architecture
ARG TORCH_VER="2.7.1+cu126" # still passed for stable builds
ARG FA_VER=""    # default is *blank* ⇒ skip on Turing
ARG TORCH_CHANNEL="stable"  # PyTorch channel: stable or nightly
ARG TORCH_CUDA_SUFFIX       # e.g., cu126, cu128
ARG CUDA_ARCH_LIST=""       # e.g., "7.5;8.0;8.6;9.0" - for flash-attn, if needed
ARG MAX_JOBS=""             # Number of parallel compilation jobs for ninja

# --- Environment Setup ---
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_LIST}
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# Set path for uv
ENV PATH="/root/.local/bin:$PATH"

# --- System Dependencies ---
# Add retry logic for apt operations
RUN set -e && \
    for i in 1 2 3; do \
        if apt-get update; then \
            break; \
        else \
            echo "Attempt $i failed, retrying in $(expr $i \* 10) seconds..." && \
            sleep $(expr $i \* 10); \
        fi; \
    done && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
        curl \
        unzip \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libfontconfig1 \
        libxrender1 \
        libgl1-mesa-glx \
        htop \
        nvtop \
        nano \
        rclone && \
    rm -rf /var/lib/apt/lists/*

# Link python3 to python3.11 and create python symlink
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# --- Install uv and ninja ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv pip install --system ninja && ninja --version

# --- PyTorch, TorchVision, Torchaudio ---
RUN if [ "$TORCH_CHANNEL" = "nightly" ]; then \
      uv pip install --system --pre \
          torch torchvision torchaudio \
          --index-url https://download.pytorch.org/whl/nightly/${TORCH_CUDA_SUFFIX}; \
    else \
      uv pip install --system \
          torch==${TORCH_VER} torchvision torchaudio \
          --index-url https://download.pytorch.org/whl/${TORCH_CUDA_SUFFIX}; \
    fi

# Install packaging utilities and psutil
RUN uv pip install --system packaging setuptools wheel psutil

# --- Flash Attention --------------------------------------------------
# ARG FA_VER # Already defined globally
# ARG TORCH_CHANNEL # Already defined globally
# ARG MAX_JOBS # Already defined globally

RUN echo "Processing Flash Attention installation..." && \
    if [ "$TORCH_CHANNEL" = "nightly" ]; then \
        echo "Building Flash-Attention v3 from source for nightly PyTorch..." && \
        git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /tmp/fa && \
        cd /tmp/fa && \
        MAX_JOBS=${MAX_JOBS:-6} uv pip install --system --no-build-isolation . && \
        cd / && rm -rf /tmp/fa; \
    elif [ -n "$FA_VER" ]; then \
        echo "Installing Flash-Attention version $FA_VER"; \
        ACTUAL_MAX_JOBS=${MAX_JOBS:-12} && \
        echo "Using MAX_JOBS=${ACTUAL_MAX_JOBS}" && \
        uv pip install --system \
            flash-attn==$FA_VER --no-build-isolation; \
    else \
        echo "FA_VER is empty or TORCH_CHANNEL is not nightly – skipping Flash-Attention"; \
    fi


# Stage 2: Runtime image
FROM base AS runtime

# Build argument for Linnaeus reference (branch or tag)
ARG LINNAEUS_REF=main

# Set PYTHONPATH to include the app directory (inherited from base, but good to be explicit)
ENV PYTHONPATH="/app/linnaeus:${PYTHONPATH}"

WORKDIR /app

# Clone Linnaeus repository
RUN git clone --depth 1 --branch ${LINNAEUS_REF} https://github.com/polli-labs/linnaeus.git linnaeus

WORKDIR /app/linnaeus

# Install non-torch dependencies first to avoid reinstalling torch
RUN uv pip install --system \
    yacs>=0.1.8 \
    numpy>=1.20 \
    tqdm \
    h5py \
    wandb \
    pyyaml \
    Pillow \
    opencv-python \
    pandas \
    matplotlib \
    termcolor \
    rich \
    polli-typus>=0.1.7 \
    huggingface-hub \
    python-dateutil \
    ruff \
    pytest

# Install Linnaeus without dependencies (since we installed them above)
RUN uv pip install --system --no-deps -e .[dev]

# Default command is bash shell
CMD ["/bin/bash"]
