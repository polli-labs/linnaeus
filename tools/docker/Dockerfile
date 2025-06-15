# Stage 1: Base image for compilation and dependencies
ARG NVIDIA_CUDA_TAG="12.8.0-cudnn-devel-ubuntu22.04"
FROM nvidia/cuda:${NVIDIA_CUDA_TAG} AS base

# Build arguments for PyTorch, Flash Attention, and CUDA architecture
ARG TORCH_VER="2.7.1+cu126" # Example: 2.7.1+cu126
ARG FA_VER="2.7.4.post1"    # Example: 2.7.4.post1
ARG TORCH_CHANNEL="stable"  # PyTorch channel: stable or nightly
ARG TORCH_CUDA_SUFFIX       # e.g., cu126, cu128 - must be passed by build.sh
ARG CUDA_ARCH_LIST=""       # e.g., "7.5;8.0;8.6;9.0" - for flash-attn, if needed

# --- Environment Setup ---
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_LIST}
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# Set path for uv
ENV PATH="/root/.local/bin:$PATH"

# --- System Dependencies ---
# Add retry logic for apt operations
RUN apt-get update || (sleep 10 && apt-get update) || (sleep 20 && apt-get update) && \
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
    && rm -rf /var/lib/apt/lists/*

# Link python3 to python3.11 and create python symlink
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# --- Install uv and ninja ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv pip install --system ninja && ninja --version

# --- Python Core Dependencies ---
# Install PyTorch, TorchVision, TorchAudio
# The TORCH_VER should be the full version string like "2.7.1+cu126"
# uv should be able to parse this and find the correct package.
# If a specific index URL is needed, the build script would have to provide it
# or this line would need to be adjusted. For now, relying on uv's capabilities.
RUN if [ "$TORCH_CHANNEL" = "nightly" ]; then \
        echo "Installing PyTorch from NIGHTLY channel: ${TORCH_VER} for CUDA ${TORCH_CUDA_SUFFIX}"; \
        uv pip install --system --pre \
            torch==${TORCH_VER} torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/nightly/${TORCH_CUDA_SUFFIX}; \
    else \
        echo "Installing PyTorch from STABLE channel: ${TORCH_VER} for CUDA ${TORCH_CUDA_SUFFIX}"; \
        uv pip install --system \
            torch==${TORCH_VER} torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/${TORCH_CUDA_SUFFIX}; \
    fi

# Install packaging utilities and psutil
RUN uv pip install --system packaging setuptools wheel psutil

# Install Flash Attention
RUN if [ -n "${FA_VER}" ]; then \
      echo "Attempting to install flash-attn version: ${FA_VER}"; \
      uv pip install --system flash-attn==${FA_VER} --no-build-isolation; \
    else \
      echo "FA_VER is empty, skipping flash-attn installation."; \
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

# Install Linnaeus and its development dependencies
RUN uv pip install --system -e .[dev]

# Set the entrypoint for the container
ENTRYPOINT ["python", "-m", "linnaeus.main"]

# Set a default command (e.g., opening a shell if entrypoint is not used or overridden)
CMD ["/bin/bash"]
