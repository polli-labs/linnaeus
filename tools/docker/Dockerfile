# tools/docker/Dockerfile

# Use the full NVIDIA NGC container for development and compilation capabilities
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# --- Environment Setup ---
# Prevents interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# Set path for uv
ENV PATH="/root/.local/bin:$PATH"
# Set PYTHONPATH to include the app directory
ENV PYTHONPATH="/app/linnaeus:${PYTHONPATH}"

# --- System Dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
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

# --- Install uv ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# --- Application Setup ---
WORKDIR /app

# Define build arguments for source selection
ARG SOURCE=github

# Setup linnaeus code based on source type
RUN if [ "$SOURCE" = "github" ]; then \
      echo "SOURCE is github, cloning from GitHub..."; \
      git clone https://github.com/polli-labs/linnaeus.git linnaeus; \
    else \
      echo "SOURCE is local, creating directory for local copy..."; \
      mkdir -p linnaeus; \
    fi

# Copy local files only if SOURCE=local
# Docker will copy regardless, but we'll remove if not needed
COPY . ./linnaeus-temp
RUN if [ "$SOURCE" = "local" ]; then \
      echo "Using local source, copying files..."; \
      cp -r ./linnaeus-temp/* ./linnaeus/ 2>/dev/null || true; \
      cp -r ./linnaeus-temp/.[^.]* ./linnaeus/ 2>/dev/null || true; \
    fi && \
    rm -rf ./linnaeus-temp

WORKDIR /app/linnaeus

# --- Python Dependencies ---
# Define build arguments for conditional Flash Attention installation
# Set to "true" or "false" during build
ARG FLASH_ATTENTION=true

# Install Python dependencies using uv
# We install torch and flash-attn separately for better control, as recommended.
RUN uv pip install --system torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Conditionally install flash-attn based on the build argument
RUN if [ "$FLASH_ATTENTION" = "true" ]; then \
      echo "FLASH_ATTENTION is true, installing flash-attn..."; \
      uv pip install --system flash-attn>=2.5.9.post1 --no-build-isolation; \
    else \
      echo "FLASH_ATTENTION is not true, skipping flash-attn installation."; \
    fi

# Install remaining dependencies from pyproject.toml (including linnaeus itself in editable mode)
RUN uv pip install --system -e .[dev]

# --- Final Configuration ---
WORKDIR /app/linnaeus

# Set a default command (e.g., opening a shell)
CMD ["/bin/bash"]