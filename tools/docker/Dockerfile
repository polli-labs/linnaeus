# ───────────────────────────────────────────────────
# Linnaeus Dev/Training Container (GPU‑optimized)
# Base: CUDA 12.4.1 + cuDNN 9 on Ubuntu 22.04
# ───────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

LABEL maintainer="Caleb Sowers <caleb@polli.ai>"
LABEL description="Linnaeus training container: PyTorch + Flash‑Attention + CUDA 12.4.1"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # Set CONFIG_DIR for your application within the container
    CONFIG_DIR=/app/linnaeus/configs \
    # Add uv bin directory to PATH
    PATH="/root/.local/bin:${PATH}"

# Set working directory early
WORKDIR /app

# 1) System dependencies & ninja for Flash‑Attention
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip build-essential git curl ca-certificates \
      ninja-build htop vim openssh-client rclone \
    && rm -rf /var/lib/apt/lists/*

# 2) Upgrade pip/toolchain and ensure python symlink
RUN pip3 install --upgrade pip setuptools wheel && \
    ln -sf /usr/bin/python3 /usr/bin/python

# 3) Install PyTorch CUDA 12.4 wheels
RUN pip3 install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu124

# 4) Install Flash‑Attention with build isolation disabled and compilation optimization
RUN MAX_JOBS=4 pip3 install flash-attn --no-build-isolation

# 5) Install uv for dependency management
RUN pip3 install --no-cache-dir uv

# 6) Copy & install Linnaeus & dependencies
COPY pyproject.toml ./
COPY linnaeus/ ./linnaeus/
COPY configs/ ./configs/
RUN uv pip install --system --no-cache-dir -e .[dev]

# 7) Copy & install ibrida
COPY ibrida /app/ibrida
RUN cd /app/ibrida && pip3 install --no-cache-dir -e .

ENV PYTHONPATH="/app:${PYTHONPATH:-}"

# Final working directory
WORKDIR /app/linnaeus

# Expose ports for potential TensorBoard access
EXPOSE 6006

# --- SSH Configuration for Git ---
# Create the .ssh directory
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh
# Add github.com to known_hosts to avoid prompts (less secure but common for CI/CD)
# Consider adding specific host keys for better security if feasible
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts && chmod 600 /root/.ssh/known_hosts
# Note: SSH keys themselves must be mounted at runtime

# 8) Environment verification
RUN python3 - <<EOF
import torch, flash_attn
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("FlashAttention:", flash_attn.__version__)
EOF

CMD ["python", "-c", "import torch, linnaeus; print('Ready:', torch.__version__, linnaeus.__version__)"]

# ─── TODO (public‑repo milestone) ───────────────────────────────
# Replace COPY lines with:
# RUN git clone --depth 1 https://github.com/polli-labs/linnaeus.git /app/linnaeus
# RUN git clone --depth 1 https://github.com/polli-labs/ibrida.git   /app/ibrida