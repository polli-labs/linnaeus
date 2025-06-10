# Building and Running the linnaeus Docker Image

This guide covers building the Docker image and running containers for training, including handling dependencies, GPUs, and secrets.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)
- Git (for cloning the repository)
- `rclone` (configured for your Backblaze B2 bucket)
- An SSH key authorized to access the `linnaeus` and `ibrida` private GitHub repositories (ideally a Deploy Key).

## Building the Linnaeus training image (local dev)

```bash
# Enable BuildKit for better caching
export DOCKER_BUILDKIT=1  

# Build locally (no push)
tools/docker/build_and_push.sh
```

Base image: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`
The Dockerfile installs:
- CUDA 12.4.1 toolkit and cuDNN 9 on Ubuntu 22.04
- PyTorch CUDA 12.4 wheels from official index
- Flash-Attention with `--no-build-isolation` for optimal GPU performance
- uv and all Python deps from pyproject.toml (`uv pip install --system .[dev]`)
- rclone, git, build-essential, ninja-build for dataset sync and native extension builds
- Local linnaeus‑temp and ibrida source trees via COPY.

**Planned change**: once `polli-labs/linnaeus` and `polli-labs/ibrida`
are public, those COPY lines will be replaced with unauthenticated
`git clone` commands so the Docker build no longer requires local source
folders.

To push manually:
```bash
tools/docker/build_and_push.sh --push   # pushes polli-labs/linnaeus:dev
```

## Running the Container

### Essential Mounts and Environment Variables

When running the container for training, you typically need to provide:

-   **GPU Access:** `--gpus all`
-   **SSH Key:** `-v $HOME/.ssh/your_deploy_key:/root/.ssh/id_rsa:ro` (Mount your *read-only* deploy key)
-   **Rclone Config:** `-v $HOME/.config/rclone:/root/.config/rclone:ro` (Mount your `rclone` config)
-   **Output Directory:** `-v /path/to/host/output:/output` (Map a host directory to store results)
-   **Cache Directory:** `-v /path/to/host/cache:/root/.cache/linnaeus` (Persistent cache for checkpoints/assets)
-   **Secrets (Environment Variables):**
    *   `-e WANDB_API_KEY=$YOUR_WANDB_KEY`
    *   *(Optional rclone)* `-e RCLONE_CONFIG_B2_ACCESS_KEY_ID=$YOUR_B2_KEY_ID`
    *   *(Optional rclone)* `-e RCLONE_CONFIG_B2_SECRET_ACCESS_KEY=$YOUR_B2_SECRET_KEY` (Less secure than config file)

### Example Runtime Command (Training)

```bash
IMAGE_NAME="polli-labs/linnaeus:dev"
HOST_OUTPUT_DIR="/scratch/linnaeus_runs" # Example host path
HOST_CACHE_DIR="/scratch/linnaeus_cache" # Example host path
HOST_SSH_KEY="$HOME/.ssh/github_deploy_key_linnaeus" # Your deploy key path
HOST_RCLONE_CONF="$HOME/.config/rclone"
WANDB_KEY="YOUR_WANDB_API_KEY" # Set your key here or load from env

mkdir -p "$HOST_OUTPUT_DIR"
mkdir -p "$HOST_CACHE_DIR"

# Ensure SSH key exists and has correct permissions locally first!
# Ensure rclone config exists locally!

docker run --gpus all -it --rm \
  -v "$HOST_OUTPUT_DIR:/output" \
  -v "$HOST_CACHE_DIR:/root/.cache/linnaeus" \
  -v "$HOST_SSH_KEY:/root/.ssh/id_rsa:ro" \
  -v "$HOST_RCLONE_CONF:/root/.config/rclone:ro" \
  -e WANDB_API_KEY="$WANDB_KEY" \
  --shm-size="8g" \ # Recommended: Increase shared memory
  $IMAGE_NAME \
  bash -c " \
    echo '--- Updating Code ---' && \
    cd /app/linnaeus && git pull origin main && \
    cd /app/ibrida_src && git pull origin main && \
    echo '--- Starting Training ---' && \
    python linnaeus/main.py \
      --cfg configs/experiments/your_experiment.yaml \
      --opts \
      ENV.OUTPUT.BASE_DIR /output \
      ENV.INPUT.CACHE_DIR /root/.cache/linnaeus \
      EXPERIMENT.WANDB.KEY '' \ # Don't use key from config if env var is set
      ENV.INPUT.BUCKET.REMOTE 'your_b2_remote' \
      ENV.INPUT.BUCKET.BUCKET 'your_b2_bucket' \
      ENV.OUTPUT.BUCKET.REMOTE 'your_b2_remote' \
      ENV.OUTPUT.BUCKET.BUCKET 'your_b2_bucket' \
      # Add other CLI overrides as needed
  "
```

**Explanation:**

-   The `bash -c "..."` allows running multiple commands.
-   `git pull` updates the code inside the container *before* training starts.
-   `python linnaeus/main.py ...` runs the training script.
-   We pass relevant paths and *override* the WandB key in the config via `--opts` to ensure the environment variable is used. Configure bucket names/remotes similarly.

### Interactive Shell

```bash
docker run --gpus all -it --rm \
  -v "$HOST_OUTPUT_DIR:/output" \
  -v "$HOST_CACHE_DIR:/root/.cache/linnaeus" \
  -v "$HOST_SSH_KEY:/root/.ssh/id_rsa:ro" \
  -v "$HOST_RCLONE_CONF:/root/.config/rclone:ro" \
  -e WANDB_API_KEY="$WANDB_KEY" \
  --entrypoint /bin/bash \
  $IMAGE_NAME
```
## Troubleshooting

If you encounter issues with FlashAttention:

1. Verify CUDA compatibility inside the container:
   ```bash
   docker run --gpus all -it --rm polli-labs/linnaeus:dev -c "import torch; print(torch.cuda.get_device_capability())"
   ```

2. Check FlashAttention installation:
   ```bash
   docker run --gpus all -it --rm polli-labs/linnaeus:dev -c "import flash_attn; print(flash_attn.__version__)"
   ```

3. For GPUs older than Ampere (compute capability < 8.0), FlashAttention will automatically fall back to regular attention.