# CI & Docker Guide

This guide covers Linnaeus's continuous integration setup and Docker image architecture.

## Overview

Linnaeus CI uses a slim Docker workflow designed to run on GitHub's free runners (14GB disk limit). The system splits heavy dependencies (CUDA, PyTorch) into pre-built base images, allowing runtime builds to complete in under 2 minutes using less than 10GB of disk space.

## Docker Image Architecture

| Image | Purpose | Typical Size | Tag Example |
|-------|---------|--------------|-------------|
| `frontierkodiak/linnaeus-base` | Heavy CUDA/DL dependencies | 6-7GB | `turing-cu126` |
| `frontierkodiak/linnaeus-dev` | Linnaeus runtime layer | 300MB layer | `turing-v0.1.1` |

**Note:** End users typically only need the runtime images. Base images exist purely for CI caching.

## Runtime Images on Free Runners

### GitHub Runner Constraints
- **Total disk space:** 14GB
- **BuildKit reserve:** 3GB (configured in workflow)
- **Available for builds:** ~11GB
- **Our peak usage:** 8-10GB

### How It Works

1. **Base images** (<8GB) contain PyTorch, CUDA runtime, and all heavy dependencies
2. **Runtime builds** pull the base and add only Linnaeus code (~300MB)
3. **Disk guard** monitors BuildKit tmp usage to ensure we stay under 12GB

### Workflow Configuration

The runtime build workflow
([`.github/workflows/build-runtime.yml`](https://github.com/polli-labs/linnaeus/blob/main/.github/workflows/build-runtime.yml))
includes several critical settings:

These GitHub links intentionally point at `polli-labs/linnaeus`, the public
release surface. If you are working in `linnaeus-dev`, use the same paths
locally in that repo.

```yaml
# Reserve disk space for BuildKit
driver-opts: |
  env.BUILDKIT_GC_KEEP_STORAGE=3g

# Target specific Docker stage to skip heavy builder
target: runtime

# Monitor disk usage after build
- name: Check BuildKit tmp size
```

## Adding New Dependencies

When adding a new Python dependency:

1. **Update `pyproject.toml`** with the new dependency
2. **Update `Dockerfile.base`** - add the same dependency to the heavy dependencies RUN command:
   ```dockerfile
   RUN uv pip install \
       numpy>=1.20 \
       pandas \
       your-new-package \  # Add here
       ...
   ```
3. **Rebuild base images** for all architectures
4. **Push updated base images** to Docker Hub
5. **Update workflow matrix** if base image tags changed

⚠️ **Important:** The runtime Dockerfile uses `--no-deps` to avoid downloading packages in CI. All dependencies MUST be in the base image.

## Rebuilding Base Images

### When to Rebuild
- PyTorch version updates
- CUDA version changes
- New Python dependencies added
- Flash Attention updates

### Build Commands

Base images use a multi-stage build that compiles in a 40GB `builder` stage but publishes only the slim `base` stage:

```bash
# Example for Ampere GPUs
docker buildx build \
  --platform linux/amd64 \
  -f tools/docker/Dockerfile.base \
  --target base \
  --build-arg MAX_JOBS=4 \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --build-arg TORCH_CHANNEL=stable \
  --build-arg TORCH_VER=2.7.1+cu126 \
  --build-arg TORCH_CUDA_SUFFIX=cu126 \
  --build-arg CUDA_ARCH_LIST="8.0;8.6" \
  --build-arg FA_VER=2.7.4.post1 \
  -t frontierkodiak/linnaeus-base:ampere-cu126 \
  --push .
```

See the [Docker build
guide](https://github.com/polli-labs/linnaeus/blob/main/tools/docker/README.md)
for complete build commands for all architectures.

### Updating the Workflow

After pushing new base images, update the matrix in `.github/workflows/build-runtime.yml`:

```yaml
matrix:
  include:
    - arch: turing
      base_tag: turing-cu126  # Update this tag
    - arch: ampere
      base_tag: ampere-cu126  # Update this tag
    - arch: hopper
      base_tag: hopper-cu128-nightly  # Update this tag
```

## Tagging and Releases

### Version Format
- **Stable releases:** `vX.Y.Z` (e.g., `v0.1.1`)
- **Pre-releases:** `vX.Y.Z-rcN` with hyphen (e.g., `v0.1.1-rc7`)

### Release Process
1. Update version in `pyproject.toml`
2. Commit changes to main branch
3. Create and push tag:
   ```bash
   git tag -a v0.1.1 -m "Release v0.1.1"
   git push origin v0.1.1
   ```
4. GitHub Actions automatically builds and pushes runtime images

## Monitoring Builds

### Build Logs
Check the Actions tab on GitHub for:
- Build duration (should be <2 minutes)
- Disk usage report (should show <10GB)
- Image push confirmation

### Common Issues

**"target stage runtime could not be found"**
- Ensure `Dockerfile.runtime` has `FROM ... AS runtime`

**Disk space errors**
- Verify base images are <8GB
- Check `target: runtime` is set in workflow
- Ensure `BUILDKIT_GC_KEEP_STORAGE=3g` is configured

**Missing dependencies**
- Add to both `pyproject.toml` AND `Dockerfile.base`
- Rebuild and push base images

## Technical Notes

### Why This Architecture?

1. **Size constraints:** Original monolithic images were 19-22GB, exceeding runner capacity
2. **Build time:** Compiling Flash Attention takes 15-20 minutes
3. **Solution:** Pre-build heavy layers, CI only adds application code

### MAX_JOBS Parameter

The `MAX_JOBS` build argument only affects base image builds (Flash Attention compilation):
- High memory systems: `MAX_JOBS=12`
- Limited memory: `MAX_JOBS=4`
- CI runtime builds: Parameter not used

For more detailed information about the Docker build system, see the [Docker
build guide](https://github.com/polli-labs/linnaeus/blob/main/tools/docker/README.md).
