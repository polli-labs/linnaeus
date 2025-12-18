# linnaeus-prof-run: Automated Trial Execution

The `linnaeus-prof-run` command orchestrates multiple training trials with different configurations, enabling systematic benchmarking and A/B testing of optimizations.

## Installation

```bash
uv pip install -e ".[profiling]"
linnaeus-prof-run --help
```

## Quick Start

### 1. Define Trials (trials.jsonl)

```jsonl
{"name": "baseline", "config_file": "/configs/experiments/test.yaml", "git_ref": "main", "commit_hash": "abc123"}
{"name": "optimized", "config_file": "/configs/experiments/test.yaml", "git_ref": "feature/opt", "commit_hash": "def456"}
```

### 2. Create Docker Template (docker-compose.template.yml)

```yaml
services:
  linnaeus-training:
    image: frontierkodiak/linnaeus-dev:{{LINNAEUS_TAG}}
    # NOTE: Avoid hard-coding `container_name` if you want concurrent trials.
    # Compose project names provide isolation; explicit container names can collide.
    volumes:
      - /datasets:/datasets:ro
    environment:
      - CUDA_VISIBLE_DEVICES={{GPU_RANK}}
    command: >
      bash -c "
        git clone https://github.com/polli-labs/linnaeus.git /workspace/linnaeus &&
        cd /workspace/linnaeus &&
        git checkout {{GIT_REF}} &&
        {{COMMIT_RESET_CMD}}
        python -m linnaeus.main --cfg {{CONFIG_FILE}} --opts {{OPTS}}
      "
```

### 3. Execute Trials

```bash
# Sequential execution (single GPU)
linnaeus-prof-run \
  --trial-params-file trials.jsonl \
  --output-dir results \
  --compose-template docker-compose.template.yml \
  --timeout 600

# Concurrent execution (multi-GPU)
linnaeus-prof-run \
  --trial-params-file trials.jsonl \
  --output-dir results \
  --compose-template docker-compose-ranked.template.yml \
  --timeout 600 \
  --max-concurrent 2 \
  --gpu-assignment auto
```

## Trial Configuration

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `name` | Unique trial identifier | `"baseline_v1"` |
| `config_file` | Path to config (in container) | `"/configs/test.yaml"` |

### Optional Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `git_ref` | Branch/tag to checkout | `"main"`, `"v0.3.2"` |
| `commit_hash` | Specific commit SHA | `"abc123def456"` |
| `opts` | CLI options list | `["TRAIN.EPOCHS", "10"]` |
| `env_yaml` | Environment file | `"env_vars/dgx_h100.yaml"` |
| `env` | Direct env vars | `{"NCCL_DEBUG": "INFO"}` |
| `extra_deps` | Additional packages | `["tensorboard==2.14"]` |
| `docker_tag` | Override image tag | `"ampere-0.3.2"` |
| `gpu_rank` | Manual GPU assignment | `0`, `1` |

## Concurrent Execution

### GPU Assignment Strategies

**1. Automatic (Recommended)**
```bash
--max-concurrent 2 --gpu-assignment auto
```
GPUs assigned from pool as available.

**2. Round-Robin**
```bash
--max-concurrent 2 --gpu-assignment round-robin
```
Trials distributed evenly across GPUs.

**3. Manual**
```jsonl
{"name": "trial_gpu0", "gpu_rank": 0, ...}
{"name": "trial_gpu1", "gpu_rank": 1, ...}
```

### Performance Expectations

| GPUs | Trials | Sequential | Concurrent | Speedup |
|------|--------|------------|------------|---------|
| 2 | 2 | 20 min | 10 min | 2.0x |
| 2 | 10 | 100 min | 50 min | 2.0x |
| 4 | 20 | 200 min | 50 min | 4.0x |

## Environment Variables

### Using Scenario Files

```jsonl
{"name": "single_gpu", "env_yaml": "configs/env_vars/single_gpu_workstation.yaml"}
{"name": "multi_gpu", "env_yaml": "configs/env_vars/multi_gpu_workstation.yaml"}
{"name": "dgx_h100", "env_yaml": "configs/env_vars/dgx_h100.yaml"}
```

Note:
- `env_yaml` is read and flattened into individual `KEY=VALUE` environment entries in the compose service (it is not passed through as a docker `env_file`).

### Direct Overrides

```jsonl
{"name": "debug", "env": {"TORCH_DISTRIBUTED_DEBUG": "DETAIL", "NCCL_DEBUG": "INFO"}}
```

### Combined Approach

```jsonl
{"name": "custom", "env_yaml": "configs/env_vars/base.yaml", "env": {"CUDA_VISIBLE_DEVICES": "0,1"}}
```

## Output Structure

```
results/
├── summary.json              # Trial results summary
├── results.json             # Detailed results with timings
├── profiling_runner.log     # Execution log
└── <trial_name>/
    ├── status.txt          # SUCCESS, FAILURE, or TIMEOUT
    ├── console_log.txt     # Console output (failures)
    └── debug_log.txt       # Debug logs (if --capture-debug-logs)
```

### Summary Format

```json
{
  "trials": [
    {
      "name": "baseline",
      "status": "success",
      "elapsed_time": 145.7,
      "git_ref": "main",
      "commit_hash": "abc123"
    },
    {
      "name": "optimized",
      "status": "failure",
      "elapsed_time": 89.3,
      "error": "CUDA out of memory"
    }
  ],
  "summary": {
    "total": 2,
    "successful": 1,
    "failed": 1,
    "success_rate": 0.5
  }
}
```

## Common Patterns

### Performance Comparison

```jsonl
{"name": "baseline", "config_file": "configs/test.yaml", "git_ref": "v0.3.2"}
{"name": "compile_enabled", "config_file": "configs/test.yaml", "git_ref": "v0.3.2", "env": {"TORCH_COMPILE_DISABLE": "0"}}
{"name": "flash_attn_disabled", "config_file": "configs/test.yaml", "git_ref": "v0.3.2", "opts": ["MODEL.ROPE_STAGES.USE_FLASH_ATTN", "False"]}
```

### Multi-Environment Testing

```jsonl
{"name": "rtx3090", "config_file": "configs/test.yaml", "env_yaml": "env_vars/rtx3090.yaml"}
{"name": "rtx4090", "config_file": "configs/test.yaml", "env_yaml": "env_vars/rtx4090.yaml"}
{"name": "a100", "config_file": "configs/test.yaml", "env_yaml": "env_vars/a100.yaml"}
{"name": "h100", "config_file": "configs/test.yaml", "env_yaml": "env_vars/h100.yaml"}
```

### Branch Comparison

```jsonl
{"name": "main", "config_file": "configs/test.yaml", "git_ref": "main"}
{"name": "feature_a", "config_file": "configs/test.yaml", "git_ref": "feature/optimization-a"}
{"name": "feature_b", "config_file": "configs/test.yaml", "git_ref": "feature/optimization-b"}
```

## CLI Options

```bash
linnaeus-prof-run [OPTIONS]

Required:
  --trial-params-file PATH    JSONL file with trial definitions
  --output-dir PATH           Directory for results
  --compose-template PATH     Docker Compose template file

Execution:
  --timeout SECONDS          Timeout per trial (default: 600)
  --exit-on-failure         Stop on first failure
  --capture-debug-logs      Collect detailed debug logs

Concurrent Execution:
  --max-concurrent N        Max concurrent trials (default: 1)
  --gpu-assignment MODE     GPU assignment: auto|round-robin|manual
  --stagger-delay SECONDS   Delay between trial starts (default: 5)
```

## Troubleshooting

### Docker Issues

**Problem**: Docker compose command not found
```bash
# Solution: Install Docker Compose v2
docker compose version
```

**Problem**: Permission denied
```bash
# Solution: Add user to docker group
sudo usermod -aG docker $USER
```

### GPU Allocation

**Problem**: GPU allocation timeout
```bash
# Solution: Reduce concurrency or ensure GPUs are free
--max-concurrent 1
```

**Problem**: CUDA out of memory
```bash
# Solution: Use single-GPU templates or reduce batch size
{"opts": ["DATA.BATCH_SIZE", "32"]}
```

### Trial Failures

**Problem**: Git checkout failures
```bash
# Solution: Ensure branches are pushed to remote
git push origin feature/optimization
```

**Problem**: Config file not found
```bash
# Solution: Use absolute paths inside container
"config_file": "/workspace/configs/test.yaml"
```

## Best Practices

1. **Always specify commit hashes** for reproducibility
2. **Use consistent profiling settings** across trials
3. **Set appropriate timeouts** (minimum 3 minutes for initialization)
4. **Capture debug logs** for failed trials
5. **Test with small trials first** before long runs
6. **Use environment files** for hardware-specific settings
7. **Monitor GPU utilization** during concurrent execution
8. **Clean up old results** to save disk space

## Integration Examples

### CI/CD Pipeline

```bash
#!/bin/bash
# ci_benchmark.sh
COMMIT=$(git rev-parse HEAD)
echo "{\"name\": \"ci_${COMMIT:0:8}\", \"commit_hash\": \"$COMMIT\"}" > trials.jsonl
linnaeus-prof-run --trial-params-file trials.jsonl --timeout 300
```

### Nightly Benchmarks

```python
#!/usr/bin/env python
import json
from datetime import datetime

trials = [
    {"name": f"nightly_{datetime.now():%Y%m%d}", "git_ref": "main"},
    {"name": f"stable_{datetime.now():%Y%m%d}", "git_ref": "v0.3.2"}
]

with open("nightly_trials.jsonl", "w") as f:
    for trial in trials:
        f.write(json.dumps(trial) + "\n")
```
