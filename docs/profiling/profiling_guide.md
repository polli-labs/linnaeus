# Linnaeus Profiling Guide

This comprehensive guide covers all aspects of profiling in Linnaeus, including automated trial execution, performance analysis, and optimization workflows.

## Table of Contents

1. [Installation](#installation)
2. [Profiling Workflow Overview](#profiling-workflow-overview)
3. [Trial Execution with linnaeus-prof-run](#trial-execution-with-linnaeus-prof-run)
4. [Performance Analysis with linnaeus-prof](#performance-analysis-with-linnaeus-prof)
5. [Concurrent GPU Execution](#concurrent-gpu-execution)
6. [Best Practices](#best-practices)

## Installation

The profiling tools are included with Linnaeus. Install with profiling dependencies:

```bash
pip install -e ".[profiling]"  # Install Linnaeus with profiling support
linnaeus-prof --help            # Verify CLI is available
```

## Profiling Workflow Overview

The Linnaeus profiling system provides a complete pipeline for:
1. **Trial Definition**: Define multiple experimental configurations in JSONL format
2. **Automated Execution**: Run trials in isolated Docker containers with proper instrumentation
3. **Performance Analysis**: Analyze PyTorch profiler traces and compare results
4. **Optimization Validation**: Quantify performance improvements from code changes

## Trial Execution with linnaeus-prof-run

The `linnaeus-prof-run` command orchestrates multiple training trials with different configurations, git branches, and environment settings.

### Quick Start

1. **Create a trials file** (`trials.jsonl`):
```jsonl
{"name": "baseline", "config_file": "/configs/experiments/tests/trial_template.yaml", "git_ref": "main", "commit_hash": "abc123", "opts": ["EXPERIMENT.NAME", "baseline_run"]}
{"name": "optimized", "config_file": "/configs/experiments/tests/trial_template.yaml", "git_ref": "feature/optimization", "commit_hash": "def456", "opts": ["EXPERIMENT.NAME", "optimized_run"]}
```

2. **Run trials sequentially** (default):
```bash
linnaeus-prof-run \
  --trial-params-file trials.jsonl \
  --output-dir results \
  --compose-template docker-compose.template.yml \
  --timeout 600 \
  --capture-debug-logs
```

3. **Run trials concurrently** (2x speedup on dual-GPU systems):
```bash
linnaeus-prof-run \
  --trial-params-file trials.jsonl \
  --output-dir results \
  --compose-template docker-compose-single-gpu-ranked.template.yml \
  --timeout 600 \
  --capture-debug-logs \
  --max-concurrent 2 \
  --gpu-assignment auto \
  --stagger-delay 10
```

### Trial Definition Format

Each line in the JSONL file defines one trial:

| Parameter | Required | Description |
|-----------|----------|-------------|
| `name` | Yes | Unique identifier for the trial |
| `config_file` | Yes | Path to experiment configuration (inside container) |
| `git_ref` | No | Git branch, tag, or commit to checkout (default: "main") |
| `commit_hash` | No | Specific commit SHA to pin to |
| `opts` | No | List of additional `--opts` parameters |
| `env_yaml` | No | Path to environment variables YAML file |
| `env` | No | Dictionary of direct environment variable overrides |
| `extra_deps` | No | List of additional pip packages to install |
| `gpu_rank` | No | Specific GPU ID for manual assignment (concurrent mode) |

### Docker Compose Templates

Templates use placeholders that are replaced at runtime:

```yaml
services:
  linnaeus-training:
    image: frontierkodiak/linnaeus-dev:{{LINNAEUS_TAG}}
    container_name: linnaeus-training-{{TRIAL_NAME}}
    command: >
      bash -c "
        git clone https://github.com/polli-labs/linnaeus.git /workspace/linnaeus;
        cd /workspace/linnaeus;
        git checkout {{GIT_REF}};
        python -m linnaeus.main --cfg {{CONFIG_FILE}} --opts {{OPTS}}
      "
```

**Placeholders:**
- `{{TRIAL_NAME}}`: Trial name from JSONL
- `{{LINNAEUS_TAG}}`: Docker image tag based on git_ref
- `{{GIT_REF}}`: Git reference to checkout
- `{{CONFIG_FILE}}`: Configuration file path
- `{{OPTS}}`: Formatted --opts parameters
- `{{GPU_RANK}}`: GPU ID for ranked templates

### Execution Output

The runner generates structured output:

```
results/
├── summary.json              # Overall trial results
├── results.json             # Detailed results with timings
├── profiling_runner.log    # Execution log
└── <trial_name>/           # Per-trial outputs
    ├── status.txt         # SUCCESS, FAILURE, or TIMEOUT
    ├── console_log.txt    # Console output (failures only)
    └── debug_log.txt      # Debug logs (if --capture-debug-logs)
```

## Performance Analysis with linnaeus-prof

The `linnaeus-prof` CLI provides tools for analyzing profiler traces and comparing performance between runs.

### Commands

#### `scan` - Discover Experiment Runs

Recursively discovers experiment runs in the standard directory structure:

```bash
linnaeus-prof scan --base-dir /datasets/modelWorkshop/mFormerV1/linnaeus-dev
```

**Options:**
- `--base-dir`: Base experiment directory to scan
- `--output-format`: Output format (`pretty`, `json`, `md`)
- `--save`: Save output to file

#### `summary` - Analyze Single Run

Analyzes profiler traces and generates performance summary:

```bash
linnaeus-prof summary /path/to/experiment/run --output-format md --save summary.md
```

**Metrics reported:**
- Average step time
- GPU/CPU utilization
- Memory bandwidth usage
- Kernel counts
- Component-level breakdowns (if Level 2+ profiling)

#### `diff` - Compare Two Runs

Generates side-by-side comparison between baseline and optimized runs:

```bash
linnaeus-prof diff \
  /datasets/modelWorkshop/mFormerV1/linnaeus-dev/baseline_run \
  /datasets/modelWorkshop/mFormerV1/linnaeus-dev/optimized_run \
  --output-format md \
  --save comparison.md
```

**Comparison includes:**
- Configuration differences
- Performance metric changes (absolute and percentage)
- Kernel count differences
- Memory usage changes
- Component timing breakdowns

#### `repair` - Fix Corrupted Traces

Automatically repairs corrupted PyTorch profiler traces (common on H100 GPUs):

```bash
linnaeus-prof repair /path/to/experiment/assets/profiler/
```

The repair tool:
- Detects JSON corruption patterns
- Applies heuristic fixes
- Creates backups before modification
- Validates repaired traces

#### `tensorboard` - Launch Visualization

Launch TensorBoard for visual profiling analysis:

```bash
linnaeus-prof tensorboard --base-dir /datasets/modelWorkshop/mFormerV1/
```

## Concurrent GPU Execution

The profiling runner supports concurrent execution on multiple GPUs for ~2x speedup.

### GPU Assignment Strategies

1. **Automatic Assignment** (recommended):
```bash
--max-concurrent 2 --gpu-assignment auto
```
Trials are automatically assigned to next available GPU.

2. **Manual Assignment**:
```jsonl
{"name": "baseline_gpu0", "gpu_rank": 0, ...}
{"name": "optimized_gpu1", "gpu_rank": 1, ...}
```
Specify `gpu_rank` in trials.jsonl for explicit GPU assignment.

3. **Round-Robin**:
```bash
--max-concurrent 2 --gpu-assignment round-robin
```
Trials distributed evenly across GPUs in rotating order.

### Concurrent Execution Features

- **GPU Pool Management**: Thread-safe allocation with FIFO queuing
- **Isolation**: Each trial runs in separate Docker container with CUDA_VISIBLE_DEVICES
- **Staggered Starts**: Optional delay between trial starts to reduce contention
- **Progress Tracking**: Real-time monitoring of GPU utilization and trial status
- **Error Recovery**: Automatic GPU release on trial failure

### Performance Expectations

| Scenario | GPUs | Trials | Sequential Time | Concurrent Time | Speedup |
|----------|------|--------|-----------------|-----------------|---------|
| A/B Testing | 2 | 2 | 20 min | 10 min | 2.0x |
| Parameter Sweep | 2 | 10 | 100 min | 50 min | 2.0x |
| Mixed Workloads | 2 | 6 | 60 min | 35 min | 1.7x |

## Best Practices

### 1. Trial Organization

- Use descriptive trial names that indicate the optimization being tested
- Always specify exact commit hashes for reproducibility
- Keep all trials in a single JSONL file for related experiments

### 2. Profiling Configuration

- Use consistent profiling schedules across baseline and optimized trials
- Level 2 profiling (~5% overhead) is usually sufficient for optimization work
- Level 3 profiling provides per-module breakdowns but has higher overhead

### 3. Timeout Settings

- Set timeout to at least 3 minutes to ensure training actually starts
- For profiling-only runs, 5-10 minutes is typically sufficient
- Account for model initialization time in timeout calculations

### 4. Environment Variables

- Always set `TORCH_DISTRIBUTED_DEBUG=OFF` to prevent hangs
- Use env_yaml files for hardware-specific configurations
- Override critical variables directly in the `env` field

### 5. Debugging Failed Trials

- Use `--capture-debug-logs` to collect detailed error information
- Check `/datasets/modelWorkshop/` for experiment-specific logs
- Review both console_log.txt and debug_log.txt for different error contexts

### 6. Performance Analysis

- Always compare against a stable baseline from main branch
- Run multiple profiling cycles to account for variance
- Focus on relative improvements rather than absolute timings
- Consider both kernel counts and execution time

## Troubleshooting

### Common Issues

**GPU Allocation Timeout**
- Reduce `--max-concurrent` value
- Check for stuck trials with `docker ps`
- Use `--gpu-timeout` to adjust wait time

**Docker Compose Failures**
- Ensure Docker daemon is running
- Check disk space for container images
- Verify network connectivity for image pulls

**Profiler Trace Corruption**
- Run `linnaeus-prof repair` on trace directory
- Common on H100 GPUs with DDP
- Automatic repair integrated in diff command

**Configuration Mismatches**
- Ensure all trials use same base template
- Verify YACS list parameters are fully specified
- Check for missing config keys in error logs

## Advanced Topics

### Custom Docker Templates

Create specialized templates for different hardware or configurations:

```yaml
# docker-compose-a100.template.yml
services:
  linnaeus-training:
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              device_ids: ['{{GPU_RANK}}']
```

### Integration with CI/CD

Automate profiling in continuous integration:

```bash
#!/bin/bash
# ci_profiling.sh
git_hash=$(git rev-parse HEAD)
echo "{\"name\": \"ci_test\", \"commit_hash\": \"$git_hash\"}" > trials.jsonl
linnaeus-prof-run --trial-params-file trials.jsonl --timeout 300
```

### Multi-Node Profiling

For distributed training across nodes, use environment scenarios:

```yaml
# env_vars/multi_node.yaml
WORLD_SIZE: 4
RANK: 0
MASTER_ADDR: node0
MASTER_PORT: 29500
```

## Summary

The Linnaeus profiling system provides a complete solution for:
- Automated trial execution with Docker isolation
- Concurrent GPU utilization for faster iteration
- Comprehensive performance analysis tools
- Reproducible benchmarking workflows

For the latest updates and examples, see the [Linnaeus repository](https://github.com/polli-labs/linnaeus).