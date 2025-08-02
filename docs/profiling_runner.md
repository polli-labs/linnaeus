# Profiling Runner Workflow

The Linnaeus profiling runner provides an automated way to execute multiple training trials with different configurations, git branches, and environment settings. This is essential for performance profiling, A/B testing optimizations, and systematic benchmarking.

## Overview

The profiling runner orchestrates multiple "trials" defined in a JSON Lines file, where each trial can specify:
- Different git branches/commits to test
- Different configuration files
- Different environment variable scenarios
- Different CLI options

## Quick Start

1. **Install with profiling dependencies:**
   ```bash
   pip install -e ".[profiling]"
   ```

2. **Create a trials file:**
   ```bash
   cat > work/my_trials.jsonl << EOF
   {"name": "baseline", "config_file": "configs/experiments/example.yaml", "git_ref": "main"}
   {"name": "optimized", "config_file": "configs/experiments/example.yaml", "git_ref": "feature/optimization"}
   EOF
   ```

3. **Run the trials:**
   ```bash
   linnaeus-prof-run \
     --trial-params-file work/my_trials.jsonl \
     --output-dir work/results \
     --compose-template work/fixtures/docker-compose.template.yml \
     --timeout 300
   ```

## Workflow Components

### 1. Trial Definition (JSONL)

Each line in the trials file defines one trial as a JSON object:

```jsonl
{"name": "baseline_v1", "config_file": "configs/exp.yaml", "git_ref": "v0.1.5"}
{"name": "optimized_v1", "config_file": "configs/exp.yaml", "git_ref": "v0.1.6", "opts": ["TRAIN.EPOCHS", "10"]}
{"name": "dgx_test", "config_file": "configs/exp.yaml", "git_ref": "main", "env_yaml": "configs/env_vars/dgx_h100.yaml"}
```

**Supported trial parameters:**
- `name` (required): Unique identifier for the trial
- `config_file` (required): Path to experiment configuration
- `git_ref`: Git branch, tag, or commit to checkout (default: "main")
- `commit_hash`: Specific commit SHA to pin to
- `opts`: List of additional `--opts` parameters
- `env_yaml`: Path to environment variables YAML file
- `env`: Dictionary of direct environment variable overrides
- `extra_deps`: List of additional pip packages to install

### 2. Docker Compose Template

The runner uses a template docker-compose.yml file with placeholders:

```yaml
services:
  linnaeus-training:
    image: linnaeus:latest
    command: >
      bash -c "
        git checkout {{GIT_REF}} &&
        {{COMMIT_RESET_CMD}}
        python -m linnaeus.main --cfg {{CONFIG_FILE}}{{OPTS_STRING}}
      "
```

**Template placeholders:**
- `{{GIT_REF}}`: Replaced with trial's git_ref
- `{{COMMIT_HASH}}`: Replaced with commit hash if specified  
- `{{COMMIT_RESET_CMD}}`: Git reset command if commit_hash provided
- `{{CONFIG_FILE}}`: Replaced with trial's config_file
- `{{OPTS_STRING}}`: Replaced with formatted --opts parameters

### 3. Execution Flow

For each trial, the runner:

1. **Preparation**
   - Creates a temporary docker-compose.yml with trial-specific substitutions
   - Adds environment file references and variable overrides

2. **Execution**
   - Launches `docker compose up` with the temporary compose file
   - Monitors log output in real-time
   - Enforces timeout limits

3. **Success/Failure Detection**
   - **Success**: Detects `"DEBUG: Early exiting main training loop"` in logs
   - **Failure**: Detects `"Emergency shutdown initiated"` in logs
   - **Timeout**: Kills container after specified timeout

4. **Cleanup**
   - Stops and removes containers
   - Saves failure logs and debug information
   - Records trial results in summary.json

## Environment Variable Integration

The profiling runner integrates with Linnaeus's environment variable system:

### Using Scenario Files

```jsonl
{"name": "single_gpu", "config_file": "configs/exp.yaml", "env_yaml": "configs/env_vars/single_gpu_workstation.yaml"}
{"name": "multi_gpu", "config_file": "configs/exp.yaml", "env_yaml": "configs/env_vars/multi_gpu_workstation.yaml"}
{"name": "dgx_h100", "config_file": "configs/exp.yaml", "env_yaml": "configs/env_vars/dgx_h100.yaml"}
```

### Direct Environment Overrides

```jsonl
{"name": "debug_run", "config_file": "configs/exp.yaml", "env": {"TORCH_DISTRIBUTED_DEBUG": "DETAIL", "NCCL_DEBUG": "INFO"}}
```

### Combined Approach

```jsonl
{"name": "custom", "config_file": "configs/exp.yaml", "env_yaml": "configs/env_vars/dgx_h100.yaml", "env": {"CUDA_VISIBLE_DEVICES": "0,1"}}
```

## Output and Results

### Directory Structure

```
work/results/
├── summary.json                    # Overall results summary
├── trial_name_failure.log         # Console output for failed trials
├── trial_name_debug_log.txt        # Full debug log (if --capture-debug-logs)
└── docker-compose.trial_name.yml  # Generated compose files (temporary)
```

### Summary Format

The `summary.json` contains results for all trials:

```json
[
  {
    "name": "baseline",
    "status": "success", 
    "returncode": 0,
    "elapsed_time": 145.7,
    "git_ref": "main",
    "commit_hash": null
  },
  {
    "name": "optimized",
    "status": "failure",
    "returncode": 2, 
    "elapsed_time": 89.3,
    "git_ref": "feature/optimization",
    "failure_log": "work/results/optimized_failure.log"
  }
]
```

## Common Patterns

### Performance Comparison

Compare different optimization approaches:

```jsonl
{"name": "baseline", "config_file": "configs/perf_test.yaml", "git_ref": "v0.1.5"}
{"name": "compile_enabled", "config_file": "configs/perf_test.yaml", "git_ref": "v0.1.5", "env": {"TORCH_COMPILE_DISABLE": "0"}}
{"name": "static_graph", "config_file": "configs/perf_test.yaml", "git_ref": "v0.1.5", "opts": ["DISTRIBUTED.DDP.static_graph", "true"]}
```

### Multi-Environment Testing

Test the same configuration across different hardware scenarios:

```jsonl
{"name": "single_gpu", "config_file": "configs/test.yaml", "env_yaml": "configs/env_vars/single_gpu_workstation.yaml"}
{"name": "multi_gpu", "config_file": "configs/test.yaml", "env_yaml": "configs/env_vars/multi_gpu_workstation.yaml"}
{"name": "dgx_h100", "config_file": "configs/test.yaml", "env_yaml": "configs/env_vars/dgx_h100.yaml"}
```

### Branch Comparison

Compare feature branches against baseline:

```jsonl
{"name": "main_baseline", "config_file": "configs/comparison.yaml", "git_ref": "main"}
{"name": "feature_A", "config_file": "configs/comparison.yaml", "git_ref": "feature/optimization-A"}
{"name": "feature_B", "config_file": "configs/comparison.yaml", "git_ref": "feature/optimization-B"}
```

## Best Practices

### Trial Design

1. **Use descriptive names** that clearly indicate what's being tested
2. **Include baseline trials** for meaningful comparisons
3. **Test one variable at a time** when possible for clear attribution
4. **Use consistent configurations** across trials when comparing specific changes

### Resource Management

1. **Set appropriate timeouts** based on expected training duration
2. **Use `--exit-on-failure`** for fast feedback during development
3. **Monitor disk space** as failed trials can generate large log files
4. **Clean up intermediate results** regularly

### Environment Variables

1. **Use scenario files** for consistent hardware-specific settings
2. **Override specific variables** only when testing those specific changes
3. **Document custom environments** in comments within JSONL files

### Integration with Analysis

After running trials, analyze results with:

```bash
# Compare profiler outputs
python -m linnaeus.profiling.cli diff \
  /path/to/baseline/profiler_output \
  /path/to/optimized/profiler_output \
  --output-format md

# Parse summary results
python -c "
import json
with open('work/results/summary.json') as f:
    results = json.load(f)
    
successful = [r for r in results if r['status'] == 'success']
print(f'Success rate: {len(successful)}/{len(results)}')

for r in successful:
    print(f'{r[\"name\"]}: {r[\"elapsed_time\"]:.1f}s')
"
```

## Troubleshooting

### Common Issues

1. **Docker compose not found**
   - Ensure Docker and Docker Compose are installed
   - The runner tries both `docker compose` and `docker-compose`

2. **Template substitution errors**
   - Check that all required placeholders are present in template
   - Verify JSONL syntax is valid

3. **Timeout too short**
   - Increase `--timeout` value for longer-running experiments
   - Check logs to see where training gets stuck

4. **Permission errors**
   - Ensure docker daemon is accessible
   - Check volume mount permissions in compose template

5. **Git checkout failures**
   - Ensure git repository is clean before running trials
   - Verify that specified git_ref exists and is accessible

### Debug Mode

For debugging, create a minimal test trial:

```jsonl
{"name": "debug", "config_file": "configs/minimal.yaml", "git_ref": "main", "opts": ["TRAIN.EPOCHS", "1", "DEBUG.ENABLED", "true"]}
```

Run with verbose output:
```bash
linnaeus-prof-run \
  --trial-params-file debug_trial.jsonl \
  --output-dir debug_results \
  --compose-template work/fixtures/docker-compose.template.yml \
  --timeout 60 \
  --capture-debug-logs \
  --exit-on-failure
```