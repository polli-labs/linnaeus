# Linnaeus Profiling Trial Runner

This tool automates the process of running a series of profiling trials for Linnaeus using Docker Compose. It enables systematic performance testing across different git branches, configurations, and environment settings.

## Overview

The `run_profiling_trials.py` script orchestrates a "round" of trials defined in a JSON Lines (`.jsonl`) file. For each trial, it:

1. Reads the trial parameters (config file, git ref, environment settings, etc.)
2. Creates a temporary, modified `docker-compose.yml` file
3. Launches the `linnaeus-training` container
4. Monitors the container's log output for success or failure conditions
5. Enforces a timeout to prevent stalled runs
6. Cleans up the container and temporary files
7. On failure, saves the console log and optionally the full `debug_log_rank0.txt` for analysis

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- A configured `docker-compose.yml` template for `linnaeus-training`
- Python dependencies: `pip install pyyaml rich` (or use the optional `[profiling]` extras)

## Installation

If Linnaeus is installed with profiling extras:
```bash
pip install -e ".[profiling]"
```

Or install dependencies manually:
```bash
pip install pyyaml rich
```

## Usage

### 1. Define Your Trials

Create a `trials.jsonl` file. Each line must be a valid JSON object describing one trial.

**Example `trials.jsonl`:**
```jsonl
{"name": "baseline", "config_file": "configs/experiments/example.yaml", "git_ref": "main"}
{"name": "optimized", "config_file": "configs/experiments/example.yaml", "git_ref": "feature/optimization", "opts": ["TRAIN.EPOCHS", "10"]}
{"name": "dgx_test", "config_file": "configs/experiments/example.yaml", "git_ref": "main", "env_yaml": "configs/env_vars/dgx_h100.yaml"}
```

### 2. Create a Docker Compose Template

The template should use placeholders that will be replaced:
- `{{GIT_REF}}` - Git branch/tag to checkout
- `{{COMMIT_HASH}}` - Optional specific commit
- `{{CONFIG_FILE}}` - Path to config file
- `{{OPTS_STRING}}` - Additional --opts parameters

See `work/fixtures/docker-compose.template.yml` for an example.

### 3. Run the Profiling Trials

Execute from the command line:
```bash
python -m linnaeus.tools.profiling.run_profiling_trials \
    --trial-params-file work/fixtures/trials.jsonl \
    --output-dir work/profiling_results/experiment1 \
    --compose-template work/fixtures/docker-compose.template.yml \
    --timeout 300 \
    --capture-debug-logs
```

Or using the CLI if installed:
```bash
linnaeus-prof-run \
    --trial-params-file work/fixtures/trials.jsonl \
    --output-dir work/profiling_results/experiment1 \
    --compose-template work/fixtures/docker-compose.template.yml \
    --timeout 300
```

### Command-Line Arguments

- `--trial-params-file` (Required): Path to the JSONL file defining the trials
- `--output-dir` (Required): Directory to save status and failure logs
- `--compose-template` (Required): Path to the base `docker-compose.yml` template
- `--timeout` (Optional, default 180): Timeout in seconds for each trial
- `--exit-on-failure` (Optional flag): Exit after the first failed trial
- `--capture-debug-logs` (Optional flag): Copy `debug_log_rank0.txt` from experiment output on failure

## Trial Parameters

Each trial in the JSONL file can specify:

- `name` (required): Unique name for the trial
- `config_file` (required): Path to the experiment config
- `git_ref`: Git branch/tag to use (default: "main")
- `commit_hash`: Specific commit to checkout
- `opts`: List of additional --opts parameters
- `env_yaml`: Path to environment variables YAML file
- `env`: Dictionary of environment variable overrides
- `extra_deps`: List of additional pip packages to install

**Example with all options:**
```json
{
  "name": "full_example",
  "config_file": "configs/experiments/test.yaml",
  "git_ref": "feature/branch",
  "commit_hash": "abc123",
  "opts": ["TRAIN.EPOCHS", "5", "DATA.BATCH_SIZE", "32"],
  "env_yaml": "configs/env_vars/dgx_h100.yaml",
  "env": {"CUDA_VISIBLE_DEVICES": "0,1"},
  "extra_deps": ["kornia>=0.8.1,<0.9"]
}
```

## Output

The script creates an output directory containing:
- `summary.json`: Overview of all trials with status and timing
- `<trial_name>_failure.log`: Console output for failed trials
- `<trial_name>_debug_log.txt`: Full debug log (if `--capture-debug-logs` is used)

## Success/Failure Detection

The tool monitors logs for specific strings:
- Success: `"DEBUG: Early exiting main training loop"`
- Failure: `"Emergency shutdown initiated"`

These can be customized by modifying the constants in the script.

## Integration with Linnaeus Profiler

After running trials, use the Linnaeus profiler CLI to analyze results:

```bash
python -m linnaeus.profiling.cli diff \
    /path/to/baseline/profiler_output \
    /path/to/optimized/profiler_output \
    --output-format md \
    --save comparison.md
```