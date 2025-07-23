# Linnaeus Profiling Tools

The `linnaeus-prof` command-line utility provides comprehensive tools for analyzing PyTorch profiler traces from Linnaeus experiments. It helps identify performance bottlenecks, compare runs, and generate reports suitable for LLM analysis.

## Installation

The profiling tools are included with Linnaeus. After installing Linnaeus, the `linnaeus-prof` command will be available:

```bash
pip install -e .  # Install Linnaeus in development mode
linnaeus-prof --help
```

## Quick Start

```bash
# Scan all experiment runs under a directory
linnaeus-prof scan --base-dir /datasets/modelWorkshop/mFormerV1/linnaeus-prod

# Analyze a specific run
linnaeus-prof summary /path/to/experiment/run

# Compare two runs
linnaeus-prof diff run_a/ run_b/ --output-format md --save comparison.md

# Launch TensorBoard
linnaeus-prof tensorboard --base-dir /datasets/modelWorkshop/mFormerV1/linnaeus-prod
```

## Commands

### `scan` - Discover Experiment Runs

Recursively discovers experiment runs following the `<PROJECT>/<GROUP>/<NAME>` directory structure.

```bash
linnaeus-prof scan --base-dir /datasets/modelWorkshop/mFormerV1/linnaeus-prod
```

**Options:**
- `--base-dir`: Base experiment directory to scan (required)
- `--output-format`: Output format (`pretty`, `json`, `md`)
- `--save`: Save output to file

**Example output:**
```
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Project       ┃ Group         ┃ Name                                       ┃ Last Modified   ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ linnaeus-prod │ aves_mFormerV1│ aves_mFormerV1_md_115e_v014_kernel_fusion │ 2025-07-22 22:35│
└───────────────┴───────────────┴────────────────────────────────────────────┴─────────────────┘
```

### `summary` - Analyze Single Run

Analyzes profiler traces and experiment configuration to generate performance summary.

```bash
linnaeus-prof summary /path/to/experiment/run --output-format md --save summary.md
```

**Options:**
- `run_dir`: Path to experiment run directory (required)
- `--output-format`: Output format (`pretty`, `json`, `md`)
- `--save`: Save output to file
- `--write-cache`: Cache computed summary for faster future access

**Example output:**
```markdown
# Run Summary: linnaeus-prod/aves_mFormerV1/aves_mFormerV1_md_115e_v014_kernel_fusion

## Configuration

- **Batch Size**: 84
- **Accumulation Steps**: 4
- **Model Type**: mFormerV1
- **Aug Pipeline Device**: gpu
- **GPU Compile Enabled**: True
- **GPU Compile Mode**: default

## Profiler Metrics

- **Avg Step Time**: 125.3 ms
- **GPU Utilization**: 67.2%
- **CPU Time**: 15.8%
- **GPU Time**: 84.2%
- **Memory Bandwidth**: 28.4%
- **Kernel Count**: 1247
- **Steps Profiled**: 10
```

### `diff` - Compare Two Runs

Generates side-by-side comparison of performance metrics between two experiment runs.

```bash
linnaeus-prof diff run_a/ run_b/ --output-format html --save comparison.html
```

**Options:**
- `run_a`, `run_b`: Paths to experiment run directories (required)
- `--output-format`: Output format (`pretty`, `json`, `md`, `html`)
- `--save`: Save output to file

**Features:**
- Highlights significant changes (>10% difference)
- Color-coded improvements vs regressions
- Self-contained HTML reports for sharing

### `tensorboard` - Launch TensorBoard

Launches TensorBoard with proper configuration for Linnaeus experiment directories.

```bash
linnaeus-prof tensorboard --base-dir /datasets/modelWorkshop/mFormerV1/linnaeus-prod --port 6006
```

**Options:**
- `--base-dir`: Base experiment directory for TensorBoard logdir (required)
- `--port`: TensorBoard port (default: 6006)
- `--bind-all`: Bind to all interfaces (allows remote access)

## Output Formats

### `pretty` (Default)
Rich console output with colored tables and formatting. Best for interactive use.

### `json`
Machine-readable JSON format. Suitable for programmatic analysis or integration with other tools.

### `md` (Markdown)
Human-readable markdown format. Perfect for documentation, reports, and LLM analysis.

### `html` (Diff only)
Self-contained HTML reports with styling. Great for sharing and archiving comparisons.

## Advanced Usage

### Programmatic Access

All functionality is available as Python APIs for custom scripts:

```python
from linnaeus.profiling import scanner, summary, diff

# Discover runs
runs = list(scanner.find_runs("/datasets/modelWorkshop/mFormerV1/linnaeus-prod"))

# Analyze a run
run_summary = summary.build_summary(runs[0].path)

# Compare runs
comparison = diff.compare_runs(
    summary.build_summary(runs[0].path),
    summary.build_summary(runs[1].path)
)

# Generate markdown report
report = diff.format_markdown(comparison)
```

### Caching

The `--write-cache` flag stores computed summaries in `.linnaeus_cache/` within each experiment directory:

```bash
linnaeus-prof summary /path/to/run --write-cache
# Creates: /path/to/run/.linnaeus_cache/summary.pkl
```

Cached summaries are automatically used on subsequent runs for faster analysis.

### Filtering and Automation

Use JSON output with command-line tools for filtering and automation:

```bash
# Find runs with GPU compilation enabled
linnaeus-prof scan --base-dir /experiments --output-format json | \
  jq '.[] | select(.has_gpu_compile == true)'

# Compare all runs in a directory
for run in /experiments/*/; do
  linnaeus-prof summary "$run" --output-format json --save "${run}/summary.json"
done
```

## Integration with Analysis Workflows

### For LLM Analysis

The markdown output format is specifically designed for LLM analysis:

```bash
# Generate comprehensive comparison for LLM review
linnaeus-prof diff \
  /experiments/baseline_run \
  /experiments/optimized_run \
  --output-format md \
  --save kernel_fusion_analysis.md
```

### For CI/CD Pipelines

Use JSON output to integrate with automated performance monitoring:

```python
import json
import subprocess

# Run analysis
result = subprocess.run([
    "linnaeus-prof", "summary", "/path/to/run", 
    "--output-format", "json"
], capture_output=True, text=True)

summary = json.loads(result.stdout)

# Check for regressions
if summary["profiler_metrics"]["avg_step_time_ms"] > 150:
    print("Performance regression detected!")
    exit(1)
```

## Troubleshooting

### No Profiler Traces Found

Ensure profiling is enabled in your experiment configuration:

```yaml
DEBUG:
  PROFILER:
    ENABLED: True
    OUTPUT_DIR: "{output_dir}/profiler"
    SCHEDULE: [2, 1, 5, 2]
```

### TensorBoard Plugin Issues

Install the required TensorBoard plugins:

```bash
pip install tensorboard torch-tb-profiler
```

Use Chrome for best TensorBoard compatibility (Safari has known issues with plugins).

### Permission Errors

Ensure read access to experiment directories and write access for cache files:

```bash
chmod -R +r /path/to/experiments
```

## Examples

### Daily Performance Monitoring

```bash
#!/bin/bash
# daily_perf_check.sh

BASE_DIR="/datasets/modelWorkshop/mFormerV1/linnaeus-prod"
REPORT_DIR="./performance_reports"

mkdir -p "$REPORT_DIR"

# Scan for new runs
linnaeus-prof scan --base-dir "$BASE_DIR" \
  --output-format md \
  --save "$REPORT_DIR/daily_scan_$(date +%Y%m%d).md"

# Find latest runs and compare with baseline
LATEST_RUN=$(linnaeus-prof scan --base-dir "$BASE_DIR" --output-format json | \
  jq -r '.[0].path')
BASELINE_RUN="/path/to/baseline/run"

if [ -n "$LATEST_RUN" ]; then
  linnaeus-prof diff "$BASELINE_RUN" "$LATEST_RUN" \
    --output-format html \
    --save "$REPORT_DIR/comparison_$(date +%Y%m%d).html"
fi
```

### Batch Analysis

```bash
# Analyze all runs and generate summaries
find /experiments -name "experiment_config.yaml" -exec dirname {} \; | \
while read run_dir; do
  echo "Analyzing: $run_dir"
  linnaeus-prof summary "$run_dir" \
    --output-format md \
    --write-cache \
    --save "${run_dir}/performance_summary.md"
done
```