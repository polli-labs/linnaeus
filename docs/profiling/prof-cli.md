# linnaeus-prof: Performance Analysis CLI

The `linnaeus-prof` command provides comprehensive tools for analyzing PyTorch profiler traces, comparing runs, and generating performance reports.

## Installation

```bash
pip install -e .  # Included with base Linnaeus
linnaeus-prof --help
```

## Commands Overview

| Command | Purpose | Output Formats |
|---------|---------|----------------|
| `scan` | Discover experiment runs | pretty, json, md |
| `summary` | Analyze single run | pretty, json, md |
| `diff` | Compare two runs | pretty, json, md, html |
| `repair` | Fix corrupted traces | - |
| `tensorboard` | Launch visualization | - |

## Command Reference

### scan - Discover Experiment Runs

Recursively discovers experiment runs in the standard directory structure.

```bash
linnaeus-prof scan --base-dir /datasets/modelWorkshop/mFormerV1/linnaeus-dev
```

**Options:**
- `--base-dir PATH`: Base experiment directory (required)
- `--output-format FORMAT`: Output format (pretty|json|md)
- `--save PATH`: Save output to file

**Example Output:**
```
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Project       ┃ Group          ┃ Name                             ┃ Last Modified    ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ linnaeus-dev  │ aves_mFormerV1 │ baseline_l3_v032                 │ 2025-08-04 15:30 │
│ linnaeus-dev  │ aves_mFormerV1 │ optimized_l3_v032                │ 2025-08-04 15:45 │
└───────────────┴────────────────┴──────────────────────────────────┴──────────────────┘
```

### summary - Analyze Single Run

Analyzes profiler traces and generates comprehensive performance summary.

```bash
linnaeus-prof summary /path/to/experiment/run --output-format md --save summary.md
```

**Options:**
- `run_dir PATH`: Experiment run directory (required)
- `--output-format FORMAT`: Output format (pretty|json|md)
- `--save PATH`: Save output to file
- `--write-cache`: Cache computed summary for faster access

**Metrics Reported:**

**Level 1 Profiling:**
- Average step time
- GPU/CPU utilization percentages
- Memory bandwidth usage
- Total kernel count

**Level 2+ Profiling:**
- Component breakdowns (data, model, loss, optimizer)
- Model stage timings (stem, convnext, rope, heads)
- Data pipeline stages (I/O, decode, transform)
- Augmentation operation timings

**Example Output:**
```markdown
# Run Summary: baseline_l3_v032

## Configuration
- **Model**: mFormerV1
- **Batch Size**: 84
- **Profiler Level**: 3
- **GPU Compile**: Enabled

## Performance Metrics
- **Avg Step Time**: 1125.3 ms
- **GPU Utilization**: 67.2%
- **Kernel Count**: 12,470

## Component Breakdown
| Component | Time (ms) | Percentage |
|-----------|-----------|------------|
| Data Loading | 45.2 | 4.0% |
| Forward Pass | 687.5 | 61.1% |
| Backward Pass | 312.8 | 27.8% |
| Optimizer Step | 79.8 | 7.1% |
```

### diff - Compare Two Runs

Generates detailed comparison between baseline and optimized runs.

```bash
linnaeus-prof diff \
  /datasets/modelWorkshop/baseline_run \
  /datasets/modelWorkshop/optimized_run \
  --output-format html \
  --save comparison.html
```

**Options:**
- `run_a PATH`: First run directory (baseline)
- `run_b PATH`: Second run directory (optimized)
- `--output-format FORMAT`: Output format (pretty|json|md|html)
- `--save PATH`: Save output to file

**Comparison Features:**
- Side-by-side configuration differences
- Percentage changes with color coding
- Kernel count differences
- Component-level timing comparisons
- Automatic significance detection (>10% changes)

**Example Output:**
```markdown
# Performance Comparison

## Summary
- **Overall Speedup**: 15.3%
- **Kernel Reduction**: 2,502 kernels (-20.1%)
- **Memory Savings**: 1.2 GB (-8.5%)

## Detailed Metrics
| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Step Time | 1125.3 ms | 953.2 ms | -15.3% ✅ |
| GPU Util | 67.2% | 78.5% | +16.8% ✅ |
| Kernels | 12,470 | 9,968 | -20.1% ✅ |

## Component Changes
| Component | Baseline | Optimized | Impact |
|-----------|----------|-----------|--------|
| rope/apply_rotary_emb | 125.3 ms | 42.1 ms | -66.4% ✅ |
| model/rope_stage_3 | 387.2 ms | 298.5 ms | -22.9% ✅ |
| loss/drop_path | 89.3 ms | 0.0 ms | -100% ✅ |
```

### repair - Fix Corrupted Traces

Automatically repairs corrupted PyTorch profiler JSON traces.

```bash
linnaeus-prof repair /path/to/experiment/assets/profiler/
```

**Features:**
- Detects common corruption patterns (H100 DDP issues)
- Creates backups before modification
- Validates repaired traces
- Integrated into diff/summary commands

**Corruption Patterns Fixed:**
- Unterminated JSON arrays
- Missing closing braces
- Truncated event records
- Invalid UTF-8 sequences

### tensorboard - Launch Visualization

Launches TensorBoard for visual profiling analysis.

```bash
linnaeus-prof tensorboard --base-dir /datasets/modelWorkshop/ --port 6006
```

**Options:**
- `--base-dir PATH`: Base directory for experiments
- `--port PORT`: TensorBoard port (default: 6006)
- `--bind-all`: Allow remote access

**TensorBoard Views:**
- **Trace View**: Timeline of GPU/CPU operations
- **Memory View**: Memory allocation patterns
- **Module View**: Per-module breakdown (Level 3)
- **Kernel View**: CUDA kernel statistics

## Output Formats

### pretty (Console)
Rich terminal output with colors and tables. Best for interactive use.

### json (Machine-Readable)
```json
{
  "run_info": {
    "name": "baseline_l3_v032",
    "path": "/datasets/...",
    "timestamp": "2025-08-04T15:30:00"
  },
  "metrics": {
    "avg_step_time_ms": 1125.3,
    "gpu_utilization": 0.672,
    "kernel_count": 12470
  }
}
```

### md (Markdown)
Human-readable format for documentation and LLM analysis.

### html (Diff Only)
Self-contained HTML with embedded CSS for sharing.

## Programmatic Usage

```python
from linnaeus.profiling import scanner, summary, diff

# Discover runs
runs = list(scanner.find_runs("/experiments"))

# Analyze single run
run_summary = summary.build_summary(runs[0].path)

# Compare runs
comparison = diff.compare_runs(
    summary.build_summary("/experiments/baseline"),
    summary.build_summary("/experiments/optimized")
)

# Generate report
report = diff.format_markdown(comparison)
print(report)
```

## Caching

Enable caching for faster repeated analysis:

```bash
linnaeus-prof summary /path/to/run --write-cache
# Creates: /path/to/run/.linnaeus_cache/summary.pkl
```

## Integration Patterns

### LLM Analysis Pipeline

```bash
#!/bin/bash
# Generate comprehensive report for LLM review
linnaeus-prof diff \
  baseline_run/ \
  optimized_run/ \
  --output-format md \
  --save optimization_analysis.md

# Feed to LLM for insights
cat optimization_analysis.md | llm-analyze
```

### CI/CD Performance Gates

```python
import json
import subprocess
import sys

# Run analysis
result = subprocess.run([
    "linnaeus-prof", "summary", "/path/to/run",
    "--output-format", "json"
], capture_output=True, text=True)

summary = json.loads(result.stdout)

# Check performance thresholds
step_time = summary["metrics"]["avg_step_time_ms"]
if step_time > 1200:
    print(f"Performance regression: {step_time}ms > 1200ms threshold")
    sys.exit(1)
```

### Batch Analysis

```bash
#!/bin/bash
# Analyze all runs in directory
for run in /experiments/*/; do
  echo "Analyzing: $(basename $run)"
  linnaeus-prof summary "$run" \
    --output-format md \
    --write-cache \
    --save "${run}/performance_summary.md"
done
```

## Troubleshooting

### No Profiler Traces Found

Ensure profiling is enabled in configuration:
```yaml
DEBUG:
  PROFILER:
    ENABLED: True
    LEVEL: 2  # or 3 for detailed analysis
    SCHEDULE: [2, 1, 5, 2]  # wait, warmup, active, repeat
```

### Corrupted Traces

Run repair before analysis:
```bash
linnaeus-prof repair /path/to/profiler/
linnaeus-prof summary /path/to/run  # Now works
```

### TensorBoard Issues

Install required plugins:
```bash
pip install tensorboard torch-tb-profiler
```

Use Chrome browser (Safari has compatibility issues).

### Memory Errors

For large traces, increase memory limit:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
linnaeus-prof summary /path/to/run
```

## Best Practices

1. **Consistent Profiling**: Use same profiler level for baseline/optimized
2. **Multiple Runs**: Average results across 3+ runs for stability
3. **Warmup Period**: Use profiler schedule with adequate warmup
4. **Cache Results**: Enable caching for large trace files
5. **Version Control**: Track profiler outputs in experiments