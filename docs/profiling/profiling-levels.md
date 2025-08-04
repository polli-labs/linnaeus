# Multi-Level Profiling Configuration

Linnaeus provides a configurable multi-level profiling system to balance instrumentation detail with runtime overhead.

## Profiling Levels Overview

| Level | Name | Overhead | Use Case |
|-------|------|----------|----------|
| 0 | OFF | 0% | Production training |
| 1 | Lite | ~1-2% | High-level timing |
| 2 | Component | ~5% | Component breakdown |
| 3 | Deep | ~10-15% | Per-module analysis |

## Configuration

Set profiling level in your experiment configuration:

```yaml
DEBUG:
  PROFILER:
    ENABLED: True
    LEVEL: 2  # Choose 1, 2, or 3
    OUTPUT_DIR: "{output_dir}/assets/profiler"
    SCHEDULE: [2, 1, 5, 2]  # [wait, warmup, active, repeat]
    RECORD_SHAPES: False
    WITH_STACK: False
    SYNC_PROFILING: False  # Enable for accurate GPU timing
```

## Level 1: Lite Profiling

Captures high-level timing for major training components.

### What's Measured
- Total step time
- Data loading time
- Forward pass duration
- Backward pass duration
- Optimizer step time
- Basic GPU utilization

### Output Example
```
Step 100: 1125.3ms (data: 45.2ms, forward: 687.5ms, backward: 312.8ms, optimizer: 79.8ms)
```

### When to Use
- Production training monitoring
- Quick performance checks
- Baseline measurements
- Long training runs

## Level 2: Component Profiling

Detailed breakdown of model stages and data pipeline components.

### What's Measured

**Model Components:**
- Stem processing
- ConvNeXt stages (1-4)
- RoPE stages (3-4)
- Classification heads
- Loss computation

**Data Pipeline:**
- Queue wait time
- I/O operations
- CPU decoding
- GPU transfer
- Augmentation stages

**Training Components:**
- Gradient computation
- All-reduce operations (DDP)
- Parameter updates
- Memory allocations

### Output Example
```
Component Breakdown:
├── data_pipeline: 45.2ms
│   ├── queue_wait: 2.1ms
│   ├── io_read: 15.3ms
│   ├── cpu_decode: 18.7ms
│   └── gpu_transform: 9.1ms
├── model_forward: 687.5ms
│   ├── stem: 23.4ms
│   ├── convnext_stages: 234.5ms
│   ├── rope_stages: 389.2ms
│   └── heads: 40.4ms
└── optimizer: 79.8ms
    ├── gradient_clip: 5.2ms
    └── parameter_update: 74.6ms
```

### When to Use
- Optimization development
- Bottleneck identification
- Architecture comparisons
- Performance debugging

## Level 3: Deep Profiling

Per-module instrumentation with full call stack analysis.

### What's Measured

**Per-Module Metrics:**
- Every nn.Module forward/backward
- Individual layer timings
- Memory allocations per module
- CUDA kernel launches

**DDP Instrumentation:**
- Per-parameter all-reduce timing
- Gradient bucket formation
- Communication overlap analysis

**Queue Statistics:**
- Real-time queue depths
- Throughput metrics
- Cache hit rates
- Written to `queue_stats.jsonl`

### Output Structure
```
assets/profiler/
├── rank0_trace.pt.trace.json     # PyTorch trace
├── rank0_trace.repaired.json     # Auto-repaired trace
├── queue_stats.jsonl              # Queue metrics
└── module_timings.json           # Per-module breakdown
```

### Module Timing Example
```json
{
  "module/stages.2.0.attn": 45.3,
  "module/stages.2.0.attn/qkv_projection": 8.2,
  "module/stages.2.0.attn/rope_apply": 12.4,
  "module/stages.2.0.attn/flash_attention": 18.7,
  "module/stages.2.0.attn/output_projection": 6.0
}
```

### When to Use
- Deep optimization work
- Memory profiling
- Kernel-level analysis
- Research and development

## Schedule Configuration

The profiler schedule controls when profiling is active:

```yaml
SCHEDULE: [wait, warmup, active, repeat]
```

| Parameter | Purpose | Typical Value |
|-----------|---------|---------------|
| wait | Steps before profiling starts | 2-10 |
| warmup | Steps to stabilize before recording | 1-3 |
| active | Steps to actively profile | 5-10 |
| repeat | Number of profiling cycles | 1-5 |

### Examples

**Quick Profile (Development)**
```yaml
SCHEDULE: [2, 1, 5, 2]  # Profile steps 3-7 and 10-14
```

**Stable Profile (Benchmarking)**
```yaml
SCHEDULE: [50, 5, 20, 1]  # Profile steps 55-74 after warmup
```

**Continuous Monitoring**
```yaml
SCHEDULE: [10, 2, 5, 100]  # Profile every ~17 steps
```

## Advanced Options

### Synchronous Profiling
```yaml
SYNC_PROFILING: True  # Forces CUDA synchronization for accurate timing
```
- Provides exact GPU timings
- Higher overhead (~2-3% additional)
- Required for kernel-level analysis

### Shape Recording
```yaml
RECORD_SHAPES: True  # Records tensor shapes in traces
```
- Helps identify shape inefficiencies
- Increases trace file size
- Useful for memory analysis

### Stack Traces
```yaml
WITH_STACK: True  # Includes Python call stacks
```
- Shows full call hierarchy
- Large trace files (can be >1GB)
- Essential for debugging

## Performance Impact

### Overhead by Level and Feature

| Configuration | Overhead | Use Case |
|---------------|----------|----------|
| Level 1 | 1-2% | Production |
| Level 2 | 4-6% | Development |
| Level 3 | 10-15% | Research |
| + SYNC_PROFILING | +2-3% | Accurate timing |
| + RECORD_SHAPES | +1-2% | Memory analysis |
| + WITH_STACK | +3-5% | Debugging |

### Memory Usage

| Level | Trace Size (10 steps) | Memory Overhead |
|-------|----------------------|-----------------|
| 1 | ~10 MB | Negligible |
| 2 | ~50 MB | ~100 MB |
| 3 | ~200 MB | ~500 MB |
| 3 + all options | ~1 GB | ~2 GB |

## Integration with Analysis Tools

### Level 1 → Basic Metrics
```bash
linnaeus-prof summary /path/to/run
# Shows: step time, GPU utilization
```

### Level 2 → Component Analysis
```bash
linnaeus-prof diff baseline/ optimized/
# Shows: component-level speedups
```

### Level 3 → Deep Dive
```bash
linnaeus-prof tensorboard --base-dir /experiments
# Enables: per-module flame graphs
```

## Best Practices

### Choosing the Right Level

1. **Start with Level 2** for most optimization work
2. **Use Level 1** for long-running production jobs
3. **Reserve Level 3** for specific deep-dive investigations
4. **Always compare at the same level** for fair comparisons

### Profiling Schedule Tips

1. **Skip initial steps** (wait ≥ 10) to avoid initialization noise
2. **Profile after warmup** for stable measurements
3. **Limit active steps** (5-20) to manage trace size
4. **Use single repeat** for benchmarking consistency

### Managing Overhead

1. **Disable in production** unless monitoring specific issues
2. **Use sampling** (repeat with gaps) for long runs
3. **Clean up traces** regularly (can consume significant disk)
4. **Consider network storage** for large trace files

## Troubleshooting

### High Overhead
- Reduce profiling level
- Disable SYNC_PROFILING
- Decrease active steps
- Remove WITH_STACK

### Corrupted Traces
```bash
linnaeus-prof repair /path/to/profiler/
```

### Missing Components (Level 2)
- Verify prof() contexts in code
- Check profiling level in config
- Ensure proper initialization

### Empty Module Timings (Level 3)
- Confirm Level 3 is enabled
- Check forward hooks registered
- Verify no module pruning

## Examples

### A/B Testing Configuration
```yaml
# Use Level 2 for component comparison
DEBUG:
  PROFILER:
    ENABLED: True
    LEVEL: 2
    SCHEDULE: [20, 3, 10, 2]  # Stable measurements
    SYNC_PROFILING: False  # Lower overhead
```

### Memory Debugging Configuration
```yaml
# Use Level 3 with shape recording
DEBUG:
  PROFILER:
    ENABLED: True
    LEVEL: 3
    SCHEDULE: [5, 1, 3, 1]  # Short burst
    RECORD_SHAPES: True
    WITH_STACK: True
```

### Production Monitoring
```yaml
# Minimal Level 1 profiling
DEBUG:
  PROFILER:
    ENABLED: True
    LEVEL: 1
    SCHEDULE: [100, 5, 10, 10]  # Periodic sampling
```