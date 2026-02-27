# Profiling Best Practices & Troubleshooting

## Best Practices

### 1. Trial Design

**Use Descriptive Names**
```jsonl
{"name": "baseline_v032_l2"}           ✅ Clear version and profiling level
{"name": "test1"}                      ❌ Ambiguous
```

**Pin Exact Commits**
```jsonl
{"git_ref": "main", "commit_hash": "abc123def"}  ✅ Reproducible
{"git_ref": "main"}                              ❌ May change
```

**One Variable at a Time**
```jsonl
{"name": "droppath_disabled", "opts": ["MODEL.DROP_PATH_RATE", "0.0"]}  ✅ Clear attribution
{"name": "all_opts", "opts": ["MODEL.DROP_PATH_RATE", "0.0", "TRAIN.AMP_OPT_LEVEL", "O2"]}  ❌ Confounded
```

### 2. Profiling Configuration

**Match Profiling Levels**
```yaml
# Both baseline and optimized should use same level
DEBUG.PROFILER.LEVEL: 2  # Consistent across trials
```

**Appropriate Schedule for Purpose**

```yaml
# Development (quick feedback)
SCHEDULE: [2, 1, 5, 2]

# Benchmarking (stable measurements)
SCHEDULE: [50, 5, 20, 1]

# Memory debugging (minimal steps)
SCHEDULE: [5, 1, 3, 1]
```

### 3. Resource Management

**Set Realistic Timeouts**
```bash
# Minimum 3 minutes for model initialization
--timeout 180  ❌ Too short, may timeout during init
--timeout 600  ✅ Allows for initialization + profiling
```

**Manage Disk Space**
```bash
# Clean up old traces regularly
find /experiments -name "*.pt.trace.json" -mtime +7 -delete

# Use compression for archives
tar -czf traces_backup.tar.gz assets/profiler/
```

**GPU Memory Management**
```yaml
# Reduce batch size for memory-constrained profiling
DATA.BATCH_SIZE: 32  # Instead of production 128
```

### 4. Concurrent Execution

**Optimal Concurrency Settings**
```bash
# For 2 GPUs
--max-concurrent 2 --gpu-assignment auto --stagger-delay 10

# For 4 GPUs with memory constraints
--max-concurrent 2 --gpu-assignment round-robin --stagger-delay 30
```

**Avoid Oversubscription**
```python
# Don't exceed physical GPU count
num_trials = 10
num_gpus = 2
max_concurrent = min(num_trials, num_gpus)  # Use 2, not 10
```

### 5. Environment Variables

**Critical Settings**
```yaml
# Always disable distributed debug in production
TORCH_DISTRIBUTED_DEBUG: "OFF"

# Enable NCCL logging only when debugging
NCCL_DEBUG: "WARN"  # Not "INFO" unless needed

# Control thread count for CPU-bound operations
OMP_NUM_THREADS: "4"
TORCH_NUM_THREADS: "4"
```

### 6. Analysis Workflow

**Compare Multiple Metrics**
```bash
# Don't rely on single metric
linnaeus-prof diff baseline/ optimized/ --output-format json | \
  jq '{
    step_time_change: .metrics.avg_step_time.change_percent,
    kernel_change: .metrics.kernel_count.change_percent,
    memory_change: .metrics.peak_memory.change_percent
  }'
```

**Verify Reproducibility**
```bash
# Run same trial multiple times
for i in {1..3}; do
  linnaeus-prof-run --trial-params-file same_trial.jsonl \
    --output-dir "run_$i"
done

# Check variance
python -c "
import json
import numpy as np
times = []
for i in range(1, 4):
    with open(f'run_{i}/summary.json') as f:
        times.append(json.load(f)['metrics']['avg_step_time'])
print(f'Mean: {np.mean(times):.1f}ms, Std: {np.std(times):.1f}ms')
"
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Docker/Container Issues

**Problem**: Docker compose not found
```bash
# Check Docker Compose version
docker compose version || docker-compose version

# Solution: Install Docker Compose v2
curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
```

**Problem**: Container permission denied
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker  # Or logout/login
```

**Problem**: Container fails to start
```bash
# Check Docker daemon
sudo systemctl status docker

# Check disk space
df -h /var/lib/docker

# Clean up old containers/images
docker system prune -a
```

#### GPU Issues

**Problem**: CUDA out of memory
```yaml
# Reduce batch size
DATA.BATCH_SIZE: 32

# Disable gradient checkpointing
TRAIN.GRADIENT_CHECKPOINTING.ENABLED_NORMAL_STEPS: False

# Use lower precision
TRAIN.AMP_OPT_LEVEL: "O1"
```

**Problem**: GPU not visible
```bash
# Check CUDA installation
nvidia-smi

# Verify CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES

# Reset GPU
sudo nvidia-smi -r
```

**Problem**: GPU allocation timeout
```bash
# Increase timeout
--gpu-timeout 7200  # 2 hours

# Reduce concurrency
--max-concurrent 1

# Check for stuck processes
nvidia-smi | grep python
```

#### Profiler Issues

**Problem**: No profiler traces found
```yaml
# Verify profiling enabled
DEBUG:
  PROFILER:
    ENABLED: True  # Must be True
    LEVEL: 2      # Must be > 0
```

**Problem**: Corrupted traces
```bash
# Auto-repair traces
linnaeus-prof repair /path/to/assets/profiler/

# Manual inspection
python -m json.tool rank0_trace.pt.trace.json > /dev/null
# If error, trace is corrupted
```

**Problem**: Trace files too large
```yaml
# Reduce profiling scope
DEBUG:
  PROFILER:
    SCHEDULE: [50, 2, 5, 1]  # Fewer active steps
    RECORD_SHAPES: False      # Disable shape recording
    WITH_STACK: False         # Disable stack traces
```

#### Trial Execution Issues

**Problem**: Git checkout fails
```bash
# Ensure branch is pushed
git push origin feature/optimization

# Or use local worktree
git worktree add /tmp/linnaeus-test feature/optimization
```

**Problem**: Config file not found
```jsonl
// Use absolute paths inside container
{"config_file": "/workspace/linnaeus/configs/test.yaml"}  ✅
{"config_file": "configs/test.yaml"}                      ❌ Relative
```

**Problem**: Trial hangs indefinitely
```bash
# Add timeout
--timeout 600

# Enable debug logging
{"env": {"TORCH_DISTRIBUTED_DEBUG": "DETAIL"}}

# Check container logs
docker logs linnaeus-training-<trial_name>
```

#### Analysis Issues

**Problem**: Diff shows no changes
```bash
# Verify both runs have profiler data
ls -la baseline/assets/profiler/
ls -la optimized/assets/profiler/

# Check profiling was at same level
grep "PROFILER.LEVEL" baseline/experiment_config.yaml
grep "PROFILER.LEVEL" optimized/experiment_config.yaml
```

**Problem**: TensorBoard won't load
```bash
# Install required plugins
uv pip install tensorboard torch-tb-profiler

# Use Chrome (not Safari)
google-chrome http://localhost:6006

# Check for port conflicts
lsof -i :6006
```

### Debug Mode Workflow

For difficult issues, use this debug workflow:

1. **Create Minimal Test Case**
```jsonl
{"name": "debug", "config_file": "configs/minimal.yaml", "opts": ["TRAIN.EPOCHS", "1", "DATA.BATCH_SIZE", "2"]}
```

2. **Enable Verbose Logging**
```yaml
# In config
DEBUG:
  VERBOSE_DEBUG: True
  TRAINING_LOOP: True
  PROFILER:
    ENABLED: True
    LEVEL: 1  # Start with minimal profiling
```

3. **Run with Debug Capture**
```bash
linnaeus-prof-run \
  --trial-params-file debug.jsonl \
  --output-dir debug_output \
  --timeout 120 \
  --capture-debug-logs \
  --exit-on-failure
```

4. **Analyze Logs**
```bash
# Check status
cat debug_output/debug/status.txt

# Review console output
less debug_output/debug/console_log.txt

# Check debug logs
tail -n 100 debug_output/debug/debug_log.txt
```

### Performance Regression Checklist

When performance regresses:

- [ ] Verify same hardware (GPU model, CPU, memory)
- [ ] Check same software versions (PyTorch, CUDA, drivers)
- [ ] Confirm identical configs (diff yaml files)
- [ ] Compare profiling levels
- [ ] Check for thermal throttling (nvidia-smi -q)
- [ ] Verify no background processes
- [ ] Test multiple runs for variance
- [ ] Review git diff for unintended changes
- [ ] Check Docker image versions
- [ ] Analyze component breakdowns for specific regression

### Recovery Procedures

**Corrupted Experiment State**
```bash
# Backup current state
tar -czf experiment_backup.tar.gz /path/to/experiment/

# Remove cache files
find /path/to/experiment -name ".linnaeus_cache" -type d -exec rm -rf {} +

# Repair traces
linnaeus-prof repair /path/to/experiment/assets/profiler/

# Regenerate summary
linnaeus-prof summary /path/to/experiment --write-cache
```

**Stuck Docker Containers**
```bash
# List all containers
docker ps -a

# Force stop and remove
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)

# Clean up volumes
docker volume prune -f
```

**GPU Memory Leak**
```bash
# Kill all Python processes
pkill -9 python

# Reset GPUs
sudo nvidia-smi --gpu-reset

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"
```
