# linnaeus Scripts

This directory contains utility scripts for common operations in the linnaeus project.

## Experiment Batch Runner

`run_experiment_batch.sh` - A template script for running multiple experiments with controlled timeouts.

### Features:

- Runs multiple experiments sequentially
- Each experiment runs for a configurable timeout period
- Automatically cleans up processes between experiments
- Configurable cooldown period between runs
- Detailed logging for each experiment
- Support for complex YAML configuration via command line arguments

### Usage:

1. Copy the template and customize it for your specific debugging or experimental needs:

```bash
cp tools/scripts/run_experiment_batch.sh work/active/myproject/my_experiment_batch.sh
```

2. Edit your copy to configure:
   - Timeout duration
   - Cooldown period
   - Base config file
   - Common options for all experiments
   - Experiment-specific options
   - Experiment order

3. Run your batch:

```bash
cd /path/to/linnaeus
./work/active/myproject/my_experiment_batch.sh
```

4. Check the logs in the generated log directory.

### Tips:

- When debugging complex issues, define experiments from simplest to most complex
- Start with minimal configurations that isolate specific components
- Use debug flags like `DEBUG.DATALOADER` and `DEBUG.AUGMENTATION` to get targeted diagnostic information
- Set appropriate timeouts - for debugging, 300-500 seconds is usually enough to collect diagnostic data
- After the batch completes, use `tools/filter_logs.py` to analyze logs by component
## Batch Size Analyzer

`analyze_batch_sizes.py` - Determine safe batch sizes for different GPU memory fractions.

```bash
python tools/analyze_batch_sizes.py --cfg configs/experiment.yaml --fractions 0.4,0.5 --modes train,val --output bs_results.jsonl
```


