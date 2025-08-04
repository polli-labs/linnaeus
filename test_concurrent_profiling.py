#!/usr/bin/env python3
"""Test concurrent profiling integration."""

import json
import tempfile
import os
from pathlib import Path

# Create test trials
trials = [
    {
        "name": "test_concurrent_1",
        "git_ref": "main",
        "config_file": "/configs/experiments/tests/trial_template_MASTER.yaml",
        "opts": ["EXPERIMENT.NAME", "test_concurrent_1", "TRAIN.EPOCHS", "1"]
    },
    {
        "name": "test_concurrent_2", 
        "git_ref": "main",
        "config_file": "/configs/experiments/tests/trial_template_MASTER.yaml",
        "opts": ["EXPERIMENT.NAME", "test_concurrent_2", "TRAIN.EPOCHS", "1"]
    }
]

# Create test docker-compose template
template = """
version: '3.8'
services:
  linnaeus-training:
    image: frontierkodiak/linnaeus-dev:{{LINNAEUS_TAG}}
    command: |
      bash -c "
        echo 'Trial: {{TRIAL_NAME}} on GPU {{GPU_RANK}}'
        echo 'Config: {{CONFIG_FILE}}'
        echo 'Opts: {{OPTS}}'
        sleep 5
        echo 'DEBUG: Early exiting main training loop'
      "
    environment:
      - CUDA_VISIBLE_DEVICES={{GPU_RANK}}
{{ENV_VARS}}
"""

# Write files
with tempfile.TemporaryDirectory() as tmpdir:
    # Write trials
    trials_file = Path(tmpdir) / "test_trials.jsonl"
    with open(trials_file, 'w') as f:
        for trial in trials:
            f.write(json.dumps(trial) + '\n')
    
    # Write template
    template_file = Path(tmpdir) / "test_template.yml"
    with open(template_file, 'w') as f:
        f.write(template)
    
    # Output directory
    output_dir = Path(tmpdir) / "output"
    output_dir.mkdir()
    
    print(f"Test files created in {tmpdir}")
    print(f"Trials file: {trials_file}")
    print(f"Template file: {template_file}")
    print(f"Output dir: {output_dir}")
    
    # Run test
    cmd = f"""
    python -m linnaeus.tools.profiling.run_profiling_trials \
        --trial-params-file {trials_file} \
        --output-dir {output_dir} \
        --compose-template {template_file} \
        --timeout 30 \
        --max-concurrent 2 \
        --gpu-assignment auto \
        --stagger-delay 2
    """
    
    print("\nRun this command to test concurrent execution:")
    print(cmd)
    
    # Keep directory for manual testing
    input("Press Enter to clean up test files...")