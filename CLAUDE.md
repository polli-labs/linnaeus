# CLAUDE.md — Local Development Guide for the Linnaeus Codebase

## ⚠️ CRITICAL SECURITY RULES - READ FIRST ⚠️

### PUBLIC vs PRIVATE CONFIG SEPARATION

1. **linnaeus/configs/** - PUBLIC EXAMPLES ONLY
   - This directory is part of the PUBLIC GitHub repository
   - NEVER put experiment configs, trial configs, or ANY configs with:
     - API keys, credentials, or tokens (e.g., Backblaze keys)
     - Internal paths or infrastructure details
     - Experiment-specific configurations
     - Personal or proprietary information
   - ONLY put example configs that demonstrate features

2. **linnaeus-deployment/linnaeus_deploy/configs/** - ALL PRIVATE CONFIGS
   - This is the PRIVATE deployment repository
   - ALL experiment configs go here
   - ALL trial templates go here
   - ALL profiling configs go here
   - Safe for credentials, internal paths, etc.

3. **work/** directory - UNTRACKED SCRATCH SPACE
   - Use for temporary files and scratch work
   - Never commit files from work/ to git

**ENFORCEMENT**: Always use absolute paths to private configs in linnaeus-deployment repo. Never copy configs to linnaeus/configs/ for any reason.

## Project Overview

Linnaeus is a PyTorch‑based toolkit for hierarchical biodiversity classification.

The project is public on github at polli-labs/linnaeus. Deployment wrappers (for local and cloud testing, training) are all in the polli-labs/linnaeus-deployment repo, which, critically, is private. Local copies of both repos are on this machine, you have access to both in your workspace.

## Profiling Workflow (Preferred Development Pattern)

We use a fully reproducible, high-observability job+runner model for development flows:

### Key Principles:
1. **Reproducibility**: Each trial specified with exact git ref, commit hash, config, and env vars
2. **Observability**: All parameters explicitly documented in JSONL format
3. **Control**: Fine-grained control over experimental params and environment variables
4. **Consistency**: Baseline constancy within branches with trial-by-trial configurability

### Workflow Components:
1. **Trial Definition** (work/active/<version>/<round>/trials.jsonl)
   - All trials use same template from work/fixtures/trial_template_MASTER.yaml
   - Differences expressed via --opts and env_yaml parameters only
   - DEBUG.PROFILER settings identical across trials for fair comparison

2. **Docker-based Execution**
   - Spin up per-job containers with specified branch/commit
   - Pull linnaeus from source at container startup
   - **Critical**: Must commit AND push branches before running trials

3. **Profiling Runner** (linnaeus-prof-run)
   - Automated trial orchestration with template substitution
   - Environment variable integration via scenario files
   - Intelligent error handling and retry logic

### Example Trial Definition:
```jsonl
{"name": "v015a_baseline", "config_file": "/home/caleb/repo/linnaeus-deployment/linnaeus_deploy/configs/experiments/tests/trial_template_v015.yaml", "git_ref": "main", "commit_hash": "6e34cee...", "env_yaml": "/home/caleb/repo/linnaeus-deployment/linnaeus_deploy/configs/env_vars/single_gpu_workstation.yaml", "opts": ["EXPERIMENT.NAME", "aves_mFormerV1_md_115e_v015a_baseline", "EXPERIMENT.CODE_VERSION", "main_v0.1.4"]}
{"name": "v015a_optimized", "config_file": "/home/caleb/repo/linnaeus-deployment/linnaeus_deploy/configs/experiments/tests/trial_template_v015.yaml", "git_ref": "v0.1.5", "commit_hash": "9573bd9...", "env_yaml": "/home/caleb/repo/linnaeus-deployment/linnaeus_deploy/configs/env_vars/single_gpu_workstation.yaml", "opts": ["EXPERIMENT.NAME", "aves_mFormerV1_md_115e_v015a_optimized", "EXPERIMENT.CODE_VERSION", "v0.1.5a"]}
```

## Common Development Commands

### 1 Rules

- NEVER use pip. Only use uv.
- NEVER save experiment configs in the linnaeus repo. Only example and model arch configs are allowed in the public repo.
- ALWAYS use linnaeus-deployment/linnaeus_deploy/configs/ for ALL experiment/trial/profiling configs
- ALWAYS push commits immediately after committing when working on feature branches (not main). Use `git push origin <branch-name>` or `git push -u origin <branch-name>` for first push.
- work/ is untracked. You can use it for scratch work and for any non-public work. We organize our work like `work/active/<second part of branch name>`, typically there is a spec file there that we are iterating on. more substantial branches are often split into multiple subdirs under there. 
  - bugs are in `work/bugs/inbox`. some branches are also currently in there but that is not a good practice, going forward we will put branch work dirs under work/active.

### 2  Common Local Commands

#### 2.1  Environment

python -m venv .venv     # first time only
source .venv/bin/activate

#### 2.2  Training (local, single‑GPU)

python -m linnaeus.main \
  --cfg configs/experiments/examples/aves_smoke.yaml \
  --opts TRAIN.EPOCHS 1 DEBUG.PROFILER.ENABLED False

#### 2.3  Modern Profiling / Trial Orchestration  :rocket:

Stop editing docker‑compose files by hand.
Use the two dedicated slash commands which wrap the orchestration script stored in linnaeus‑deployment/tools/profiling:

##### 1) Implement spec + generate baseline/opt trials.jsonl
./.claude/commands/prof_impl work/active/shared_hybrid_img_dir/v016a_spec.md

##### 2) Launch both trials under Docker Compose, timeout after 5 min each
./.claude/commands/prof_run  work/active/shared_hybrid_img_dir/v016a_spec.md --timeout 300

Behind the scenes this:
	1.	Checks out the correct Git refs (baseline vs. feature branch)
	2.	Re‑writes a temporary docker-compose.yml per trial
	3.	Streams logs & captures debug_log_rank0.txt on failure
	4.	Produces summary.json ready for /prof_analyze
Note that those are claude slash commands-- not scripts, you can't run them directly, but you can read them and then gather required info to execute the test (or whatever) it is you need to do.

**For manual execution**, use the installed CLI directly:
```bash
# Run profiling trials with the installed CLI
linnaeus-prof-run \
  --trial-params-file work/active/v016/v016a/trials.jsonl \
  --output-dir work/active/v016/v016a/results \
  --compose-template work/fixtures/docker-compose.template.yml \
  --timeout 300 \
  --capture-debug-logs
```

##### 3) Set up a wrapper script for a round of trials

Example from branch v0.1.5, revision d:

```example trial wrapper
#!/bin/bash

# v015d Final Vectorization Profiling Trials (Stable)
# Tests the complete end-to-end vectorization of selective mixing
# Uses steps 81-90 and 101-110 for stable measurements

echo "=� Starting v015d Final Vectorization Profiling Trials (Stable)"
echo "=============================================================="
echo "Baseline: v0.1.5a (9573bd9) vs Optimized: v0.1.5d (a25fa94)"
echo "Profiling: Steps 81-90 and 101-110 (20 total steps)"
echo ""
echo "Expected improvements:"
echo "  - Selective mixing: 120ms � <50ms per step (-60%)"
echo "  - Kernel count: 1,800 � <900 per step (-50%)"
echo "  - Overall step time: 1,120ms � <1,050ms (-7%)"

# Create output directory  
mkdir -p /home/caleb/repo/linnaeus/work/bugs/inbox/v015/v015d/v015d_results_stable

# Run trials with stable profiling schedule
PYTHONPATH=/home/caleb/repo/linnaeus-deployment uv run --python 3.10 \
  /home/caleb/repo/linnaeus-deployment/tools/profiling/run_profiling_trials.py \
  --trial-params-file /home/caleb/repo/linnaeus/work/bugs/inbox/v015/v015d/v015d_trials_stable.jsonl \
  --output-dir /home/caleb/repo/linnaeus/work/bugs/inbox/v015/v015d/v015d_results_stable \
  --compose-template /home/caleb/repo/linnaeus-deployment/tools/profiling/docker-compose.template.yml \
  --timeout 320 \
  --capture-debug-logs

echo ""
echo " v015d stable trials completed. Results saved to:"
echo "   /home/caleb/repo/linnaeus/work/bugs/inbox/v015/v015d/v015d_results_stable/"
echo ""
echo "=
 To analyze results:"
echo "   cd /home/caleb/repo/linnaeus"
echo "   source .venv/bin/activate"  
echo "   python -m linnaeus.profiling.cli diff \\"
echo "     /datasets/modelWorkshop/mFormerV1/linnaeus-dev/aves_mFormerV1/aves_mFormerV1_md_115e_v015a_baseline_stable \\"
echo "     /datasets/modelWorkshop/mFormerV1/linnaeus-dev/aves_mFormerV1/aves_mFormerV1_md_115e_v015d_optimized_stable \\"
echo "     --output-format md --save /home/caleb/repo/linnaeus/work/bugs/inbox/v015/v015d/v015d_results_stable/comparison.md"


# Note: baseline trial skipped (already done), but the jsonl looks like:
### // skip (already done) {"name": "v015a_baseline_stable", "config_file": "/configs/experiments/tests/v015_trials/trial_template_v015.yaml", "git_ref": "v0.1.5", "commit_hash": "9573bd97e0bcf3638e7a8544e21cd8329e6acb35", "extra_deps": ["kornia>=0.8.1,<0.9"], "opts": ["EXPERIMENT.NAME", "aves_mFormerV1_md_115e_v015a_baseline_stable", "EXPERIMENT.CODE_VERSION", "v0.1.5a", "DEBUG.PROFILER.SYNC_PROFILING", "True", "DEBUG.PROFILER.SCHEDULE", "[77, 3, 10, 2]"]}
```

## Architecture Overview

### Core Components

1. **Model System**: Two main families - mFormerV0 and mFormerV1, with hybrid CNN-Transformer architectures
2. **Dataset System**: HDF5-based with support for hierarchical labels and metadata
3. **Training Pipeline**: Step-based scheduling, gradient accumulation, distributed training support
4. **Configuration**: YACS-based configuration system with hierarchical config files

### Key Design Patterns

- **Component Registration**: Uses decorators for registering models, losses, and other components
- **Two-tiered Design**: Registered Components (via decorators) and Building Blocks
- **Schedule-based Operations**: Central OpsSchedule manages all timed events during training

### Critical Implementation Details

1. **Dataset Logging**: Dataset wrappers don't propagate logger instances properly. Use `logging.getLogger('h5data')` directly
2. **Step Counting**: `global_step` counts optimizer updates (not batches) - critical for scheduling
3. **Schedule Parameters**: Only use ONE method per parameter (e.g., don't set both `INTERVAL_STEPS` and `INTERVAL_EPOCHS`)
4. **Metadata Systems**: 
   - `parameter_groups_metadata`: Semantic parameter groupings (informational)
   - `pretrained_ckpt_handling_metadata`: Checkpoint loading/adaptation instructions

### File Structure

(minimal summary, poke around in the repo as needed to see more)
- `linnaeus/`: Core framework code
  - `models/`: Model implementations
  - `h5data/`: Dataset handling
  - `utils/`: Training utilities
  - `main.py`: Training entry point
- `configs/`: YAML configuration files
  - `model/archs/`: Model architecture configs
  - `experiments/`: Experiment configurations
- `tools/`: Utility scripts and analysis tools
- `docs/`: Comprehensive documentation
  - `dev/`: Architecture notes and design decisions

## Docker Testing and Debugging Guidelines

You should almost always use the run_profiling_trials.py script to run trials (see `/home/caleb/repo/linnaeus-deployment/tools/profiling`). Extra instructions are in the .claude/commands/prof_run file. This requires that you create a jsonl according to the examples in the .claude/commands/prof_impl (also in `/home/caleb/repo/linnaeus-deployment/tools/profiling/<README.md/trials.example.jsonl`), inheriting from a template (see `/home/caleb/repo/linnaeus-deployment/linnaeus_deploy/configs/experiments/tests/trial_template_MASTER.yaml`, you must create a new one for each branch, stored in `/home/caleb/repo/linnaeus-deployment/linnaeus_deploy/configs/experiments/tests/<branch_name or target version formatted like v016>`). trials are defined by the --opts passed in the jsonl on top of the template.

### Trial Management Best Practices:
1. **Git Hygiene**: Always commit AND push branches before running trials (containers pull from remote)
2. **Reproducibility**: Record exact commit hashes in trial definitions
3. **Environment Variables**: Use scenario files (env_yaml) for hardware-specific settings
4. **Debugging**: Capture debug logs with --capture-debug-logs for failed trials
5. **Organization**: Keep all trials for a branch in one jsonl file for investigative debugging

### Testing Training Runs
- **Always use 3+ minute timeout** when testing docker compose training runs to verify training steps actually run
- Previous 1-minute timeouts only showed initialization, not actual training progress
- Look for "Training step X" messages in logs to confirm training is working
- Only declare victory after seeing actual training steps complete

### Debugging Failed Training Runs
- Main console output may not show all errors
- Check detailed logs in: `/datasets/modelWorkshop/mFormerV1/<EXPERIMENT.PROJECT>/<EXPERIMENT.GROUP>/<EXPERIMENT.NAME>/logs/`
- Look for: `h5data_debug_log_rank0` and other rank-specific logs
- These contain detailed error traces that may be hidden from main output

## Release Management

### Version Management and Releases

When preparing to create a new tagged release from a branch back to main:

1. **Update CHANGELOG.md**:
   - Add a new section with the target version number (e.g., `## [0.1.3] - YYYY-MM-DD`)
   - Document all changes in appropriate sections (Added, Changed, Fixed, Performance, etc.)
   - Include any breaking changes and migration notes

2. **Update pyproject.toml**:
   - Bump the version number in the `[project]` section
   - Example: `version = "0.1.3"`

3. **Version Tagging Process**:
   - Use proper semver format: `vX.Y.Z` for stable releases
   - Use `vX.Y.Z-rcN` for pre-releases (note the hyphen!)
   - Ensure base Docker images are up-to-date before tagging

4. **Documentation Review**:
   - Check if any documentation needs updating based on the changes
   - Update README.md if new features or usage patterns are introduced
   - Review and update any architectural documentation in `docs/`

### Release Checklist

Before merging a release branch:
- [ ] CHANGELOG.md updated with new version entry
- [ ] pyproject.toml version bumped
- [ ] All tests passing
- [ ] Docker images build successfully
- [ ] Documentation reviewed and updated as needed
- [ ] Breaking changes documented with migration guide

After merging to main:
- [ ] Tag the release: `git tag vX.Y.Z`
- [ ] Push the tag: `git push origin vX.Y.Z`
- [ ] Update Docker image tags if needed
- [ ] Create GitHub release with changelog notes