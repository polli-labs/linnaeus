# CLAUDE.md - Local Development Guide for the Linnaeus Codebase

## CRITICAL SECURITY RULES - READ FIRST

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

## Quick Reference: Complete Profiling Workflow

1. **Design**: Create spec in `work/active/<feature>/spec.md`
2. **Implement**: Make changes in `linnaeus`, commit and push
3. **Configure**: `/prof_impl work/active/<feature>/spec.md` (creates trials in linnaeus-deployment)
4. **Execute**: `/prof_run work/active/<feature>/spec.md --timeout 600` (runs trials concurrently)
5. **Analyze**: `/prof_analyze work/active/<feature>/spec.md` (generates performance reports)

**Key Points**:
- All trial configs go in PRIVATE linnaeus-deployment repo
- All results/analysis go in PUBLIC linnaeus/work/ directory
- Always use concurrent execution (`--max-concurrent 2`) for 2x speedup
- Always commit AND push before running trials

## Project Overview

Linnaeus is a PyTorch-based toolkit for hierarchical biodiversity classification.

The project is public on github at polli-labs/linnaeus. Deployment wrappers (for local and cloud testing, training) are all in the polli-labs/linnaeus-deployment repo, which, critically, is private. Local copies of both repos are on this machine, you have access to both in your workspace.

## Profiling Workflow (Preferred Development Pattern)

We use a fully reproducible, high-observability job+runner model for development flows with **strict separation** between public code and private configurations.

### Complete Workflow Summary:
1. **Design**: Create spec in `linnaeus/work/active/<feature>/spec.md`
2. **Implement**: Code changes in `linnaeus` repo, commit and push
3. **Configure**: Use `/prof_impl` to prepare trials in `linnaeus-deployment` (private)
4. **Execute**: Run `linnaeus-prof-run` with concurrent GPU execution (2x speedup)
5. **Analyze**: Use `/prof_analyze` to generate performance reports

### Key Principles:
1. **Reproducibility**: Each trial specified with exact git ref, commit hash, config, and env vars
2. **Observability**: All parameters explicitly documented in JSONL format
3. **Control**: Fine-grained control over experimental params and environment variables
4. **Consistency**: Baseline constancy within branches with trial-by-trial configurability
5. **Security**: All experiment configs stay in private repo (linnaeus-deployment)
6. **Performance**: Concurrent GPU execution by default for ~2x speedup

### Critical Directory Structure:

#### Trial Fixtures (PRIVATE - linnaeus-deployment repo):
```
/home/caleb/repo/linnaeus-deployment/linnaeus_deploy/configs/experiments/tests/
├── trial_template_MASTER.yaml          # Master template - copy for each version
├── v035/                               # Target release version
│   ├── trial_template_v035.yaml       # Version-specific base template
│   └── test0/                         # Feature or test name
│       └── trials.jsonl               # Trial definitions
├── v040/
│   ├── trial_template_v040.yaml
│   └── mFormerV1-downsample/
│       └── p0/
│           └── trials.jsonl
```

#### Working Documents (PUBLIC - linnaeus repo):
```
/home/caleb/repo/linnaeus/work/
├── active/                            # Active development work
│   └── mFormerV1-downsample/         # Feature branch work
│       └── p0/                       # Round/phase
│           ├── spec.md               # Design spec
│           └── results/              # Output from prof-run
└── bugs/inbox/                       # Bug tracking
    └── v015/                         # Version with issue
        └── issue_analysis.md         # Must reference absolute paths to fixtures
```

**CRITICAL**: Working docs in `work/` MUST use absolute paths to reference fixtures in linnaeus-deployment!

### Workflow Components:
1. **Trial Definition** (`/home/caleb/repo/linnaeus-deployment/linnaeus_deploy/configs/experiments/tests/<version>/<feature>/trials.jsonl`)
   - All trials inherit from version-specific template in same directory tree
   - Differences expressed via --opts and env_yaml parameters only
   - Container paths (`/configs/...`) map to linnaeus-deployment at runtime

2. **Docker-based Execution**
   - Spin up per-job containers with specified branch/commit
   - Pull linnaeus from source at container startup
   - **Critical**: Must commit AND push branches before running trials

3. **Profiling Runner** (linnaeus-prof-run) - Now with concurrent execution by default!
   - Automated trial orchestration with template substitution
   - Multi-GPU concurrent execution for ~2x speedup
   - Environment variable integration via scenario files
   - Intelligent error handling and retry logic

### Example Trial Definition:
```jsonl
{"name": "v040_baseline", "config_file": "/configs/experiments/tests/v040/trial_template_v040.yaml", "git_ref": "main", "commit_hash": "6e34cee", "env_yaml": "/configs/env_vars/single_gpu_workstation.yaml", "env": {"TORCH_DISTRIBUTED_DEBUG": "OFF"}, "opts": ["EXPERIMENT.NAME", "aves_mFormerV1_md_v040_baseline", "EXPERIMENT.CODE_VERSION", "main_6e34cee"]}
{"name": "v040_optimized", "config_file": "/configs/experiments/tests/v040/trial_template_v040.yaml", "git_ref": "experiment/feature-branch", "commit_hash": "9573bd9", "env_yaml": "/configs/env_vars/single_gpu_workstation.yaml", "env": {"TORCH_DISTRIBUTED_DEBUG": "OFF"}, "opts": ["EXPERIMENT.NAME", "aves_mFormerV1_md_v040_optimized", "EXPERIMENT.CODE_VERSION", "exp_9573bd9"]}
```

**Note**: Paths in trials.jsonl use container mount points (`/configs/...`) which map to `/home/caleb/repo/linnaeus-deployment/linnaeus_deploy/configs/...` on host.

## Common Development Commands

### 1 Rules

- NEVER use pip. Only use uv.
- NEVER save experiment configs in the linnaeus repo. Only example and model arch configs are allowed in the public repo.
- ALWAYS use linnaeus-deployment/linnaeus_deploy/configs/ for ALL experiment/trial/profiling configs
- ALWAYS push commits immediately after committing when working on feature branches (not main). Use `git push origin <branch-name>` or `git push -u origin <branch-name>` for first push.
- work/ is untracked. You can use it for scratch work and for any non-public work.

### Work Organization:

1. **Bug/Issue Tracking** (`work/bugs/inbox/`):
   - All bugs and issues flow through the tracker system
   - Use `/file_issue` command or describe issues naturally
   - Issues are auto-triaged by priority (P0-P3)
   - GitHub-migration ready format
   - Move to `work/closed/` after PR merge
   - Must include context (project, git worktree) for non-main branches

2. **Active Development** (`work/active/`):
   - Complex iterative experimental workflows (e.g., optimization)
   - Structure: `work/active/<feature-name>/`
   - Contains specs, hypotheses, analyses, trial results
   - Actionable code changes from here should flow through issue tracker
   - Always reference the git branch/worktree context

3. **Issue Filing**:
   ```bash
   # Use the Claude command
   /file_issue "The validation is triggering too early"
   
   # Or with explicit type/priority
   /file_issue bug P0 "Config breaks" "YACS inheritance issue"
   ```

## YAML Frontmatter Enforcement (MANDATORY)

### Policy
**ALL** working markdown documents in linnaeus MUST include valid YAML frontmatter conforming to the appropriate DocType schema. This includes all files in:
- `work/` (specs, issues, state, outcomes, etc.)
- `docs/dev/` and `docs/profiling/` (documentation pages)
- Any markdown files tracking progress, issues, or project state

### Required Behavior
1. **On Create**: Insert full frontmatter template for the inferred DocType
2. **On Edit**: Preserve `created` timestamp, update `updated` to current UTC
3. **On Save**: Validate frontmatter; block save if invalid and propose fix
4. **Always**: Use UTC ISO-8601 format (YYYY-MM-DDTHH:MM:SSZ) for timestamps

### DocType Schemas

#### 1. SprintGoals
```yaml
---
title: "Sprint Goals: Week of Aug 4, 2025"
doc_type: sprint_goals
project: linnaeus
sprint: "2025-WeekAug4"
status: active|closed
objective: "Complete drop path optimization"
deadline: 2025-08-09
version_target: "v0.4.0"
created: 2025-08-04T09:00:00Z
updated: 2025-08-04T17:00:00Z
tags: [sprint, goals]
x:
  repo: linnaeus
  worktree: main
  canonical: work/sprints/2025-WeekAug4.md
  related: []
---
```

#### 2. Standup
```yaml
---
title: "Daily Standup – 2025-08-06"
doc_type: standup
project: linnaeus
date: 2025-08-06
sprint: "2025-WeekAug4"
status: logged|amended
created: 2025-08-06T09:05:00Z
updated: 2025-08-06T17:40:00Z
tags: [standup]
x:
  blockers: []
  links: []
---
```

#### 3. Spec
```yaml
---
title: "Drop Path Optimization Spec"
doc_type: spec
project: linnaeus
status: draft|review|approved|superseded
owners: ["caleb"]
reviewers: []
version: "0.1"
created: 2025-08-06T18:00:00Z
updated: 2025-08-07T02:40:00Z
tags: [spec, optimization]
x:
  decision_log: []
  risks: []
  related: []
---
```

#### 4. Outcome/Report
```yaml
---
title: "Drop Path Optimization Results"
doc_type: outcome
project: linnaeus
status: final|interim
created: 2025-08-07T03:00:00Z
updated: 2025-08-07T03:20:00Z
tags: [results, report, optimization]
x:
  sources: ["work/active/mFormerV1-droppath/results/"]
  metrics: {improvement: "6%", method: "batch_rng"}
  decisions: []
  next_steps: []
---
```

#### 5. State/Progress
```yaml
---
title: "Current Project State – 2025-08-07"
doc_type: state
project: linnaeus
status: active
period: "Day 3 of 5"
created: 2025-08-07T01:00:00Z
updated: 2025-08-07T19:30:00Z
tags: [status]
x:
  summary: "Batch RNG optimization complete, Triton blocked"
  risks: []
  todos: []
---
```

#### 6. Issue
```yaml
---
title: "[P2][bug] Triton segfaults with L3 profiling"
doc_type: issue
project: linnaeus
type: bug|feature|enhancement|task
priority: P0|P1|P2|P3
status: open|in_progress|blocked|done
component: profiling
created: 2025-08-04T10:00:00Z
updated: 2025-08-05T14:11:00Z
tags: [issue, P2, triton]
x:
  id: "triton-l3-segfault"
  related: ["v0.3.4-segfault"]
  version: "v0.4.0-dev"
---
```

#### 7. LogExcerpt
```yaml
---
title: "Training Log Excerpt – drop_path_trial"
doc_type: log_excerpt
project: linnaeus
status: captured
created: 2025-08-06T22:10:00Z
updated: 2025-08-06T22:10:00Z
tags: [logs, profiling]
x:
  source_path: "/datasets/modelWorkshop/logs/trial.log"
  time_range: "2025-08-06T20:00Z..2025-08-06T22:00Z"
  grep: 'rg -n --context 5 "(ERROR|SIGSEGV)"'
  error_signatures: ["SIGSEGV", "exit code 139"]
---
```

#### 8. DocsPage (for MkDocs)
```yaml
---
title: "Profiling Workflow Guide"
summary: "How to run profiling trials with concurrent execution"
tags: [docs, profiling]
date: 2025-08-07
lastmod: 2025-08-07
x:
  project: linnaeus
  doc_type: docs_page
---
```

### Template Selection Heuristics
- `work/active/*/spec.md` → spec
- `work/bugs/**/*.md` → issue  
- `work/sprints/*.md` → sprint_goals
- `work/standups/*.md` → standup
- `work/outcomes/*.md` → outcome
- `work/state/*.md` → state
- `docs/**/*.md` → docs_page
- Files with "log" or "excerpt" → log_excerpt

### Frontmatter Commands
- `/frontmatter-new <DocType> [title]` - Create new doc with template
- `/frontmatter-repair` - Fix current file's frontmatter
- `/frontmatter-sync [path]` - Batch repair directory tree

### Validation
The `validate_frontmatter.py` script runs automatically before saves. If validation fails:
1. The save is blocked
2. You'll be prompted to run `/frontmatter-repair`
3. The repair will preserve `created` and update `updated`

### 2  Common Local Commands

#### 2.1  Environment

python -m venv .venv     # first time only
source .venv/bin/activate

#### 2.2  Training (local, single-GPU)

python -m linnaeus.main \
  --cfg configs/experiments/examples/aves_smoke.yaml \
  --opts TRAIN.EPOCHS 1 DEBUG.PROFILER.ENABLED False

#### 2.3  Modern Profiling / Trial Orchestration

Complete workflow for performance optimization with concurrent GPU execution:

##### Step 1: Implement spec + prepare trial configurations
```
/prof_impl work/active/mFormerV1-downsample/spec.md
```
This creates trial configurations in linnaeus-deployment (private repo).

##### Step 2: Execute trials with concurrent GPU support (2x speedup)
```
/prof_run work/active/mFormerV1-downsample/spec.md --timeout 600
```
This runs baseline and optimized trials concurrently on separate GPUs.

##### Step 3: Analyze results
```
/prof_analyze work/active/mFormerV1-downsample/spec.md
```
This generates comprehensive performance reports.

**Behind the scenes**:
- Trials are defined in: `/home/caleb/repo/linnaeus-deployment/linnaeus_deploy/configs/experiments/tests/<version>/<feature>/trials.jsonl`
- Docker templates from: `/home/caleb/repo/linnaeus-deployment/linnaeus_deploy/docker/runtime/profiling/blade/templates/`
- Results saved to: `/home/caleb/repo/linnaeus/work/active/<feature>/results/`
- Concurrent execution allocates GPUs automatically for ~2x speedup

**For manual execution**, use the installed CLI directly with concurrent execution (default):
```bash
# Run profiling trials with concurrent execution (2x speedup on 2 GPUs)
linnaeus-prof-run \
  --trial-params-file /home/caleb/repo/linnaeus-deployment/linnaeus_deploy/configs/experiments/tests/v040/feature-name/trials.jsonl \
  --output-dir /home/caleb/repo/linnaeus/work/active/feature-name/results \
  --compose-template /home/caleb/repo/linnaeus-deployment/linnaeus_deploy/docker/runtime/profiling/blade/templates/docker-compose.template.yml \
  --timeout 600 \
  --capture-debug-logs \
  --max-concurrent 2 \
  --gpu-assignment auto \
  --stagger-delay 10
```

**Note**: For complex multi-round profiling campaigns, you can create wrapper scripts that orchestrate multiple trial runs with different configurations.

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

### Quick Start: Running Profiling Trials

**IMPORTANT**: Always use concurrent execution for ~2x speedup on our dual-GPU system.

**⚠️ TRITON/DOCKER ISSUE**: Standard CI/CD images lack CUDA tools for Triton. Use local profiling image for any trials involving Triton kernels or torch.compile:

```bash
# For trials WITHOUT Triton/torch.compile (standard images)
linnaeus-prof-run \
  --trial-params-file /home/caleb/repo/linnaeus-deployment/linnaeus_deploy/configs/experiments/tests/v040/feature-name/trials.jsonl \
  --output-dir /home/caleb/repo/linnaeus/work/active/feature-name/results \
  --compose-template /home/caleb/repo/linnaeus-deployment/linnaeus_deploy/docker/runtime/profiling/blade/templates/docker-compose.template.yml \
  --timeout 600 \
  --max-concurrent 2 \
  --gpu-assignment auto \
  --capture-debug-logs

# For trials WITH Triton/torch.compile (local profiling image)
linnaeus-prof-run \
  --trial-params-file /home/caleb/repo/linnaeus-deployment/linnaeus_deploy/configs/experiments/tests/v040/feature-name/trials.jsonl \
  --output-dir /home/caleb/repo/linnaeus/work/active/feature-name/results \
  --compose-template /home/caleb/repo/linnaeus-deployment/linnaeus_deploy/docker/runtime/profiling/blade/templates/docker-compose-local-image.template.yml \
  --timeout 600 \
  --max-concurrent 2 \
  --gpu-assignment auto \
  --capture-debug-logs
```

**Building Local Profiling Image** (if needed):
```bash
cd /home/caleb/repo/linnaeus
./tools/docker/build_profiling.sh ampere main
```

### Manual Trial Setup Process (if not using /prof_impl):

1. **Create version-specific template** in linnaeus-deployment (private repo):
   ```bash
   cd /home/caleb/repo/linnaeus-deployment
   cp linnaeus_deploy/configs/experiments/tests/trial_template_MASTER.yaml \
      linnaeus_deploy/configs/experiments/tests/v040/trial_template_v040.yaml
   # Edit template to use amphibia dataset for faster iteration
   ```

2. **Create trials.jsonl** with baseline and optimized configurations:
   ```bash
   mkdir -p linnaeus_deploy/configs/experiments/tests/v040/feature-name/
   # Create trials.jsonl with exact commit hashes and proper opts
   ```

**Remember**: All trial configs stay in linnaeus-deployment (private), results go to linnaeus/work/ (public).

### Trial Management Best Practices:
1. **Git Hygiene**: Always commit AND push branches before running trials (containers pull from remote)
2. **Reproducibility**: Record exact commit hashes in trial definitions
3. **Environment Variables**: Use scenario files (env_yaml) for hardware-specific settings
4. **Debugging**: Capture debug logs with --capture-debug-logs for failed trials
5. **Organization**: Keep all trials for a branch in one jsonl file for investigative debugging
6. **Concurrent Execution**: Always use `--max-concurrent 2` for optimal performance
7. **Timeout Settings**: Use minimum 600s (10 min) for meaningful profiling data

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