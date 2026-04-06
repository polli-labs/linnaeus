# Profiling Overview

The profiling docs cover two related surfaces:

- the public CLI and analysis tools that live in this repo
- operator workflows that often depend on private trial manifests, private
  compose templates, and host-specific runtime setup that are not published
  here

If you are reading this from the public repo, treat this section as a guide to
the tooling contract, not as a promise that every launch recipe is runnable from
public assets alone.

## CLI Surface

The current discovery path is:

```bash
uv run linnaeus --help
```

Current mapping:

- `linnaeus run ...` delegates to the trial runner surface
- `linnaeus prof ...` delegates to the profiler analysis surface
- `linnaeus config render|validate|validation-plan ...` handles preflight
  config work without launching a run
- legacy `linnaeus-prof` and `linnaeus-prof-run` commands still exist

## Public-Safe Quickstart

You can verify the profiling toolchain from source without any private
manifests:

```bash
uv sync --extra dev --extra profiling --extra cpu
uv run linnaeus --help
uv run linnaeus prof --help
uv run linnaeus config validate --help
```

If you already have a profiler output directory, you can analyze it directly:

```bash
uv run linnaeus prof summary /path/to/run --output-format md
uv run linnaeus prof diff /path/to/baseline /path/to/candidate --output-format md
```

## What Usually Stays Private

Real profiling launches often need inputs that are not part of the public repo:

- host-specific Docker Compose templates
- private experiment config banks
- local dataset paths
- machine-specific environment overlays

That split is intentional. The public repo documents the code and the CLI
contract. Internal operator runbooks add the deployment details on top.

## Main Components

### 1. [Prof Run](./prof-run.md)

Trial orchestration for repeated or comparative runs.

### 2. [Prof CLI](./prof-cli.md)

Trace inspection, summaries, and run-to-run comparison.

### 3. [Prof Validate](./prof-validate.md)

Preflight checks for config, trial manifests, and runtime assumptions.

### 4. [Profiling Levels](./profiling-levels.md)

Instrumentation depth from coarse timing to detailed module-level traces.

## Common Workflows

### Analyze an existing run

Use this when you already have profiler output and want a readable summary.

### Compare baseline vs candidate

Use this when you want to test a model or runtime change against a known
baseline.

### Preflight a launch surface

Use `linnaeus config validate` and `prof-validate` before long profiling waves.
That catches path, config, and contract failures sooner than a failed Docker
launch.

## Scope Notes

These docs are current for the profiling toolchain itself. They are not a full
recipe book for every internal operator workflow. If you need those details,
you are in the private-runtime lane, not the public-docs lane.

## Read Next

- [Prof Run](./prof-run.md)
- [Prof CLI](./prof-cli.md)
- [Prof Validate](./prof-validate.md)
- [Best Practices](./best-practices.md)
