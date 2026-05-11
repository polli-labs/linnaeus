# Polli Linnaeus

Polli Linnaeus is a research codebase for taxonomy-aware visual classification.
The active line today is a frozen-backbone DINOv3 system with hierarchical
heads and optional multi-view pooling. The repository still contains older
mFormer-era model families, configs, and migration records because they remain
part of the project's provenance.

[View changelog](CHANGELOG.md)

## Current Public Position

- The active architecture is `DINOv3MultiHead`: frozen ViT-B/16 backbone,
  per-rank classification heads, and optional MIL pooling over multi-view bags.
- The documented inference path is bundle-first. Start from a local
  `inference_config.yaml`; do not assume a bare model-ID loader or a hosted API
  surface.
- No verified public Linnaeus model registry is live from this repo today.
- The older plan to publish a North American `mFormerV1_sm` suite has been
  retired.
- The config system is mid-transition. YACS still drives runtime resolution
  today while typed/Pydantic validation work continues.

If you want the shortest honest overview, start with
[Current State](docs/current_state.md).

## Private Development Mirror

`polli-labs/linnaeus-dev` is the private development mirror for this codebase.
`polli-labs/linnaeus` is the public release repository.

For the remote contract (`origin` = private dev, `public` = public upstream),
promotion flow, and cutover guardrails, see
[docs/migration/dev_public_release_contract.md](docs/migration/dev_public_release_contract.md).

## Install From Source

```bash
uv venv .venv
uv sync --extra dev --extra cpu
uv run python -c "import linnaeus; print(linnaeus.__version__)"
uv run linnaeus-prof --help
uv run mkdocs build --strict
```

Use `--extra cuda` instead of `--extra cpu` on a CUDA host.

For setup details, see [Installation](docs/installation.md) and the
[UV guide](docs/dev/uv.md).

For the repo gate that mirrors the core CI baseline, run:

```bash
bash tools/ci/run_core_baseline_gate.sh
```

See [docs/ci.md](docs/ci.md) for CI scope and failure triage.

## What This Repo Covers

- Source training for hierarchical taxonomic classifiers
- HDF5-first and hybrid image/label data pipelines
- Bundle-based inference via `LinnaeusInferenceHandler`
- Experiment preflight, validation, profiling, and operator tooling
- Migration-era records that explain how the current layout replaced older
  deployment and path conventions

## Start Here

- [Current State](docs/current_state.md): what is true today, what is in
  flight, and what not to assume
- [Documentation Hub](docs/index.md): top-level routing
- [Getting Started](docs/getting_started.md): source checkout orientation
- [Training Overview](docs/training/overview.md): current training surface
- [Inference Overview](docs/inference/overview.md): current bundle-first
  inference contract
- [Model System Overview](docs/models/model_system_overview.md): active vs
  legacy model families
- [Profiling Overview](docs/profiling/README.md): operator-facing run and
  analysis surface

## Docker

Linnaeus ships a two-stage Docker build system so dependency-heavy base images
can be reused while the runtime layer changes quickly with the code.

For image layout, tags, and build workflow, see
[tools/docker/README.md](tools/docker/README.md).

## Research Use

If you use Polli Linnaeus in research, cite:

```text
@software{pollilinnaeus2024,
  author = {Sowers, Caleb},
  title = {Polli Linnaeus: A Deep Learning Framework for Taxonomic Recognition},
  year = {2024},
  publisher = {Polli Labs Inc.},
  url = {https://github.com/polli-labs/linnaeus}
}
```

## Community and Contributions

Open issues and public discussion belong in
[`polli-labs/linnaeus`](https://github.com/polli-labs/linnaeus). Internal work
lands in the private dev mirror first and is promoted outward deliberately.

## License

Apache License 2.0
