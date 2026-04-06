# Documentation Hub

This hub routes readers to the current surfaces first and keeps historical
material out of the main path.

## Start Here

- **[Current State](./current_state.md):** what Linnaeus is today, what is
  still in flight, and what not to assume
- **[Project README](https://github.com/polli-labs/linnaeus/blob/main/README.md):**
  repo-level overview and source install path
- **[Installation](./installation.md):** environment setup
- **[Getting Started](./getting_started.md):** orientation for working from
  source

## Models and Release Posture

- **[Model Zoo](./models/model_zoo.md):** current release posture; this is not a
  live public registry yet
- **[Model System Overview](./models/model_system_overview.md):** active vs
  legacy model families and the current extension surface

## Training

- **[Training Overview](./training/overview.md):** current training entrypoint,
  config preflight, metrics, and operator guardrails
- **[Training a Custom Model](./training/training_custom_model_example.md):**
  current source-training workflow for your own data
- **[Data Loading for Training](./training/data_loading.md):** HDF5 and hybrid
  dataset contract
- **[Phase 2 Abstention RL](./training/phase2_abstention_rl.md):** experimental
  follow-on work for abstention training, not the main release path

## Inference

- **[Inference Overview](./inference/overview.md):** current handler and bundle
  contract
- **[Running Inference from a Bundle](./inference/running_inference_with_pretrained_models.md):**
  hands-on bundle workflow
- **[Inference Bundle](./inference/inference_bundle.md):** artifact format and
  export contract
- **[LitServe Sketch](./inference/litserve.md):** narrow service integration
  example, not a production deployment guide

## Evaluation and Profiling

- **[Evaluation Overview](./evaluation/overview.md):** validation surfaces and
  metrics routing
- **[Validation](./evaluation/validation.md):** scheduling, masking modes, and
  validation-only constraints
- **[Profiling Overview](./profiling/README.md):** profiling and operator
  tooling

## Provenance and Historical Material

- **[Migration and Historical Overview](./migration/index.md):** cutover records
  and legacy routing
- **[Official Dataset Provenance (ibrida-v0-r1)](./datasets/dataset_generation.md):**
  historical provenance for the retired initial release program

## Developer References

- **[CI and Docker Guide](./ci.md):** build, CI, and image layout
- **[Docker Build Guide](https://github.com/polli-labs/linnaeus/blob/main/tools/docker/README.md):**
  repo-root Docker build docs
- **[Training Loop Guide](./dev/01_training_loop_and_progress.md):** deeper
  developer architecture notes
- **[Known Limitations](./known_limitations.md):** current caveats and gaps

If you need more than this hub provides, open an issue in
[`polli-labs/linnaeus`](https://github.com/polli-labs/linnaeus/issues).
