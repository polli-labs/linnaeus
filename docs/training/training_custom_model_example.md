# Train a Custom Model

This guide covers the current source-training workflow for your own data. It
does not assume access to private Polli experiment manifests or to unreleased
public artifacts.

## What You Need

1. A working Linnaeus checkout and environment
2. A `labels.h5` file plus image files in the shape described by
   [Data Loading for Training](./data_loading.md)
3. A taxonomy surface if you are doing hierarchical classification
4. An experiment config you can render and validate before launch

## 1. Prepare The Dataset

Linnaeus expects a hybrid dataset surface:

- images on disk
- labels and optional metadata in `labels.h5`

At minimum, your `labels.h5` needs:

- `img_identifiers`
- one or more `taxa_LXX` label datasets
- any metadata datasets you intend to enable in the config

Do not rely on this page as the schema reference. Use
[Data Loading for Training](./data_loading.md) for the actual contract.

## 2. Write An Experiment Config

Start from a small config you understand. The active research line uses
`DINOv3MultiHead`, but the codebase still contains older mFormer configs. Be
explicit about what you intend to run.

This schematic example shows the shape of a current custom experiment:

```yaml
EXPERIMENT:
  PROJECT: "my-linnaeus-project"
  NAME: "my-custom-run"

MODEL:
  TYPE: "DINOv3MultiHead"
  NAME: "my_dinov3_run"
  BASE: [""]
  CLASSIFICATION:
    HEADS:
      taxa_L10:
        TYPE: "ConditionalClassifier"
      taxa_L20:
        TYPE: "ConditionalClassifier"

DATA:
  TASK_KEYS_H5: ["taxa_L10", "taxa_L20"]
  HYBRID:
    USE_HYBRID: True
    IMAGES_DIR: "/abs/path/to/images"
  H5:
    LABELS_PATH: "/abs/path/to/labels.h5"
    TRAIN_VAL_SPLIT_RATIO: 0.9
    TRAIN_VAL_SPLIT_SEED: 42

TRAIN:
  EPOCHS: 10
```

The point of the example is the contract, not the exact hyperparameters.
Before long operator runs, inspect what your real config resolves to through
the private-runtime preflight path that owns your trial manifests.

## 3. Inspect Before Launch

```bash
uv run python -m linnaeus.main --help
```

If private-runtime validation fails, fix that first. Do not start training and
hope the runtime will sort it out.

## 4. Launch Training

The current training entrypoint is `linnaeus.main`:

```bash
uv run python -m linnaeus.main \
  --cfg /abs/path/to/my_experiment.yaml
```

To override values at launch time:

```bash
uv run python -m linnaeus.main \
  --cfg /abs/path/to/my_experiment.yaml \
  --opts TRAIN.BATCH_SIZE 32 EXPERIMENT.WANDB.ENABLED False
```

## 5. Watch The Right Signals

During training, pay attention to:

- partial chain accuracy (PCA)
- DWPCA
- per-rank accuracies
- validation loss
- schedule summary output

If you only watch scalar loss, you will miss the failures that matter most for
hierarchical prediction quality.

## 6. Export For Inference

Once you have a trained experiment you trust, build an inference bundle:

```bash
uv run python tools/prepare_inference_bundle.py \
  --experiment-dir /abs/path/to/experiment \
  --epoch 10
```

Then follow [Inference Overview](../inference/overview.md) and
[Running Inference from a Bundle](../inference/running_inference_with_pretrained_models.md).

## Next Steps

- [Training Overview](./overview.md)
- [Data Loading for Training](./data_loading.md)
- [Validation](../evaluation/validation.md)
- [Model System Overview](../models/model_system_overview.md)
