# Running Inference from a Linnaeus Bundle

> Status: current workflow for released or locally prepared artifacts. This
> page assumes you already have a bundle. It does not assume a public model
> registry exists.

If you have an inference bundle, this is the straight path:

1. point the handler at `inference_config.yaml`
2. load one or more images
3. pass metadata only if the bundle expects it
4. read `typus` hierarchical outputs

## Prerequisites

- a Linnaeus environment (`uv sync --extra dev --extra cpu` is enough for local
  CPU use)
- a bundle directory containing at least:
  - `inference_config.yaml`
  - model weights
  - `taxonomy.json`
  - `class_index_map.json`
- an image to classify

If you do not already have a bundle, export one from a training run:

```bash
uv run python tools/prepare_inference_bundle.py \
  --experiment-dir /abs/path/to/experiment \
  --epoch 40
```

## Minimal Script

```python
from pathlib import Path
from PIL import Image
from linnaeus.inference.handler import LinnaeusInferenceHandler

BUNDLE_DIR = Path("/abs/path/to/inference_bundle")
CONFIG_FILE_PATH = BUNDLE_DIR / "inference_config.yaml"
IMAGE_PATH = Path("/abs/path/to/image.jpg")

handler = LinnaeusInferenceHandler.load_from_artifacts(
    config_file_path=CONFIG_FILE_PATH
)

image = Image.open(IMAGE_PATH).convert("RGB")
results = handler.predict(images=[image], metadata_list=None)

if results:
    print(results[0].model_dump_json(indent=2))
```

Run it with:

```bash
uv run python run_linnaeus_inference.py
```

## Metadata

If the bundle uses metadata, pass canonical raw fields when possible:

```python
metadata = {
    "lat": 34.0522,
    "lon": -118.2437,
    "datetime_utc": "2024-07-15T10:30:00Z",
    "elevation_m": 185.0,
    "component_vectors": {
        "weather_embed": [0.12, 0.98],
    },
}

results = handler.predict(images=[image], metadata_list=[metadata])
```

Use `component_vectors` only for components the bundle cannot reconstruct from
the raw scalar fields above.

## Remote-Backed Weights

If `model.weights_path` inside the bundle uses an `hf://...` URI, the handler
will resolve that file during `load_from_artifacts(...)`.

That is not the same thing as a documented bare model-ID loader. The local
bundle config is still the entrypoint.

## What This Page Does Not Cover

- discovering a public Linnaeus model registry
- observation-level or multi-view inference APIs
- service deployment

Those are separate questions. This page is only about using a concrete bundle
you already have.

## Next Steps

- [Inference Overview](./overview.md)
- [Inference Bundle](./inference_bundle.md)
- [Model Zoo](../models/model_zoo.md)
