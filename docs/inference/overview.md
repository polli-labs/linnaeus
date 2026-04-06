# Inference Overview

> Status: the documented inference contract is bundle-first. Start from a local
> `inference_config.yaml` and load the handler with
> `LinnaeusInferenceHandler.load_from_artifacts(...)`. This repo does not
> document a bare model-ID loader on the handler itself.

Linnaeus inference is a Python-level, bundle-backed surface. It is good enough
for local use, bundle validation, and lightweight service adapters. It should
not be described as a finished public inference platform.

## What Is Stable Today

The core inference seam is `linnaeus.inference.handler.LinnaeusInferenceHandler`.
The handler is responsible for:

- loading a bundle and reconstructing the model
- preprocessing images and optional metadata
- producing `typus` hierarchical classification results
- exposing bundle metadata through `info()`

The documented load surface is:

```python
handler = LinnaeusInferenceHandler.load_from_artifacts(
    config_file_path="/abs/path/to/inference_bundle/inference_config.yaml",
    artifacts_base_dir=None,
    model_weights_path_override=None,
    taxonomy_tree_path_override=None,
    class_index_map_path_override=None,
)
```

## The Current Contract

Inference starts from a local bundle directory. That bundle contains:

- `inference_config.yaml`
- model weights
- taxonomy data
- class index maps

The bundle may still refer to remote-backed weights by using `hf://...` inside
`model.weights_path`, but the entrypoint is still the local config file.

When `metadata_preprocessing.components` is present, it is the authoritative
metadata contract. For older local bundles that lack that section, the handler
can still recover metadata semantics from sibling training config artifacts in
some cases.

See [Inference Bundle](./inference_bundle.md) for the full artifact format.

## What Is Not Promised

Do not assume any of the following:

- a bare Hugging Face repo ID is the primary loading surface
- a public bundle registry exists
- observation-level or multi-view inference is first-class in the public API
- the LitServe page is a production deployment guide

The current inference architecture is deliberately narrow. That is by design.

## Minimal Example

```python
from pathlib import Path
from PIL import Image
from linnaeus.inference.handler import LinnaeusInferenceHandler

bundle_config_path = Path("/abs/path/to/inference_bundle/inference_config.yaml")
handler = LinnaeusInferenceHandler.load_from_artifacts(config_file_path=bundle_config_path)

image = Image.open("/abs/path/to/image.jpg").convert("RGB")
results = handler.predict(images=[image], metadata_list=None)

if results:
    print(results[0].model_dump_json(indent=2))
```

If the bundle expects metadata, pass canonical raw fields such as `lat`, `lon`,
`datetime_utc`, and `elevation_m`, plus `component_vectors` for any component
the bundle cannot derive from those raw scalars.

## Testing

The repo keeps targeted inference coverage in:

- `tests/test_inference_handler.py`
- `tests/test_inference_components.py`
- `tests/test_inference_bundle_contract.py`

Those tests are the best way to tell whether a handler or bundle change is
still honest.

## Further Reading

- [Running Inference from a Bundle](./running_inference_with_pretrained_models.md)
- [Inference Bundle](./inference_bundle.md)
- [LitServe Sketch](./litserve.md)
