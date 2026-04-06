# The Linnaeus Inference Bundle

> Status: current contract for handler-backed inference. Bundle-backed loading
> is the documented inference path in this repo.

An inference bundle is the artifact set the handler needs in order to produce
predictions honestly. If you do not have a bundle, you do not have a supported
inference input for the current docs.

## Bundle Structure

A typical inference bundle is a directory containing the following key files:

*   **`inference_config.yaml`**: local entrypoint for the handler
*   **Model weights**: the trained state dictionary or remote-backed weight URI
    referenced by the config
*   **Taxonomy data**: the hierarchy used to interpret predictions
*   **Class index map**: translation between model output indices and `typus`
    taxon IDs

### Example Directory Layout:

```
my_model_inference_bundle/
├── inference_config.yaml
├── pytorch_model.bin
├── taxonomy.json
└── class_index_map.json
```

## Key Components in Detail

### 1. `inference_config.yaml`

This YAML file is the entry point for the inference handler. It contains several sections:

*   **`model`**:
    *   `architecture_name`: human-readable model label
    *   `architecture_type`: registered top-level model type, for example
        `DINOv3MultiHead`
    *   `resolved_model_config`: Optional base-free resolved config fragment used to reconstruct the training model without needing a colocated experiment config. New bundles produced by `tools/prepare_inference_bundle.py` should include this.
    *   `weights_path`: Path (relative to the bundle root or absolute) to the model's weights file, or an `hf://org/repo/path/to/file` reference.
    *   `model_task_keys_ordered`: Ordered list of internal Linnaeus task keys the model predicts (e.g., `["taxa_L70", "taxa_L60", ..., "taxa_L10"]`). This order must match the model's output structure.
    *   `num_classes_per_task`: List of class counts (including null) for each task in `model_task_keys_ordered`.
    *   `null_class_indices`: Dictionary mapping each Linnaeus `task_key` to the model's output index that represents the "null" or "unknown" class for that task.
    *   `expected_aux_vector_length`: (Optional) The expected length of the auxiliary feature vector if metadata is used. If `null` or not provided, the `LinnaeusInferenceHandler` will attempt to derive this from the `metadata_preprocessing` section. It's recommended to set this explicitly if metadata is used.

*   **`input_preprocessing`**:
    *   `image_size`: Expected image input dimensions `[C, H, W]`.
    *   `image_mean`: Mean values for image normalization.
    *   `image_std`: Standard deviation values for image normalization.
    *   `image_interpolation`: Interpolation method for resizing (e.g., `bilinear`).

*   **`metadata_preprocessing`**:
    *   `use_geolocation`: Boolean, whether latitude/longitude are used.
    *   `use_temporal`: Boolean, whether date/time are used.
    *   `temporal_use_julian_day`: Boolean, use day-of-year (if true) or month-of-year (if false) for temporal encoding.
    *   `temporal_use_hour`: Boolean, include hour-of-day sinusoidal features.
    *   `use_elevation`: Boolean, whether elevation is used.
    *   `elevation_scales`: List of scale values for elevation encoding.
    *   `components`: Optional explicit metadata component contract. When present,
        this is the authoritative description of aux-vector ordering, per-component
        dimensions, encoding semantics, and any raw fields that can be projected at
        inference time. New bundles exported by
        `tools/prepare_inference_bundle.py` should include this.

*   **`taxonomy_data`**:
    *   `source_name`: Source of the taxonomy (e.g., `CoL2024`).
    *   `version`: Version of the taxonomy.
    *   `root_identifier`: Root taxon ID or name covered by the model (for context).
    *   `taxonomy_tree_path`: Path to the `taxonomy.json` file.
    *   `class_index_map_path`: Path to the `class_index_map.json` file.

*   **`inference_options`**:
    *   `default_top_k`: Default K for top-K predictions.
    *   `device`: Device for inference (`cpu`, `cuda`, `mps`, or `auto`).
    *   `batch_size`: Maximum batch size for the handler's internal processing.
    *   `enable_hierarchical_consistency_check`: Boolean, whether to enforce parent-child consistency in predictions.
    *   `handler_version`: Version of the `LinnaeusInferenceHandler` this bundle is intended for.
    *   `artifacts_source_uri`: (Optional) URI indicating where the bundle was sourced from. This is metadata for the bundle itself; the current handler contract still starts from a local `config_file_path`.

*   **`model_description`**: (Optional) A brief human-readable description of the model configuration.

### 2. Model Weights (e.g., `pytorch_model.bin`)

This is the saved model state the handler loads at inference time.

### 3. Taxonomy Data (`taxonomy.json`)

This file stores the taxonomic hierarchy relevant to the model. It's created by calling the `.save()` method of a `linnaeus.utils.taxonomy.taxonomy_tree.TaxonomyTree` instance. The `TaxonomyTree` is typically built from the `hierarchy_map` generated during dataset processing. The JSON file includes:
*   `task_keys`: Ordered list of Linnaeus task keys representing hierarchy levels (typically lowest rank to highest, e.g., `["taxa_L10_species", "taxa_L20_genus", ...]`).
*   `num_classes`: Dictionary mapping each task key to the number of classes at that level.
*   `hierarchy_map_raw`: The core map defining parent-child relationships: `Dict[child_task_key, Dict[child_model_idx, parent_model_idx]]`.

### 4. Class Index Map (`class_index_map.json`)

This JSON file provides the critical mappings needed to translate the model's numerical outputs into meaningful taxonomic information using `typus` standards. It contains:
*   `idx_to_taxon_id`: Maps `RankLevel.value` to a dictionary of `{model_class_index: typus_taxon_id}`.
*   `taxon_id_to_idx`: The inverse of `idx_to_taxon_id`. Maps `RankLevel.value` to `{typus_taxon_id: model_class_index}`.
*   `null_taxon_ids`: Maps `RankLevel.value` to the `typus_taxon_id` that represents the "null" or "unknown" concept for that rank.
*   `num_classes_per_rank`: Maps `RankLevel.value` to the total number of classes (including null) that the model predicts for that rank.

## Creating an Inference Bundle

There are two practical ways to create a bundle today.

### Option A: Use the repo helper

The repo includes `tools/prepare_inference_bundle.py`, which exports a bundle
from a training experiment directory:

```bash
uv run python tools/prepare_inference_bundle.py \
  --experiment-dir /abs/path/to/experiment \
  --epoch 40 \
  --output-dir /abs/path/to/inference_bundle
```

If `--output-dir` is omitted, the script writes the bundle to `inference/`
inside the experiment directory.

For phase-1 contract safety, new bundles now carry an embedded
`model.resolved_model_config` so the handler can rebuild the exact training-side
model without guessing from a variant name alone.

They also carry an explicit `metadata_preprocessing.components` contract when
metadata is enabled. That lets the handler distinguish between:

*   raw scalar fields it knows how to project itself today
    (`lat`/`lon`, `datetime_utc`, `elevation_m`)
*   pre-encoded component vectors that must be supplied directly by the caller

For local legacy bundles that still lack `components`, the handler will attempt
to hydrate the metadata contract from a sibling `../configs/experiment_config.yaml`
plus the label-HDF5 dataset attrs referenced there.

### Option B: Assemble a bundle manually

Manual creation still works, but it is easier to get wrong:

1.  **Training a Model**: Train your Linnaeus model and save its state dictionary.
2.  **Preparing Taxonomy Artifacts**:
    *   During your data preparation or training setup, you should have access to the `TaxonomyTree` instance. Save it using `taxonomy_tree.save("taxonomy.json")`.
    *   You will need to construct the `class_index_map.json` file. This requires knowing:
        *   How your model's output indices for each task head map to specific `typus_taxon_id`s.
        *   Which `typus_taxon_id` represents the "null" class for each rank.
        *   The total number of classes (outputs) for each rank-specific head in your model.
        This mapping is often established during dataset creation and model configuration.
3.  **Writing `inference_config.yaml`**:
    *   Carefully create this file, ensuring all paths correctly point to your artifact files (model weights, taxonomy.json, class_index_map.json) relative to the bundle's root.
    *   Fill in all model parameters (`model_task_keys_ordered`, `num_classes_per_task`, `null_class_indices`) to match your trained model's architecture precisely.
    *   Configure preprocessing and inference options as needed.
4.  **Assembling the Bundle**: Place all these files into a single directory.

## Using the Bundle

Once created, the bundle can be used with `LinnaeusInferenceHandler`:

```python
from pathlib import Path
from linnaeus.inference.handler import LinnaeusInferenceHandler

bundle_config_path = Path("/path/to/my_model_inference_bundle/inference_config.yaml")
handler = LinnaeusInferenceHandler.load_from_artifacts(config_file_path=bundle_config_path)

# Now the handler is ready for predictions
# image = Image.open(...)
# results = handler.predict(images=[image])
# print(results[0].model_dump_json(indent=2))
```

If the bundle uses metadata, prefer the canonical request shape below:

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

Use `component_vectors` for any component that the bundle marks as a
`passthrough_vector` or otherwise cannot reconstruct from the canonical raw
scalar fields above.

This contract exists so the handler can reconstruct a trained model without
guessing from a loose model nickname or an implicit sidecar directory.
