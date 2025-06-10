# Developer Guide: Model Metadata, Parameter Filtering & Checkpoint Loading

## 1. Introduction

This guide explains the system used in linnaeus for managing model parameters and handling pretrained checkpoints through model-defined metadata. This approach promotes modularity, consistency, and simplifies the integration of new model architectures and diverse pretrained weights.

**Core Concepts:**

1.  **Model-Defined Metadata:** Each model architecture class (inheriting from `BaseModel`) defines two crucial properties:
    *   `parameter_groups_metadata` (`@property`): Describes the semantic grouping of its parameters (e.g., which parameters belong to convolution stages, transformer stages, specific heads, embeddings). This *informs* training-time operations but doesn't *execute* them.
    *   `pretrained_ckpt_handling_metadata` (`@property`): Specifies *instructions* for the `load_pretrained` utility on how checkpoints should be processed *before* being loaded into an instance of *this specific* model architecture (e.g., which layers/buffers to drop, whether to interpolate positional embeddings).
2.  **Configuration-Driven Filtering:** The *actual rules* for grouping parameters for multi-optimizer, multi-LR scheduling, or GradNorm exclusion are defined in the **YAML configuration file**.
3.  **Unified Filtering Utility:** The `UnifiedParamFilter` system (`utils/unified_filtering.py`) *reads the rules from the YAML config* and applies them to the model's parameters. It does *not* directly parse the model's `parameter_groups_metadata`.
4.  **Metadata-Aware Checkpoint Loading:** The `load_pretrained` utility (`utils/checkpoint.py`) *reads* the target model's `pretrained_ckpt_handling_metadata` to adapt the checkpoint dictionary *before* attempting to load it using `model.load_state_dict`.

## 2. Parameter Grouping Metadata (`parameter_groups_metadata`)

### Purpose

This `@property` within a model class allows the model architecture to **declare the semantic roles** of its different parameter sets. It serves as self-documentation and provides hints for setting up filtering rules in the configuration.

**It does NOT directly control filtering.** Filtering logic is defined in the YAML config.

### Definition within a Model Class

Models should define `parameter_groups_metadata` returning a dictionary. Keys are categories, values are lists of representative *name patterns* associated with that category.

```python
# Inside your model class (e.g., linnaeus/models/mFormerV1.py)
@property
def parameter_groups_metadata(self) -> Dict[str, Any]:
    return {
        "stages": {
            "convnext_stages": ["stem.", "stages.0.", "stages.1.", "downsample_layers.0", "downsample_layers.1"],
            "rope_stages": ["stages.2.", "stages.3.", "downsample_layers.2", "downsample_layers.3"],
            "rope_freqs": ["freqs"], # Specific learnable parameter
        },
        "heads": {
            "classification_heads": ["head."], # Use dot to avoid matching 'headroom' etc.
            "meta_heads": ["meta_"],
        },
        "embeddings": ["cls_token"],
        "norm_layers": ["norm", ".bn", "LayerNorm"], # Include specific layer types if needed
        "aggregation": ["cl_1_fc.", "aggregate.", "final_norm."],
    }
```

### How Filtering Actually Happens

Filtering for multi-optimizer, multi-LR, or GradNorm is configured in YAML and executed by `UnifiedParamFilter`. The YAML `FILTER` definitions use rules based on name, dimension, layer type, etc.

Example (`config.yaml` for GradNorm exclusion):

```yaml
LOSS:
  GRAD_WEIGHTING:
    TASK:
      # ... other GradNorm params ...
      EXCLUDE_CONFIG: # Filter definition read by UnifiedParamFilter
        TYPE: "or"    # Combine rules with OR
        FILTERS:
          - TYPE: "name" # Rule 1: Match by name
            MATCH_TYPE: "startswith"
            PATTERNS: ["head.", "meta_"] # Exclude classification and meta heads
          # Add more rules if needed, e.g., excluding LayerNorm biases:
          # - TYPE: "and"
          #   FILTERS:
          #     - {TYPE: "layer_type", LAYER_TYPES: ["LayerNorm"], model: "@MODEL"} # Needs model access
          #     - {TYPE: "name", MATCH_TYPE: "endswith", PATTERNS: [".bias"]}
```

-   `UnifiedParamFilter` is instantiated with this config and the model instance.
-   It applies the defined rules (e.g., `NameFilter`, `LayerTypeFilter`) to categorize parameters.
-   The model's `parameter_groups_metadata` is **not directly used** by the filter *execution* but serves as a reference for writing the filter rules in the config.

## 3. Pretrained Checkpoint Handling Metadata (`pretrained_ckpt_handling_metadata`)

### Purpose

This `@property` within a model class provides **instructions to the `load_pretrained` utility** on how to process a checkpoint *before* calling `model.load_state_dict`. This allows the *target model architecture* to define how it wants to adapt potentially incompatible checkpoints.

### Definition within a Model Class

```python
# Inside your model class (e.g., linnaeus/models/mFormerV1.py)
@property
def pretrained_ckpt_handling_metadata(self) -> Dict[str, Any]:
    return {
        "drop_buffers": [], # e.g., ["relative_position_index"] for mFormerV0
        "drop_params": ["head.", "meta_", "pos_embed"], # Always drop classifier, meta heads, and any absolute pos embeds
        "interpolate_rel_pos_bias": False, # mFormerV1 uses RoPE, no relative bias tables
        "supports_module_prefix": True, # Allow handling DDP 'module.' prefix
        "strict": False, # Use strict=False for load_state_dict by default
    }
```

### How Checkpoint Loading Actually Happens (`utils/checkpoint.py`)

1.  `load_pretrained` is called with the target `model` instance and the `config`.
2.  It loads the raw checkpoint dictionary from the path specified in `config.MODEL.PRETRAINED`.
3.  **Crucially, it accesses `model.pretrained_ckpt_handling_metadata` from the *target model instance***.
4.  It *modifies* the loaded checkpoint dictionary based on the rules specified in the metadata (`drop_buffers`, `drop_params`, handling `module.` prefix).
5.  If `interpolate_rel_pos_bias` is `True` in the metadata, it calls `relative_bias_interpolate`.
6.  If a custom source mapping is specified (e.g., `config.MODEL.PRETRAINED_SOURCE == 'metaformer'` or `'stitched_convnext_ropevit'`), it calls the corresponding `map_*_checkpoint` function *before* applying the metadata rules (or as part of a dedicated loading function like the planned `load_stitched_pretrained`).
7.  Finally, it calls `model.load_state_dict(modified_checkpoint['model'], strict=metadata.get('strict', False))`.

## 4. Key Differences Summarized

| Feature                        | Defined In                                   | Used By                                                        | Purpose                                                                                                |
| :----------------------------- | :------------------------------------------- | :------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| Parameter Grouping Rules       | **YAML Config** (`OPTIMIZER`, `LR_SCHEDULER`, `LOSS`) | `UnifiedParamFilter` (called by `optimizers/build.py`, `loss/gradient_weighting.py`) | **Executes** the filtering logic for multi-optimizer, multi-LR, GradNorm exclusion.                    |
| `parameter_groups_metadata`  | Model Class (`@property`)                    | Developers (for writing YAML filters), Future Tools              | **Declares** semantic structure of the model's parameters. *Informs* filter design but doesn't execute it. |
| Checkpoint Handling Rules    | Model Class (`@property`)                    | `load_pretrained` utility (`utils/checkpoint.py`)             | **Instructs** the loading utility on how to modify a checkpoint *before* loading into *this* model type. |
| Checkpoint Key/Shape Mapping | Custom `map_*` functions (`utils/checkpoint.py`) | `load_pretrained` utility (if `PRETRAINED_SOURCE` triggers it) | Performs complex renaming/reshaping for specific checkpoint sources *before* metadata rules are applied. |

## 5. Debugging (Recap)

*   **Filtering Issues:** Check YAML filter definitions. Use `inspect_multilr_filters` / `inspect_gradnorm_filters` or `UnifiedParamFilter.log_matches` with `DEBUG.OPTIMIZER=True`.
*   **Loading Issues:** Check `pretrained_ckpt_handling_metadata` in your model. Use `debug_load_checkpoint` *before* modifications in `load_pretrained`. Check the `load_state_dict` return value. Use `DEBUG.CHECKPOINT=True`.

This revised structure emphasizes that model metadata *informs* configuration and utilities, while the *execution* of filtering and loading is handled by dedicated configuration sections (YAML) and utility functions (`UnifiedParamFilter`, `load_pretrained`).