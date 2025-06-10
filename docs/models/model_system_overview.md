# Linnaeus Architecture Overview

This document explains the Linnaeus model architecture system, focusing on its component design, configuration approach, and extensibility patterns.

## Core Design: Component Registry and Building Blocks

Linnaeus uses a two-tiered approach for organizing neural network components:

### 1. Registered Components

These are the primary swappable units selected via configuration:

- **Models**: Complete architectures like `mFormerV1` and `mFormerV0`
- **Classification Heads**: Output layers mapping features to predictions (e.g., `LinearHead`, `ConditionalClassifierHead`)
- **Registration**: Registered with factory system via decorators (`@register_model`, `@register_head`)
- **Selection**: Chosen via configuration parameters (`MODEL.TYPE`, `MODEL.CLASSIFICATION.HEADS.<task>.TYPE`)

### 2. Building Blocks

Fundamental, reusable modules that form the internal structures:

- **Purpose**: Encapsulate specific architectural components (attention blocks, MLP blocks, etc.)
- **Examples**: `RoPE2DMHSABlock`, `ConvNeXtBlock`, `Mlp`, `DropPath`
- **Instantiation**: Imported and used directly within model implementations
- **Configuration**: Parameters set within parent component's configuration section

## Example Architecture: mFormerV1

```
mFormerV1 (Registered Model)
├── ConvNeXtBlock stages (Building Blocks)
│   ├── Configuration in MODEL.CONVNEXT_STAGES
│   └── Parameters: depths, dims, drop_path_rate, etc.
├── RoPE2DMHSABlock stages (Building Blocks)
│   ├── Configuration in MODEL.ROPE_STAGES
│   └── Parameters: depths, num_heads, window_size, etc.
└── Classification heads (Registered Components)
    ├── Configuration in MODEL.CLASSIFICATION.HEADS
    └── One head per task, each with its own TYPE and parameters
```

## Factory System (`model_factory.py`)

The factory pattern enables runtime component selection and instantiation:

- **Registration**: Maps string identifiers to component classes
- **Purpose**: Creates requested component instances based on configuration
- **Primary registries**:
  - `MODEL_REGISTRY`: Top-level model architectures
  - `HEAD_REGISTRY`: Classification output heads
  - `ATTENTION_REGISTRY`: Attention mechanisms (optional use)
  - `AGGREGATION_REGISTRY`: Feature aggregation methods (optional use)
  - `RESOLVER_REGISTRY`: Feature resolution strategies (optional use)

## Configuration System (YACS)

Configuration files (`*.yaml`) define model and experiment parameters:

```yaml
MODEL:
  TYPE: "mFormerV1"
  EMBED_DIM: 192
  META_DIMS: [16]
  CONVNEXT_STAGES:
    DEPTHS: [3, 3, 9, 3]
    DIMS: [192, 384, 768, 1536]
  ROPE_STAGES:
    DEPTHS: [3, 3, 3]
    NUM_HEADS: [6, 12, 24]
    WINDOW_SIZE: 7
  CLASSIFICATION:
    HEADS:
      task_taxonomy:
        TYPE: "LinearHead"
        NUM_CLASSES: 1000
```

Configuration handles:
- Component selection (`TYPE` fields)
- Hyperparameter setting (dimensions, depths, learning rates)
- Training dynamics (schedules, optimizers, augmentations)
- Environment setup (paths, devices, output locations)

## Model Extension Patterns

To create a new model architecture in Linnaeus:

1. **Create the model class** in `linnaeus/models/` directory:
   ```python
   @register_model
   class NewArchitecture(BaseModel):
       """New architecture implementation"""
       
       def __init__(self, cfg):
           super().__init__(cfg)
           # Build internal structure using Building Blocks
           # Configure based on cfg parameters
           
       def forward(self, x, metadata=None):
           # Implement forward pass
           return output
   ```

2. **Create configuration files** in `configs/model/archs/NewArchitecture/`:
   ```yaml
   MODEL:
     TYPE: "NewArchitecture"
     # Architecture-specific parameters
   ```

3. **Register with factory**:
   ```python
   from linnaeus.models.model_factory import register_model
   ```

## Core Available Models

### mFormerV0
- Hybrid CNN-Transformer architecture
- Implements MetaFormer paradigm with RelativeAttention
- 3 model sizes: Small (15M), Medium (35M), Large (55M)

### mFormerV1
- Enhanced hybrid architecture with improved attention
- Implements 2D RoPE (Rotary Position Embedding)
- Flash Attention compatible for faster training
- 4 model sizes: Small (18M), Medium (38M), Large (65M), XLarge (120M)

## Building Block Libraries

Linnaeus includes various implementation blocks:

- **Attention Mechanisms**:
  - `RoPE2DMHSABlock`: 2D Rotary Position Embedding with Multi-Head Self Attention
  - `RelativeMHSA`: Relative position bias attention
  - `LinformerSelfAttention`: Linear attention for efficiency
  - Other specialized attention variants

- **Convolution Blocks**:
  - `ConvNeXtBlock`: Modern CNN block with depthwise convolutions
  - `MBConv`: Mobile inverted bottleneck convolution

- **Common Components**:
  - `Mlp`: Multi-layer perceptron with configurable activation
  - `DropPath`: Stochastic depth for regularization
  - `ProgressivePatchEmbed`: Progressive patch embedding

## Customization Guidelines

When extending Linnaeus:
- Create new Building Blocks for reusable architectural components
- Register new top-level Models for major architecture changes
- Register new Classification Heads for output transformations
- Use configuration for experiment-level parameter tuning
- Prefer code changes for deep architectural modifications