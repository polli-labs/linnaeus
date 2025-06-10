# Polli Linnaeus

Polli Linnaeus is a deep learning framework for taxonomic image classification, designed for biodiversity monitoring applications. Built on PyTorch, it combines hierarchical classification, multitask learning, and metadata integration with efficient training and inference pipelines. Polli Linnaeus is an open-source deep learning framework. It serves as a research platform for developing and evaluating novel approaches to ecological image analysis.

[View Changelog](CHANGELOG.md)

## Key Features

- **Advanced Architectures**: Implements `mFormerV1` and `mFormerV0` with hybrid CNN-Transformer designs
- **Hierarchical Classification**: Natively supports taxonomy-aware classification with specialized loss functions
- **Metadata Integration**: Combines image data with categorical/numerical metadata for improved accuracy
- **H5 Optimized Dataloading**: High-throughput data loading using HDF5 datasets
- **Reproducible Workflows**: Configuration-driven experiments with deterministic training
- **Streamlined Inference:** Includes a comprehensive inference pipeline (`LinnaeusInferenceHandler`) with support for Hugging Face Hub models, `typus` structured outputs, and metadata integration.
- **Deployment Ready**: Includes `LinnaeusInferenceHandler` for use with LitServe and other serving platforms.

## Open Source and Pre-trained Models

We are excited to announce that Polli Linnaeus is fully open-source and will be hosted at [https://github.com/polli-labs/linnaeus](https://github.com/polli-labs/linnaeus). We are also in the process of releasing a suite of pre-trained `mFormerV1_sm` models on Hugging Face Hub under the user `polli-caleb` (project: `linnaeus`). The first set of models, planned for release in early July, will cover the following North American taxa:

*   North American Aves (Birds) - `mFormerV1_sm`
*   North American Amphibia (Amphibians) - `mFormerV1_sm`
*   North American Reptilia (Reptiles) - `mFormerV1_sm`
*   North American Primary Terrestrial Arthropoda (Insects, Spiders, etc.) - `mFormerV1_sm`
*   North American Angiospermae (Flowering Plants) - `mFormerV1_sm`
*   North American Mammalia (Mammals) - `mFormerV1_sm`

## Installation

```bash
# Via pip from GitHub (recommended)
pip install git+https://github.com/polli-labs/linnaeus.git

# Development installation
git clone https://github.com/polli-labs/linnaeus.git
cd linnaeus
pip install -e .
```

For detailed installation instructions, see [Installation Guide](docs/installation.md).

## Quick Start

```python
import torch
from linnaeus.models import build_model
from linnaeus.config import get_default_config

# Load a configuration
cfg = get_default_config()
cfg.merge_from_file("configs/model/archs/mFormerV1/mFormerV1_sm.yaml")

# Build the model
model = build_model(cfg)

# Run inference
image = torch.randn(1, 3, 224, 224)
metadata = torch.randn(1, cfg.MODEL.META_DIMS[0])
with torch.no_grad():
    predictions = model(image, metadata)
```

For more examples, see [Getting Started](docs/getting_started.md).

The example above shows how to build a model from a configuration. For a guide on running inference with our upcoming pre-trained models, please see our [Inference Tutorial](docs/inference/running_inference_with_pretrained_models.md) (coming soon!) and the [Getting Started Guide](docs/getting_started.md).

## Model Zoo

Learn more about available models in the [Model Zoo](docs/models/model_zoo.md).

| Model | Size | Parameters | GFLOPs | Features |
|-------|------|------------|--------|----------|
| mFormerV0 | Small | 15M | 2.8 | Hybrid Conv-Transformer with RelativeAttention |
| mFormerV0 | Medium | 35M | 4.5 | Hybrid Conv-Transformer with RelativeAttention |
| mFormerV0 | Large | 55M | 7.3 | Hybrid Conv-Transformer with RelativeAttention |
| mFormerV1 | Small | 18M | 3.1 | 2D RoPE, FlashAttention compatible |
| mFormerV1 | Medium | 38M | 5.0 | 2D RoPE, FlashAttention compatible |
| mFormerV1 | Large | 65M | 8.2 | 2D RoPE, FlashAttention compatible |
| mFormerV1 | XLarge | 120M | 15.1 | 2D RoPE, FlashAttention compatible |

## Documentation

Explore our comprehensive documentation to get the most out of Polli Linnaeus. Start with our [Documentation Hub](docs/index.md) or use the links below:

- **[Installation](docs/installation.md)**: Detailed installation steps
- **[Getting Started](docs/getting_started.md)**: Quick introduction to Linnaeus
- **[Training](docs/training/overview.md)**: Guide to training models
- **[Inference](docs/inference/overview.md)**: Running models for predictions
- **[Model System](docs/models/model_system_overview.md)**: Architecture details
- **[Advanced Topics](docs/advanced_topics/index.md)**: In-depth guides

## Batch Size Analysis Workflow

Use `tools/analyze_batch_sizes.py` to estimate the largest train and validation batch sizes for different GPU memory budgets.

```bash
python tools/analyze_batch_sizes.py --cfg my_exp.yaml --fractions 0.5,0.8 --modes train,val
```

Review the results and then enable AutoBatch in your config (or set the batch size manually) before starting training.

## Research Use

If you use Polli Linnaeus in your research, please cite:

```
@software{pollilinnaeus2024,
  author = {Sowers, Caleb},
  title = {Polli Linnaeus: A Deep Learning Framework for Taxonomic Recognition},
  year = {2024},
  publisher = {Polli Labs Inc.},
  url = {https://github.com/polli-labs/linnaeus}
}
```

## Community and Contributions

Polli Linnaeus is an open-source project, and we welcome contributions from the community. Whether it's reporting issues, suggesting new features, or contributing code, please visit our [GitHub repository](https://github.com/polli-labs/linnaeus) to learn more. Check our [Contribution Guidelines](CONTRIBUTING.md) (to be created) for more details.

## License

Apache License 2.0