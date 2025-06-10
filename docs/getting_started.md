# Getting Started with Polli Linnaeus

This guide provides a quick introduction to using Polli Linnaeus for taxonomic image classification.

## Installation

Follow the [installation instructions](installation.md) to set up Linnaeus.

## Basic Usage

### Loading a Pre-configured Model

```python
import torch
from linnaeus.models import build_model
from linnaeus.config import get_default_config

# Load configuration for a small mFormerV1 model
cfg = get_default_config()
cfg.merge_from_file("configs/model/archs/mFormerV1/mFormerV1_sm.yaml")

# Optional: override configuration parameters
cfg.MODEL.META_DIMS = [16]  # Set metadata dimensions

# Build the model
model = build_model(cfg)
```

### Running Inference

The following example demonstrates running inference with the model built in the previous step. It uses a randomly generated tensor as input. For loading and processing real images, you would typically use libraries like Pillow (PIL) and `torchvision.transforms`.
```python
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Prepare input data
# The 'transform' object would typically normalize an image tensor if it came from an image.
# For this example, we define it but note that direct normalization of a random tensor isn't standard.
# Real models expect normalized inputs according to their training.
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create a dummy image tensor (batch_size, channels, height, width)
# Replace this with actual image loading and transformation if you have an image file.
# For example:
# from PIL import Image
# image = Image.open("path/to/your/image.jpg").convert("RGB")
# image_tensor = transform(image).unsqueeze(0)

# For this example, we use a random tensor:
image_tensor = torch.randn(1, 3, 224, 224)
# The 'transform' object would typically normalize this if it came from an image.
# Since it's random, direct normalization isn't standard here,
# but be aware that models expect normalized inputs.

# Generate sample metadata (taxonomy level feature vector)
metadata_tensor = torch.zeros(1, cfg.MODEL.META_DIMS[0])

# Run inference
model.eval()
with torch.no_grad():
    outputs = model(image_tensor, metadata_tensor)

# Process outputs (example for classification task)
if isinstance(outputs, dict):
    # Multi-task output
    for task_name, task_output in outputs.items():
        probs = torch.nn.functional.softmax(task_output, dim=1)
        top_p, top_class = probs.topk(1, dim=1)
        print(f"Task: {task_name}, Class: {top_class.item()}, Confidence: {top_p.item():.4f}")
else:
    # Single task output
    probs = torch.nn.functional.softmax(outputs, dim=1)
    top_p, top_class = probs.topk(1, dim=1)
    print(f"Class: {top_class.item()}, Confidence: {top_p.item():.4f}")
```

> _**Note:** This example shows inference with a model built directly from a configuration. For a detailed guide on using our officially pre-trained models (e.g., from Hugging Face Hub), please see the **[Running Inference with Pre-trained Models tutorial](./inference/running_inference_with_pretrained_models.md)**._

## Training a Model

For training, you'll need to:
1. Prepare your dataset in the appropriate format
2. Configure your training parameters
3. Launch training

### Dataset Preparation

Polli Linnaeus uses an optimized H5 dataset format for efficient loading. For detailed information on this format and how to prepare your data, see [Data Loading for Training](./training/data_loading.md). For a step-by-step guide on converting a sample dataset to HDF5, refer to our tutorial on [Training Your First Polli Linnaeus Model](./training/training_custom_model_example.md#1-preparing-your-dataset).

### Basic Training Launch

```bash
# Basic training command
python -m linnaeus.train \
  --cfg configs/model/archs/mFormerV1/mFormerV1_sm.yaml \
  --opts \
  DATA.TRAIN_DATASET /path/to/your/dataset.h5 \
  DATA.VAL_DATASET /path/to/your/validation.h5 \
  EXPERIMENT.OUTPUT_DIR /path/to/output/directory \
  EXPERIMENT.WANDB.ENABLED True \
  EXPERIMENT.WANDB.PROJECT your_project_name
```

See [Training Overview](training/overview.md) for detailed information.

## Working with Configurations

Polli Linnaeus uses YACS for configuration management:

```python
from linnaeus.config import get_default_config

# Load base config
cfg = get_default_config()

# Load from file
cfg.merge_from_file("path/to/config.yaml")

# Override from command line-style arguments
cfg.merge_from_list(["MODEL.NUM_CLASSES", 100, "TRAIN.BATCH_SIZE", 32])

# Access configuration values
print(cfg.MODEL.NUM_CLASSES)  # 100
print(cfg.TRAIN.BATCH_SIZE)   # 32

# Freeze config to prevent further modifications
cfg.freeze()
```

## Using Hierarchical Classification

Polli Linnaeus specializes in hierarchical classification:

```python
from linnaeus.utils.taxonomy import TaxonomyTree

# Load a taxonomy
taxonomy = TaxonomyTree.from_file("path/to/taxonomy.json")

# Access taxonomy information
print(f"Number of nodes: {len(taxonomy)}")
print(f"Leaf nodes: {taxonomy.get_leaf_nodes()}")
print(f"Root nodes: {taxonomy.get_root_nodes()}")

# Get parent-child relationships
children = taxonomy.get_children("class_name")
parents = taxonomy.get_parents("class_name")

# Create hierarchical loss
from linnaeus.loss import create_hierarchical_loss
h_loss = create_hierarchical_loss(taxonomy, alpha=0.1)
```

For more details, see [Hierarchical Approaches](advanced_topics/hierarchical_approaches.md).

## Next Steps

- [Model System Overview](models/model_system_overview.md): Learn about the model architecture
- [Training Guide](training/overview.md): Detailed training instructions
- [Inference Guide](inference/overview.md): How to deploy models for inference
- [Advanced Topics](advanced_topics/hierarchical_approaches.md): Specialized features