# Training Your First Polli Linnaeus Model

This guide is for ML researchers and developers who want to train their own image classification models using the Polli Linnaeus framework. We'll walk through understanding dataset requirements, setting up an experiment configuration, and launching a training run.

## Prerequisites

1.  **Installation:** Ensure Polli Linnaeus is installed. See the [Installation Guide](../installation.md).
2.  **Familiarity:** Basic understanding of PyTorch and YAML configuration files will be helpful.
3.  **Computational Resources:** Access to a machine with an NVIDIA GPU is highly recommended for training.

## Overview of the Training Process

1.  **Prepare Your Dataset:**
    *   Organize your images into a directory.
    *   Create a `labels.h5` file that conforms to the Polli Linnaeus schema, containing image identifiers, taxonomic labels, and any metadata.
2.  **Define a Taxonomy (if applicable):** If you're doing hierarchical classification, prepare a `taxonomy.json` file defining your class relationships.
3.  **Configure Your Experiment:** Create or adapt a YAML configuration file (`.yaml`) to define the model architecture, dataset paths, metadata usage, training parameters, etc.
4.  **Launch Training:** Use the `linnaeus.train` module to start the training process.
5.  **Monitor and Evaluate:** Track training progress (e.g., with Weights & Biases) and evaluate your trained model.

## 1. Preparing Your Dataset (`labels.h5` and Images)

Polli Linnaeus primarily uses a **hybrid data model**:
*   **Images:** Stored as individual files (e.g., `.jpg`, `.png`) in a directory.
*   **Labels and Metadata:** Stored in a single HDF5 file (commonly named `labels.h5`).

This approach is flexible and efficient for large datasets.

**The `labels.h5` File:**
This is the most critical part of your custom dataset. It links your image files to their labels and any associated metadata.

**For the definitive and detailed HDF5 schema, including mandatory and optional datasets, their `dtypes`, `shapes`, and the `metadata` group structure, YOU MUST REFER TO: [Data Loading for Training](./data_loading.md).**

Briefly, your `labels.h5` will need:
*   `img_identifiers`: A dataset of strings matching your image filenames.
*   `taxa_LXX` datasets: Datasets like `taxa_L10` (species), `taxa_L20` (genus), etc., containing integer taxon IDs. The specific `taxa_LXX` datasets you need are determined by `DATA.TASK_KEYS_H5` in your experiment configuration.
*   **Optional metadata datasets:** Such as `spatial` (for geolocation), `temporal` (for time), `elevation_broadrange_2` (for elevation features), etc. These are only used if you enable and configure them in the `DATA.META.COMPONENTS` section of your experiment config.

**Image Directory:**
Simply a directory containing all your image files. The filenames should match the entries in the `img_identifiers` dataset within your `labels.h5`.

**Example:**
If `labels.h5` has an `img_identifiers` entry `"image_001.jpg"`, you should have an image file named `image_001.jpg` in your image directory.

**Train/Validation Split:**
Polli Linnaeus supports dynamic train/validation splitting from a single `labels.h5` file and image directory. This is configured via `DATA.H5.TRAIN_VAL_SPLIT_RATIO` and `DATA.H5.TRAIN_VAL_SPLIT_SEED` in your experiment YAML, and is the recommended approach for flexibility.

## 2. Define a Taxonomy (If Applicable)

If your model involves hierarchical classification (predicting multiple taxonomic ranks like genus and species), you'll need a `taxonomy.json` file. This file defines the parent-child relationships between your classes.

Example `taxonomy.json` structure:
```json
{
  "nodes": [
    {"id": "unique_taxon_id_1", "rank": "species", "name": "Species Name A"},
    {"id": "unique_taxon_id_2", "rank": "genus", "name": "Genus Name B"},
    // ... more nodes
  ],
  "edges": [
    {"source": "unique_taxon_id_2", "target": "unique_taxon_id_1"}, // Genus B is parent of Species A
    // ... more edges
  ]
}
```
Polli Linnaeus uses `polli_typus.TaxonomyTree` (via the `polli-typus` package) to load and work with this data. The `id` fields should correspond to the integer taxon IDs used in your `taxa_LXX` datasets in `labels.h5`.

## 3. Configure Your Experiment (`my_experiment.yaml`)

Training runs are controlled by YAML configuration files. Start by adapting an existing configuration from the `configs/` directory (e.g., `configs/experiments/generic_mformer_example.yaml` or one of the model-specific configs like those for `mFormerV1_sm`).

Below is an example snippet for a custom experiment (`my_custom_experiment.yaml`), highlighting key sections to customize. **This example assumes you are using the preferred hybrid data mode and dynamic train/val splitting.**

```yaml
# my_custom_experiment.yaml
BASE: ['configs/model/archs/mFormerV1/mFormerV1_sm.yaml'] # Inherit base model settings

EXPERIMENT:
  PROJECT: 'MyLinnaeusProject'
  GROUP: 'CustomModelTraining'
  NAME: 'MyFirstCustomRun_mFormerV1_sm'
  TAGS: ['custom_dataset', 'mFormerV1_sm', 'tutorial_example']
  WANDB:
    ENABLED: True
    PROJECT: 'MyLinnaeusExperiments_ProjectName' # Your WandB project
    ENTITY: 'your_wandb_username_or_team'     # Your WandB entity

MODEL:
  # TYPE and NAME might be inherited from BASE, or override here
  # Example: Ensure IMG_SIZE matches your image preprocessing
  IMG_SIZE: 384 # Ensure your images are processed to this size
  CLASSIFICATION:
    HEADS: # Define output heads for each taxonomic task you want to predict
      taxa_L10: # Corresponds to 'taxa_L10' dataset in labels.h5 & TASK_KEYS_H5
        TYPE: 'LinearHead' # Or 'ConditionalClassifier', 'HierarchicalSoftmaxHead'
        NUM_CLASSES: 150   # Number of unique species classes in your taxa_L10 dataset
        LOSS:
          TYPE: "CrossEntropyLoss" # Or "TaxonomyAwareLabelSmoothing"
          PARAMS: { label_smoothing: 0.1 }
      taxa_L20: # Corresponds to 'taxa_L20' dataset in labels.h5 & TASK_KEYS_H5
        TYPE: 'LinearHead'
        NUM_CLASSES: 50    # Number of unique genus classes in your taxa_L20 dataset
        LOSS:
          TYPE: "CrossEntropyLoss"
          PARAMS: { label_smoothing: 0.1 }
  # META_DIMS will be automatically calculated if DATA.META.ACTIVE is True

DATA:
  USE_VECTORIZED_PROCESSOR: True # Recommended
  PIN_MEMORY: True
  BATCH_SIZE: 64
  BATCH_SIZE_VAL: 512 # Typically larger for validation
  NUM_WORKERS: 4 # Adjust based on your CPU cores and I/O
  IMG_SIZE: 384 # Must match MODEL.IMG_SIZE

  # Define which taxonomic label datasets from labels.h5 to use
  TASK_KEYS_H5: ['taxa_L10', 'taxa_L20'] # Model will train on species and genus

  # Configure paths for hybrid mode (labels.h5 + image directory)
  HYBRID:
    USE_HYBRID: True
    IMAGES_DIR: '/path/to/your/custom_image_directory/' # All images here
    FILE_EXTENSION: '.jpg' # Or '.png', etc. if not in img_identifiers

  H5: # Settings for the HDF5 labels file
    LABELS_PATH: '/path/to/your/custom_labels.h5' # Single labels file
    TRAIN_VAL_SPLIT_RATIO: 0.90 # e.g., 90% for training, 10% for validation
    TRAIN_VAL_SPLIT_SEED: 42   # For reproducible splits

  # (Optional) Configure usage of metadata from labels.h5
  META:
    ACTIVE: True # Set to False if not using any metadata
    COMPONENTS:
      # Example: Using a 'temporal' dataset from labels.h5 for temporal features
      TEMPORAL: {ENABLED: True, SOURCE: "temporal", COLUMNS: ["month_sin", "month_cos"], DIM: 2, IDX: 0, ALLOW_MISSING: True, OOR_MASK: False}
      # Example: Using a 'spatial' dataset from labels.h5 for spatial features
      SPATIAL: {ENABLED: True, SOURCE: "spatial", COLUMNS: [], DIM: 3, IDX: 1, ALLOW_MISSING: True, OOR_MASK: False}
      # Example: Using 'elevation_broadrange_2' dataset from labels.h5 for elevation
      ELEVATION: {ENABLED: True, SOURCE: "elevation_broadrange_2", COLUMNS: [], DIM: 10, IDX: 2, ALLOW_MISSING: True, OOR_MASK: False}
      # Ensure the SOURCE dataset names and expected DIM match your labels.h5!

  # (Optional) Path to your taxonomy.json if doing hierarchical classification
  TAXONOMY_PATH: "/path/to/your/taxonomy.json" # Set to null or remove if not hierarchical

  # NUM_CLASSES_PER_LEVEL is usually inferred automatically if using VectorizedDatasetProcessorOnePass
  # and your HDF5 metadata is set up correctly.
  # If needed, you can specify it manually:
  # NUM_CLASSES_PER_LEVEL:
  #   taxa_L10: 150
  #   taxa_L20: 50

TRAIN:
  EPOCHS: 50
  AMP_OPT_LEVEL: 'O1' # For mixed-precision training
  # ... other training parameters like optimizer, scheduler, augmentations ...

# For a full list of configuration options, refer to the default config
# in linnaeus/config/defaults.py and explore example configs in configs/
```

**Key Configuration Points for Custom Datasets:**

*   **`DATA.HYBRID.IMAGES_DIR`**: Path to your image directory.
*   **`DATA.H5.LABELS_PATH`**: Path to your `labels.h5` file.
*   **`DATA.TASK_KEYS_H5`**: List of HDF5 dataset names for taxonomic labels (e.g., `['taxa_L10', 'taxa_L20']`). These directly map to the `taxa_LXX` datasets in your `labels.h5`.
*   **`MODEL.CLASSIFICATION.HEADS`**: Define a head for each task key in `DATA.TASK_KEYS_H5`. Ensure `NUM_CLASSES` for each head matches the number of unique classes in the corresponding `taxa_LXX` dataset.
*   **`DATA.META.COMPONENTS`**: If using metadata, configure each component (e.g., `TEMPORAL`, `SPATIAL`, `ELEVATION`).
    *   `ENABLED: True` activates the component.
    *   `SOURCE`: Specifies the HDF5 dataset name within `labels.h5` (e.g., `"temporal"`, `"elevation_broadrange_2"`).
    *   `DIM`: Expected dimension of the metadata feature vector from this source.
    *   `COLUMNS`: Optionally specify which columns from the source dataset to use (if it's multi-dimensional).
*   **`DATA.TAXONOMY_PATH`**: Path to your `taxonomy.json` if your model or loss functions are hierarchy-aware.

## 4. Launch Training

Once your dataset is ready (image directory and `labels.h5` conform to schema) and your configuration file is set up, start training:

```bash
python -m linnaeus.train --cfg /path/to/your/my_custom_experiment.yaml
```

Override configuration parameters directly from the command line using `--opts`:
```bash
python -m linnaeus.train --cfg /path/to/your/my_custom_experiment.yaml   --opts TRAIN.BATCH_SIZE 32 EXPERIMENT.WANDB.ENABLED False

```

## 5. Monitor and Evaluate

*   **Console Output:** Training progress, loss values, and metrics will be printed.
*   **Weights & Biases (WandB):** If `EXPERIMENT.WANDB.ENABLED: True`, detailed logs, metrics, and visualizations will be available on your WandB dashboard.
*   **Checkpoints:** Model checkpoints are saved in `EXPERIMENT.OUTPUT_DIR`.
*   **Evaluation:**
    *   Validation runs periodically during training.
    *   After training, use saved checkpoints for inference on a test set. Adapt the script from the [Inference Tutorial](../inference/running_inference_with_pretrained_models.md), pointing it to your custom model's checkpoint and inference configuration.

## Next Steps

*   **Master the HDF5 Schema:** Thoroughly review **[Data Loading for Training](./data_loading.md)** to ensure your `labels.h5` is perfectly structured.
*   **Explore Example Configurations:** Study the various `.yaml` files in the `configs/` directory of the Polli Linnaeus repository to understand advanced setups for different models, augmentations, optimizers, and schedulers.
*   **Custom Model Architectures:** To define new model architectures, see `docs/models/model_system_overview.md`.
*   **Official Dataset Generation:** To understand how the official Polli Linnaeus pre-trained models were created, see `docs/datasets/dataset_generation.md` (Coming Soon).

Training deep learning models is an iterative process. Start with a small subset of your data if possible, verify your pipeline, and then scale up. Good luck!
