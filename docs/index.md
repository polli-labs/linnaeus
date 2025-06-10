# Polli Linnaeus Documentation Hub

Welcome to the official documentation for Polli Linnaeus, an open-source deep learning framework designed for taxonomic image classification and biodiversity monitoring applications. This hub provides a central point for accessing all documentation resources, whether you're looking to quickly use a pre-trained model, train your own custom models, or dive deep into the framework's architecture.

## Getting Started

New to Polli Linnaeus? These guides will help you get up and running.

*   **[Project README](../../README.md):** Start here for a general overview of the project, its goals, and quick installation/usage notes.
*   **[Installation Guide](./installation.md):** Detailed instructions for installing Polli Linnaeus and its dependencies.
*   **[Getting Started Tutorial](./getting_started.md):** A beginner-friendly introduction to the basic functionalities of Linnaeus.

## Using Pre-trained Models

Leverage our suite of pre-trained models for your classification tasks.

*   **[Model Zoo](./models/model_zoo.md):** Discover available pre-trained models, their taxonomic scope, and links to Hugging Face Hub.
*   **[Running Inference with Pre-trained Models](./inference/running_inference_with_pretrained_models.md):** A step-by-step tutorial on how to use our pre-trained models for inference on your own images.
*   **[Inference Overview](./inference/overview.md):** Learn about the core components of the Linnaeus inference pipeline, including the `LinnaeusInferenceHandler`.

## Training Models

Guides for researchers and developers looking to train or fine-tune models.

*   **[Training Your First Polli Linnaeus Model](./training/training_custom_model_example.md):** A comprehensive tutorial on preparing your custom dataset (HDF5 format), configuring experiments, and launching training runs.
*   **[Training Overview](./training/overview.md):** An overview of the training system, including data loading, augmentations, multi-task learning, and more.
*   **[Data Loading for Training](./training/data_loading.md):** Detailed information on how Linnaeus handles data, with a focus on the HDF5 format.
*   **[Phase 2: Abstention Training with RL (Experimental)](./training/phase2_abstention_rl.md):** Learn how to fine-tune pre-trained models to learn abstention behavior using Reinforcement Learning.

## Datasets

Understanding the data used to train and evaluate models.

*   **[Official Dataset Provenance (ibrida-v0-r1)](./datasets/dataset_generation.md):** Detailed information on the provenance and filtering logic used to create the initial batch of pre-trained models from the iNaturalist Open Data. This is crucial for understanding model scope and limitations.

## Advanced Topics

For users who want to explore more specialized features of Linnaeus.

*   **[Advanced Topics Index](./advanced_topics/index.md):** Links to guides on:
    *   Automatic Batch Sizing
    *   Development Features & Debug Flags
    *   Hierarchical Approaches (Taxonomy-Guided Label Smoothing, Hierarchical Heads)
    *   Taxonomy Representation
    *   Training Progress Tracking

## Developer Documentation

Resources for those looking to extend or contribute to Polli Linnaeus.

*   **[Model System Overview](./models/model_system_overview.md):** Understand the architectural design, component registries, and how to create new model architectures.
*   **[Development Guides](./dev/):** (Link to `docs/dev/` directory if it contains an index or relevant files - e.g., `docs/dev/01_training_loop_and_progress.md`) Explore notes on the training loop, scheduling system, metrics, and other internal design choices.
*   **[Contributing Guidelines](../../CONTRIBUTING.md):** (To be created) How to contribute to the Polli Linnaeus project.

---

If you can't find what you're looking for, please consider [opening an issue](https://github.com/polli-labs/linnaeus/issues) on our GitHub repository.
