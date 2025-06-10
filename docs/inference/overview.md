# Inference with Polli Linnaeus Models

Polli Linnaeus provides robust mechanisms for performing inference with its trained hierarchical classification models. This includes direct inference using PyTorch for local processing and integration with `LitServe` for deploying models as scalable services.

## Core Component: `LinnaeusInferenceHandler`

The primary interface for inference is the `linnaeus.inference.handler.LinnaeusInferenceHandler`. This class is responsible for:

*   Loading all necessary artifacts (model weights, taxonomy data, class mappings, and configurations). This can be from a self-contained **Inference Bundle** (a local directory) or directly from a **Hugging Face Hub model identifier** (which may download and cache the bundle).
*   Preprocessing input images and optional metadata (such as geolocation, timestamp, and elevation) using `typus` projection utilities.
*   Executing the model to obtain raw predictions.
*   Postprocessing the predictions, which includes:
    *   Mapping model output indices to `typus` `taxon_id`s.
    *   Applying softmax and selecting top-K predictions for each taxonomic rank.
    *   Optionally, enforcing hierarchical consistency to ensure that child predictions align with their parent predictions in the taxonomy.
*   Returning structured results as a list of `typus.models.classification.HierarchicalClassificationResult` objects.

## The Inference Bundle

To ensure portability and ease of use, Linnaeus employs an **Inference Bundle**. This is a directory that packages the model's weights, the `inference_config.yaml` (which orchestrates the loading and behavior of the handler), taxonomic data, and class mapping files. While pre-trained models from Hugging Face Hub are distributed as inference bundles, the `LinnaeusInferenceHandler` aims to abstract away the manual management of these bundles when loading directly via a model identifier.

For detailed information on the structure of the inference bundle, its components, and how to create one, please see the [Inference Bundle Documentation](./inference_bundle.md).

## Running Inference

Inference can be performed in several ways, primarily differing in how the model assets are obtained and managed:

1.  **Using a Local Inference Bundle**: Instantiate `LinnaeusInferenceHandler` by pointing it to the `inference_config.yaml` within your bundle. This is suitable for batch processing, integration into larger Python applications, or debugging.
    ```python
    from pathlib import Path
    from PIL import Image
    from linnaeus.inference.handler import LinnaeusInferenceHandler

    # Path to the configuration file within your inference bundle
    bundle_config_path = Path("path/to/your_model_bundle/inference_config.yaml")

    # Load the handler
    handler = LinnaeusInferenceHandler.load_from_artifacts(config_file_path=bundle_config_path)

    # Load an image
    image = Image.open("path/to/your_image.jpg")

    # Make predictions
    # Metadata can optionally be provided as a list of dictionaries
    results = handler.predict(images=[image], metadata_list=None)

    # Print the top result (example)
    if results:
        print(results[0].model_dump_json(indent=2))
    ```

2.  **Using a Hugging Face Model Identifier**:
    For models available on Hugging Face Hub, you can often load the `LinnaeusInferenceHandler` by providing the model's Hugging Face identifier. The handler may manage the download and caching of the necessary bundle components. This is the recommended approach for using official pre-trained Polli Linnaeus models.
    ```python
    from PIL import Image
    from linnaeus.inference.handler import LinnaeusInferenceHandler
    # Assuming LinnaeusInferenceHandler or a helper can resolve HF IDs
    # This is a simplified conceptual example.
    # Actual implementation might involve LinnaeusInferenceHandler.load_from_hf_hub("polli-caleb/...")
    # or ensuring the HF model is cached locally such that load_from_artifacts can find it.
    # Please refer to the [Running Inference with Pre-trained Models](./running_inference_with_pretrained_models.md) tutorial for detailed examples.

    MODEL_ID = "polli-caleb/linnaeus-aves-mformerV1_sm-v1" # Example HF Model ID
    # handler = LinnaeusInferenceHandler.load_from_artifacts(hf_model_id=MODEL_ID) # Or similar API

    # --- Placeholder: Illustrative ---
    # The exact API for direct HF load needs to be shown as per its implementation.
    # For now, we point to the tutorial which has a more detailed, albeit temporary, approach.
    print(f"For direct Hugging Face model loading, please see the tutorial: running_inference_with_pretrained_models.md")
    # image = Image.open("path/to/your_image.jpg")
    # results = handler.predict(images=[image])
    # if results:
    #     print(results[0].model_dump_json(indent=2))
    ```

3.  **As a Service with `LitServe`**: The `LinnaeusInferenceHandler` can be easily integrated into a `LitServe` API, allowing you to serve your model over a network. See the [LitServe Integration Guide](./litserve.md) for more details.

## Automated Testing

The `linnaeus.inference` module is equipped with a suite of automated tests to ensure its correctness and robustness. These tests cover:

*   **Unit Tests (`tests/test_inference_components.py`)**: For individual functions responsible for metadata preprocessing (`preprocess_metadata_batch`) and hierarchical consistency enforcement (`enforce_hierarchical_consistency`).
*   **Integration Tests (`tests/test_inference_handler.py`)**: For the end-to-end functionality of `LinnaeusInferenceHandler`. These tests use a mock inference bundle to verify:
    *   Successful loading of the handler from artifacts.
    *   Correctness of the `handler.info()` method.
    *   Proper output types and structure from `handler.predict()`.
    *   Behavior of `handler.predict()` with and without metadata, and for batch inputs.

These tests are crucial for maintaining code quality and preventing regressions as the inference system evolves. They can be run using `pytest` from the project root.

## Further Reading

*   **[Running Inference with Pre-trained Models](./running_inference_with_pretrained_models.md)**: A step-by-step guide for using models from Hugging Face Hub.
*   [Inference Bundle Details](./inference_bundle.md)
*   [Serving with LitServe](./litserve.md)
