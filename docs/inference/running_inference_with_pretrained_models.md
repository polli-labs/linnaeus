# Running Inference with Pre-trained Polli Linnaeus Models

This guide will walk you through the process of using a pre-trained Polli Linnaeus model to classify an image. We'll focus on using models from the Hugging Face Hub. This guide is designed for users like ecological researchers who want to apply these models to their data with minimal setup.

## Prerequisites

1.  **Install Polli Linnaeus:** If you haven't already, please follow the [Installation Guide](../installation.md). This will ensure you have Polli Linnaeus and its core dependencies, including `polli-typus` and `huggingface-hub`.

2.  **Sample Image:** For this tutorial, you'll need an image to classify.
    *   You can use your own image of a plant, animal, or insect that you believe might be covered by one of our [North American models](../models/model_zoo.md).
    *   Alternatively, for testing purposes, you can save a publicly available image of a common North American species. For example, search for an image of a "Monarch Butterfly" or a "Bald Eagle". Ensure you have the rights to use any image you download. Let's assume you have an image saved as `my_sample_image.jpg`.

## Steps to Run Inference

### 1. Choose a Model

Visit our [Model Zoo](../models/model_zoo.md) to see the list of available pre-trained models. For this example, let's assume you want to use a model for classifying birds. The Model Zoo will provide a Hugging Face model identifier (e.g., `polli-caleb/linnaeus-aves-mformerV1_sm-v1`).

### 2. Create an Inference Script

Create a Python script (e.g., `run_linnaeus_inference.py`) with the following content:

```python
from pathlib import Path
from PIL import Image
from linnaeus.inference.handler import LinnaeusInferenceHandler
from linnaeus.config import get_default_config

# --- Configuration ---
# Replace with the actual path to your inference bundle's config OR a Hugging Face model identifier
# For Hugging Face models (once released and handler supports direct HF loading):
MODEL_IDENTIFIER = "polli-caleb/linnaeus-aves-mformerV1_sm-v1" # Example, replace with actual model ID
# If using a downloaded Inference Bundle:
# ARTIFACT_DIR = Path("path/to/your_model_bundle/")
# CONFIG_FILE_PATH = ARTIFACT_DIR / "inference_config.yaml"

IMAGE_PATH = Path("my_sample_image.jpg") # Replace with the path to your image

def main():
    print(f"Loading model... This might take a moment.")

    # Load the handler
    # Option 1: From Hugging Face (Preferred for pre-trained models)
    # Note: Direct HuggingFace ID loading in LinnaeusInferenceHandler is a new feature.
    # Ensure your Linnaeus version supports this.
    # If the model needs to be downloaded first, this might happen automatically
    # or you might need to use tools from huggingface_hub library explicitly if direct loading isn't implemented.
    # For now, we assume direct loading or user has downloaded the bundle via other Hugging Face tools.

    # Create a minimal configuration for the handler if not using a full bundle config
    cfg = get_default_config()
    # Potentially override cfg settings if needed, e.g. device
    # cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # This part assumes LinnaeusInferenceHandler.load_from_artifacts can take an HF string
    # or that the user has downloaded the bundle and CONFIG_FILE_PATH is set.
    # The exact mechanism for HF loading might need refinement based on LinnaeusInferenceHandler's API.
    # For now, let's assume a common pattern where HF models are downloaded to a cache
    # and can be loaded by pointing to the cached path or directly by identifier.

    # Placeholder for actual loading logic - this needs to align with LinnaeusInferenceHandler capabilities
    # For a local bundle:
    # handler = LinnaeusInferenceHandler.load_from_artifacts(config_file_path=CONFIG_FILE_PATH)
    # For HF (conceptual):
    # One way this MIGHT work (consult LinnaeusInferenceHandler docs/examples):
    try:
        # Attempt to load directly if handler supports HF model ID
        # This is a conceptual representation. The actual API might differ.
        # It might involve downloading the bundle first using huggingface_hub
        # and then loading from the downloaded bundle path.
        print(f"Attempting to load model: {MODEL_IDENTIFIER}")
        # For this example, we will proceed as if the user has a local bundle,
        # as direct HF string loading in load_from_artifacts needs verification.
        print("Please ensure you have downloaded the model bundle from Hugging Face and unpacked it.")
        print("Then, update ARTIFACT_DIR and CONFIG_FILE_PATH in this script.")

        # --- TEMPORARY: User needs to set up a local bundle path ---
        # This section should be updated once direct HF loading is confirmed and documented.
        use_local_bundle_example = True
        if use_local_bundle_example:
            # User needs to replace these paths:
            example_bundle_path = Path("path/to/downloaded_hf_bundle/linnaeus-aves-mformerV1_sm-v1")
            if not example_bundle_path.exists():
                print(f"Error: Example bundle path {example_bundle_path} does not exist.")
                print("Please download the model from Hugging Face, unpack it, and update the path.")
                return
            cfg_file = example_bundle_path / "inference_config.yaml"
            if not cfg_file.exists():
                print(f"Error: Inference config not found at {cfg_file}")
                return
            handler = LinnaeusInferenceHandler.load_from_artifacts(config_file_path=cfg_file)
        else:
            # Conceptual direct HF loading (needs API confirmation)
            # handler = LinnaeusInferenceHandler.load_from_hf_hub(MODEL_IDENTIFIER, cfg_updates={})
            print("Direct Hugging Face loading not yet fully exemplified here. See docs for LinnaeusInferenceHandler.")
            return
        # --- END TEMPORARY ---

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have specified the correct model identifier or bundle path,")
        print("and that the model files are accessible.")
        return

    print(f"Model '{handler.model_name}' loaded successfully.")
    print(f"Description: {handler.model_description}")
    print(f"Taxonomic Scope: {handler.model_scope_description}")

    # Load the image
    if not IMAGE_PATH.exists():
        print(f"Error: Image not found at {IMAGE_PATH}")
        return

    print(f"Loading image: {IMAGE_PATH}")
    try:
        image = Image.open(IMAGE_PATH).convert("RGB")
    except Exception as e:
        print(f"Error opening or converting image: {e}")
        return

    # Make predictions
    # Metadata can optionally be provided as a list of dictionaries
    # For example: metadata_list = [{"latitude": 34.5, "longitude": -118.2, "date": "2024-05-29"}]
    # The specific metadata fields depend on the model's training.
    # For the initial mFormerV1_sm models, no specific metadata is required by default.
    metadata_list = None

    print("Running inference...")
    try:
        results = handler.predict(images=[image], metadata_list=metadata_list)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # Process and print results
    if results:
        print(f"
Prediction Results for {IMAGE_PATH.name}:")
        # result is a typus.models.classification.HierarchicalClassificationResult object
        result = results[0]

        # Print all top predictions (across ranks where available)
        if result.predictions:
            for pred_item in result.predictions:
                # pred_item is a typus.models.classification.HierarchicalPredictionItem
                print(f"  Rank: {pred_item.rank if pred_item.rank else 'Default'}")
                print(f"    Taxon ID: {pred_item.taxon_id}")
                print(f"    Scientific Name: {pred_item.scientific_name}")
                print(f"    Confidence: {pred_item.confidence:.4f}")
                if pred_item.common_name:
                    print(f"    Common Name: {pred_item.common_name}")
        else:
            print("No predictions returned.")

        # Example: Get the top prediction at a specific rank (e.g., species)
        species_prediction = result.get_top_prediction_for_rank("species")
        if species_prediction:
            print("
  Top Species Prediction:")
            print(f"    Taxon ID: {species_prediction.taxon_id}")
            print(f"    Scientific Name: {species_prediction.scientific_name}")
            print(f"    Confidence: {species_prediction.confidence:.4f}")
    else:
        print("No results returned from prediction.")

if __name__ == "__main__":
    main()

```

### 3. Understanding the Script

*   **`MODEL_IDENTIFIER` / `CONFIG_FILE_PATH`**: You'll need to set this to point to your chosen model. The script currently has a placeholder for direct Hugging Face loading and an example for loading a downloaded "Inference Bundle". *The ability to load directly from a Hugging Face identifier in `LinnaeusInferenceHandler` is a newer feature; the example temporarily defaults to using a local bundle path. Please refer to the primary Linnaeus documentation for the most up-to-date way to load HF models.*
*   **`IMAGE_PATH`**: Set this to the path of your image.
*   **`LinnaeusInferenceHandler.load_from_artifacts`**: This is the core class for loading the model and its associated assets (taxonomy, configurations).
*   **`handler.predict(images=[image], metadata_list=metadata_list)`**: This method takes a list of PIL Images and an optional list of metadata dictionaries.
*   **`HierarchicalClassificationResult`**: The output from `predict` is a list of these objects (one for each input image). This object (from the `polli-typus` library) contains detailed, multi-rank predictions.
    *   `result.predictions`: A list of `HierarchicalPredictionItem` objects, each representing the top prediction for a given taxonomic rank (e.g., kingdom, phylum, class, order, family, genus, species).
    *   `result.get_top_prediction_for_rank("species")`: A helper method to get the top prediction at a specific rank.

### 4. Running the Script

```bash
python run_linnaeus_inference.py
```

You should see output detailing the loaded model and then the classification results for your image, including scientific names, common names (if available), and confidence scores at various taxonomic ranks.

## Supplying Metadata (Optional)

Some Polli Linnaeus models can leverage metadata (like geolocation, date/time of observation) to improve predictions. The initial `mFormerV1_sm` models for North America generally do not require specific metadata by default, but future models might.

If a model uses metadata, you would pass it to the `predict` method like this:

```python
metadata_for_image1 = {
    "latitude": 34.0522,  # Decimal degrees
    "longitude": -118.2437, # Decimal degrees
    "date": "2024-07-15T10:30:00Z",  # ISO 8601 format
    # ... any other metadata fields the model expects
}
results = handler.predict(images=[image1], metadata_list=[metadata_for_image1])
```
The `LinnaeusInferenceHandler` will preprocess this metadata according to the model's configuration. Refer to the specific model's documentation in the [Model Zoo](../models/model_zoo.md) for details on expected metadata fields.

## Next Steps

*   **Explore Other Models:** Try different pre-trained models from our Model Zoo.
*   **Batch Inference:** The `predict` method can accept multiple images: `handler.predict(images=[image1, image2, image3])`.
*   **Advanced Usage:** For details on training your own models or more advanced topics, please see the main [Documentation Hub](../index.md).

If you encounter any issues or have questions, please feel free to [open an issue](https://github.com/polli-labs/linnaeus/issues) on our GitHub repository.
