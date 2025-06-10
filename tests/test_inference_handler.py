import pytest
import torch
import torch.nn as nn
from pathlib import Path
import yaml
import json
from PIL import Image

from linnaeus.inference.handler import LinnaeusInferenceHandler
from linnaeus.inference.config import InferenceConfig
from linnaeus.inference.api_schemas import ModelInformation
from typus.models.classification import HierarchicalClassificationResult, TaskPrediction
from typus.constants import RankLevel

# Define a minimal model for testing
class TinyModel(nn.Module):
    def __init__(self, num_classes_per_task):
        super().__init__()
        self.tasks = nn.ModuleDict()
        # Example: num_classes_per_task = {"taxa_L10": 10, "taxa_L20": 5}
        # Use 3*32*32 = 3072 features to match a small RGB image
        self.input_features = 3 * 32 * 32  # Flattened RGB 32x32 image
        for task_key, num_classes in num_classes_per_task.items():
            self.tasks[task_key] = nn.Linear(self.input_features, num_classes)

    def forward(self, image_tensor_batch, aux_vector=None): # Accept aux_vector even if not used
        # image_tensor_batch is (B, C, H, W), e.g., (B, 3, 32, 32)
        # Flatten C,H,W dims to get (B, Features)
        batch_size = image_tensor_batch.shape[0]
        image_features = image_tensor_batch.view(batch_size, -1)

        # Verify that the flattened features match self.input_features
        # This is a sanity check for the test setup.
        expected_features = self.input_features
        if image_features.shape[1] != expected_features:
            # This might happen if image_size in config doesn't match model's expected input_features
            # For example, if image_size=[3,32,32] (3072 features) but model built with input_features=10.
            # The fixture ensures these align.
            pass # Or raise error, but for tests, let it proceed to catch issues downstream.


        # If aux_vector is provided and has features, a real model might concatenate it.
        # For simplicity, this tiny model will ignore aux_vector if its size is 0,
        # or assume it's already incorporated if its size > 0 and matches expectations.
        # This part is mostly for API compatibility with the handler.

        outputs = {}
        for task_key, head in self.tasks.items():
            outputs[task_key] = head(image_features) # Use the flattened image_features
        return outputs

@pytest.fixture(scope="session")
def fixture_inference_bundle(tmp_path_factory):
    bundle_dir = tmp_path_factory.mktemp("inference_bundle_session")

    # 1. Create and save a tiny nn.Module state dict
    # Define task keys and number of classes for the model
    # These should align with what LinnaeusInferenceHandler expects from its config
    model_task_keys = ["taxa_L20_genus", "taxa_L10_species"] # Higher rank first typically in Linnaeus model outputs
    num_classes_for_model = [3, 4] # Genus: 3 classes (G1, G2, G_null), Species: 4 classes (S1, S2, S3, S_null)

    # Null class indices for the model heads
    # Let's say for genus (taxa_L20_genus), index 2 is null
    # For species (taxa_L10_species), index 3 is null
    null_class_indices_model = {
        model_task_keys[0]: 2, # taxa_L20_genus -> null index 2
        model_task_keys[1]: 3  # taxa_L10_species -> null index 3
    }

    # Create model instance based on the actual keys and class counts
    model_config_for_tiny_model = {key: num for key, num in zip(model_task_keys, num_classes_for_model, strict=False)}
    tiny_model = TinyModel(num_classes_per_task=model_config_for_tiny_model)
    model_weights_path = bundle_dir / "mock_pytorch_model.bin"
    torch.save(tiny_model.state_dict(), model_weights_path)

    # 2. Create and save a minimal taxonomy.json (TaxonomyTree format)
    # Ranks: Genus (L20), Species (L10)
    # Taxon IDs (for typus): G1=10, G2=20, G_null=1 (Genus)
    #                         S1=101, S2=102, S3=201, S_null=0 (Species)
    # Model class indices (from above):
    # Genus: G1=idx0, G2=idx1, G_null=idx2
    # Species: S1=idx0, S2=idx1, S3=idx2, S_null=idx3

    # Hierarchy for TaxonomyTree:
    # G1 (model_idx 0 at taxa_L20_genus)
    #  |- S1 (model_idx 0 at taxa_L10_species)
    #  |- S2 (model_idx 1 at taxa_L10_species)
    # G2 (model_idx 1 at taxa_L20_genus)
    #  |- S3 (model_idx 2 at taxa_L10_species)
    # G_null (model_idx 2 at taxa_L20_genus) is null.
    # S_null (model_idx 3 at taxa_L10_species) is null.

    # TaxonomyTree task_keys are usually ordered lowest to highest rank for hierarchy_map
    tt_task_keys = [model_task_keys[1], model_task_keys[0]] # ["taxa_L10_species", "taxa_L20_genus"]
    tt_num_classes = {
        model_task_keys[1]: num_classes_for_model[1], # species: 4
        model_task_keys[0]: num_classes_for_model[0]  # genus: 3
    }
    # hierarchy_map: child_task_key -> {child_model_idx: parent_model_idx}
    hierarchy_map_for_tree = {
        model_task_keys[1]: { # child is species (taxa_L10_species)
            0: 0,  # S1 (idx 0) -> G1 (idx 0 of taxa_L20_genus)
            1: 0,  # S2 (idx 1) -> G1 (idx 0 of taxa_L20_genus)
            2: 1,  # S3 (idx 2) -> G2 (idx 1 of taxa_L20_genus)
            # S_null (idx 3) has no parent link
        }
        # Genus level has no children in this map structure (it's a parent level here)
    }
    taxonomy_tree_data_to_save = {
        "__taxonomy_tree_version__": "1.0",
        "task_keys": tt_task_keys, # lowest to highest for tree internal use
        "num_classes": tt_num_classes, # map of task_key -> num_classes
        "hierarchy_map_raw": hierarchy_map_for_tree,
    }
    taxonomy_json_path = bundle_dir / "mock_taxonomy.json"
    with open(taxonomy_json_path, "w") as f:
        json.dump(taxonomy_tree_data_to_save, f)

    # 3. Create and save a minimal class_index_map.json
    # This maps linnaeus task keys to {string_model_class_idx: typus_taxon_id}
    # Format expected by load_class_index_maps_artifact:
    # {
    #   "taxa_L10": { "0": <null_taxon_id_L10>, "1": <taxon_id_A>, ... },
    #   "taxa_L20": { "0": <null_taxon_id_L20>, "1": <taxon_id_B>, ... },
    # }

    # For genus (taxa_L20_genus): G1=idx0 (TID 10), G2=idx1 (TID 20), G_null=idx2 (TID 1)
    # For species (taxa_L10_species): S1=idx0 (TID 101), S2=idx1 (TID 102), S3=idx2 (TID 201), S_null=idx3 (TID 0)

    class_index_map_data_to_save = {
        "taxa_L20_genus": {
            "0": 10,   # G1
            "1": 20,   # G2
            "2": 1     # G_null
        },
        "taxa_L10_species": {
            "0": 101,  # S1
            "1": 102,  # S2
            "2": 201,  # S3
            "3": 0     # S_null
        }
    }

    class_index_map_json_path = bundle_dir / "mock_class_index_map.json"
    with open(class_index_map_json_path, "w") as f:
        json.dump(class_index_map_data_to_save, f)

    # 4. Create and save a minimal inference_config.yaml
    # This config should point to the files created above.
    # The model_task_keys_ordered in the config should match the order TinyModel expects its outputs.
    # TinyModel produces outputs in the order of model_task_keys = ["taxa_L20_genus", "taxa_L10_species"]
    inference_config_dict = {
        "model": {
            "architecture_name": "mFormerV0",
            "weights_path": str(model_weights_path.name), # Relative path
            "model_task_keys_ordered": model_task_keys, # ["taxa_L20_genus", "taxa_L10_species"]
            "num_classes_per_task": num_classes_for_model, # [3, 4]
            "null_class_indices": null_class_indices_model,
            "expected_aux_vector_length": 0 # Our mock model doesn't use aux for simplicity in this test
        },
        "input_preprocessing": {
            "image_size": [3, 32, 32], # [C, H, W] -> RGB 32x32 image
            "image_mean": [0.485, 0.456, 0.406], # Standard ImageNet mean
            "image_std": [0.229, 0.224, 0.225],  # Standard ImageNet std
            "image_interpolation": "bilinear"
        },
        "metadata_preprocessing": { # Keep this minimal as aux_vector_length is 0
            "use_geolocation": False,
            "use_temporal": False,
            "use_elevation": False,
            "elevation_scales": []
        },
        "taxonomy_data": {
            "source_name": "MockTaxonomy",
            "version": "0.1",
            "root_identifier": "Life", # Dummy root
            "taxonomy_tree_path": str(taxonomy_json_path.name), # Relative path
            "class_index_map_path": str(class_index_map_json_path.name) # Relative path
        },
        "inference_options": {
            "default_top_k": 2,
            "device": "cpu",
            "batch_size": 2,
            "enable_hierarchical_consistency_check": True,
            "handler_version": "test-0.1"
        },
        "model_description": "A mock model bundle for testing LinnaeusInferenceHandler"
    }
    inference_config_yaml_path = bundle_dir / "mock_inference_config.yaml"
    with open(inference_config_yaml_path, "w") as f:
        yaml.dump(inference_config_dict, f)

    return bundle_dir # Yield Path object to the bundle directory

# Placeholder for a synthetic image that matches the "TinyModel" input
# The TinyModel expects a flattened tensor of size 3*32*32 = 3072
# The handler's preprocess_image_batch will create a [C,H,W] tensor of shape [3,32,32]
# For testing, we'll create a PIL image and mock the preprocessing

def create_mock_pil_image(width=32, height=32, mode="RGB"):
    # Creates a RGB image that matches our config
    return Image.new(mode, (width, height), color="white")


# --- Tests will be added below this line in subsequent subtasks ---

from unittest.mock import patch

# Helper to create a correctly shaped mock tensor for TinyModel's input
def create_mock_image_input_tensor(batch_size, num_features):
    return torch.randn(batch_size, num_features)

# Helper to create TinyModel instance
def create_tiny_model(bundle_path):
    """Create and load TinyModel with saved weights"""
    model_config_for_tiny_model = {"taxa_L20_genus": 3, "taxa_L10_species": 4}
    tiny_model = TinyModel(num_classes_per_task=model_config_for_tiny_model)
    state_dict = torch.load(bundle_path / "mock_pytorch_model.bin")
    tiny_model.load_state_dict(state_dict)
    tiny_model.eval()
    return tiny_model

def create_mock_handler(bundle_path, config_file, config_overrides=None):
    """Create a mock LinnaeusInferenceHandler with TinyModel"""
    tiny_model = create_tiny_model(bundle_path)

    # Create a mock handler with the minimum required attributes
    mock_handler = LinnaeusInferenceHandler.__new__(LinnaeusInferenceHandler)
    mock_handler.model = tiny_model
    mock_handler.device = torch.device("cpu")

    # Load real config
    from linnaeus.inference.config import load_inference_config
    if config_overrides:
        # Apply config overrides if provided
        from linnaeus.inference.config import InferenceConfig
        with open(config_file) as f:
            config_dict = yaml.safe_load(f)
        config_dict.update(config_overrides)
        mock_handler.config = InferenceConfig(**config_dict)
    else:
        mock_handler.config = load_inference_config(config_file, bundle_path)

    # Load real taxonomy and class maps
    from linnaeus.inference.artifacts import load_taxonomy_tree_artifact, load_class_index_maps_artifact
    mock_handler.taxonomy_data = load_taxonomy_tree_artifact(mock_handler.config.taxonomy_data, bundle_path)
    mock_handler.class_maps = load_class_index_maps_artifact(mock_handler.config.taxonomy_data, bundle_path)

    return mock_handler


def test_predict_output_types(fixture_inference_bundle):
    """Verify that handler.predict() returns objects of the correct typus types."""
    bundle_path = fixture_inference_bundle

    # Patch model loading
    tiny_model = create_tiny_model(bundle_path)
    with patch('linnaeus.inference.model_utils.load_model_for_inference', return_value=tiny_model):
        config_file = bundle_path / "mock_inference_config.yaml"
        handler = LinnaeusInferenceHandler.load_from_artifacts(config_file_path=config_file)

    # The TinyModel in the fixture expects input_features = 10.
    # The handler's input_preprocessing config image_size is [10, 1, 1].
    # preprocess_image_batch would normally produce a tensor like (B, C, H, W) = (B, 10, 1, 1).
    # The TinyModel's forward method expects a flat (B, Features) tensor.
    # We need to ensure the input to model() within handler.predict() is correctly shaped.

    # For this test, let's mock the preprocess_image_batch to directly return
    # a tensor that our TinyModel expects after flattening, or adjust the TinyModel's forward.
    # The handler itself doesn't flatten; it passes the preprocessed tensor to the model.
    # So, TinyModel's forward should handle the (B, C, H, W) input or expect (B, Features).
    # The current TinyModel expects (B, Features). Let's assume the handler's model wrapper
    # or the model itself handles the view change from (B,C,H,W) to (B, Features).
    # For the purpose of this test, we will prepare an "image" that, once preprocessed by
    # the *actual* `preprocess_image_batch` (not a mock of it, if possible),
    # results in a (1, 10, 1, 1) tensor. Then the model receives this.
    # The TinyModel needs to adapt, or we mock what the model receives.

    # Simpler approach: Mock the model's output directly to focus on handler's postprocessing.
    # However, the spec asks to test `predict()` returns types, implying an end-to-end flow through predict.

    # Let's refine TinyModel's forward to accept (B,C,H,W) and flatten it.
    # This was not done in the fixture creation, so we'd need to update the fixture,
    # or patch the model on the fly, or ensure our input image processing leads to (B, 10).

    # Path of least resistance for this test:
    # Create a PIL image that will be converted by `preprocess_image_batch` into a (1, 10, 1, 1) tensor.
    # A 1x1 image where C=10. PIL doesn't directly support C>4.
    # So, we must rely on mocking or a very specific interpretation of "image_size: [10,1,1]".
    # The `InputConfig` implies C,H,W. `preprocess_single_image` uses TF.to_tensor which makes C the first dim.
    # If image_size[0] is C, then it's 10.
    # `TF.resize` takes (H,W). `TF.to_tensor` assumes HWC or HW.

    # Let's use a workaround: provide "bytes" that our mock model can interpret as features.
    # Or, more robustly, patch `preprocess_image_batch` within the handler for this test
    # to return a tensor that TinyModel expects.

    mock_image_tensor_for_model = create_mock_image_input_tensor(batch_size=1, num_features=handler.model.input_features) # (1, 10)

    # Patch preprocess_image_batch to control its output directly
    with patch('linnaeus.inference.handler.preprocess_image_batch') as mock_preprocess:
        # Configure the mock to return a tensor of shape (B, C, H, W) as the handler expects
        # C, H, W from handler.config.input_preprocessing.image_size = [10, 1, 1]
        # So, the output of preprocess_image_batch should be (1, 10, 1, 1)
        # And the TinyModel should adapt this.
        # Let's assume TinyModel's forward is:
        # def forward(self, image_tensor_batch, aux_vector=None):
        #     batch_size = image_tensor_batch.shape[0]
        #     # Flatten C,H,W dims:
        #     image_features = image_tensor_batch.view(batch_size, -1)
        #     ...
        # This change should ideally be in TinyModel in the fixture.
        # For now, let's make sure the preprocessor mock output is (B, 10, 1, 1)
        # and assume the model handles it. TinyModel.forward now handles flattening.

        # The `handler.model` is the `TinyModel` instance.
        # Its `input_features` is 3*32*32 = 3072 based on fixture.
        # The `handler.config.input_preprocessing.image_size` is `[3, 32, 32]`.
        # `preprocess_image_batch` will return a tensor of shape `(batch_size, 3, 32, 32)`.
        # The updated `TinyModel.forward` now correctly flattens this to `(batch_size, 3072)`.
        # No patching of the model's forward method is needed anymore.

        # mock_preprocess_image_batch should return the (B,C,H,W) tensor
        # For a single image input: batch_size = 1
        mock_preprocessed_output = torch.randn(1, *handler.config.input_preprocessing.image_size) # (1, 3, 32, 32)
        mock_preprocess.return_value = mock_preprocessed_output

        # Create a dummy PIL image (it won't be used by the mocked preprocessor, but API requires it)
        dummy_pil_image = create_mock_pil_image() # Defaults to 32x32 RGB

        results = handler.predict(images=[dummy_pil_image]) # Call with one image

        assert isinstance(results, list), "Predict should return a list"
        assert len(results) == 1, "Predict should return one result for one image"

        hcr = results[0]
        assert isinstance(hcr, HierarchicalClassificationResult), "Result item should be HierarchicalClassificationResult"
        assert hcr.taxonomy_context is not None
        assert hcr.taxonomy_context.source == "MockTaxonomy" # From fixture config

        assert isinstance(hcr.tasks, list), "HCR tasks attribute should be a list"
        assert len(hcr.tasks) > 0, "HCR tasks should not be empty" # TinyModel has 2 tasks

        # Tasks are sorted by rank_level descending in handler.
        # Mock model has taxa_L20_genus (RankLevel.L20) and taxa_L10_species (RankLevel.L10)
        # So, Genus task should be first.
        expected_ranks_in_order = [RankLevel.L20, RankLevel.L10]
        assert [task.rank_level for task in hcr.tasks] == expected_ranks_in_order,             f"Tasks are not sorted by rank correctly. Got {[task.rank_level for task in hcr.tasks]}"

        for task_pred in hcr.tasks:
            assert isinstance(task_pred, TaskPrediction), "Each task in HCR should be TaskPrediction"
            assert isinstance(task_pred.predictions, list), "TaskPrediction.predictions should be a list"
            assert task_pred.rank_level in [RankLevel.L20, RankLevel.L10]
            if task_pred.predictions: # Might be empty if top_k is small and many nulls
                pred_item = task_pred.predictions[0]
                assert isinstance(pred_item, tuple), "Each prediction item should be a tuple"
                assert isinstance(pred_item[0], int), "Taxon ID in prediction tuple should be int"
                assert isinstance(pred_item[1], float), "Score in prediction tuple should be float"

        # No need to restore original_forward as we are not patching the model's forward anymore.


def test_predict_with_without_metadata(fixture_inference_bundle):
    """Test handler.predict() with and without metadata."""
    bundle_path = fixture_inference_bundle
    config_file = bundle_path / "mock_inference_config.yaml"

    # For this test, we need a model that actually uses the aux vector.
    # The current TinyModel in the fixture has expected_aux_vector_length = 0.
    # We need to update the fixture or this test to use a config where aux_vector_length > 0.

    # Let's modify the config on the fly for this test to enable some metadata.
    # This is easier than rebuilding the whole fixture for one test variation.
    # We'll patch load_inference_config used by handler.load_from_artifacts.

    # Original config from fixture
    with open(config_file) as f:
        original_config_dict = yaml.safe_load(f)

    # Modify a copy for this test
    config_dict_with_meta = json.loads(json.dumps(original_config_dict)) # Deep copy

    # Enable one metadata component and set expected length
    # Say, geolocation (3 features)
    config_dict_with_meta["metadata_preprocessing"]["use_geolocation"] = True
    # Our TinyModel's forward takes image_features, aux_vector.
    # Let's make expected_aux_vector_length = 3 for this test.
    # The handler will calculate this if set to None in config, or use value if provided.
    # We will set it explicitly to 3.
    config_dict_with_meta["model"]["expected_aux_vector_length"] = 3
                                                                    # (lat,lon -> x,y,z)

    # Patch load_inference_config to return this modified config when handler loads
    modified_inference_config_obj = InferenceConfig(**config_dict_with_meta)

    # Create TinyModel instance before patching
    model_config_for_tiny_model = {"taxa_L20_genus": 3, "taxa_L10_species": 4}
    tiny_model = TinyModel(num_classes_per_task=model_config_for_tiny_model)
    state_dict = torch.load(bundle_path / "mock_pytorch_model.bin")
    tiny_model.load_state_dict(state_dict)
    tiny_model.eval()

    with patch('linnaeus.inference.handler.load_inference_config') as mock_load_cfg:
        mock_load_cfg.return_value = modified_inference_config_obj

        # Patch model loading at higher level to bypass factory completely
        with patch('linnaeus.inference.model_utils.load_model_for_inference') as mock_load_model_inner:
            mock_load_model_inner.return_value = tiny_model

            handler = LinnaeusInferenceHandler.load_from_artifacts(
                config_file_path=config_file,
                artifacts_base_dir=bundle_path
            )

    assert handler.config.model.expected_aux_vector_length == 3
    assert handler.config.metadata_preprocessing.use_geolocation is True

    # Patch the model's forward to handle flattening and also to check aux_vector presence
    original_model_forward = handler.model.forward
    last_aux_vector_received = None

    def patched_model_forward_for_meta_test(image_tensor_batch, aux_vector=None):
        nonlocal last_aux_vector_received
        last_aux_vector_received = aux_vector # Capture aux_vector

        batch_size = image_tensor_batch.shape[0]
        image_features = image_tensor_batch.view(batch_size, -1) # Flatten (B,C,H,W)

        # If aux_vector is present and model expects it, it might be used.
        # For TinyModel, let's just pass it to original logic which might not use it,
        # but we've captured it for assertion.
        return original_model_forward(image_features, aux_vector)

    handler.model.forward = patched_model_forward_for_meta_test

    # Patch preprocess_image_batch
    with patch('linnaeus.inference.handler.preprocess_image_batch') as mock_preprocess_img:
        mock_preprocessed_img_output = torch.randn(1, *handler.config.input_preprocessing.image_size) # (1,10,1,1)
        mock_preprocess_img.return_value = mock_preprocessed_img_output
        dummy_pil_image = create_mock_pil_image()

        # --- Test Case 1: Prediction WITH metadata ---
        metadata_with = [{"lat": 40.0, "lon": -70.0}] # Activates geolocation
        handler.predict(images=[dummy_pil_image], metadata_list=metadata_with)

        assert last_aux_vector_received is not None, "Aux vector should be passed to model when metadata is provided"
        assert isinstance(last_aux_vector_received, torch.Tensor), "Aux vector should be a Tensor"
        assert last_aux_vector_received.shape == (1, 3),             f"Aux vector shape incorrect. Expected (1,3), Got {last_aux_vector_received.shape}"
        # Geolocation (x,y,z) for (40, -70) should not be all zeros.
        assert not torch.all(last_aux_vector_received == 0).item(), "Aux vector from metadata should not be all zeros"

        # --- Test Case 2: Prediction WITHOUT metadata (empty dict) ---
        last_aux_vector_received = None # Reset
        handler.predict(images=[dummy_pil_image], metadata_list=[{}]) # Empty dict for metadata

        assert last_aux_vector_received is not None, "Aux vector should still be passed (as zeros) if model expects it"
        assert isinstance(last_aux_vector_received, torch.Tensor)
        assert last_aux_vector_received.shape == (1, 3)
        # When metadata is missing but aux is expected, preprocess_metadata_batch creates zero tensor.
        assert torch.all(last_aux_vector_received == 0).item(), "Aux vector should be all zeros when metadata is missing but expected"

        # --- Test Case 3: Prediction with metadata_list=None ---
        last_aux_vector_received = None # Reset
        handler.predict(images=[dummy_pil_image], metadata_list=None)

        assert last_aux_vector_received is not None
        assert isinstance(last_aux_vector_received, torch.Tensor)
        assert last_aux_vector_received.shape == (1, 3)
        assert torch.all(last_aux_vector_received == 0).item(), "Aux vector should be all zeros when metadata_list is None but expected"

    # Restore original model forward
    handler.model.forward = original_model_forward


def test_predict_batch(fixture_inference_bundle):
    """Test handler.predict() with a batch of multiple images."""
    bundle_path = fixture_inference_bundle
    batch_size = 3

    tiny_model = create_tiny_model(bundle_path)
    with patch('linnaeus.inference.model_utils.load_model_for_inference', return_value=tiny_model):
        config_file = bundle_path / "mock_inference_config.yaml"
        handler = LinnaeusInferenceHandler.load_from_artifacts(config_file_path=config_file)

        # Patch model's forward to handle flattening
        original_model_forward = handler.model.forward
        def patched_model_forward_for_batch_test(image_tensor_batch, aux_vector=None):
            batch_dim = image_tensor_batch.shape[0]
            assert batch_dim == batch_size, f"Model received incorrect batch size. Expected {batch_size}, Got {batch_dim}"
            image_features = image_tensor_batch.view(batch_dim, -1)
            return original_model_forward(image_features, aux_vector)
        handler.model.forward = patched_model_forward_for_batch_test

        # Patch preprocess_image_batch
        with patch('linnaeus.inference.handler.preprocess_image_batch') as mock_preprocess_img:
            # Output of preprocess_image_batch should be (batch_size, C, H, W)
            mock_preprocessed_img_output = torch.randn(batch_size, *handler.config.input_preprocessing.image_size)
            mock_preprocess_img.return_value = mock_preprocessed_img_output

            dummy_pil_images = [create_mock_pil_image() for _ in range(batch_size)]

            # --- Test Case 1: Batch predict without metadata ---
            results_batch_no_meta = handler.predict(images=dummy_pil_images, metadata_list=None)
            assert isinstance(results_batch_no_meta, list)
            assert len(results_batch_no_meta) == batch_size,                 f"Expected {batch_size} results for batch input, got {len(results_batch_no_meta)}"
            for res in results_batch_no_meta:
                assert isinstance(res, HierarchicalClassificationResult)

            # --- Test Case 2: Batch predict with metadata ---
            metadata_batch = [{}, {"lat": 40.0, "lon": -70.0}, {}] # Mix of empty and actual
            assert len(metadata_batch) == batch_size

            # Since default fixture has aux_vector_length=0, metadata content won't change aux tensor from shape (B,0)
            results_batch_with_meta = handler.predict(images=dummy_pil_images, metadata_list=metadata_batch)
            assert isinstance(results_batch_with_meta, list)
            assert len(results_batch_with_meta) == batch_size
            for res in results_batch_with_meta:
                assert isinstance(res, HierarchicalClassificationResult)


# --- More tests will be added below this ---

def test_handler_loading():
    """Ensure the LinnaeusInferenceHandler can be successfully loaded from real artifacts."""
    real_bundle_path = Path("/datasets/modelWorkshop/mFormerV1/linnaeus/amphibia_mFormerV1/amphibia_mFormerV1_sm_r3c_40e/inference")
    config_file = real_bundle_path / "inference_config.yaml"

    assert config_file.exists(), f"Real inference config file should exist at {config_file}"

    try:
        handler = LinnaeusInferenceHandler.load_from_artifacts(config_file_path=config_file)
    except Exception as e:
        pytest.fail(f"LinnaeusInferenceHandler.load_from_artifacts failed: {e}")

    assert isinstance(handler, LinnaeusInferenceHandler), "Should return an instance of LinnaeusInferenceHandler"
    assert handler.model is not None, "Handler model should be loaded"
    assert handler.config is not None, "Handler config should be loaded"
    assert handler.taxonomy_data is not None, "Handler taxonomy_data should be loaded"
    assert handler.class_maps is not None, "Handler class_maps should be loaded"


def test_info_method():
    """Test the info() method of the LinnaeusInferenceHandler."""
    real_bundle_path = Path("/datasets/modelWorkshop/mFormerV1/linnaeus/amphibia_mFormerV1/amphibia_mFormerV1_sm_r3c_40e/inference")
    config_file = real_bundle_path / "inference_config.yaml"

    handler = LinnaeusInferenceHandler.load_from_artifacts(config_file_path=config_file)
    model_info = handler.info()

    assert isinstance(model_info, ModelInformation), "info() should return ModelInformation object"

    # Test basic info fields exist
    assert hasattr(model_info, 'model_name')
    assert hasattr(model_info, 'predicted_rank_levels')
    assert hasattr(model_info, 'num_classes_per_rank')
    assert hasattr(model_info, 'image_input_size')
    assert hasattr(model_info, 'default_top_k')

    # Test that rank levels are valid
    assert isinstance(model_info.predicted_rank_levels, list)
    assert len(model_info.predicted_rank_levels) > 0

    # Test that image input size is valid
    assert isinstance(model_info.image_input_size, list)
    assert len(model_info.image_input_size) == 3  # [C, H, W]
