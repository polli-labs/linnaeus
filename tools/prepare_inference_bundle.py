# tools/prepare_inference_bundle.py

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import torch
import yaml
from yacs.config import CfgNode as CN


# --- Add project root to sys.path to allow imports from linnaeus ---
def get_project_root() -> Path:
    """Find the project root by looking for the 'linnaeus' directory."""
    current_path = Path(__file__).resolve()
    while current_path.parent != current_path:
        if (current_path / 'linnaeus').is_dir():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError("Could not find project root. Make sure 'linnaeus' directory is in the path.")

PROJECT_ROOT = get_project_root()
sys.path.append(str(PROJECT_ROOT))
# --- End sys.path setup ---

# Now we can import from linnaeus and typus
try:
    from typus.constants import RankLevel

    from linnaeus.inference.config import InferenceConfig, InferenceOptionsConfig, InputConfig, MetaConfig, ModelConfig, TaxonomyConfig
except ImportError as e:
    print(f"Error: Failed to import linnaeus/typus modules. Ensure they are installed and PYTHONPATH is correct. Details: {e}")
    sys.exit(1)


# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def process_weights(checkpoint_path: Path, output_dir: Path) -> None:
    """
    Loads a training checkpoint, extracts the model's state dictionary,
    cleans it for inference, and saves it as 'pytorch_model.bin'.
    """
    logger.info(f"Processing model weights from: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load checkpoint onto CPU to avoid GPU memory usage
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # The state dict is usually under the 'model' key
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        raise KeyError("Checkpoint does not contain a 'model' key with the state dictionary.")

    # Remove the 'module.' prefix if it exists (from DDP training)
    cleaned_state_dict = {
        k.replace('module.', ''): v for k, v in state_dict.items()
    }
    logger.info(f"Cleaned {len(cleaned_state_dict)} keys from the state dictionary.")

    # Save the cleaned state dictionary
    output_path = output_dir / "pytorch_model.bin"
    torch.save(cleaned_state_dict, output_path)
    logger.info(f"Saved cleaned model weights to: {output_path}")


def process_taxonomy_tree(assets_dir: Path, output_dir: Path) -> None:
    """Copies the taxonomy_tree.json to the inference bundle."""
    source_path = assets_dir / "taxonomy_tree.json"
    dest_path = output_dir / "taxonomy.json"
    logger.info(f"Copying taxonomy tree from: {source_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"Taxonomy tree file not found: {source_path}")

    shutil.copy(source_path, dest_path)
    logger.info(f"Copied taxonomy tree to: {dest_path}")


def process_class_index_map(assets_dir: Path, output_dir: Path) -> None:
    """
    Loads the class_to_idx map, inverts it to create an idx_to_taxon_id map,
    and saves it to the inference bundle.
    """
    source_path = assets_dir / "class_to_idx.json"
    dest_path = output_dir / "class_index_map.json"
    logger.info(f"Processing class index map from: {source_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"Class index map file not found: {source_path}")

    with open(source_path) as f:
        class_to_idx = json.load(f)

    # Invert the mapping: {taxon_id_str_or_null: class_idx} -> {class_idx_str: taxon_id_int}
    idx_to_taxon_id = {}
    for task_key, mapping in class_to_idx.items():
        inverted_mapping = {}
        for taxon_id_str, class_idx in mapping.items():
            # Handle the special "null" key by mapping it to taxon ID 0.
            if taxon_id_str == 'null':
                taxon_id_int = 0
            else:
                taxon_id_int = int(taxon_id_str)

            inverted_mapping[str(class_idx)] = taxon_id_int

        idx_to_taxon_id[task_key] = inverted_mapping

    with open(dest_path, 'w') as f:
        json.dump(idx_to_taxon_id, f, indent=2)
    logger.info(f"Saved inverted class index map to: {dest_path}")


def generate_inference_config(exp_dir: Path, output_dir: Path, epoch: int) -> None:
    """Generates the self-contained inference_config.yaml."""
    logger.info("Generating inference_config.yaml...")

    # --- Load Source Files ---
    exp_config_path = exp_dir / "configs" / "experiment_config.yaml"
    num_classes_path = exp_dir / "assets" / "num_classes.json"

    if not exp_config_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {exp_config_path}")
    if not num_classes_path.is_file():
        raise FileNotFoundError(f"Number of classes file not found: {num_classes_path}")

    with open(exp_config_path) as f:
        # --- FIX: Use yaml.load with FullLoader to handle python/tuple tags ---
        cfg = CN(yaml.load(f, Loader=yaml.FullLoader))
        # --- END OF FIX ---
    with open(num_classes_path) as f:
        num_classes_dict = json.load(f)

    # --- Populate Pydantic Models ---
    task_keys = cfg.DATA.TASK_KEYS_H5
    num_classes_list = [num_classes_dict[key] for key in task_keys]

    # Model Config
    model_cfg = ModelConfig(
        architecture_name=cfg.MODEL.NAME,
        weights_path="pytorch_model.bin",
        model_task_keys_ordered=task_keys,
        num_classes_per_task=num_classes_list,
        null_class_indices={key: 0 for key in task_keys}, # Convention
        expected_aux_vector_length=sum(
            comp.DIM for comp in cfg.DATA.META.COMPONENTS.values() if comp.ENABLED
        )
    )

    # Input Preprocessing Config
    input_cfg = InputConfig(
        image_size=[3, cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE]
    )

    # Metadata Preprocessing Config
    meta_cfg = MetaConfig(
        use_geolocation=cfg.DATA.META.COMPONENTS.SPATIAL.ENABLED,
        use_temporal=cfg.DATA.META.COMPONENTS.TEMPORAL.ENABLED,
        use_elevation=cfg.DATA.META.COMPONENTS.ELEVATION.ENABLED,
    )

    # Taxonomy Data Config
    tax_cfg = TaxonomyConfig(
        source_name=f"iNat_{cfg.DATA.DATASET.CLADE}",
        version=cfg.DATA.DATASET.VERSION,
        root_identifier=47169, # Example: Amphibia
        taxonomy_tree_path="taxonomy.json",
        class_index_map_path="class_index_map.json"
    )

    # Inference Options Config
    options_cfg = InferenceOptionsConfig(
        default_top_k=5,
        artifacts_source_uri=str(output_dir.resolve())
    )

    # Final Inference Config
    inference_config = InferenceConfig(
        model=model_cfg,
        input_preprocessing=input_cfg,
        metadata_preprocessing=meta_cfg,
        taxonomy_data=tax_cfg,
        inference_options=options_cfg,
        model_description=f"Exported from {cfg.EXPERIMENT.NAME} at epoch {epoch}"
    )

    # Save to YAML
    output_path = output_dir / "inference_config.yaml"
    with open(output_path, 'w') as f:
        # Dump Pydantic model to dict, then to YAML
        yaml.dump(inference_config.model_dump(mode='json'), f, default_flow_style=False, sort_keys=False)

    logger.info(f"Generated and saved inference config to: {output_path}")


def main():
    """Main script execution."""
    parser = argparse.ArgumentParser(
        description="Prepare a training checkpoint and assets for inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Path to the training experiment output directory.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
        help="The epoch number of the checkpoint to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the inference bundle. Defaults to 'inference/' inside the experiment directory.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level.",
    )
    args = parser.parse_args()

    # Setup logging
    logger.setLevel(args.log_level.upper())

    try:
        exp_dir = Path(args.experiment_dir).resolve()
        assets_dir = exp_dir / "assets"
        ckpt_path = exp_dir / "checkpoints" / f"ckpt_epoch_{args.epoch}.pth"

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir).resolve()
        else:
            output_dir = exp_dir / "inference"

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Inference bundle will be created at: {output_dir}")

        # --- Run processing steps ---
        process_weights(ckpt_path, output_dir)
        process_taxonomy_tree(assets_dir, output_dir)
        process_class_index_map(assets_dir, output_dir)
        generate_inference_config(exp_dir, output_dir, args.epoch)

        logger.info("\n--- Inference Bundle Creation Complete ---")
        logger.info(f"Bundle contents saved to: {output_dir}")
        logger.info("The bundle is now ready for use with the LinnaeusInferenceHandler.")

    except FileNotFoundError as e:
        logger.error(f"A required file was not found: {e}")
        sys.exit(1)
    except KeyError as e:
        logger.error(f"A required key was missing from a configuration or asset file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
