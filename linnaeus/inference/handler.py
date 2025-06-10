"""
Linnaeus Inference Handler Implementation.
"""
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from typus.constants import RankLevel
from typus.models.classification import (
    HierarchicalClassificationResult,
    TaskPrediction,
    TaxonomyContext,
)

from linnaeus.inference.api_schemas import InferenceRequestMetadata, ModelInformation
from linnaeus.inference.artifacts import (
    ClassIndexMapData,
    TaxonomyData,
    load_class_index_maps_artifact,
    load_taxonomy_tree_artifact,
)
from linnaeus.inference.config import InferenceConfig, load_inference_config
from linnaeus.inference.model_utils import load_model_for_inference
from linnaeus.inference.postprocessing import enforce_hierarchical_consistency
from linnaeus.inference.preprocessing import (
    preprocess_image_batch,
    preprocess_metadata_batch,
)

logger = logging.getLogger("linnaeus.inference")


class LinnaeusInferenceHandler:
    def __init__(
        self,
        model: nn.Module,
        config: InferenceConfig,
        taxonomy_data: TaxonomyData,
        class_maps: ClassIndexMapData,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.taxonomy_data = taxonomy_data
        self.class_maps = class_maps
        self.device = device

        if self.config.model.expected_aux_vector_length is None:
            length = 0
            meta_cfg = self.config.metadata_preprocessing
            if meta_cfg.use_geolocation: length += 3
            if meta_cfg.use_temporal:
                length += 2
                if meta_cfg.temporal_use_hour: length += 2
            if meta_cfg.use_elevation: length += 2 * len(meta_cfg.elevation_scales)
            self.config.model.expected_aux_vector_length = length
            logger.info(f"Derived expected_aux_vector_length: {self.config.model.expected_aux_vector_length}")


    @classmethod
    def load_from_artifacts(
        cls,
        config_file_path: str | Path,
        artifacts_base_dir: str | Path | None = None,
        model_weights_path_override: str | Path | None = None,
        taxonomy_tree_path_override: str | Path | None = None,
        class_index_map_path_override: str | Path | None = None,
    ) -> "LinnaeusInferenceHandler":
        config_path = Path(config_file_path)
        cfg = load_inference_config(config_path)
        logger.info(f"Loaded inference configuration from: {config_path}")

        base_path = Path(artifacts_base_dir) if artifacts_base_dir else Path(cfg.inference_options.artifacts_source_uri or config_path.parent)

        # Update config with resolved paths if they were relative
        # Model weights path
        weights_path_str = model_weights_path_override or cfg.model.weights_path
        if not Path(weights_path_str).is_absolute() and not weights_path_str.startswith("hf://"):
            weights_path_str = str(base_path / weights_path_str)
        cfg.model.weights_path = weights_path_str # Update in Pydantic model

        # Taxonomy tree path
        tax_tree_path_str = taxonomy_tree_path_override or cfg.taxonomy_data.taxonomy_tree_path
        if not Path(tax_tree_path_str).is_absolute():
            tax_tree_path_str = str(base_path / tax_tree_path_str)
        cfg.taxonomy_data.taxonomy_tree_path = tax_tree_path_str

        # Class index map path
        class_map_path_str = class_index_map_path_override or cfg.taxonomy_data.class_index_map_path
        if not Path(class_map_path_str).is_absolute():
            class_map_path_str = str(base_path / class_map_path_str)
        cfg.taxonomy_data.class_index_map_path = class_map_path_str

        if cfg.inference_options.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        else:
            device = torch.device(cfg.inference_options.device)
        logger.info(f"Using device: {device}")

        taxonomy_data = load_taxonomy_tree_artifact(
            Path(cfg.taxonomy_data.taxonomy_tree_path),
            taxonomy_source_name=cfg.taxonomy_data.source_name,
            taxonomy_version_name=cfg.taxonomy_data.version,
            taxonomy_root_identifier=cfg.taxonomy_data.root_identifier,
        )

        class_maps = load_class_index_maps_artifact(
            Path(cfg.taxonomy_data.class_index_map_path),
            model_linnaeus_task_keys_ordered=cfg.model.model_task_keys_ordered,
            model_num_classes_per_task=cfg.model.num_classes_per_task,
            model_null_class_indices=cfg.model.null_class_indices
        )

        model = load_model_for_inference(cfg.model, cfg, taxonomy_data, device)

        return cls(model, cfg, taxonomy_data, class_maps, device)

    def _get_rank_level_from_linnaeus_task_key(self, linnaeus_task_key: str) -> RankLevel:
        try:
            numeric_part_str = linnaeus_task_key.split('_L')[-1]
            if '.' in numeric_part_str or '_' in numeric_part_str:
                 numeric_part_str = numeric_part_str.replace('_', '') # L33_5 -> L335 for typus
                 numeric_part = int("".join(filter(str.isdigit, numeric_part_str)))
            else:
                numeric_part = int(numeric_part_str)
            return RankLevel(numeric_part)
        except ValueError as e:
            logger.error(f"Cannot convert Linnaeus task key '{linnaeus_task_key}' to RankLevel. Error: {e}")
            raise

    def predict(
        self,
        images: list[bytes | Image.Image],
        metadata_list: list[dict[str, Any]] | None = None,
        per_sample_overrides: list[InferenceRequestMetadata | None] | None = None,
    ) -> list[HierarchicalClassificationResult]:
        start_time = time.monotonic()
        batch_size = len(images)

        if metadata_list is None: metadata_list = [{} for _ in range(batch_size)]
        if per_sample_overrides is None: per_sample_overrides = [None] * batch_size

        if len(metadata_list) != batch_size or len(per_sample_overrides) != batch_size:
            raise ValueError("Images, metadata_list, and per_sample_overrides must have the same length.")

        image_tensor_batch = preprocess_image_batch(images, self.config.input_preprocessing).to(self.device)

        aux_info_list: list[torch.Tensor] = []
        final_top_k_values: list[int] = []

        for i in range(batch_size):
            sample_meta_dict = metadata_list[i]
            sample_override = per_sample_overrides[i]

            final_top_k_values.append(
                sample_override.top_k if sample_override and sample_override.top_k is not None
                else self.config.inference_options.default_top_k
            )

            if sample_override and sample_override.unsafe_aux_override and sample_override.aux_vector:
                aux_tensor = torch.tensor(sample_override.aux_vector, dtype=torch.float32)
                if self.config.model.expected_aux_vector_length is not None and \
                   aux_tensor.shape[0] != self.config.model.expected_aux_vector_length:
                    raise ValueError(f"Provided aux_vector length mismatch for sample {i}.")
            else:
                single_sample_aux_tensor_batch = preprocess_metadata_batch(
                    [sample_meta_dict], self.config.metadata_preprocessing, self.config.model.expected_aux_vector_length
                )
                aux_tensor = single_sample_aux_tensor_batch[0]
            aux_info_list.append(aux_tensor)

        aux_tensor_batch = torch.stack(aux_info_list).to(self.device) if aux_info_list else torch.empty((batch_size, 0), device=self.device)

        # Handle case where no metadata components are active
        model_input_aux = aux_tensor_batch if (self.config.model.expected_aux_vector_length or 0) > 0 else None


        with torch.no_grad():
            raw_outputs: dict[str, torch.Tensor] = self.model(image_tensor_batch, model_input_aux)

        batch_results: list[HierarchicalClassificationResult] = []
        for i in range(batch_size):
            sample_task_predictions: list[TaskPrediction] = []
            current_top_k = final_top_k_values[i]

            for linnaeus_task_key in self.config.model.model_task_keys_ordered:
                if linnaeus_task_key not in raw_outputs:
                    logger.warning(f"Output for task '{linnaeus_task_key}' not found. Skipping.")
                    continue

                logits = raw_outputs[linnaeus_task_key][i]
                probs = torch.softmax(logits, dim=-1)

                rank_level = self._get_rank_level_from_linnaeus_task_key(linnaeus_task_key)
                num_classes = self.class_maps.num_classes_per_rank[rank_level]
                actual_k = min(current_top_k, num_classes)

                top_k_probs, top_k_indices = torch.topk(probs, k=actual_k)
                predictions_for_typus: list[tuple[int, float]] = []

                for k_idx in range(actual_k):
                    model_class_idx = top_k_indices[k_idx].item()
                    probability = top_k_probs[k_idx].item()
                    try:
                        taxon_id = self.class_maps.idx_to_taxon_id[rank_level][model_class_idx]
                        predictions_for_typus.append((taxon_id, probability))
                    except KeyError:
                        logger.error(f"Cannot map class index {model_class_idx} for task {linnaeus_task_key}.")

                sample_task_predictions.append(
                    TaskPrediction(rank_level=rank_level, temperature=1.0, predictions=predictions_for_typus)
                )

            sample_task_predictions.sort(key=lambda t: t.rank_level.value, reverse=True)
            hcr = HierarchicalClassificationResult(
                taxonomy_context=TaxonomyContext(source=self.taxonomy_data.source, version=self.taxonomy_data.version),
                tasks=sample_task_predictions,
                subtree_roots= {self.taxonomy_data.root_id} if self.taxonomy_data.root_id is not None else None,
            )

            if self.config.inference_options.enable_hierarchical_consistency_check:
                hcr = enforce_hierarchical_consistency(hcr, self.taxonomy_data, self.class_maps)

            batch_results.append(hcr)

        logger.info(f"Inference for batch of {batch_size} completed in {time.monotonic() - start_time:.4f}s.")
        return batch_results

    def info(self) -> ModelInformation:
        predicted_levels = [
            self._get_rank_level_from_linnaeus_task_key(key)
            for key in self.config.model.model_task_keys_ordered
        ]
        num_classes_per_rank_typus = {
            rl: self.class_maps.num_classes_per_rank[rl] for rl in predicted_levels
        }
        null_class_info_typus = {
            rl: self.class_maps.null_taxon_ids[rl] for rl in predicted_levels
        }

        meta_components_enabled: list[str] = []
        meta_feature_encoding: dict[str, str] = {}
        meta_cfg = self.config.metadata_preprocessing
        if meta_cfg.use_geolocation:
            meta_components_enabled.append("geolocation")
            meta_feature_encoding["geolocation"] = "lat/lon -> 3-dim unit sphere vector"
        if meta_cfg.use_temporal:
            meta_components_enabled.append("temporal")
            time_enc = f"{'day_of_year' if meta_cfg.temporal_use_julian_day else 'month_of_year'}"
            if meta_cfg.temporal_use_hour: time_enc += " + hour_of_day"
            meta_feature_encoding["temporal"] = f"{time_enc} -> cyclical (sin/cos) features"
        if meta_cfg.use_elevation:
            meta_components_enabled.append("elevation")
            meta_feature_encoding["elevation"] = f"elevation_m with scales {meta_cfg.elevation_scales} -> multi-scale sin/cos features"

        return ModelInformation(
            model_name=self.config.model.architecture_name,
            model_version=self.config.model_description,
            model_description=self.config.model_description,
            taxonomy_source=self.taxonomy_data.source,
            taxonomy_version=self.taxonomy_data.version,
            taxonomy_root_id=self.taxonomy_data.root_id,
            predicted_rank_levels=predicted_levels,
            num_classes_per_rank=num_classes_per_rank_typus,
            null_class_info=null_class_info_typus,
            image_input_size=self.config.input_preprocessing.image_size,
            image_normalization_mean=self.config.input_preprocessing.image_mean,
            image_normalization_std=self.config.input_preprocessing.image_std,
            metadata_components_enabled=meta_components_enabled,
            metadata_feature_encoding=meta_feature_encoding,
            aux_vector_length=self.config.model.expected_aux_vector_length or 0,
            default_top_k=self.config.inference_options.default_top_k,
            inference_handler_version=self.config.inference_options.handler_version,
            artifacts_source_uri=self.config.inference_options.artifacts_source_uri,
        )
