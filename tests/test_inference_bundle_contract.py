from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest
import torch
import torch.nn as nn
import yaml
from PIL import Image
from pydantic import ValidationError
from yacs.config import CfgNode as CN

from linnaeus.inference.api_schemas import ModelInformation
from linnaeus.inference.config import ModelConfig
from linnaeus.inference.handler import LinnaeusInferenceHandler
from linnaeus.models.model_factory import _model_registry
from linnaeus.utils.config_utils import load_yaml_data
from tools.prepare_inference_bundle import generate_inference_config


class FixtureInferenceModelContract(nn.Module):
    def __init__(self, config: CN, **kwargs):
        super().__init__()
        img_size = int(config.MODEL.IMG_SIZE)
        in_features = 3 * img_size * img_size
        self.heads = nn.ModuleDict()

        for task_key, head_cfg in config.MODEL.CLASSIFICATION.HEADS.items():
            self.heads[task_key] = nn.Linear(in_features, int(head_cfg.OUT_FEATURES))

    def forward(self, image_tensor_batch, aux_vector=None):
        batch_size = image_tensor_batch.shape[0]
        flat = image_tensor_batch.view(batch_size, -1)
        return {task_key: head(flat) for task_key, head in self.heads.items()}


@pytest.fixture(autouse=True)
def register_fixture_model_contract():
    previous = _model_registry.get("FixtureInferenceModelContract")
    _model_registry["FixtureInferenceModelContract"] = FixtureInferenceModelContract
    try:
        yield
    finally:
        if previous is None:
            _model_registry.pop("FixtureInferenceModelContract", None)
        else:
            _model_registry["FixtureInferenceModelContract"] = previous


def _taxonomy_tree_payload() -> dict:
    return {
        "__taxonomy_tree_version__": "1.0",
        "task_keys": ["taxa_L10", "taxa_L20"],
        "num_classes": {
            "taxa_L10": 4,
            "taxa_L20": 3,
        },
        "hierarchy_map_raw": {
            "taxa_L10": {
                0: 0,
                1: 0,
                2: 1,
            }
        },
    }


def _class_index_map_payload() -> dict:
    return {
        "taxa_L20": {
            "0": 10,
            "1": 20,
            "2": 1,
        },
        "taxa_L10": {
            "0": 101,
            "1": 102,
            "2": 201,
            "3": 0,
        },
    }


def _resolved_model_config_payload() -> dict:
    return {
        "MODEL": {
            "TYPE": "FixtureInferenceModelContract",
            "NAME": "fixture_contract_model",
            "IMG_SIZE": 32,
            "PRETRAINED": "",
            "AGGREGATION": {
                "PARAMETERS": {
                    "out_channels": 3 * 32 * 32,
                }
            },
            "CLASSIFICATION": {
                "HEADS": {
                    "taxa_L20": {
                        "TYPE": "Linear",
                        "OUT_FEATURES": 3,
                    },
                    "taxa_L10": {
                        "TYPE": "Linear",
                        "OUT_FEATURES": 4,
                    },
                }
            },
        },
        "DATA": {
            "TASK_KEYS_H5": ["taxa_L20", "taxa_L10"],
            "META": {
                "ACTIVE": False,
                "COMPONENTS": {
                    "TEMPORAL": {"ENABLED": False, "DIM": 2, "IDX": 0},
                    "SPATIAL": {"ENABLED": False, "DIM": 3, "IDX": 1},
                    "ELEVATION": {"ENABLED": False, "DIM": 0, "IDX": 2},
                },
            },
        },
    }


def _write_contract_bundle(bundle_dir: Path, *, artifacts_source_uri: str = ".") -> Path:
    resolved_model_config = _resolved_model_config_payload()
    model = FixtureInferenceModelContract(CN(resolved_model_config))
    torch.save(model.state_dict(), bundle_dir / "pytorch_model.bin")

    (bundle_dir / "taxonomy.json").write_text(json.dumps(_taxonomy_tree_payload(), indent=2), encoding="utf-8")
    (bundle_dir / "class_index_map.json").write_text(json.dumps(_class_index_map_payload(), indent=2), encoding="utf-8")

    inference_config = {
        "model": {
            "architecture_name": "fixture_contract_model",
            "architecture_type": "FixtureInferenceModelContract",
            "resolved_model_config": resolved_model_config,
            "weights_path": "pytorch_model.bin",
            "model_task_keys_ordered": ["taxa_L20", "taxa_L10"],
            "num_classes_per_task": [3, 4],
            "null_class_indices": {
                "taxa_L20": 2,
                "taxa_L10": 3,
            },
            "expected_aux_vector_length": 0,
        },
        "input_preprocessing": {
            "image_size": [3, 32, 32],
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "image_interpolation": "bilinear",
        },
        "metadata_preprocessing": {
            "use_geolocation": False,
            "use_temporal": False,
            "temporal_use_julian_day": False,
            "temporal_use_hour": False,
            "use_elevation": False,
            "elevation_scales": [],
        },
        "taxonomy_data": {
            "source_name": "MockTaxonomy",
            "version": "0.1",
            "root_identifier": "Life",
            "taxonomy_tree_path": "taxonomy.json",
            "class_index_map_path": "class_index_map.json",
        },
        "inference_options": {
            "default_top_k": 2,
            "device": "cpu",
            "batch_size": 2,
            "enable_hierarchical_consistency_check": True,
            "handler_version": "test-0.1",
            "artifacts_source_uri": artifacts_source_uri,
        },
        "model_description": "Synthetic bundle for inference contract tests",
    }

    config_path = bundle_dir / "inference_config.yaml"
    config_path.write_text(yaml.safe_dump(inference_config, sort_keys=False), encoding="utf-8")
    return config_path


def _write_generate_inference_config_fixture(
    exp_dir: Path,
    *,
    backbone_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
) -> Path:
    config_dir = exp_dir / "configs"
    assets_dir = exp_dir / "assets"
    output_dir = exp_dir / "inference"
    labels_path = assets_dir / "labels.h5"
    config_dir.mkdir(parents=True)
    assets_dir.mkdir(parents=True)

    with h5py.File(labels_path, "w") as labels_h5:
        spatial = labels_h5.create_dataset("spatial", data=[[0.1, 0.2, 0.3]])
        spatial.attrs["method"] = "unit_sphere"
        spatial.attrs["column_names"] = ["spatial_x", "spatial_y", "spatial_z"]

        temporal = labels_h5.create_dataset("temporal", data=[[0.1, 0.2]])
        temporal.attrs["method"] = "sinusoidal"
        temporal.attrs["column_names"] = ["month_sin", "month_cos"]

        elevation = labels_h5.create_dataset("elevation_broadrange_2", data=[[0.1] * 10])
        elevation.attrs["method"] = "sinusoidal"
        elevation.attrs["column_names"] = [
            "elev_200_sin",
            "elev_200_cos",
            "elev_300_sin",
            "elev_300_cos",
            "elev_1000_sin",
            "elev_1000_cos",
            "elev_3000_sin",
            "elev_3000_cos",
            "elev_6000_sin",
            "elev_6000_cos",
        ]
        elevation.attrs["scales"] = [200, 300, 1000, 3000, 6000]

    experiment_config = {
        "EXPERIMENT": {
            "PROJECT": "linnaeus-dev",
            "GROUP": "tests",
            "NAME": "fixture-export",
        },
        "MODEL": {
            "TYPE": "DINOv3MultiHead",
            "NAME": "dinov3_fixture_contract",
            "DINOV3": {
                "BACKBONE_ID": backbone_id,
                "PATCH_SIZE": 16,
                "EMBED_DIM": 32,
                "USE_STUB": True,
            },
            "CLASSIFICATION": {
                "HEADS": {
                    "taxa_L20": {"TYPE": "Linear"},
                    "taxa_L10": {"TYPE": "Linear"},
                }
            },
        },
        "DATA": {
            "IMG_SIZE": 32,
            "TASK_KEYS_H5": ["taxa_L20", "taxa_L10"],
            "H5": {
                "LABELS_PATH": str(labels_path),
            },
            "META": {
                "ACTIVE": True,
                "COMPONENTS": {
                    "TEMPORAL": {"ENABLED": True, "SOURCE": "temporal", "DIM": 2, "IDX": 0},
                    "SPATIAL": {"ENABLED": True, "SOURCE": "spatial", "DIM": 3, "IDX": 1},
                    "ELEVATION": {"ENABLED": True, "SOURCE": "elevation_broadrange_2", "DIM": 10, "IDX": 2},
                },
            },
        },
    }
    (config_dir / "experiment_config.yaml").write_text(yaml.safe_dump(experiment_config, sort_keys=False), encoding="utf-8")
    (assets_dir / "num_classes.json").write_text(
        json.dumps({"taxa_L20": 3, "taxa_L10": 4}, indent=2),
        encoding="utf-8",
    )
    return output_dir


def test_model_config_rejects_task_length_mismatch():
    with pytest.raises(ValidationError, match="num_classes_per_task must have the same length"):
        ModelConfig(
            architecture_name="fixture_contract_model",
            weights_path="pytorch_model.bin",
            model_task_keys_ordered=["taxa_L20", "taxa_L10"],
            num_classes_per_task=[3],
            null_class_indices={"taxa_L20": 0, "taxa_L10": 0},
        )


def test_load_yaml_data_handles_python_tuple_tags(tmp_path: Path):
    yaml_path = tmp_path / "tuple_config.yaml"
    yaml_path.write_text("BETAS: !!python/tuple [0.9, 0.999]\n", encoding="utf-8")

    payload = load_yaml_data(str(yaml_path), allow_python_tuple=True)

    assert payload == {"BETAS": (0.9, 0.999)}


def test_load_from_artifacts_uses_embedded_resolved_model_config(tmp_path: Path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    config_path = _write_contract_bundle(bundle_dir)

    handler = LinnaeusInferenceHandler.load_from_artifacts(config_file_path=config_path)
    info = handler.info()
    results = handler.predict(images=[Image.new("RGB", (32, 32), color="white")])

    assert isinstance(handler, LinnaeusInferenceHandler)
    assert isinstance(info, ModelInformation)
    assert info.model_name == "fixture_contract_model"
    assert len(results) == 1
    assert len(results[0].tasks) == 2


def test_load_from_artifacts_resolves_portable_artifacts_relative_to_config_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    bundle_dir = tmp_path / "portable-bundle"
    bundle_dir.mkdir()
    config_path = _write_contract_bundle(bundle_dir, artifacts_source_uri=".")

    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    handler = LinnaeusInferenceHandler.load_from_artifacts(config_file_path=config_path)

    assert Path(handler.config.inference_options.artifacts_source_uri) == bundle_dir.resolve()
    assert Path(handler.config.model.weights_path) == bundle_dir.resolve() / "pytorch_model.bin"
    assert Path(handler.config.taxonomy_data.taxonomy_tree_path) == bundle_dir.resolve() / "taxonomy.json"
    assert Path(handler.config.taxonomy_data.class_index_map_path) == bundle_dir.resolve() / "class_index_map.json"


def test_load_from_artifacts_supports_legacy_absolute_artifacts_source_uri(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    bundle_dir = tmp_path / "legacy-bundle"
    bundle_dir.mkdir()
    config_path = _write_contract_bundle(bundle_dir, artifacts_source_uri=str(bundle_dir.resolve()))

    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    handler = LinnaeusInferenceHandler.load_from_artifacts(config_file_path=config_path)

    assert Path(handler.config.inference_options.artifacts_source_uri) == bundle_dir.resolve()
    assert Path(handler.config.model.weights_path) == bundle_dir.resolve() / "pytorch_model.bin"
    assert Path(handler.config.taxonomy_data.taxonomy_tree_path) == bundle_dir.resolve() / "taxonomy.json"
    assert Path(handler.config.taxonomy_data.class_index_map_path) == bundle_dir.resolve() / "class_index_map.json"


def test_generate_inference_config_embeds_resolved_model_contract(tmp_path: Path):
    exp_dir = tmp_path / "experiment"
    output_dir = _write_generate_inference_config_fixture(exp_dir)

    generate_inference_config(exp_dir=exp_dir, output_dir=output_dir, epoch=1)

    payload = yaml.safe_load((output_dir / "inference_config.yaml").read_text(encoding="utf-8"))

    assert payload["model"]["architecture_type"] == "DINOv3MultiHead"
    assert payload["model"]["resolved_model_config"]["MODEL"]["TYPE"] == "DINOv3MultiHead"
    assert payload["model"]["resolved_model_config"]["MODEL"]["BASE"] == []
    assert payload["model"]["resolved_model_config"]["MODEL"]["PRETRAINED"] == ""
    assert (
        payload["model"]["resolved_model_config"]["MODEL"]["DINOV3"]["BACKBONE_ID"]
        == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    )
    assert payload["metadata_preprocessing"]["use_geolocation"] is True
    assert payload["metadata_preprocessing"]["use_temporal"] is True
    assert payload["metadata_preprocessing"]["use_elevation"] is True
    assert payload["metadata_preprocessing"]["elevation_scales"] == [200.0, 300.0, 1000.0, 3000.0, 6000.0]
    assert payload["inference_options"]["artifacts_source_uri"] == "."
    assert payload["metadata_preprocessing"]["components"] == [
        {
            "name": "TEMPORAL",
            "source": "temporal",
            "dim": 2,
            "idx": 0,
            "columns": ["month_sin", "month_cos"],
            "method": "sinusoidal",
            "encoding_kind": "temporal_sinusoids",
            "scales": None,
            "temporal_basis": "month_of_year",
            "temporal_include_hour": False,
            "raw_input_fields": ["datetime_utc"],
        },
        {
            "name": "SPATIAL",
            "source": "spatial",
            "dim": 3,
            "idx": 1,
            "columns": ["spatial_x", "spatial_y", "spatial_z"],
            "method": "unit_sphere",
            "encoding_kind": "latlon_unit_sphere",
            "scales": None,
            "temporal_basis": None,
            "temporal_include_hour": False,
            "raw_input_fields": ["lat", "lon"],
        },
        {
            "name": "ELEVATION",
            "source": "elevation_broadrange_2",
            "dim": 10,
            "idx": 2,
            "columns": [
                "elev_200_sin",
                "elev_200_cos",
                "elev_300_sin",
                "elev_300_cos",
                "elev_1000_sin",
                "elev_1000_cos",
                "elev_3000_sin",
                "elev_3000_cos",
                "elev_6000_sin",
                "elev_6000_cos",
            ],
            "method": "sinusoidal",
            "encoding_kind": "elevation_sinusoids",
            "scales": [200.0, 300.0, 1000.0, 3000.0, 6000.0],
            "temporal_basis": None,
            "temporal_include_hour": False,
            "raw_input_fields": ["elevation_m"],
        },
    ]


def test_generate_inference_config_normalizes_known_absolute_dinov3_backbone_path(tmp_path: Path):
    exp_dir = tmp_path / "experiment"
    output_dir = _write_generate_inference_config_fixture(
        exp_dir,
        backbone_id="/datasets/modelZoo/dinov3/dinov3-vitb16-pretrain-lvd1689m",
    )

    generate_inference_config(exp_dir=exp_dir, output_dir=output_dir, epoch=1, portable=True)

    payload = yaml.safe_load((output_dir / "inference_config.yaml").read_text(encoding="utf-8"))

    assert (
        payload["model"]["resolved_model_config"]["MODEL"]["DINOV3"]["BACKBONE_ID"]
        == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    )
    assert payload["inference_options"]["artifacts_source_uri"] == "."


def test_real_bundle_loading_uses_legacy_experiment_config_fallback():
    real_bundle_path = Path(
        "/datasets/modelWorkshop/mFormerV1/linnaeus/amphibia_mFormerV1/amphibia_mFormerV1_sm_r3c_40e/inference"
    )
    config_file = real_bundle_path / "inference_config.yaml"

    if not config_file.exists():
        pytest.skip(f"Real inference bundle not present on this machine: {config_file}")

    handler = LinnaeusInferenceHandler.load_from_artifacts(config_file_path=config_file)

    assert isinstance(handler, LinnaeusInferenceHandler)
    assert handler.model.__class__.__name__ == "mFormerV1"
    assert handler.config.model.architecture_name == "mFormerV1_sm"
    assert handler.config.metadata_preprocessing.components is not None
    assert [component.name for component in handler.config.metadata_preprocessing.components] == ["TEMPORAL", "SPATIAL", "ELEVATION"]
    assert handler.config.metadata_preprocessing.elevation_scales == [200.0, 300.0, 1000.0, 3000.0, 6000.0]
