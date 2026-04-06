"""
Helpers for describing and reconstructing metadata component contracts used by inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
from yacs.config import CfgNode as CN

from linnaeus.config import get_default_config
from linnaeus.config_resolution import temporary_config_dir
from linnaeus.utils.config_utils import load_model_base_config, load_yaml_data

from .config import MetaConfig, MetadataComponentConfig


def _node_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if hasattr(node, "get"):
        return node.get(key, default)
    return getattr(node, key, default)


def _node_items(node: Any):
    if node is None:
        return []
    if hasattr(node, "items"):
        return list(node.items())
    return []


def _normalize_attr_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        value = [value]
    normalized = []
    for item in value:
        if isinstance(item, bytes):
            normalized.append(item.decode("utf-8", errors="replace"))
        else:
            normalized.append(item)
    return normalized


def _component_attrs(labels_path: Path | None, source: str) -> dict[str, Any]:
    if labels_path is None or not labels_path.is_file():
        return {}

    with h5py.File(labels_path, "r") as labels_h5:
        if source not in labels_h5:
            return {}

        dataset = labels_h5[source]
        return {key: dataset.attrs[key] for key in dataset.attrs.keys()}


def _infer_component_contract(
    *,
    name: str,
    source: str,
    dim: int,
    idx: int,
    columns: list[str],
    method: str | None,
    scales: list[float] | None,
) -> MetadataComponentConfig:
    upper_name = name.upper()
    encoding_kind = "passthrough_vector"
    raw_input_fields: list[str] = []
    temporal_basis: str | None = None
    temporal_include_hour = False

    lowered_columns = [str(column).lower() for column in columns]

    if upper_name == "SPATIAL" and source == "spatial" and method == "unit_sphere" and dim == 3:
        encoding_kind = "latlon_unit_sphere"
        raw_input_fields = ["lat", "lon"]
    elif upper_name == "TEMPORAL" and source == "temporal" and method == "sinusoidal":
        encoding_kind = "temporal_sinusoids"
        raw_input_fields = ["datetime_utc"]
        temporal_basis = "day_of_year" if any(column.startswith("jd_") for column in lowered_columns) else "month_of_year"
        temporal_include_hour = any(column.startswith("hour_") for column in lowered_columns)
    elif upper_name == "ELEVATION" and method == "sinusoidal" and scales:
        encoding_kind = "elevation_sinusoids"
        raw_input_fields = ["elevation_m"]

    return MetadataComponentConfig(
        name=upper_name,
        source=source,
        dim=dim,
        idx=idx,
        columns=columns,
        method=method,
        encoding_kind=encoding_kind,
        scales=scales,
        temporal_basis=temporal_basis,
        temporal_include_hour=temporal_include_hour,
        raw_input_fields=raw_input_fields,
    )


def build_metadata_components_from_data_cfg(data_cfg: Any, labels_path: Path | None = None) -> list[MetadataComponentConfig]:
    meta_cfg = _node_get(data_cfg, "META")
    components_cfg = _node_get(meta_cfg, "COMPONENTS")
    components: list[MetadataComponentConfig] = []

    for comp_name, comp_cfg in _node_items(components_cfg):
        if not _node_get(comp_cfg, "ENABLED", False):
            continue

        idx = int(_node_get(comp_cfg, "IDX", -1))
        if idx < 0:
            continue

        source = str(_node_get(comp_cfg, "SOURCE", comp_name.lower()))
        dim = int(_node_get(comp_cfg, "DIM", 0))
        configured_columns = [str(column) for column in (_node_get(comp_cfg, "COLUMNS", []) or [])]

        attrs = _component_attrs(labels_path, source)
        attr_columns = [str(column) for column in _normalize_attr_list(attrs.get("column_names"))]
        method = attrs.get("method")
        if isinstance(method, bytes):
            method = method.decode("utf-8", errors="replace")
        elif method is not None:
            method = str(method)

        scales_raw = _normalize_attr_list(attrs.get("scales"))
        scales = [float(scale) for scale in scales_raw] if scales_raw else None

        columns = configured_columns or attr_columns
        components.append(
            _infer_component_contract(
                name=str(comp_name),
                source=source,
                dim=dim,
                idx=idx,
                columns=columns,
                method=method,
                scales=scales,
            )
        )

    return sorted(components, key=lambda component: component.idx)


def load_metadata_components_from_experiment_config(exp_config_path: str | Path) -> list[MetadataComponentConfig]:
    exp_path = Path(exp_config_path)
    with temporary_config_dir(str(exp_path)):
        config = get_default_config()
        config.set_new_allowed(True)
        exp_config = CN(load_yaml_data(str(exp_path), allow_python_tuple=True))
        config.merge_from_other_cfg(exp_config)
        resolved_cfg = load_model_base_config(config)

    data_cfg = getattr(resolved_cfg, "DATA", None)
    h5_cfg = _node_get(data_cfg, "H5")
    labels_path_str = (
        _node_get(h5_cfg, "LABELS_PATH")
        or _node_get(h5_cfg, "TRAIN_LABELS_PATH")
        or _node_get(h5_cfg, "VAL_LABELS_PATH")
    )
    labels_path = Path(labels_path_str) if labels_path_str else None

    return build_metadata_components_from_data_cfg(data_cfg, labels_path=labels_path)


def ordered_metadata_components(meta_cfg: MetaConfig) -> list[MetadataComponentConfig]:
    if meta_cfg.components:
        return sorted(meta_cfg.components, key=lambda component: component.idx)

    components: list[MetadataComponentConfig] = []
    if meta_cfg.use_geolocation:
        components.append(
            MetadataComponentConfig(
                name="SPATIAL",
                source="spatial",
                dim=3,
                idx=0,
                method="unit_sphere",
                encoding_kind="latlon_unit_sphere",
                raw_input_fields=["lat", "lon"],
            )
        )
    if meta_cfg.use_temporal:
        components.append(
            MetadataComponentConfig(
                name="TEMPORAL",
                source="temporal",
                dim=2 + (2 if meta_cfg.temporal_use_hour else 0),
                idx=1,
                method="sinusoidal",
                encoding_kind="temporal_sinusoids",
                temporal_basis="day_of_year" if meta_cfg.temporal_use_julian_day else "month_of_year",
                temporal_include_hour=meta_cfg.temporal_use_hour,
                raw_input_fields=["datetime_utc"],
            )
        )
    if meta_cfg.use_elevation:
        components.append(
            MetadataComponentConfig(
                name="ELEVATION",
                source="elevation",
                dim=2 * len(meta_cfg.elevation_scales),
                idx=2,
                method="sinusoidal",
                encoding_kind="elevation_sinusoids",
                scales=[float(scale) for scale in meta_cfg.elevation_scales],
                raw_input_fields=["elevation_m"],
            )
        )
    return sorted(components, key=lambda component: component.idx)


def derive_expected_aux_vector_length(meta_cfg: MetaConfig) -> int:
    return sum(component.dim for component in ordered_metadata_components(meta_cfg))


def sync_legacy_meta_fields(meta_cfg: MetaConfig) -> None:
    components = ordered_metadata_components(meta_cfg)
    names = {component.name for component in components}

    meta_cfg.use_geolocation = "SPATIAL" in names
    meta_cfg.use_temporal = "TEMPORAL" in names
    meta_cfg.use_elevation = "ELEVATION" in names

    temporal = next((component for component in components if component.name == "TEMPORAL"), None)
    if temporal is not None:
        meta_cfg.temporal_use_julian_day = temporal.temporal_basis == "day_of_year"
        meta_cfg.temporal_use_hour = temporal.temporal_include_hour

    elevation = next((component for component in components if component.name == "ELEVATION"), None)
    if elevation is not None and elevation.scales:
        meta_cfg.elevation_scales = [float(scale) for scale in elevation.scales]


def describe_metadata_component(component: MetadataComponentConfig) -> str:
    if component.encoding_kind == "latlon_unit_sphere":
        return "lat/lon -> 3-dim unit sphere vector"
    if component.encoding_kind == "temporal_sinusoids":
        basis = component.temporal_basis or "time"
        description = f"{basis} -> cyclical (sin/cos) features"
        if component.temporal_include_hour:
            description += " + hour_of_day"
        return description
    if component.encoding_kind == "elevation_sinusoids":
        if component.scales:
            return f"elevation_m with scales {component.scales} -> multi-scale sin/cos features"
        return "elevation_m -> sinusoidal features"
    if component.source:
        return f"pre-encoded vector supplied for source '{component.source}'"
    return "pre-encoded vector supplied directly by component name"
