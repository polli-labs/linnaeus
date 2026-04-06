"""Shared config resolution and validation helpers for CLI preflight and runtime bootstrap."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, overload

_BACKBONE_PATCH_RE = re.compile(r"vit[a-z0-9]*?(\d{2})\b", re.IGNORECASE)


@dataclass
class ValidationReport:
    """Validation envelope returned to config preflight and runtime bootstrap."""

    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def normalize_opts(opts: list[str] | None) -> list[str]:
    """Normalize YACS-style CLI overrides and enforce key/value pairing."""

    normalized = opts or []
    if len(normalized) % 2 != 0:
        raise ValueError("--opts must contain key/value pairs")
    return normalized


def _infer_config_dir(abs_cfg_path: Path) -> str | None:
    """Infer CONFIG_DIR from the nearest ``.../configs/...`` ancestor."""

    parts = abs_cfg_path.parts
    for idx, part in enumerate(parts):
        if part == "configs":
            return str(Path(*parts[: idx + 1]))
    return None


@contextmanager
def temporary_config_dir(config_path: str) -> Iterator[None]:
    """Temporarily infer CONFIG_DIR from a config path when the caller did not set one."""

    previous = os.environ.get("CONFIG_DIR")
    inferred = _infer_config_dir(Path(config_path).resolve())
    if previous is None and inferred is not None:
        os.environ["CONFIG_DIR"] = inferred
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CONFIG_DIR", None)
        else:
            os.environ["CONFIG_DIR"] = previous


def resolve_config_bundle(cfg_path: str, opts: list[str]) -> Any:
    """Resolve config and return a sidecar bundle with provenance trace."""

    from linnaeus.config_resolution_trace import resolve_config_with_trace

    return resolve_config_with_trace(cfg_path=cfg_path, opts=opts)


@overload
def resolve_config(cfg_path: str, opts: list[str], *, trace: Literal[False] = False) -> Any: ...


@overload
def resolve_config(cfg_path: str, opts: list[str], *, trace: Literal[True]) -> Any: ...


def resolve_config(cfg_path: str, opts: list[str], *, trace: bool = False) -> Any:
    """Resolve defaults -> experiment config -> MODEL.BASE -> CLI overrides."""

    if trace:
        return resolve_config_bundle(cfg_path=cfg_path, opts=opts)

    from linnaeus.config import get_default_config
    from linnaeus.utils.config_utils import load_config, load_model_base_config

    config = get_default_config()
    exp_config = load_config(cfg_path)
    config.merge_from_other_cfg(exp_config)
    config = load_model_base_config(config)
    if opts:
        config.defrost()
        config.merge_from_list(opts)
        config.freeze()
    return config


def validate_dinov3_consistency(config: Any) -> ValidationReport:
    """Validate lightweight DINOv3-specific config consistency checks."""

    errors: list[str] = []
    warnings: list[str] = []

    dinov3 = getattr(config.MODEL, "DINOV3", None)
    if dinov3 is None:
        return ValidationReport(errors=errors, warnings=warnings)

    img_size = int(getattr(config.DATA, "IMG_SIZE", getattr(config.MODEL, "IMG_SIZE", 0)))
    patch_size = int(getattr(dinov3, "PATCH_SIZE", 0))
    if img_size > 0 and patch_size > 0 and img_size % patch_size != 0:
        errors.append(
            f"DATA.IMG_SIZE ({img_size}) must be divisible by MODEL.DINOV3.PATCH_SIZE ({patch_size})."
        )

    backbone_id = str(getattr(dinov3, "BACKBONE_ID", "")).strip()
    if backbone_id:
        match = _BACKBONE_PATCH_RE.search(backbone_id)
        if match:
            inferred_patch = int(match.group(1))
            if patch_size > 0 and inferred_patch != patch_size:
                warnings.append(
                    "MODEL.DINOV3.BACKBONE_ID appears to encode patch size "
                    f"{inferred_patch}, but MODEL.DINOV3.PATCH_SIZE is {patch_size}. "
                    "Using warning severity because some templates intentionally override patch settings."
                )
        else:
            warnings.append(
                "Could not infer DINOv3 patch size from MODEL.DINOV3.BACKBONE_ID; "
                "patch/backbone consistency check was skipped."
            )

    return ValidationReport(errors=errors, warnings=warnings)


def validate_parameter_selection_config(config: Any) -> ValidationReport:
    """Validate the lightweight schema for TRAIN.PARAMETER_SELECTION rules."""

    from linnaeus.utils.param_filters import collect_filter_config_errors

    errors: list[str] = []
    warnings: list[str] = []

    selection_cfg = getattr(getattr(config, "TRAIN", None), "PARAMETER_SELECTION", None)
    if selection_cfg is None or not getattr(selection_cfg, "ENABLED", False):
        return ValidationReport(errors=errors, warnings=warnings)

    rules_node = selection_cfg.get("RULES", None) if hasattr(selection_cfg, "get") else None
    if rules_node is None or not hasattr(rules_node, "items") or len(list(rules_node.items())) == 0:
        errors.append("TRAIN.PARAMETER_SELECTION.ENABLED=True requires TRAIN.PARAMETER_SELECTION.RULES")
        return ValidationReport(errors=errors, warnings=warnings)

    for rule_name, rule_cfg in rules_node.items():
        rule_path = f"TRAIN.PARAMETER_SELECTION.RULES.{rule_name}"
        if not hasattr(rule_cfg, "get"):
            errors.append(f"{rule_path} must be a mapping")
            continue

        filter_cfg = rule_cfg.get("FILTER", None)
        if not filter_cfg:
            errors.append(f"{rule_path}.FILTER is required")
            continue

        errors.extend(collect_filter_config_errors(filter_cfg, path=f"{rule_path}.FILTER"))

    return ValidationReport(errors=errors, warnings=warnings)


def validate_resolved_config(config: Any, *, runtime_contract_policy: str = "error") -> ValidationReport:
    """Run shared config validation used by both CLI preflight and runtime bootstrap."""

    from pydantic import ValidationError

    from linnaeus.config_schema import (
        DINOv3Config,
        EnvironmentConfig,
        ExperimentConfig,
        LossShapeValidator,
        MetricsConfig,
        ModelIdentityConfig,
        format_pydantic_errors,
    )
    from linnaeus.utils.bbox_config_validation import validate_bbox_key_consistency
    from linnaeus.utils.runtime_contract_checks import collect_runtime_contract_findings
    from linnaeus.utils.schedule_utils import validate_schedule_config
    from linnaeus.utils.training_consistency import validate_gradnorm_config, validate_loss_config

    errors: list[str] = []
    warnings: list[str] = []

    try:
        validate_bbox_key_consistency(config)
    except ValueError as exc:
        errors.append(str(exc))

    try:
        ExperimentConfig.from_cfg_node(config.EXPERIMENT).validate_resolved()
    except (ValidationError, ValueError) as exc:
        errors.extend(format_pydantic_errors("EXPERIMENT", exc))

    try:
        ModelIdentityConfig.from_cfg_node(
            {
                "TYPE": getattr(config.MODEL, "TYPE", ""),
                "NAME": getattr(config.MODEL, "NAME", ""),
            }
        )
    except (ValidationError, ValueError) as exc:
        errors.extend(format_pydantic_errors("MODEL", exc))

    try:
        dinov3_config = DINOv3Config.from_cfg_node(config.MODEL.DINOV3)
        warnings.extend(dinov3_config.collect_warnings(model_type=str(getattr(config.MODEL, "TYPE", ""))))
    except (ValidationError, ValueError) as exc:
        errors.extend(format_pydantic_errors("MODEL.DINOV3", exc))

    try:
        EnvironmentConfig.from_cfg_node(config.ENV)
    except (ValidationError, ValueError) as exc:
        errors.extend(format_pydantic_errors("ENV", exc))

    try:
        MetricsConfig.from_cfg_node(config.METRICS)
    except (ValidationError, ValueError) as exc:
        errors.extend(format_pydantic_errors("METRICS", exc))

    try:
        # Guard chained attribute access to avoid AttributeError when
        # intermediate sections are missing (e.g. no LOSS.TASK_SPECIFIC).
        _loss = getattr(config, "LOSS", None)
        _ts = getattr(_loss, "TASK_SPECIFIC", None) if _loss else None
        _ts_train = getattr(_ts, "TRAIN", None) if _ts else None
        _ts_val = getattr(_ts, "VAL", None) if _ts else None
        _tax = getattr(_loss, "TAXONOMY_SMOOTHING", None) if _loss else None

        LossShapeValidator.from_cfg_node(
            {
                "DATA": {
                    "TASK_KEYS_H5": list(getattr(config.DATA, "TASK_KEYS_H5", [])),
                },
                "LOSS": {
                    "TASK_SPECIFIC": {
                        "TRAIN": {
                            "FUNCS": list(getattr(_ts_train, "FUNCS", [])) if _ts_train else [],
                        },
                        "VAL": {
                            "FUNCS": list(getattr(_ts_val, "FUNCS", [])) if _ts_val else [],
                        },
                    },
                    "TAXONOMY_SMOOTHING": {
                        "ENABLED": list(getattr(_tax, "ENABLED", [])) if _tax else [],
                    },
                },
            }
        )
    except (ValidationError, ValueError) as exc:
        errors.extend(format_pydantic_errors("", exc))

    schedule_errors, schedule_warnings = validate_schedule_config(config)
    errors.extend(schedule_errors)
    warnings.extend(schedule_warnings)

    errors.extend(validate_gradnorm_config(config))
    errors.extend(validate_loss_config(config))

    dinov3_report = validate_dinov3_consistency(config)
    errors.extend(dinov3_report.errors)
    warnings.extend(dinov3_report.warnings)

    parameter_selection_report = validate_parameter_selection_config(config)
    errors.extend(parameter_selection_report.errors)
    warnings.extend(parameter_selection_report.warnings)

    runtime_findings = collect_runtime_contract_findings(config, policy=runtime_contract_policy)
    for finding in runtime_findings:
        formatted = f"{finding.code}: {finding.message}"
        if finding.remediation:
            formatted += f" Remediation: {finding.remediation}"
        if finding.severity == "error":
            errors.append(formatted)
        else:
            warnings.append(formatted)

    return ValidationReport(errors=errors, warnings=warnings)


def format_validation_failures(report: ValidationReport) -> str:
    """Render shared validation failures for CLI/runtime exception messages."""

    lines = ["Resolved configuration validation failed:"]
    for error in report.errors:
        lines.append(f"- {error}")
    return "\n".join(lines)
