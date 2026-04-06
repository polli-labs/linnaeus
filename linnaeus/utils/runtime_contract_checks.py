"""Cross-layer config-to-runtime contract checks.

These checks are intentionally lightweight and side-effect free. They verify
that critical configuration toggles have observable runtime consumers before we
spend GPU time on a launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from yacs.config import CfgNode as CN

RuntimeContractPolicy = Literal["warn", "error"]
DEPRECATED_KEY_MAP = {
    "MODEL.FIND_UNUSED_PARAMETERS": "DISTRIBUTED.DDP.find_unused_parameters",
}


@dataclass(frozen=True)
class RuntimeContractFinding:
    severity: str
    code: str
    message: str
    remediation: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.remediation:
            payload["remediation"] = self.remediation
        return payload


def _severity_for(policy: RuntimeContractPolicy) -> str:
    return "warning" if policy == "warn" else "error"


def _normalize_policy(policy: str) -> RuntimeContractPolicy:
    return "warn" if policy == "warn" else "error"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _file_contains_tokens(path: Path, tokens: list[str]) -> tuple[bool, list[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False, list(tokens)
    missing = [token for token in tokens if token not in text]
    return len(missing) == 0, missing


def _has_config_path(config: CN, dotted_path: str) -> bool:
    node: object = config
    for part in dotted_path.split("."):
        if not hasattr(node, part):
            return False
        node = getattr(node, part)
    return True


def _task_level(task_key: str) -> int | None:
    if not task_key.startswith("taxa_L"):
        return None
    suffix = task_key.removeprefix("taxa_L")
    if not suffix.isdigit():
        return None
    return int(suffix)


def _is_default_fine_to_coarse_order(task_keys: list[str]) -> bool:
    levels = [_task_level(task_key) for task_key in task_keys]
    if not levels or any(level is None for level in levels):
        return False
    normalized_levels = [level for level in levels if level is not None]
    return normalized_levels == sorted(normalized_levels)


def _check_mask_pooling_bbox_runtime(config: CN, *, policy: RuntimeContractPolicy) -> list[RuntimeContractFinding]:
    mask_cfg = config.MODEL.MASK_POOLING
    if not (bool(mask_cfg.ENABLED) and bool(mask_cfg.get("USE_BBOX_IF_AVAILABLE", False))):
        return []

    bbox_key = str(mask_cfg.get("BBOX_KEY", "")).strip()
    valid_key = str(mask_cfg.get("BBOX_VALID_KEY", "")).strip()
    if not bbox_key or not valid_key:
        return [
            RuntimeContractFinding(
                severity=_severity_for(policy),
                code="MASK_POOLING_BBOX_KEYS_MISSING",
                message=(
                    "MODEL.MASK_POOLING.USE_BBOX_IF_AVAILABLE=True but bbox key fields are empty "
                    f"(BBOX_KEY='{bbox_key}', BBOX_VALID_KEY='{valid_key}')."
                ),
                remediation="Set non-empty MODEL.MASK_POOLING.BBOX_KEY and MODEL.MASK_POOLING.BBOX_VALID_KEY.",
            )
        ]

    findings: list[RuntimeContractFinding] = []
    pad_fraction = float(mask_cfg.get("BBOX_PAD_FRACTION", 0.0))
    pad_max_fraction = float(mask_cfg.get("BBOX_PAD_MAX_FRACTION", -1.0))
    if pad_fraction < 0.0:
        findings.append(
            RuntimeContractFinding(
                severity=_severity_for(policy),
                code="MASK_POOLING_BBOX_PAD_FRACTION_INVALID",
                message=(
                    "MODEL.MASK_POOLING.BBOX_PAD_FRACTION must be >= 0.0; "
                    f"got {pad_fraction}."
                ),
                remediation="Set MODEL.MASK_POOLING.BBOX_PAD_FRACTION to a non-negative value (typically 0.0, 0.05, 0.10).",
            )
        )
    if pad_max_fraction < 0.0 and pad_max_fraction != -1.0:
        findings.append(
            RuntimeContractFinding(
                severity=_severity_for(policy),
                code="MASK_POOLING_BBOX_PAD_MAX_FRACTION_INVALID",
                message=(
                    "MODEL.MASK_POOLING.BBOX_PAD_MAX_FRACTION must be -1.0 (disabled cap) "
                    f"or >= 0.0; got {pad_max_fraction}."
                ),
                remediation="Use -1.0 to disable pad cap, or set a non-negative max fraction.",
            )
        )

    root = _repo_root()
    contract_checks = [
        (
            root / "linnaeus" / "runtime_core" / "forward.py",
            ["resolve_mask_pooling_weights(", "mask_weights=mask_weights"],
            "runtime-core forward wiring",
        ),
        (
            root / "linnaeus" / "models" / "dinov3_vnext.py",
            ["bbox_xywh_norm_to_patch_mask", "weights.shape[-1] == 4"],
            "DINOv3 runtime bbox->patch conversion",
        ),
        (
            root / "linnaeus" / "h5data" / "prefetching_hybrid_dataset.py",
            ["resolve_bbox_keys(", "targets[bbox_key]"],
            "hybrid dataloader bbox target attachment",
        ),
        (
            root / "linnaeus" / "h5data" / "prefetching_h5_dataset.py",
            ["resolve_bbox_keys(", "targets[bbox_key]"],
            "h5 dataloader bbox target attachment",
        ),
    ]

    missing_signals: list[str] = []
    for path, tokens, label in contract_checks:
        ok, missing_tokens = _file_contains_tokens(path, tokens)
        if ok:
            continue
        missing_desc = ", ".join(missing_tokens)
        missing_signals.append(f"{label} ({path}): missing [{missing_desc}]")

    if missing_signals:
        findings.append(
            RuntimeContractFinding(
                severity=_severity_for(policy),
                code="MASK_POOLING_BBOX_RUNTIME_SIGNAL_MISSING",
                message=(
                    "Mask-pooling bbox mode is enabled, but one or more runtime-consumer signals are missing: "
                    + "; ".join(missing_signals)
                ),
                remediation=(
                    "Wire bbox mask-pooling end-to-end: dataloader targets -> runtime-core "
                    "mask_weights forwarding -> DINOv3 bbox mask conversion."
                ),
            )
        )
    return findings


def _check_dinov3_stub_backbone(config: CN, *, policy: RuntimeContractPolicy) -> list[RuntimeContractFinding]:
    """Flag when DINOv3 stub backbone is active — features will be random."""
    model_type = str(getattr(config.MODEL, "TYPE", ""))
    if "DINOv3" not in model_type and "dinov3" not in model_type.lower():
        return []

    dinov3_cfg = getattr(config.MODEL, "DINOV3", None)
    if dinov3_cfg is None:
        return []

    use_stub = bool(getattr(dinov3_cfg, "USE_STUB", False))
    if not use_stub:
        return []

    return [
        RuntimeContractFinding(
            severity=_severity_for(policy),
            code="DINOV3_STUB_BACKBONE_ACTIVE",
            message=(
                "MODEL.DINOV3.USE_STUB=True: the DINOv3 backbone will produce random features. "
                "Training and evaluation results are NOT meaningful."
            ),
            remediation="Set MODEL.DINOV3.USE_STUB=False for real training/evaluation. The stub exists only for CI test fixtures.",
        )
    ]


def _check_warmup_exceeds_training(config: CN, *, policy: RuntimeContractPolicy) -> list[RuntimeContractFinding]:
    """Warn when warmup epochs exceed total training epochs."""
    warmup_epochs = float(getattr(config.LR_SCHEDULER, "WARMUP_EPOCHS", 0))
    total_epochs = int(getattr(config.TRAIN, "EPOCHS", 1))

    if warmup_epochs <= 0 or total_epochs <= 0:
        return []

    if warmup_epochs >= total_epochs:
        return [
            RuntimeContractFinding(
                severity="warning",  # Always warning, not error — could be intentional for profiling
                code="WARMUP_EXCEEDS_TRAINING_EPOCHS",
                message=(
                    f"LR_SCHEDULER.WARMUP_EPOCHS ({warmup_epochs}) >= TRAIN.EPOCHS ({total_epochs}). "
                    "The learning rate will never leave the warmup phase. "
                    "Training signal will be minimal."
                ),
                remediation=f"Either increase TRAIN.EPOCHS above {warmup_epochs} or reduce WARMUP_EPOCHS.",
            )
        ]
    return []


def _check_taxonomy_smoothing_loss_coupling(
    config: CN, *, policy: RuntimeContractPolicy
) -> list[RuntimeContractFinding]:
    task_keys = list(getattr(config.DATA, "TASK_KEYS_H5", []))
    train_funcs = list(getattr(config.LOSS.TASK_SPECIFIC.TRAIN, "FUNCS", []))
    enabled_flags = list(getattr(config.LOSS.TAXONOMY_SMOOTHING, "ENABLED", []))

    findings: list[RuntimeContractFinding] = []
    for idx, task_key in enumerate(task_keys):
        if idx >= len(train_funcs) or idx >= len(enabled_flags):
            continue

        func_name = str(train_funcs[idx]).strip()
        smoothing_enabled = bool(enabled_flags[idx])
        if func_name == "TaxonomyAwareLabelSmoothing" and not smoothing_enabled:
            findings.append(
                RuntimeContractFinding(
                    severity=_severity_for(policy),
                    code="TAXONOMY_SMOOTHING_LOSS_REQUIRES_ENABLED",
                    message=(
                        f"LOSS.TASK_SPECIFIC.TRAIN.FUNCS[{idx}] for task '{task_key}' is "
                        "TaxonomyAwareLabelSmoothing, but "
                        f"LOSS.TAXONOMY_SMOOTHING.ENABLED[{idx}] is False. "
                        "The loss expects a taxonomy smoothing matrix that will not be generated."
                    ),
                    remediation=(
                        f"Either set LOSS.TAXONOMY_SMOOTHING.ENABLED[{idx}]=True for '{task_key}', "
                        "or switch the loss back to CrossEntropyLoss."
                    ),
                )
            )
        elif smoothing_enabled and func_name == "CrossEntropyLoss":
            findings.append(
                RuntimeContractFinding(
                    severity="warning",
                    code="TAXONOMY_SMOOTHING_ENABLED_WITH_CROSS_ENTROPY",
                    message=(
                        f"LOSS.TAXONOMY_SMOOTHING.ENABLED[{idx}] is True for task '{task_key}', "
                        f"but LOSS.TASK_SPECIFIC.TRAIN.FUNCS[{idx}] is CrossEntropyLoss. "
                        "Taxonomy smoothing matrices will be computed but unused."
                    ),
                    remediation=(
                        f"Either disable LOSS.TAXONOMY_SMOOTHING.ENABLED[{idx}] for '{task_key}', "
                        "or switch the loss to TaxonomyAwareLabelSmoothing."
                    ),
                )
            )
    return findings


def _check_deprecated_key_aliases(config: CN, *, policy: RuntimeContractPolicy) -> list[RuntimeContractFinding]:
    findings: list[RuntimeContractFinding] = []
    for deprecated_key, replacement_key in DEPRECATED_KEY_MAP.items():
        if not _has_config_path(config, deprecated_key):
            continue
        findings.append(
            RuntimeContractFinding(
                severity=_severity_for(policy),
                code="DEPRECATED_CONFIG_KEY_ALIAS",
                message=(
                    f"Deprecated config key '{deprecated_key}' is set. "
                    f"Use '{replacement_key}' instead."
                ),
                remediation=f"Rename '{deprecated_key}' to '{replacement_key}' and remove the deprecated alias.",
            )
        )
    return findings


def _check_missingness_sweep_cost(config: CN, *, policy: RuntimeContractPolicy) -> list[RuntimeContractFinding]:
    sweep_cfg = getattr(config.VAL, "MISSINGNESS_SWEEP", None)
    if sweep_cfg is None or not bool(getattr(sweep_cfg, "ENABLED", False)):
        return []

    interval_epochs = int(getattr(sweep_cfg, "INTERVAL_EPOCHS", 0))
    total_epochs = int(getattr(config.TRAIN, "EPOCHS", 1))
    if interval_epochs != 1 or total_epochs <= 3:
        return []

    patterns = list(getattr(sweep_cfg, "PATTERNS", []))
    extra_passes = len(patterns)
    return [
        RuntimeContractFinding(
            severity="warning",
            code="MISSINGNESS_SWEEP_EVERY_EPOCH_COSTLY",
            message=(
                "VAL.MISSINGNESS_SWEEP.ENABLED=True with INTERVAL_EPOCHS=1 and "
                f"TRAIN.EPOCHS={total_epochs}. Running the full missingness sweep every epoch adds "
                f"~{extra_passes} extra validation passes per epoch."
            ),
            remediation=(
                "Consider setting VAL.MISSINGNESS_SWEEP.INTERVAL_EPOCHS to match TRAIN.EPOCHS "
                "for a final-epoch-only sweep."
            ),
        )
    ]


def _check_init_weights_length(config: CN, *, policy: RuntimeContractPolicy) -> list[RuntimeContractFinding]:
    task_keys = list(getattr(config.DATA, "TASK_KEYS_H5", []))
    init_weights = list(getattr(config.LOSS.GRAD_WEIGHTING.TASK, "INIT_WEIGHTS", []))
    if not init_weights:
        return []

    findings: list[RuntimeContractFinding] = []
    if len(init_weights) != len(task_keys):
        findings.append(
            RuntimeContractFinding(
                severity=_severity_for(policy),
                code="INIT_WEIGHTS_LENGTH_MISMATCH",
                message=(
                    "LOSS.GRAD_WEIGHTING.TASK.INIT_WEIGHTS length "
                    f"({len(init_weights)}) does not match DATA.TASK_KEYS_H5 length ({len(task_keys)})."
                ),
                remediation="Provide one INIT_WEIGHTS entry per task key, in DATA.TASK_KEYS_H5 order.",
            )
        )
        return findings

    try:
        normalized_weights = [float(weight) for weight in init_weights]
    except (TypeError, ValueError):
        return findings

    if (
        len(normalized_weights) > 1
        and task_keys
        and task_keys[0] == "taxa_L10"
        and _is_default_fine_to_coarse_order(task_keys)
        and all(left >= right for left, right in zip(normalized_weights, normalized_weights[1:], strict=False))
        and normalized_weights[0] > normalized_weights[-1]
    ):
        findings.append(
            RuntimeContractFinding(
                severity="info",
                code="INIT_WEIGHTS_FINE_HEAVY_RECOMMENDED",
                message=(
                    "INIT_WEIGHTS gives highest weight to L10 (species); "
                    "this is the recommended fine-heavy configuration."
                ),
            )
        )
    return findings


def collect_runtime_contract_findings(
    config: CN,
    *,
    policy: str = "error",
) -> list[RuntimeContractFinding]:
    """Return runtime contract findings for critical feature toggles."""
    normalized_policy = _normalize_policy(policy)
    findings: list[RuntimeContractFinding] = []
    findings.extend(_check_mask_pooling_bbox_runtime(config, policy=normalized_policy))
    findings.extend(_check_dinov3_stub_backbone(config, policy=normalized_policy))
    findings.extend(_check_warmup_exceeds_training(config, policy=normalized_policy))
    findings.extend(_check_taxonomy_smoothing_loss_coupling(config, policy=normalized_policy))
    findings.extend(_check_deprecated_key_aliases(config, policy=normalized_policy))
    findings.extend(_check_missingness_sweep_cost(config, policy=normalized_policy))
    findings.extend(_check_init_weights_length(config, policy=normalized_policy))
    return findings
