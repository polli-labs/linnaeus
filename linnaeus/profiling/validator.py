"""Validation contract for profiling trial preflight.

This module powers the `linnaeus-prof validate` command and intentionally keeps
validation side-effect-free.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from linnaeus.config import get_default_config

EXIT_CODE_VALID = 0
EXIT_CODE_USAGE_ERROR = 2
EXIT_CODE_VALIDATION_FAILED = 3
EXIT_CODE_RUNTIME_FAILURE = 4

DOCKER_SERVICE_NAME = "linnaeus-training"
_COMMIT_HASH_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
_CONTAINER_CONFIG_PREFIX = "/configs/"
_ALLOWED_TRIAL_KEYS = {
    "name",
    "git_ref",
    "commit_hash",
    "config_file",
    "opts",
    "env_yaml",
    "env",
    "extra_deps",
    "docker_tag",
    "gpu_rank",
}


@dataclass
class ValidationReport:
    """Serializable report emitted by `linnaeus-prof validate`."""

    status: str = "valid"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_paths: list[str] = field(default_factory=list)
    checked_refs: list[str] = field(default_factory=list)

    def add_checked_path(self, path_like: Path | str) -> None:
        self.checked_paths.append(str(Path(path_like).expanduser().resolve()))

    def add_checked_ref(self, ref: str) -> None:
        self.checked_refs.append(str(ref))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
            "checked_paths": sorted(set(self.checked_paths)),
            "checked_refs": sorted(set(self.checked_refs)),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass
class GitCommandResult:
    returncode: int
    stdout: str
    stderr: str


def _run_git_command(args: Sequence[str], *, cwd: Path) -> GitCommandResult:
    """Execute `git` and return captured output."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    return GitCommandResult(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def _normalize_opts(opts: Sequence[str] | None) -> tuple[list[str], list[str]]:
    if opts is None:
        return [], []
    normalized = [str(item) for item in opts]
    if len(normalized) % 2 != 0:
        return [], ["--opts must contain KEY VALUE pairs (received an odd number of tokens)."]
    return normalized, []


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _config_roots(repo_root: Path) -> list[Path]:
    roots: list[Path] = []
    env_roots = os.environ.get("LINNAEUS_PROF_CONFIG_ROOTS")
    if env_roots:
        for raw in env_roots.split(os.pathsep):
            raw = raw.strip()
            if raw:
                roots.append(Path(raw).expanduser())

    roots.extend(
        [
            Path("/home/caleb/repo/linnaeus-deployment/linnaeus_deploy/configs"),
            repo_root / "private" / "configs",
            repo_root / "configs",
        ]
    )
    deduped = _dedupe_preserve_order([str(root.resolve()) for root in roots])
    return [Path(item) for item in deduped]


def _candidate_host_paths(raw_path: str, *, repo_root: Path) -> list[Path]:
    path = Path(raw_path).expanduser()
    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)
        if raw_path.startswith(_CONTAINER_CONFIG_PREFIX):
            rel = raw_path[len(_CONTAINER_CONFIG_PREFIX):]
            for root in _config_roots(repo_root):
                candidates.append(Path(root) / rel)
    else:
        candidates.append(repo_root / path)

    deduped = _dedupe_preserve_order([str(candidate.resolve()) for candidate in candidates])
    return [Path(item) for item in deduped]


def _resolve_trial_path(raw_path: str, *, repo_root: Path) -> tuple[Path | None, list[Path]]:
    candidates = _candidate_host_paths(raw_path, repo_root=repo_root)
    for candidate in candidates:
        if candidate.exists():
            return candidate, candidates
    return None, candidates


def _validate_config_schema(cfg_path: Path, opts: Sequence[str], validation_errors: list[str]) -> None:
    config = get_default_config()
    config.defrost()

    try:
        config.merge_from_file(str(cfg_path))
    except Exception as exc:  # pragma: no cover - exact exception type varies with yacs internals
        validation_errors.append(f"Config schema validation failed for {cfg_path}: {exc}")
        return

    if opts:
        try:
            config.merge_from_list(list(opts))
        except Exception as exc:  # pragma: no cover - exact exception type varies with yacs internals
            validation_errors.append(f"Config option validation failed for --opts: {exc}")


def _load_trials(trial_params_path: Path, validation_errors: list[str]) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []

    with trial_params_path.open("r") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                validation_errors.append(f"Invalid JSON in trial params at line {line_no}: {exc.msg}.")
                continue

            if not isinstance(payload, dict):
                validation_errors.append(f"Trial entry at line {line_no} must be a JSON object.")
                continue

            trials.append(payload)

    if not trials:
        validation_errors.append(f"No trial definitions found in {trial_params_path}.")

    return trials


def _validate_trial_contract(
    trials: list[dict[str, Any]],
    *,
    repo_root: Path,
    report: ValidationReport,
    validation_errors: list[str],
) -> None:
    seen_names: set[str] = set()

    for index, trial in enumerate(trials, start=1):
        prefix = f"trial[{index}]"
        name = trial.get("name")
        if not isinstance(name, str) or not name.strip():
            validation_errors.append(f"{prefix}: missing required string field `name`.")
            continue

        name = name.strip()
        if name in seen_names:
            validation_errors.append(f"{prefix}: duplicate trial name `{name}`.")
        seen_names.add(name)

        unknown_keys = sorted(set(trial.keys()) - _ALLOWED_TRIAL_KEYS)
        if unknown_keys:
            report.warnings.append(f"{prefix} ({name}): unknown keys ignored: {', '.join(unknown_keys)}.")

        config_file = trial.get("config_file")
        if not isinstance(config_file, str) or not config_file.strip():
            validation_errors.append(f"{prefix} ({name}): missing required string field `config_file`.")
        else:
            resolved, candidates = _resolve_trial_path(config_file.strip(), repo_root=repo_root)
            for candidate in candidates:
                report.add_checked_path(candidate)
            if resolved is None:
                candidate_text = ", ".join(str(candidate) for candidate in candidates)
                validation_errors.append(
                    f"{prefix} ({name}): config_file `{config_file}` was not found. Tried: {candidate_text}."
                )

        env_yaml = trial.get("env_yaml")
        if env_yaml is not None:
            if not isinstance(env_yaml, str) or not env_yaml.strip():
                validation_errors.append(f"{prefix} ({name}): `env_yaml` must be a non-empty string when provided.")
            else:
                resolved, candidates = _resolve_trial_path(env_yaml.strip(), repo_root=repo_root)
                for candidate in candidates:
                    report.add_checked_path(candidate)
                if resolved is None:
                    candidate_text = ", ".join(str(candidate) for candidate in candidates)
                    validation_errors.append(
                        f"{prefix} ({name}): env_yaml `{env_yaml}` was not found. Tried: {candidate_text}."
                    )

        opts = trial.get("opts")
        if opts is not None:
            if not isinstance(opts, list):
                validation_errors.append(f"{prefix} ({name}): `opts` must be a list of KEY VALUE tokens.")
            elif len(opts) % 2 != 0:
                validation_errors.append(f"{prefix} ({name}): `opts` must contain KEY VALUE pairs.")

        env = trial.get("env")
        if env is not None and not isinstance(env, dict):
            validation_errors.append(f"{prefix} ({name}): `env` must be an object/dict when provided.")

        git_ref = trial.get("git_ref")
        commit_hash = trial.get("commit_hash")
        if not commit_hash and not git_ref:
            validation_errors.append(f"{prefix} ({name}): provide either `git_ref` or `commit_hash`.")


def _load_compose_template(compose_template_path: Path, validation_errors: list[str]) -> dict[str, Any] | None:
    try:
        data = yaml.safe_load(compose_template_path.read_text())
    except Exception as exc:
        validation_errors.append(f"Failed to parse compose template {compose_template_path}: {exc}")
        return None

    if not isinstance(data, dict):
        validation_errors.append(f"Compose template {compose_template_path} must parse to a YAML object.")
        return None

    services = data.get("services")
    if not isinstance(services, dict):
        validation_errors.append(f"Compose template {compose_template_path} must contain a `services` object.")
        return None

    service = services.get(DOCKER_SERVICE_NAME)
    if not isinstance(service, dict):
        validation_errors.append(
            f"Compose template {compose_template_path} must define `services.{DOCKER_SERVICE_NAME}`."
        )
        return None

    if "command" not in service:
        validation_errors.append(
            f"Compose template {compose_template_path} is missing `services.{DOCKER_SERVICE_NAME}.command`."
        )

    return data


def _resolve_repo_root(cwd: Path) -> tuple[Path | None, str | None]:
    try:
        result = _run_git_command(["rev-parse", "--show-toplevel"], cwd=cwd)
    except FileNotFoundError:
        return None, "git executable not found while resolving repository root."

    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "unknown git error"
        return None, f"Unable to resolve git repository root: {stderr}."

    root = result.stdout.strip()
    if not root:
        return None, "git returned an empty repository root."

    return Path(root), None


def _preflight_refs(
    trials: list[dict[str, Any]],
    *,
    repo_root: Path,
    report: ValidationReport,
    validation_errors: list[str],
    runtime_errors: list[str],
) -> None:
    needs_remote_lookup = any(not trial.get("commit_hash") for trial in trials)
    if needs_remote_lookup:
        try:
            remote_result = _run_git_command(["remote", "get-url", "origin"], cwd=repo_root)
        except FileNotFoundError:
            runtime_errors.append("git executable not found while validating remote refs.")
            return
        if remote_result.returncode != 0:
            stderr = remote_result.stderr.strip() or remote_result.stdout.strip() or "unknown git error"
            runtime_errors.append(f"Unable to resolve git remote `origin`: {stderr}.")
            return

    for index, trial in enumerate(trials, start=1):
        name = str(trial.get("name", f"trial-{index}"))
        git_ref = trial.get("git_ref")
        commit_hash = trial.get("commit_hash")

        if commit_hash:
            commit_hash = str(commit_hash).strip()
            report.add_checked_ref(f"{name}:commit:{commit_hash}")

            if not _COMMIT_HASH_RE.match(commit_hash):
                validation_errors.append(
                    f"trial[{index}] ({name}): commit_hash `{commit_hash}` must be 7-40 hexadecimal characters."
                )
                continue

            try:
                commit_result = _run_git_command(["cat-file", "-e", f"{commit_hash}^{{commit}}"], cwd=repo_root)
            except FileNotFoundError:
                runtime_errors.append("git executable not found while validating commit hashes.")
                return

            if commit_result.returncode != 0:
                validation_errors.append(
                    f"trial[{index}] ({name}): commit_hash `{commit_hash}` is not resolvable in this repository."
                )
            continue

        if not isinstance(git_ref, str) or not git_ref.strip():
            validation_errors.append(f"trial[{index}] ({name}): missing `git_ref` for remote preflight.")
            continue

        git_ref = git_ref.strip()
        report.add_checked_ref(f"{name}:origin:{git_ref}")
        try:
            ref_result = _run_git_command(["ls-remote", "--exit-code", "origin", git_ref], cwd=repo_root)
        except FileNotFoundError:
            runtime_errors.append("git executable not found while validating remote refs.")
            return

        if ref_result.returncode == 0:
            refs = [line.strip() for line in ref_result.stdout.splitlines() if line.strip()]
            if not refs:
                validation_errors.append(
                    f"trial[{index}] ({name}): git_ref `{git_ref}` resolved with empty output from git ls-remote."
                )
            else:
                for line in refs:
                    sha = line.split("\t", 1)[0].strip()
                    if sha:
                        report.add_checked_ref(f"{name}:origin:{git_ref}@{sha}")
            continue

        if ref_result.returncode == 2:
            validation_errors.append(
                f"trial[{index}] ({name}): git_ref `{git_ref}` was not found on remote `origin`."
            )
            continue

        stderr = ref_result.stderr.strip() or ref_result.stdout.strip() or "unknown git error"
        runtime_errors.append(
            f"trial[{index}] ({name}): unable to query remote ref `{git_ref}` from `origin`: {stderr}."
        )
        return


def run_validation_contract(
    *,
    cfg: Path | None,
    opts: Sequence[str] | None,
    trial_params_file: Path | None,
    compose_template: Path | None,
    dry_run: bool,
    repo_root: Path | None = None,
) -> tuple[int, ValidationReport]:
    """Validate config/trial/template + provenance contract."""
    del dry_run  # Validation is always dry-run; we keep the flag for contract ergonomics.
    report = ValidationReport()

    usage_errors: list[str] = []
    validation_errors: list[str] = []
    runtime_errors: list[str] = []

    normalized_opts, opts_usage_errors = _normalize_opts(opts)
    usage_errors.extend(opts_usage_errors)

    if cfg is None:
        usage_errors.append("Missing required argument: --cfg.")
    if trial_params_file is None:
        usage_errors.append("Missing required argument: --trial-params-file.")
    if compose_template is None:
        usage_errors.append("Missing required argument: --compose-template.")

    if usage_errors:
        report.status = "usage_error"
        report.errors.extend(usage_errors)
        return EXIT_CODE_USAGE_ERROR, report

    assert cfg is not None
    assert trial_params_file is not None
    assert compose_template is not None

    report.add_checked_path(cfg)
    report.add_checked_path(trial_params_file)
    report.add_checked_path(compose_template)

    if not cfg.exists():
        validation_errors.append(f"Config file not found: {cfg}.")
    if not trial_params_file.exists():
        validation_errors.append(f"Trial params file not found: {trial_params_file}.")
    if not compose_template.exists():
        validation_errors.append(f"Compose template not found: {compose_template}.")

    resolved_repo_root = repo_root
    if resolved_repo_root is None:
        resolved_repo_root, repo_error = _resolve_repo_root(Path.cwd())
        if repo_error:
            runtime_errors.append(repo_error)
    if resolved_repo_root is not None:
        report.add_checked_path(resolved_repo_root)

    trials: list[dict[str, Any]] = []

    if cfg.exists():
        _validate_config_schema(cfg, normalized_opts, validation_errors)

    if compose_template.exists():
        _load_compose_template(compose_template, validation_errors)

    if trial_params_file.exists():
        trials = _load_trials(trial_params_file, validation_errors)

    if resolved_repo_root is not None and trials:
        _validate_trial_contract(
            trials,
            repo_root=resolved_repo_root,
            report=report,
            validation_errors=validation_errors,
        )

    if resolved_repo_root is not None and trials and not runtime_errors:
        _preflight_refs(
            trials,
            repo_root=resolved_repo_root,
            report=report,
            validation_errors=validation_errors,
            runtime_errors=runtime_errors,
        )

    if runtime_errors:
        report.status = "runtime_error"
        report.errors.extend(runtime_errors)
        report.errors.extend(validation_errors)
        return EXIT_CODE_RUNTIME_FAILURE, report

    if validation_errors:
        report.status = "validation_failed"
        report.errors.extend(validation_errors)
        return EXIT_CODE_VALIDATION_FAILED, report

    report.status = "valid"
    return EXIT_CODE_VALID, report
