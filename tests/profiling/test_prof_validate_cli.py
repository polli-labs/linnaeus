"""Contract tests for `linnaeus-prof validate`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from linnaeus.profiling import cli, validator


def _write_fixture_files(tmp_path: Path, *, cfg_body: str, trial_payload: dict[str, object]) -> tuple[Path, Path, Path]:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(cfg_body)

    trials_path = tmp_path / "trials.jsonl"
    trials_path.write_text(json.dumps(trial_payload) + "\n")

    compose_path = tmp_path / "docker-compose.template.yml"
    compose_path.write_text(
        "services:\n"
        "  linnaeus-training:\n"
        "    image: linnaeus:test\n"
        "    command: python -m linnaeus.main --cfg {{CONFIG_FILE}}\n"
    )

    return cfg_path, trials_path, compose_path


def _fake_git_success(args, *, cwd):
    command = tuple(args)
    if command[:2] == ("remote", "get-url"):
        return validator.GitCommandResult(0, "git@github.com:polli-labs/linnaeus.git\n", "")
    if command[:2] == ("ls-remote", "--exit-code"):
        return validator.GitCommandResult(0, "0123456789abcdef\trefs/heads/main\n", "")
    if command[:2] == ("cat-file", "-e"):
        return validator.GitCommandResult(0, "", "")
    if command[:2] == ("rev-parse", "--show-toplevel"):
        return validator.GitCommandResult(0, str(cwd), "")
    return validator.GitCommandResult(0, "", "")


def test_validation_contract_valid_commit_pin(monkeypatch, tmp_path: Path):
    cfg_path, trials_path, compose_path = _write_fixture_files(
        tmp_path,
        cfg_body="EXPERIMENT:\n  NAME: unit-test\n",
        trial_payload={
            "name": "baseline",
            "config_file": str(tmp_path / "cfg.yaml"),
            "commit_hash": "abc1234",
        },
    )
    monkeypatch.setattr(validator, "_run_git_command", _fake_git_success)

    exit_code, report = validator.run_validation_contract(
        cfg=cfg_path,
        opts=None,
        trial_params_file=trials_path,
        compose_template=compose_path,
        dry_run=True,
        repo_root=tmp_path,
    )

    payload = report.to_dict()
    assert exit_code == validator.EXIT_CODE_VALID
    assert payload["status"] == "valid"
    assert payload["errors"] == []
    assert {"status", "errors", "warnings", "checked_paths", "checked_refs"} == set(payload.keys())
    assert any(item.startswith("baseline:commit:abc1234") for item in payload["checked_refs"])


def test_validation_contract_unknown_config_key_fails(monkeypatch, tmp_path: Path):
    cfg_path, trials_path, compose_path = _write_fixture_files(
        tmp_path,
        cfg_body="EXPERIMENT:\n  NAME: unit-test\nBROKEN:\n  UNKNOWN: true\n",
        trial_payload={
            "name": "baseline",
            "config_file": str(tmp_path / "cfg.yaml"),
            "commit_hash": "abc1234",
        },
    )
    monkeypatch.setattr(validator, "_run_git_command", _fake_git_success)

    exit_code, report = validator.run_validation_contract(
        cfg=cfg_path,
        opts=None,
        trial_params_file=trials_path,
        compose_template=compose_path,
        dry_run=True,
        repo_root=tmp_path,
    )

    assert exit_code == validator.EXIT_CODE_VALIDATION_FAILED
    assert report.status == "validation_failed"
    assert any("Config schema validation failed" in err for err in report.errors)


def test_validation_contract_missing_remote_ref_fails(monkeypatch, tmp_path: Path):
    cfg_path, trials_path, compose_path = _write_fixture_files(
        tmp_path,
        cfg_body="EXPERIMENT:\n  NAME: unit-test\n",
        trial_payload={
            "name": "baseline",
            "config_file": str(tmp_path / "cfg.yaml"),
            "git_ref": "feature/does-not-exist",
        },
    )

    def fake_git_missing_ref(args, *, cwd):
        command = tuple(args)
        if command[:2] == ("remote", "get-url"):
            return validator.GitCommandResult(0, "git@github.com:polli-labs/linnaeus.git\n", "")
        if command[:2] == ("ls-remote", "--exit-code"):
            return validator.GitCommandResult(2, "", "")
        return validator.GitCommandResult(0, "", "")

    monkeypatch.setattr(validator, "_run_git_command", fake_git_missing_ref)

    exit_code, report = validator.run_validation_contract(
        cfg=cfg_path,
        opts=None,
        trial_params_file=trials_path,
        compose_template=compose_path,
        dry_run=True,
        repo_root=tmp_path,
    )

    assert exit_code == validator.EXIT_CODE_VALIDATION_FAILED
    assert report.status == "validation_failed"
    assert any("was not found on remote `origin`" in err for err in report.errors)
    assert any("baseline:origin:feature/does-not-exist" == ref for ref in report.to_dict()["checked_refs"])


def test_validation_contract_usage_error_for_odd_opts(monkeypatch, tmp_path: Path):
    cfg_path, trials_path, compose_path = _write_fixture_files(
        tmp_path,
        cfg_body="EXPERIMENT:\n  NAME: unit-test\n",
        trial_payload={
            "name": "baseline",
            "config_file": str(tmp_path / "cfg.yaml"),
            "commit_hash": "abc1234",
        },
    )
    monkeypatch.setattr(validator, "_run_git_command", _fake_git_success)

    exit_code, report = validator.run_validation_contract(
        cfg=cfg_path,
        opts=["EXPERIMENT.NAME"],
        trial_params_file=trials_path,
        compose_template=compose_path,
        dry_run=True,
        repo_root=tmp_path,
    )

    assert exit_code == validator.EXIT_CODE_USAGE_ERROR
    assert report.status == "usage_error"
    assert any("KEY VALUE pairs" in err for err in report.errors)


def test_validation_contract_runtime_error_when_git_missing(monkeypatch, tmp_path: Path):
    cfg_path, trials_path, compose_path = _write_fixture_files(
        tmp_path,
        cfg_body="EXPERIMENT:\n  NAME: unit-test\n",
        trial_payload={
            "name": "baseline",
            "config_file": str(tmp_path / "cfg.yaml"),
            "git_ref": "main",
        },
    )

    def fake_git_missing(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(validator, "_run_git_command", fake_git_missing)

    exit_code, report = validator.run_validation_contract(
        cfg=cfg_path,
        opts=None,
        trial_params_file=trials_path,
        compose_template=compose_path,
        dry_run=True,
        repo_root=tmp_path,
    )

    assert exit_code == validator.EXIT_CODE_RUNTIME_FAILURE
    assert report.status == "runtime_error"
    assert any("git executable not found" in err for err in report.errors)


def test_cli_validate_json_contract(monkeypatch, tmp_path: Path, capsys):
    cfg_path, trials_path, compose_path = _write_fixture_files(
        tmp_path,
        cfg_body="EXPERIMENT:\n  NAME: unit-test\n",
        trial_payload={
            "name": "baseline",
            "config_file": str(tmp_path / "cfg.yaml"),
            "commit_hash": "abc1234",
        },
    )
    monkeypatch.setattr(validator, "_run_git_command", _fake_git_success)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "linnaeus-prof",
            "validate",
            "--cfg",
            str(cfg_path),
            "--trial-params-file",
            str(trials_path),
            "--compose-template",
            str(compose_path),
            "--dry-run",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == validator.EXIT_CODE_VALID
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "valid"
    assert payload["errors"] == []
    assert set(payload.keys()) == {"status", "errors", "warnings", "checked_paths", "checked_refs"}
