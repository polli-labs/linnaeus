#!/usr/bin/env python3
"""
Canonical local/CI quality gate scaffold for linnaeus-dev.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_PYTEST_TARGETS = [
    "tests/test_bbox_config_validation.py",
    "tests/test_bbox_lane_smoke.py",
    "tests/test_bbox_observability_metrics.py",
    "tests/test_bbox_metrics_contract_smoke.py",
]

DEFAULT_TYPECHECK_TARGETS = [
    "linnaeus/utils/metrics/bbox_observability.py",
    "tools/smoke/bbox_metrics_contract_smoke.py",
    "tools/quality_gate.py",
]


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"+ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run linnaeus-dev baseline quality gate.")
    parser.add_argument("--skip-lint", action="store_true", help="Skip ruff lint check")
    parser.add_argument("--skip-typecheck", action="store_true", help="Skip ty typecheck")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest stage")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip deterministic bbox metrics smoke stage")
    parser.add_argument(
        "--pytest-target",
        action="append",
        default=None,
        help="Pytest target path/pattern; can be passed multiple times. Defaults to bbox observability regression set.",
    )
    parser.add_argument(
        "--typecheck-target",
        action="append",
        default=None,
        help="ty target path; can be passed multiple times. Defaults to bbox observability gate modules.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    pytest_targets = args.pytest_target or list(DEFAULT_PYTEST_TARGETS)
    typecheck_targets = args.typecheck_target or list(DEFAULT_TYPECHECK_TARGETS)

    try:
        if not args.skip_lint:
            _run(
                [
                    "uv",
                    "run",
                    "--extra",
                    "dev",
                    "ruff",
                    "check",
                    "--select",
                    "E9,F63,F7,F82",
                    "linnaeus",
                    "tests",
                    "tools",
                ],
                cwd=repo_root,
            )

        if not args.skip_typecheck:
            _run(["uv", "run", "--extra", "dev", "ty", "check", *typecheck_targets], cwd=repo_root)

        if not args.skip_tests:
            _run(["uv", "run", "--extra", "dev", "pytest", "-q", *pytest_targets], cwd=repo_root)

        if not args.skip_smoke:
            _run(["uv", "run", "python", "tools/smoke/bbox_metrics_contract_smoke.py", "--json"], cwd=repo_root)
    except subprocess.CalledProcessError as exc:
        print(f"quality_gate failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode

    print("quality_gate completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
