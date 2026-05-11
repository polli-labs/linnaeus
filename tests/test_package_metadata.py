from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path


def test_public_scripts_do_not_advertise_missing_root_cli() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    scripts = pyproject["project"]["scripts"]

    assert "linnaeus" not in scripts
    assert scripts["linnaeus-prof"] == "linnaeus.profiling.cli:main"
    assert scripts["linnaeus-prof-run"] == "linnaeus.tools.profiling.run_profiling_trials:main"
    assert importlib.util.find_spec("linnaeus.cli") is None
