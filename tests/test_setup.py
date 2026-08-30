from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_package_and_local_web_sources_are_present() -> None:
    assert (REPOSITORY_ROOT / "game" / "src" / "engine.ts").is_file()
    assert (REPOSITORY_ROOT / "game" / "dist" / "batch.html").is_file()


def test_system_command_reports_the_selected_controller_accelerator() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_system.py", "--json"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    information = json.loads(result.stdout)
    assert information["operating_system"]
    assert information["python_version"]
    assert information["logical_cpu_count"]
    assert information["controller_compute"]
    assert information["controller_compute_diagnostic"] is None or isinstance(
        information["controller_compute_diagnostic"], str
    )
    assert information["controller_compute"].startswith("cpu") or information[
        "controller_compute"
    ].startswith("cuda:0")


def test_public_runtime_has_no_superseded_sarsa_scope_and_cuda_is_optional() -> None:
    public_paths = (
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "pyproject.toml",
        REPOSITORY_ROOT / "src",
        REPOSITORY_ROOT / "scripts",
        REPOSITORY_ROOT / "game",
    )
    sources: list[str] = []
    for path in public_paths:
        files = path.rglob("*") if path.is_dir() else (path,)
        for file in files:
            if (
                file.is_file()
                and file.suffix in {".md", ".py", ".toml", ".ts", ".html", ".json"}
                and "node_modules" not in file.parts
                and "dist" not in file.parts
            ):
                sources.append(file.read_text(encoding="utf-8"))
    combined = "\n".join(sources).lower()
    assert "sarsa" not in combined
    assert re.search(r"\bcma(?:[-_ ]?es)?\b", combined) is None
    project = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "cuda = [" in project
    assert '"torch>=2.13,<2.14"' in project
    assert 'name = "pytorch-cu132"' in project
    assert '"cma' not in project
