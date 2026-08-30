from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path: str) -> str:
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


def test_browser_bridges_only_transport_pixels_and_canonical_actions() -> None:
    bridge = _source("game/src/batch_bridge.ts")
    assert "from './engine'" not in bridge
    assert "renderModel" not in bridge
    assert "getState" not in bridge
    assert "private_clean_candidate_frames" in bridge
    assert "getImageData" in bridge
    assert "zlib_xor_grayscale_u8" in bridge
    assert "CompressionStream" in bridge
    assert "action_batch" in bridge


def test_python_boundary_does_not_evaluate_or_read_browser_game_state() -> None:
    browser = _source("src/dino_er/browser.py")
    environment = _source("src/dino_er/environment.py")
    perception = _source("src/dino_er/perception.py")
    combined = browser + environment + perception
    assert ".evaluate(" not in browser
    assert "renderModel" not in combined
    assert "getState" not in combined
    assert "game/src/engine" not in combined
    assert 'observation_source": "current_rendered_frame_pixels_only"' in perception


def test_runtime_code_has_no_pygame_dependency() -> None:
    for relative_path in (
        "src",
        "scripts/evolve.py",
        "scripts/replay.py",
        "pyproject.toml",
    ):
        path = REPOSITORY_ROOT / relative_path
        files = path.rglob("*.py") if path.is_dir() else (path,)
        for file in files:
            assert "pygame" not in file.read_text(encoding="utf-8").lower()
