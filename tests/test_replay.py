from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dino_er.controllers import default_controller_spec
from scripts import replay


def test_latest_checkpoint_uses_newest_campaign_then_best_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / f"model_v002_{replay.model_hash()}"
    paths = [
        model_dir / "condition-a" / "old" / "reactive-checkpoint.pkl",
        model_dir / "condition-b" / "new-low" / "reactive-checkpoint.pkl",
        model_dir / "condition-c" / "new-high" / "reactive-checkpoint.pkl",
    ]
    for path in paths:
        path.parent.mkdir(parents=True)
        path.touch()
    metadata = {
        paths[0]: ("2026-08-28_12-00-00_CEST", 999.0, 7),
        paths[1]: ("2026-08-29_12-00-00_CEST", 400.0, 7),
        paths[2]: ("2026-08-29_12-00-00_CEST", 500.0, 17),
    }

    def fake_load(path: Path) -> tuple[np.ndarray, Any, dict[str, Any]]:
        campaign, fitness, seed = metadata[path]
        return (
            np.zeros(93),
            default_controller_spec("reactive", input_size=5),
            {
                "best_fitness": fitness,
                "config": {
                    "campaign_timestamp": campaign,
                    "study_id": "rq1-main",
                    "evolution_seed": seed,
                },
            },
        )

    monkeypatch.setattr(replay, "load_best_from_checkpoint", fake_load)

    selected, selected_metadata = replay._latest_checkpoint(
        controller_type="reactive",
        evolution_seed=None,
        study_id="rq1-main",
        model_version=2,
        artifacts_root=tmp_path,
    )
    assert selected == paths[2].resolve()
    assert selected_metadata["best_fitness"] == 500.0

    seeded, _ = replay._latest_checkpoint(
        controller_type="reactive",
        evolution_seed=7,
        study_id="rq1-main",
        model_version=2,
        artifacts_root=tmp_path,
    )
    assert seeded == paths[1].resolve()
