"""Focused regression tests for scientific viewer aggregations and figures."""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dino_er.scientific import (
    _mean_ci,
    _rq1_generation_fitness_figure,
    _rq1_heldout_figure,
    _rq2_controller_heatmaps,
    _rq3_strategy_figure,
    _write_summary_table,
)


def test_rq2_heatmaps_separate_controllers_on_one_performance_scale() -> None:
    cells = {
        ("reactive", 20, 0.1): [2800.0, 2900.0, 3000.0],
        ("proactive", 20, 0.1): [2100.0, 2200.0, 2300.0],
        ("reactive", 10, 0.3): [2700.0, 2800.0, 2900.0],
        ("proactive", 10, 0.3): [1400.0, 1500.0, 1600.0],
    }
    planned = {(20, 0.1), (10, 0.3)}
    figure = _rq2_controller_heatmaps(cells, planned)
    try:
        reactive_image = figure.axes[0].images[0]
        proactive_image = figure.axes[1].images[0]
        assert reactive_image.get_clim() == (0.0, 3600.0)
        assert proactive_image.get_clim() == (0.0, 3600.0)
        assert figure.axes[0].get_title() == "A  Reactive"
        assert figure.axes[1].get_title() == "B  Proactive"
        assert reactive_image.get_array().mask[0, 0]
        assert proactive_image.get_array().mask[0, 0]
    finally:
        figure.clf()


def test_learning_figure_uses_full_censored_range() -> None:
    rows = [
        {
            "controller_type": controller,
            "run_id": f"{controller}-{seed}",
            "generation": generation,
            "cumulative_candidate_evaluations": (generation + 1) * 100,
            "selected_parent_fitness_mean": 300 + seed + generation,
            "fitness_mean": 300 + seed + generation,
        }
        for controller in ("reactive", "proactive")
        for seed in (7, 17, 27)
        for generation in (0, 1)
    ]
    figure = _rq1_generation_fitness_figure(rows)
    try:
        assert figure.axes[0].get_ylim() == pytest.approx((0, 3600))
        assert len(figure.axes[0].lines) >= 4
    finally:
        figure.clf()


def test_heldout_pairs_architecture_specific_run_ids() -> None:
    validation = {
        (c, f"{c}-seed-{s}"): float(s + (10 if c == "proactive" else 0))
        for c in ("reactive", "proactive")
        for s in (7, 17, 27)
    }
    figure = _rq1_heldout_figure(
        {},
        validation,
        [
            {
                "optimizer_seed": s,
                "reactive_validation": float(s),
                "proactive_validation": float(s + 10),
                "proactive_minus_reactive": 10.0,
            }
            for s in (7, 17, 27)
        ],
    )
    try:
        assert sum(line.get_color() == "#999999" for line in figure.axes[0].lines) == 3
        assert len(figure.axes[1].collections) >= 1
    finally:
        figure.clf()


def test_table_writer_preserves_exact_rows(tmp_path: Path) -> None:
    rows = [
        {"world_seed": 7, "reactive_survival": 100, "proactive_survival": 200},
        {
            "world_seed": 17,
            "reactive_survival": 3600,
            "proactive_survival": 3500,
            "reactive_minus_proactive": 100,
        },
    ]
    path = _write_summary_table(tmp_path, "final_test_paired_data", rows)
    assert path is not None
    csv_text = (tmp_path / "final_test_paired_data.csv").read_text(encoding="utf-8")
    assert csv_text.count("\n") == 3
    assert "reactive_minus_proactive" in csv_text.splitlines()[0]
    assert "reactive_minus_proactive" in pq.read_schema(path).names


def test_rq3_strategy_figure_has_six_visual_intervention_categories() -> None:
    rows = []
    for controller in ("reactive", "proactive"):
        for intervention in (
            "normal",
            "constant_first_frame",
            *(f"ablate_sensor_{i}" for i in range(1, 6)),
        ):
            rows.append(
                {
                    "controller_type": controller,
                    "run_id": controller,
                    "world_seed": 1,
                    "intervention": intervention,
                    "survival_steps": 100,
                    "no_action_steps": 1,
                    "jump_steps": 1,
                    "duck_steps": 1,
                }
            )
    figure = _rq3_strategy_figure(rows)
    try:
        assert len(figure.axes[1].get_xticks()) == 6
    finally:
        figure.clf()


def test_mean_ci_singleton_is_undefined() -> None:
    assert _mean_ci([1.0]) == (1.0, None)
