from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from matplotlib import pyplot as plt

import dino_er.evolution as evolution_module
from dino_er.controllers import ControllerSpec, genome_hash
from dino_er.evaluation import CandidateEvaluation, EpisodeConfig, EpisodeResult
from dino_er.evolution import (
    EvolutionConfig,
    EvolutionRunner,
    adapt_mutation_scale,
    initial_parent_population,
    load_candidate_model,
    mutate_offspring,
    rank_biased_offspring_parents,
    save_candidate_model,
    select_parent_indices,
)
from dino_er.scientific import (
    ParquetResultStore,
    _central_curve,
    _latest_archive_training_means,
    _paired_trace_effect,
    _rq1_generation_fitness_figure,
    aggregate_study_results,
    export_behaviour_trace,
    export_parquet_figures,
    model_hash,
    model_result_directory,
    read_parquet_parts,
    result_location,
)
from scripts.evolve import _local_campaign_timestamp
from scripts.run_experiments import (
    RQ1,
    _planned_configs,
    _resolve_workers,
    _run_phase,
    _selected_final_configs,
)
from scripts.view_results import _resolve_view_model_directory


def _config(
    output_dir: Path,
    *,
    generations: int,
) -> EvolutionConfig:
    return EvolutionConfig(
        controller_type="reactive",
        training_seeds=(11, 12),
        generations=generations,
        parent_count=2,
        offspring_count=2,
        mutation_scale=0.3,
        evolution_seed=9,
        max_steps=10,
        run_id="resume-test",
        output_dir=output_dir,
        results_root=output_dir / "results",
    )


@pytest.fixture
def stub_evolution_session(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubSession:
        stop_after_generation = False

        def __init__(self, *_: object, **__: object) -> None:
            pass

        def announce_status(self, *_: object, **__: object) -> None:
            pass

        def poll_controls(self) -> None:
            pass

        def consume_candidate_save_requests(self) -> tuple[int, ...]:
            return ()

        def requeue_candidate_save_requests(self, _: tuple[int, ...]) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(evolution_module, "PopulationEvaluationSession", StubSession)


def test_results_viewer_falls_back_to_latest_manifested_scientific_model(
    tmp_path: Path,
) -> None:
    older = tmp_path / "model_v002_old"
    active = tmp_path / "model_v002_active"
    empty_current = tmp_path / "model_v002_edit"
    for path in (older, active, empty_current):
        path.mkdir()
    (active / "results-manifest.json").write_text("{}", encoding="utf-8")

    assert (
        _resolve_view_model_directory(
            tmp_path,
            2,
            None,
            current_model_hash="edit",
        )
        == active
    )
    assert _resolve_view_model_directory(tmp_path, 2, "old") == older


def test_results_viewer_prefers_complete_evidence_over_a_newer_failed_model(
    tmp_path: Path,
) -> None:
    complete = tmp_path / "model_v002_complete"
    failed = tmp_path / "model_v002_failed"
    for path, statuses in (
        (complete, {"rq1-main": {"complete": 6}}),
        (failed, {"rq1-main": {"failed": 6}}),
    ):
        path.mkdir()
        (path / "results-manifest.json").write_text(
            json.dumps({"run_statuses": statuses}), encoding="utf-8"
        )

    assert _resolve_view_model_directory(tmp_path, 2, None, current_model_hash="edit") == complete


def test_generated_campaign_timestamp_is_filesystem_safe_on_localised_windows(
    tmp_path: Path,
) -> None:
    timestamp = _local_campaign_timestamp()
    config = EvolutionConfig(
        **{
            **_config(tmp_path, generations=1).__dict__,
            "campaign_timestamp": timestamp,
        }
    )

    assert config.campaign_timestamp == timestamp
    assert "UTC" in timestamp


def test_aggregation_can_target_an_existing_model_hash(tmp_path: Path) -> None:
    index = aggregate_study_results(
        tmp_path,
        model_version=2,
        planned_runs=[],
        selected_model_hash="existing",
    )

    assert index["model_hash"] == "existing"
    assert (tmp_path / "model_v002_existing" / "scientific-results.json").is_file()


def test_result_location_can_target_a_valid_existing_campaign_hash(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path / "artifacts", generations=1),
        result_model_hash="0123456789ab",
    )

    location = result_location(config)

    assert location.model_hash == "0123456789ab"
    assert "model_v001_0123456789ab" in location.run_dir.as_posix()
    assert (
        model_result_directory(config.results_root, config.model_version, location.model_hash)
        == config.results_root / "model_v001_0123456789ab"
    )


def test_result_model_hash_rejects_unsafe_or_ambiguous_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="12-character lowercase hexadecimal"):
        replace(
            _config(tmp_path / "artifacts", generations=1),
            result_model_hash="NOT-A-HASH",
        )


def _fake_population(
    capture: list[tuple[str, ...]],
) -> Callable[..., list[CandidateEvaluation]]:
    def evaluate(
        genomes: list[np.ndarray],
        spec: ControllerSpec,
        config: EpisodeConfig,
        **_: object,
    ) -> list[CandidateEvaluation]:
        del config
        hashes = tuple(genome_hash(genome, spec) for genome in genomes)
        capture.append(hashes)
        evaluations: list[CandidateEvaluation] = []
        for index, genome in enumerate(genomes):
            fitness = -float(np.sum((genome - 0.1) ** 2))
            evaluations.append(
                CandidateEvaluation(
                    candidate_id=index,
                    genome_hash=hashes[index],
                    fitness=fitness,
                    selection_score=fitness,
                    mean_survival_fraction=0.0,
                    observation_changes=0,
                    score_changes=0,
                    action_switches=0,
                    responsive_transitions=0,
                    response_rate=None,
                    action_response_rate=None,
                    duration_seconds=0.0,
                    episodes=(),
                )
            )
        return evaluations

    return evaluate


def _episode(seed: int, steps: int = 10) -> EpisodeResult:
    return EpisodeResult(
        seed=seed,
        visual_passes=0,
        steps=steps,
        terminated=False,
        truncated=True,
        perception_failures=0,
        actions=(0,) * steps,
        action_scores=((0.0, 0.0, 0.0),) * steps,
        sensory_trace=((0.0,) * 5,) * steps,
        hidden_trace=(),
        obstacle_pass_steps=(),
        observation_changes=0,
        score_changes=0,
        action_switches=0,
        responsive_transitions=0,
        observation_variation_l1=0.0,
        score_variation_l1=0.0,
        action_counts=(steps, 0, 0),
        pterodactyl_visible_steps=0,
        duck_on_pterodactyl_steps=0,
        cactus_visible_steps=0,
        duck_on_cactus_steps=0,
    )


def _fake_population_with_episodes(
    capture: list[tuple[str, ...]],
) -> Callable[..., list[CandidateEvaluation]]:
    def evaluate(
        genomes: list[np.ndarray],
        spec: ControllerSpec,
        config: EpisodeConfig,
        **_: object,
    ) -> list[CandidateEvaluation]:
        hashes = tuple(genome_hash(genome, spec) for genome in genomes)
        capture.append(hashes)
        episodes = tuple(_episode(seed) for seed in config.seeds)
        return [
            CandidateEvaluation(
                candidate_id=index,
                genome_hash=hashes[index],
                fitness=float(index),
                selection_score=float(index),
                mean_survival_fraction=1.0,
                observation_changes=0,
                score_changes=0,
                action_switches=0,
                responsive_transitions=0,
                response_rate=None,
                action_response_rate=None,
                duration_seconds=0.0,
                episodes=episodes,
                action_counts=(20, 0, 0),
            )
            for index in range(len(genomes))
        ]

    return evaluate


def _archived_evaluation(*, generation: int, steps: int) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate_id=0,
        genome_hash=f"archive-{generation}",
        fitness=float(steps),
        selection_score=float(steps),
        mean_survival_fraction=0.5,
        observation_changes=1,
        score_changes=1,
        action_switches=0,
        responsive_transitions=0,
        response_rate=0.0,
        action_response_rate=0.0,
        duration_seconds=0.0,
        episodes=(_episode(11, steps),),
        action_counts=(steps, 0, 0),
    )


def test_initial_parent_population_uses_deterministic_rng() -> None:
    first = initial_parent_population(5, 3, 0.4, np.random.default_rng(17))
    second = initial_parent_population(5, 3, 0.4, np.random.default_rng(17))

    assert first.shape == (3, 5)
    assert np.array_equal(first, second)
    assert len({row.tobytes() for row in first}) == 3


def test_rank_biased_reproduction_preserves_tied_diversity_and_favours_ranked_elites() -> None:
    tied = rank_biased_offspring_parents(
        np.array([10.0, 10.0, 10.0]),
        9_000,
        np.random.default_rng(17),
    )
    ranked = rank_biased_offspring_parents(
        np.array([30.0, 20.0, 10.0]),
        9_000,
        np.random.default_rng(17),
    )

    assert np.array_equal(
        tied,
        rank_biased_offspring_parents(
            np.array([10.0, 10.0, 10.0]),
            9_000,
            np.random.default_rng(17),
        ),
    )
    assert set(tied[:3]) == {0, 1, 2}
    assert np.ptp(np.bincount(tied, minlength=3)) < 300
    assert np.bincount(ranked, minlength=3)[0] > np.bincount(ranked, minlength=3)[2]


def test_gaussian_mutation_matches_theta_plus_sigma_epsilon() -> None:
    parents = np.asarray([[1.0, 2.0], [-1.0, -2.0]])
    expected_rng = np.random.default_rng(23)
    epsilon = expected_rng.standard_normal((4, 2))
    expected = parents[np.asarray([0, 1, 0, 1])] + 0.25 * epsilon

    actual = mutate_offspring(parents, 4, 0.25, np.random.default_rng(23))

    assert np.array_equal(actual, expected)
    assert not np.shares_memory(actual, parents)


def test_one_fifth_success_rule_adapts_sigma_and_treats_ties_as_failures() -> None:
    parents = np.asarray([10.0, 20.0])
    source_indices = np.asarray([0, 0, 0, 1, 1])

    low_rate, reduced, low_reason = adapt_mutation_scale(
        0.3,
        parents,
        np.asarray([10.0, 10.0, 9.0, 20.0, 19.0]),
        source_indices,
        initial_mutation_scale=0.3,
    )
    exact_rate, unchanged, exact_reason = adapt_mutation_scale(
        0.3,
        parents,
        np.asarray([11.0, 10.0, 9.0, 20.0, 19.0]),
        source_indices,
        initial_mutation_scale=0.3,
    )
    high_rate, increased, high_reason = adapt_mutation_scale(
        0.3,
        parents,
        np.asarray([11.0, 12.0, 9.0, 21.0, 19.0]),
        source_indices,
        initial_mutation_scale=0.3,
    )

    assert low_rate == pytest.approx(0.0)
    assert exact_rate == pytest.approx(0.2)
    assert unchanged == pytest.approx(0.3)
    assert reduced == pytest.approx(0.3 * 0.85)
    assert low_reason == "success_below_target_contract"
    assert high_rate == pytest.approx(0.6)
    assert increased == pytest.approx(0.3)
    assert exact_reason == "success_at_target_hold"
    assert high_reason == "success_above_target_expand_or_cap"


def test_adaptive_sigma_expands_instead_of_contracting_on_a_flat_plateau() -> None:
    success_rate, expanded, reason = adapt_mutation_scale(
        0.3,
        np.asarray([270.0, 270.0]),
        np.asarray([270.0, 269.0, 270.0, 268.0]),
        np.asarray([0, 0, 1, 1]),
        initial_mutation_scale=0.3,
    )

    assert success_rate == 0.0
    assert expanded == pytest.approx(0.6)
    assert reason == "flat_plateau_expand"


def test_top_mu_selection_is_ranked_and_stable_for_elite_ties() -> None:
    scores = np.asarray([9.0, 7.0, 9.0, 8.0, 7.0])

    selected = select_parent_indices(scores, 3)

    assert selected.tolist() == [0, 2, 3]


def test_population_is_bounded_by_the_one_app_arena(tmp_path: Path) -> None:
    config = _config(tmp_path, generations=1)
    with pytest.raises(ValueError, match="at most 100"):
        EvolutionConfig(
            **{
                **config.__dict__,
                "parent_count": 20,
                "offspring_count": 81,
            }
        )


def test_final_standard_generation_is_twenty_parents_and_eighty_offspring(
    tmp_path: Path,
) -> None:
    config = EvolutionConfig(
        **{
            **_config(tmp_path, generations=1).__dict__,
            "parent_count": 20,
            "offspring_count": 80,
        }
    )

    assert config.population_size == 100
    assert config.simulation_speed == 1.0


def test_campaign_timestamp_names_results_and_isolates_versioned_artifacts(
    tmp_path: Path,
) -> None:
    plan = {
        "model_version": 2,
        "results_root": str(tmp_path / "results"),
        "output_root": str(tmp_path / "artifacts"),
        "common": {
            "campaign_timestamp": "2026-08-29_10-47-10_CEST",
            "training_seeds": [101],
            "validation_seeds": [303],
            "final_test_seeds": [505],
        },
        "experiments": [
            {
                "id": RQ1,
                "adaptive_sigma": True,
                "controllers": ["reactive"],
                "optimizer_seeds": [7],
                "parent_counts": [20],
                "offspring_counts": [80],
                "mutation_scales": [0.3],
                "generations": 1,
                "max_steps": 10,
            }
        ],
    }

    config = _planned_configs(
        plan,
        Namespace(section=None, accelerator="cpu"),
    )[0]

    assert result_location(config).run_dir.name.startswith("2026-08-29_10-47-10_CEST__")
    assert "model_v002_" in config.output_dir.as_posix()
    assert config.output_dir.name.startswith("2026-08-29_10-47-10_CEST__")


def test_complete_plan_keeps_adaptive_rq1_separate_from_fixed_sigma_rq2() -> None:
    plan = json.loads(Path("configs/experiment-plan.json").read_text(encoding="utf-8"))

    complete = _planned_configs(plan, Namespace(section=None, accelerator="cpu"))
    rq2_only = _planned_configs(
        plan,
        Namespace(section=["rq2-sensitivity"], accelerator="cpu"),
    )
    targeted = _planned_configs(
        plan,
        Namespace(section=None, accelerator="cpu", model_hash="46ccc63ff585"),
    )

    assert len(complete) == 36
    assert len(rq2_only) == 30
    assert all(config.generations == 25 for config in complete)
    selected_reactive = next(
        config
        for config in targeted
        if config.run_id == "rq1-main-reactive-seed-17-mu-20-lambda-80-sigma-0.3"
    )
    assert result_location(selected_reactive).experiment_hash == "85137ffa5b1d"
    assert "model_v002_46ccc63ff585" in selected_reactive.output_dir.as_posix()
    assert all(config.adaptive_sigma is True for config in complete if config.study_id == RQ1)
    assert all(
        config.adaptive_sigma is False
        for config in complete
        if config.study_id == "rq2-sensitivity"
    )
    assert (
        sum(
            config.study_id == "rq2-sensitivity"
            and config.parent_count == 20
            and config.mutation_scale == pytest.approx(0.3)
            for config in complete
        )
        == 6
    )
    assert (
        sum(
            config.parent_count == 20 and config.mutation_scale == pytest.approx(0.3)
            for config in rq2_only
        )
        == 6
    )
    assert {
        (config.parent_count, config.mutation_scale)
        for config in rq2_only
        if config.controller_type == "reactive" and config.evolution_seed == 7
    } == {(20, 0.1), (20, 0.3), (20, 0.6), (10, 0.3), (40, 0.3)}


def test_rq2_aggregation_rejects_an_adaptive_sigma_plan(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="RQ2 sensitivity analysis requires"):
        aggregate_study_results(
            tmp_path,
            model_version=2,
            planned_runs=[
                {
                    "experiment_group": "rq2-sensitivity",
                    "run_id": "invalid-adaptive-rq2",
                    "controller_type": "reactive",
                    "parent_count": 20,
                    "offspring_count": 80,
                    "mutation_scale": 0.3,
                    "adaptive_sigma": True,
                }
            ],
        )


def test_learning_curve_uses_only_the_common_complete_run_horizon() -> None:
    rows = [
        {
            "run_id": run_id,
            "cumulative_candidate_evaluations": point,
            "generation": point // 100 - 1,
            "selected_parent_fitness_mean": value,
            "fitness_mean": -1.0,
        }
        for run_id, point, value in (
            ("run-a", 100, 10.0),
            ("run-b", 100, 20.0),
            ("run-c", 100, 30.0),
            ("run-a", 200, 40.0),
            ("run-b", 200, 50.0),
        )
    ]

    x, mean, ci = _central_curve(rows, {"run-a", "run-b", "run-c"})

    np.testing.assert_array_equal(x, np.asarray([100]))
    np.testing.assert_allclose(mean, np.asarray([20.0]))
    assert ci.shape == (1,)
    assert ci[0] > 0.0

    generation, generation_mean, _generation_ci = _central_curve(
        rows,
        {"run-a", "run-b", "run-c"},
        x_field="generation",
        x_offset=1,
    )
    np.testing.assert_array_equal(generation, np.asarray([1]))
    np.testing.assert_allclose(generation_mean, np.asarray([20.0]))


def test_rq1_figure_uses_compact_journal_style_without_range_shading() -> None:
    rows = [
        {
            "controller_type": controller,
            "run_id": f"{controller}-{run_id}",
            "generation": generation,
            "cumulative_candidate_evaluations": (generation + 1) * 100,
            "selected_parent_fitness_mean": value + 8.0,
            "fitness_mean": value,
        }
        for controller, controller_offset in (("proactive", 0.0), ("reactive", 4.0))
        for run_id, run_offset in enumerate((0.0, 3.0, 6.0))
        for generation, value in enumerate(
            (272.0 + controller_offset + run_offset, 284.0 + controller_offset + run_offset)
        )
    ]
    required = {
        controller: {f"{controller}-{run_id}" for run_id in range(3)}
        for controller in ("proactive", "reactive")
    }

    figure = _rq1_generation_fitness_figure(rows, required)
    try:
        axis = figure.axes[0]
        assert tuple(figure.get_size_inches()) == pytest.approx((7.2, 4.15))
        assert not axis.collections
        assert not axis.spines["top"].get_visible()
        assert not axis.spines["right"].get_visible()
        assert axis.get_ylim() == pytest.approx((0.0, 3600.0))
    finally:
        plt.close(figure)


def test_archive_training_keeps_immutable_updates_but_aggregates_latest(
    tmp_path: Path,
) -> None:
    config = EvolutionConfig(
        **{
            **_config(tmp_path / "artifacts", generations=1).__dict__,
            "results_root": tmp_path / "results",
            "campaign_timestamp": "2026-08-29_10-47-10_CEST",
        }
    )
    store = ParquetResultStore(config)
    store.record_held_out("archive_training", 0, _archived_evaluation(generation=0, steps=10))
    store.record_held_out("archive_training", 3, _archived_evaluation(generation=3, steps=20))

    parts = sorted((store.run_dir / "episode-results").glob("archive_training-*.parquet"))
    rows = read_parquet_parts(store.run_dir / "episode-results")
    means = _latest_archive_training_means(rows)

    assert [path.name for path in parts] == [
        "archive_training-generation-0000.parquet",
        "archive_training-generation-0003.parquet",
    ]
    assert means[(config.controller_type, config.run_id)] == pytest.approx(20.0)


def test_paired_trace_effect_reports_causal_action_and_score_changes() -> None:
    action_effect, score_effect, shared_steps = _paired_trace_effect(
        (0, 1, 0, 2),
        (0, 0, 0),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )

    assert shared_steps == 3
    assert action_effect == pytest.approx(1 / 3)
    assert score_effect == pytest.approx(2 / 3)


def test_rq3_aggregation_exposes_paired_sensor_dependence_in_results_gui_data(
    tmp_path: Path,
) -> None:
    config = EvolutionConfig(
        **{
            **_config(tmp_path / "artifacts", generations=1).__dict__,
            "results_root": tmp_path / "results",
            "study_id": RQ1,
            "scientific_study_dir": tmp_path / "study",
            "validation_seeds": (303,),
            "final_test_seeds": (505,),
            "campaign_timestamp": "2026-08-29_10-47-10_CEST",
        }
    )
    normal_episode = replace(
        _episode(303, 3),
        actions=(0, 1, 0),
        action_scores=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        sensory_trace=((0.0,) * 5, (1.0,) * 5, (0.0,) * 5),
        action_counts=(2, 1, 0),
    )
    constant_episode = replace(
        _episode(303, 3),
        actions=(0, 0, 0),
        action_scores=((1.0, 0.0, 0.0),) * 3,
        sensory_trace=((0.0,) * 5,) * 3,
        action_counts=(3, 0, 0),
    )

    def evaluation(episode: EpisodeResult) -> CandidateEvaluation:
        return replace(
            _archived_evaluation(generation=1, steps=episode.steps),
            episodes=(episode,),
            action_counts=episode.action_counts,
        )

    store = ParquetResultStore(config)
    for label, episode in (
        ("normal", normal_episode),
        ("constant_first_frame", constant_episode),
    ):
        current = evaluation(episode)
        store.record_intervention(intervention=label, generation=1, evaluation=current)
        store.record_trace(
            trace_id=f"rq3-{label}",
            evaluation=current,
            controller_type=config.controller_type,
            intervention=label,
        )
    store.mark_complete("complete")
    model_dir = model_result_directory(config.results_root, config.model_version)
    (model_dir / "results-manifest.json").write_text(
        json.dumps({"campaign_timestamp": config.campaign_timestamp}),
        encoding="utf-8",
    )

    index = aggregate_study_results(
        config.results_root,
        model_version=config.model_version,
        planned_runs=[],
    )

    causal = index["rq3_causal_summary"][0]
    assert causal["paired_trace_seeds"] == 1
    assert causal["mean_action_disagreement_fraction"] == pytest.approx(1 / 3)
    assert causal["discrete_actions_depend_on_visual_input"] is True
    assert "RQ3 causal sensor dependence" in index["tables"]


def test_manifest_progress_and_timestamps_update_after_each_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_evolution_session: None,
) -> None:
    monkeypatch.setattr(
        evolution_module,
        "evaluate_population_batch",
        _fake_population_with_episodes([]),
    )
    base = _config(tmp_path / "artifacts", generations=1)
    config = EvolutionConfig(
        **{
            **base.__dict__,
            "results_root": tmp_path / "results",
            "campaign_timestamp": "2026-08-29_10-47-10_CEST",
        }
    )

    EvolutionRunner(config).run()
    manifest = json.loads(
        (result_location(config).run_dir / "run-manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["completed_generations"] == 1
    assert manifest["status"] == "complete"
    assert manifest["campaign_timestamp"] == config.campaign_timestamp
    assert manifest["started_at_utc"].endswith("Z")
    assert manifest["updated_at_utc"].endswith("Z")
    assert manifest["finished_at_utc"].endswith("Z")


def test_locked_final_test_uses_only_the_validation_frozen_archives(tmp_path: Path) -> None:
    reactive = EvolutionConfig(
        **{
            **_config(tmp_path / "reactive", generations=1).__dict__,
            "run_id": "reactive-selected",
            "study_id": RQ1,
            "scientific_study_dir": tmp_path / "study",
            "validation_seeds": (303,),
            "final_test_seeds": (505,),
        }
    )
    proactive = EvolutionConfig(
        **{
            **_config(tmp_path / "proactive", generations=1).__dict__,
            "controller_type": "proactive",
            "run_id": "proactive-selected",
            "study_id": RQ1,
            "scientific_study_dir": tmp_path / "study",
            "validation_seeds": (303,),
            "final_test_seeds": (505,),
        }
    )
    non_selected = EvolutionConfig(
        **{
            **_config(tmp_path / "other", generations=1).__dict__,
            "run_id": "reactive-not-selected",
            "study_id": RQ1,
            "scientific_study_dir": tmp_path / "study",
            "validation_seeds": (303,),
            "final_test_seeds": (505,),
        }
    )

    selected = _selected_final_configs(
        (reactive, proactive, non_selected),
        {
            "selected_models": [
                {"run_id": reactive.run_id},
                {"run_id": proactive.run_id},
            ]
        },
    )

    assert [config.run_id for config in selected] == [reactive.run_id, proactive.run_id]
    assert all(config.evaluate_final_test for config in selected)


def test_final_test_progress_counts_only_the_validation_frozen_archives(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    model_dir = model_result_directory(results_root, model_version=1)
    model_dir.mkdir(parents=True)
    (model_dir / "results-manifest.json").write_text(
        json.dumps({"frozen_selection": {"selected_models": [{"run_id": "reactive-selected"}]}}),
        encoding="utf-8",
    )
    for run_id in ("reactive-selected", "reactive-not-selected"):
        run_dir = model_dir / "rq1-main" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "run-manifest.json").write_text(
            json.dumps(
                {
                    "experiment_group": RQ1,
                    "run_id": run_id,
                    "status": "complete",
                    "final_test_recorded": True,
                }
            ),
            encoding="utf-8",
        )

    index = aggregate_study_results(
        results_root,
        model_version=1,
        planned_runs=[
            {"experiment_group": RQ1, "run_id": "reactive-selected"},
            {"experiment_group": RQ1, "run_id": "reactive-not-selected"},
        ],
    )

    assert index["progress"]["final_test"] == {
        "completed": 1,
        "planned": 1,
        "failed": 0,
        "running": 0,
        "interrupted": 0,
        "not_started": 0,
    }


def test_pipeline_retries_only_operational_failures_without_stopping_other_runs(
    tmp_path: Path,
) -> None:
    retry = EvolutionConfig(
        **{**_config(tmp_path / "retry", generations=1).__dict__, "run_id": "retry"}
    )
    permanent = EvolutionConfig(
        **{**_config(tmp_path / "permanent", generations=1).__dict__, "run_id": "permanent"}
    )
    attempts: dict[str, int] = {}

    def worker(config: EvolutionConfig, *, require_final_test: bool) -> dict[str, str]:
        del require_final_test
        attempts[config.run_id] = attempts.get(config.run_id, 0) + 1
        if config.run_id == "retry" and attempts[config.run_id] == 1:
            return {"run_id": config.run_id, "status": "failed", "detail": "OSError: transient"}
        if config.run_id == "permanent":
            return {
                "run_id": config.run_id,
                "status": "failed",
                "detail": "ValueError: invalid plan",
            }
        return {"run_id": config.run_id, "status": "complete", "detail": "done"}

    reports = _run_phase(
        "test",
        (retry, permanent),
        require_final_test=False,
        workers=1,
        retries=1,
        worker=worker,
    )

    assert attempts == {"retry": 2, "permanent": 1}
    assert [(report["run_id"], report["status"]) for report in reports] == [
        ("retry", "failed"),
        ("permanent", "failed"),
        ("retry", "complete"),
    ]


def test_worker_policy_parallelises_independent_runs_without_backend_special_cases() -> None:
    assert 1 <= _resolve_workers("auto") <= 6
    assert _resolve_workers("3") == 3
    with pytest.raises(ValueError, match="cannot exceed 6"):
        _resolve_workers("10")


def test_elites_survive_unchanged_into_the_next_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_evolution_session: None,
) -> None:
    candidate_hashes: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        evolution_module,
        "evaluate_population_batch",
        _fake_population(candidate_hashes),
    )

    result = EvolutionRunner(_config(tmp_path, generations=2)).run()

    assert candidate_hashes[1][:2] == result.records[0].selected_parent_hashes
    assert result.records[1].fitness_max >= result.records[0].fitness_max
    assert result.records[0].best_genome_hash in candidate_hashes[1][:2]


def test_completed_generation_has_a_parquet_only_audit_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_evolution_session: None,
) -> None:
    capture: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        evolution_module,
        "evaluate_population_batch",
        _fake_population_with_episodes(capture),
    )
    base = _config(tmp_path / "outputs", generations=1)
    config = EvolutionConfig(
        **{
            **base.__dict__,
            "results_root": tmp_path / "results",
            "simulation_speed": 100.0,
        }
    )

    EvolutionRunner(config).run()
    location = result_location(config)
    generation = pq.read_table(next((location.run_dir / "generation-results").glob("*.parquet")))
    episodes = pq.read_table(
        next((location.run_dir / "episode-results").glob("generation-*.parquet"))
    )

    assert generation.num_rows == 1
    assert episodes.num_rows == config.population_size * len(config.training_seeds)
    assert {
        "selected_parent_fitness_mean",
        "offspring_fitness_mean",
    } <= set(generation.column_names)
    assert {
        "candidate_id",
        "candidate_role",
        "genome_id",
        "parent_genome_id",
        "selected_as_parent",
        "world_seed",
        "survival_steps",
        "terminated",
        "truncated",
    } <= set(episodes.column_names)
    assert set(episodes.column("candidate_role").to_pylist()) == {"parent", "offspring"}
    assert not list(location.run_dir.rglob("*.csv"))
    assert (location.run_dir / "run-manifest.json").is_file()
    figures = export_parquet_figures(location)
    assert set(figures) == {"RQ1 learning"}
    assert figures["RQ1 learning"].is_file()


def test_rq3_parquet_trace_is_exported_as_a_scientific_figure(tmp_path: Path) -> None:
    config = EvolutionConfig(
        **{
            **_config(tmp_path / "outputs", generations=1).__dict__,
            "results_root": tmp_path / "results",
        }
    )
    location = result_location(config)
    trace_path = export_behaviour_trace(
        location.run_dir / "behavioural-traces" / "rq3-selected-normal.parquet",
        evaluation=CandidateEvaluation(
            candidate_id=0,
            genome_hash="test-genome",
            fitness=10.0,
            selection_score=10.0,
            mean_survival_fraction=1.0,
            observation_changes=0,
            score_changes=0,
            action_switches=0,
            responsive_transitions=0,
            response_rate=None,
            action_response_rate=None,
            duration_seconds=0.0,
            episodes=(_episode(11),),
        ),
        controller_type="reactive",
        source_id="selected archive",
        intervention="normal replay",
    )
    (location.run_dir / "run-manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (location.run_dir / "run-manifest.json").write_text(
        '{"experiment_group": "engineering"}\n', encoding="utf-8"
    )

    figures = export_parquet_figures(location)
    trace_rows = pq.read_table(trace_path).to_pylist()

    assert trace_path.is_file()
    assert trace_rows[-1]["control_time_seconds"] == pytest.approx(9 / 60)
    assert figures["RQ3 behavioural trace (rq3 selected normal)"].is_file()


def test_model_hash_is_stable_with_gameplay_sources() -> None:
    assert model_hash() == model_hash()


def test_checkpoint_resume_reproduces_uninterrupted_candidate_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_evolution_session: None,
) -> None:
    uninterrupted: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        evolution_module,
        "evaluate_population_batch",
        _fake_population(uninterrupted),
    )
    full = EvolutionRunner(_config(tmp_path / "full", generations=2)).run()
    saved_genome, saved_spec, saved_metadata = load_candidate_model(
        tmp_path / "full" / "best-candidate.npz"
    )
    assert genome_hash(saved_genome, saved_spec) == full.best_genome_hash
    assert saved_metadata["fitness"] == full.best_fitness

    resumed: list[tuple[str, ...]] = []
    uninterrupted_evaluator = _fake_population(resumed)
    calls = 0

    def interrupt_after_first_generation(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return uninterrupted_evaluator(*args, **kwargs)

    monkeypatch.setattr(
        evolution_module,
        "evaluate_population_batch",
        interrupt_after_first_generation,
    )
    interrupted_config = _config(tmp_path / "resume", generations=2)
    with pytest.raises(KeyboardInterrupt):
        EvolutionRunner(interrupted_config).run()
    assert (tmp_path / "resume" / "reactive-checkpoint.pkl").is_file()
    monkeypatch.setattr(
        evolution_module,
        "evaluate_population_batch",
        uninterrupted_evaluator,
    )
    second = EvolutionRunner(_config(tmp_path / "resume", generations=2)).run()

    assert resumed == uninterrupted
    assert second.best_genome_hash == full.best_genome_hash
    assert second.best_fitness == pytest.approx(full.best_fitness)
    assert second.records[-1].evaluations == 8


def test_fixed_sigma_mode_never_changes_scale_between_generations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_evolution_session: None,
) -> None:
    evaluated: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        evolution_module,
        "evaluate_population_batch",
        _fake_population(evaluated),
    )
    base = _config(tmp_path / "fixed-sigma", generations=3)
    result = EvolutionRunner(EvolutionConfig(**{**base.__dict__, "adaptive_sigma": False})).run()

    assert len(evaluated) == 3
    assert [record.mutation_scale for record in result.records] == [0.3, 0.3, 0.3]
    assert [record.mutation_scale_next for record in result.records] == [0.3, 0.3, 0.3]
    assert {record.mutation_scale_reason for record in result.records} == {
        "fixed_by_experiment_design"
    }


def test_continuous_visible_run_stops_after_a_complete_generation_and_waits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSession:
        def __init__(self, *_: object, **__: object) -> None:
            self.stop_after_generation = True
            self.statuses: list[str] = []
            self.save_requests: list[tuple[int, ...]] = [(1,)]
            self.waited = False
            self.closed = False
            sessions.append(self)

        def poll_controls(self) -> None:
            return None

        def announce_status(self, phase: str, _message: str, **_: object) -> None:
            self.statuses.append(phase)

        def consume_candidate_save_requests(self) -> tuple[int, ...]:
            return self.save_requests.pop(0) if self.save_requests else ()

        def requeue_candidate_save_requests(self, requests: tuple[int, ...]) -> None:
            self.save_requests.insert(0, requests)

        def wait_until_window_closed(self, *, on_poll: object | None = None) -> None:
            del on_poll
            self.waited = True

        def close(self) -> None:
            self.closed = True

    sessions: list[FakeSession] = []
    evaluated: list[tuple[str, ...]] = []
    monkeypatch.setattr(evolution_module, "PopulationEvaluationSession", FakeSession)
    monkeypatch.setattr(
        evolution_module,
        "evaluate_population_batch",
        _fake_population(evaluated),
    )
    base = _config(tmp_path / "continuous", generations=2)
    config = EvolutionConfig(
        **{
            **base.__dict__,
            "continuous": True,
            "render_mode": "human",
        }
    )

    result = EvolutionRunner(config, keep_open_after_run=True).run()

    session = sessions[0]
    assert len(evaluated) == 1
    assert len(result.records) == 1
    assert len(result.saved_candidate_paths) == 1
    assert Path(result.saved_candidate_paths[0]).is_file()
    assert session.statuses[-1] == "stopped"
    assert session.waited is True
    assert session.closed is True


def test_headless_generations_recycle_browser_and_validation_uses_one_archived_dino(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_hashes: list[str] = []

    class FakeSession:
        def __init__(self, *_: object, **__: object) -> None:
            self.stop_after_generation = False
            sessions.append(self)

        def poll_controls(self) -> None:
            return None

        def announce_status(self, *_: object, **__: object) -> None:
            return None

        def consume_candidate_save_requests(self) -> tuple[int, ...]:
            return ()

        def evaluate(
            self,
            genomes: list[np.ndarray],
            *,
            episode_config: object,
            **_: object,
        ) -> list[CandidateEvaluation]:
            del episode_config
            validation_hashes.extend(genome_hash(genome, runner.spec) for genome in genomes)
            return [
                CandidateEvaluation(
                    candidate_id=index,
                    genome_hash=validation_hashes[index],
                    fitness=77.0,
                    selection_score=77.0,
                    mean_survival_fraction=1.0,
                    observation_changes=0,
                    score_changes=0,
                    action_switches=0,
                    responsive_transitions=0,
                    response_rate=None,
                    action_response_rate=None,
                    duration_seconds=0.0,
                    episodes=(),
                )
                for index in range(len(genomes))
            ]

        def close(self) -> None:
            return None

    sessions: list[FakeSession] = []
    evaluated: list[tuple[str, ...]] = []
    monkeypatch.setattr(evolution_module, "PopulationEvaluationSession", FakeSession)
    monkeypatch.setattr(
        evolution_module,
        "evaluate_population_batch",
        _fake_population(evaluated),
    )
    study_dir = tmp_path / "study"
    base = _config(study_dir / "reactive" / "seed-9", generations=5)
    config = EvolutionConfig(
        **{
            **base.__dict__,
            "study_id": "validation-session-test",
            "scientific_study_dir": study_dir,
            "validation_seeds": (303,),
            "final_test_seeds": (505,),
        }
    )
    runner = EvolutionRunner(config)
    result = runner.run()

    assert len(sessions) == 6
    assert len(validation_hashes) == 1
    assert set(validation_hashes) == {result.best_genome_hash}
    assert {genome_hash(genome, runner.spec) for genome in runner._latest_solutions} == {
        result.best_genome_hash
    }
    assert runner._latest_fitness == (77.0,)


def test_gui_saved_candidate_model_round_trips_without_pickle(tmp_path: Path) -> None:
    config = _config(tmp_path, generations=1)
    spec = evolution_module.default_controller_spec(
        "reactive",
        len(evolution_module.SENSORY_NAMES),
    )
    genome = np.linspace(-0.2, 0.2, spec.parameter_count)
    path = tmp_path / "selected.npz"

    save_candidate_model(
        path,
        genome,
        spec,
        metadata={
            "generation": 4,
            "candidate_id": 2,
            "fitness": 7.0,
            "run_id": config.run_id,
        },
    )
    restored, restored_spec, metadata = load_candidate_model(path)

    assert np.array_equal(restored, genome)
    assert restored_spec == spec
    assert metadata["candidate_id"] == 2
    assert metadata["fitness"] == 7.0
