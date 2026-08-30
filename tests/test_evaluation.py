from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from dino_er.controllers import ControllerSpec
from dino_er.evaluation import (
    EpisodeConfig,
    PopulationEvaluationSession,
    PopulationRunRestartRequested,
    evaluate_population_batch,
    sensory_intervention_analysis,
)


class _FakeBatchEnvironment:
    reset_seeds: list[int] = []

    def __init__(self, num_envs: int, **_: Any) -> None:
        self.num_envs = num_envs
        self._frames = np.zeros((num_envs, 2, 2, 3), dtype=np.uint8)
        self.controls: list[dict[str, Any]] = []
        self._last_perception = self._results()

    def _results(self) -> tuple[Any, ...]:
        return tuple(
            SimpleNamespace(
                failure_reason=None,
                estimate=SimpleNamespace(obstacle_class=None),
            )
            for _ in range(self.num_envs)
        )

    def configure_population(self, **_: Any) -> None:
        return None

    def reset(
        self,
        *,
        seed: int,
    ) -> np.ndarray:
        self.reset_seeds.append(seed)
        self._last_perception = self._results()
        return np.zeros((self.num_envs, 2), dtype=np.float32)

    def step(
        self,
        actions: np.ndarray,
        *,
        active: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        return (
            np.zeros((self.num_envs, 2), dtype=np.float32),
            np.asarray((actions == 1) & active, dtype=np.float32),
            np.ones(self.num_envs, dtype=np.bool_),
            np.zeros(self.num_envs, dtype=np.bool_),
            [{"failure_reason": None} for _ in range(self.num_envs)],
        )

    def step_perceptions(
        self,
        actions: np.ndarray,
        *,
        active: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[Any, ...]]:
        sensory, rewards, terminated, truncated, _ = self.step(actions, active=active)
        self._last_perception = self._results()
        return sensory, rewards, terminated, truncated, self._last_perception

    @property
    def last_perception(self) -> tuple[Any, ...]:
        return self._last_perception

    def poll_controls(self) -> list[dict[str, Any]]:
        controls = list(self.controls)
        self.controls.clear()
        return controls

    def close(self) -> None:
        return None


def _action_genome(spec: ControllerSpec, action: int) -> np.ndarray:
    genome = np.zeros(spec.parameter_count)
    genome[-spec.action_size + action] = 1.0
    return genome


def test_episode_config_rejects_a_negative_visible_hold() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        EpisodeConfig(seeds=(7,), post_episode_hold_seconds=-0.1)


def test_waiting_for_window_close_requires_visible_rendering() -> None:
    with pytest.raises(ValueError, match="requires render_mode='human'"):
        EpisodeConfig(seeds=(7,), wait_for_window_close=True)


def test_gui_save_and_new_run_controls_are_validated() -> None:
    spec = ControllerSpec("reactive", input_size=2, hidden_size=1)
    environment = _FakeBatchEnvironment(4)
    session = PopulationEvaluationSession(
        spec,
        EpisodeConfig(seeds=(7,), max_steps=2),
        population_size=4,
        environment=environment,
    )
    environment.controls = [
        {"action": "save_candidate", "candidateId": 3},
    ]
    session.poll_controls()
    assert session.consume_candidate_save_requests() == (3,)

    environment.controls = [
        {
            "action": "start_new_run",
            "parentCount": 3,
            "offspringCount": 9,
            "controllerType": "proactive",
            "accelerator": "cuda",
            "mode": "continuous",
            "generations": 30,
            "maxSteps": 600,
            "mutationScale": 0.3,
            "evolutionSeed": 17,
            "simulationSpeed": 3.0,
            "trainingSeeds": [101, 202],
            "validationSeeds": [303, 404],
            "finalTestSeeds": [505, 606],
            "studyId": "",
        }
    ]
    with pytest.raises(PopulationRunRestartRequested) as requested:
        session.poll_controls()
    assert requested.value.population_size == 12
    assert requested.value.parent_count == 3
    assert requested.value.offspring_count == 9
    assert requested.value.controller_type == "proactive"
    assert requested.value.accelerator == "cuda"
    assert requested.value.continuous is True
    assert requested.value.max_steps == 600
    assert requested.value.evolution_seed == 17
    assert requested.value.simulation_speed == 3.0


def test_live_simulation_speed_changes_pacing_only() -> None:
    spec = ControllerSpec("reactive", input_size=2, hidden_size=1)
    environment = _FakeBatchEnvironment(1)
    config = EpisodeConfig(seeds=(7,), max_steps=2, simulation_speed=1.0)
    session = PopulationEvaluationSession(
        spec,
        config,
        population_size=1,
        environment=environment,
    )
    genome = _action_genome(spec, 0)
    baseline = session.evaluate(
        [genome],
        generation=0,
        mutation_scale=0.3,
    )[0]
    environment.controls = [
        {"action": "set_simulation_speed", "simulationSpeed": 100.0},
    ]
    session.poll_controls()
    accelerated = session.evaluate(
        [genome],
        generation=0,
        mutation_scale=0.3,
    )[0]
    session.close()

    assert session.simulation_speed == 100.0
    assert baseline.genome_hash == accelerated.genome_hash
    assert baseline.fitness == accelerated.fitness
    assert baseline.selection_score == accelerated.selection_score
    assert baseline.episodes[0].steps == accelerated.episodes[0].steps
    assert baseline.episodes[0].actions == accelerated.episodes[0].actions


def test_batch_population_uses_distinct_genomes_and_identical_episode_seeds() -> None:
    _FakeBatchEnvironment.reset_seeds.clear()
    spec = ControllerSpec("reactive", input_size=2, hidden_size=1)
    genomes = [_action_genome(spec, action) for action in (0, 1, 2)]
    config = EpisodeConfig(seeds=(7, 8), max_steps=2)
    session = PopulationEvaluationSession(
        spec,
        config,
        population_size=3,
        environment=_FakeBatchEnvironment(3),
    )
    results = evaluate_population_batch(
        genomes,
        spec,
        config,
        generation=0,
        mutation_scale=0.3,
        session=session,
    )
    session.close()
    assert _FakeBatchEnvironment.reset_seeds == [7, 8]
    assert [result.candidate_id for result in results] == [0, 1, 2]
    assert [result.fitness for result in results] == [1.0, 1.0, 1.0]
    assert [result.selection_score for result in results] == [1.0, 1.0, 1.0]
    assert [tuple(episode.visual_passes for episode in result.episodes) for result in results] == [
        (0, 0),
        (1, 1),
        (0, 0),
    ]


def test_one_hundred_unique_candidates_share_one_complete_generation() -> None:
    _FakeBatchEnvironment.reset_seeds.clear()
    spec = ControllerSpec("reactive", input_size=2, hidden_size=1)
    genomes = [_action_genome(spec, index % 3) for index in range(100)]
    environment = _FakeBatchEnvironment(100)
    session = PopulationEvaluationSession(
        spec,
        EpisodeConfig(seeds=(17,), max_steps=2),
        population_size=100,
        environment=environment,
    )
    try:
        results = evaluate_population_batch(
            genomes,
            spec,
            EpisodeConfig(seeds=(17,), max_steps=2),
            generation=3,
            mutation_scale=0.3,
            session=session,
        )
        assert len(results) == 100
        assert [result.candidate_id for result in results] == list(range(100))
        assert _FakeBatchEnvironment.reset_seeds == [17]
        assert session._candidate_ids == tuple(range(100))
        session._apply_control({"action": "save_candidate", "candidateId": 99})
        session._apply_control({"action": "save_candidate", "candidateId": 100})
        assert session.consume_candidate_save_requests() == (99,)
    finally:
        session.close()


def test_population_larger_than_the_shared_arena_is_rejected() -> None:
    spec = ControllerSpec("reactive", input_size=2, hidden_size=1)
    genomes = [_action_genome(spec, index % 3) for index in range(101)]
    with pytest.raises(ValueError, match="at most 100"):
        evaluate_population_batch(
            genomes,
            spec,
            EpisodeConfig(seeds=(17,), max_steps=2),
            generation=3,
            mutation_scale=0.3,
        )


def test_existing_population_session_can_use_validation_seeds() -> None:
    _FakeBatchEnvironment.reset_seeds.clear()
    spec = ControllerSpec("reactive", input_size=2, hidden_size=1)
    training = EpisodeConfig(seeds=(7,), max_steps=2)
    session = PopulationEvaluationSession(
        spec,
        training,
        population_size=2,
        environment=_FakeBatchEnvironment(2),
    )
    genome = _action_genome(spec, 0)
    try:
        results = session.evaluate(
            [genome, genome],
            generation=1,
            mutation_scale=0.2,
            episode_config=EpisodeConfig(seeds=(303, 404), max_steps=2),
        )
    finally:
        session.close()

    assert _FakeBatchEnvironment.reset_seeds == [303, 404]
    assert [episode.seed for episode in results[0].episodes] == [303, 404]


def test_candidate_order_does_not_change_genome_fitness() -> None:
    spec = ControllerSpec("reactive", input_size=2, hidden_size=1)
    genomes = [_action_genome(spec, action) for action in (0, 1, 2)]
    config = EpisodeConfig(seeds=(17,), max_steps=2)

    def evaluated_by_hash(ordered: list[np.ndarray]) -> dict[str, float]:
        session = PopulationEvaluationSession(
            spec,
            config,
            population_size=3,
            environment=_FakeBatchEnvironment(3),
        )
        try:
            results = session.evaluate(
                ordered,
                generation=0,
                mutation_scale=0.3,
            )
        finally:
            session.close()
        return {result.genome_hash: result.fitness for result in results}

    baseline = evaluated_by_hash(genomes)
    permuted = evaluated_by_hash([genomes[2], genomes[0], genomes[1]])
    assert permuted == baseline


class _StaggeredTerminationEnvironment(_FakeBatchEnvironment):
    def __init__(self, num_envs: int) -> None:
        super().__init__(num_envs)
        self.step_index = 0

    def reset(
        self,
        *,
        seed: int,
    ) -> np.ndarray:
        self.step_index = 0
        return super().reset(seed=seed)

    def step(
        self,
        actions: np.ndarray,
        *,
        active: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        del actions, active
        self.step_index += 1
        terminated = np.asarray(
            [self.step_index >= 1, self.step_index >= 3],
            dtype=np.bool_,
        )
        return (
            np.zeros((self.num_envs, 2), dtype=np.float32),
            np.zeros(self.num_envs, dtype=np.float32),
            terminated,
            np.zeros(self.num_envs, dtype=np.bool_),
            [{"failure_reason": None} for _ in range(self.num_envs)],
        )


class _NeverTerminatingEnvironment(_FakeBatchEnvironment):
    def step(
        self,
        actions: np.ndarray,
        *,
        active: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        del actions, active
        return (
            np.zeros((self.num_envs, 2), dtype=np.float32),
            np.zeros(self.num_envs, dtype=np.float32),
            np.zeros(self.num_envs, dtype=np.bool_),
            np.zeros(self.num_envs, dtype=np.bool_),
            [{"failure_reason": None} for _ in range(self.num_envs)],
        )


class _ChangingSensoryEnvironment(_FakeBatchEnvironment):
    def __init__(self, num_envs: int) -> None:
        super().__init__(num_envs)
        self.step_index = 0

    def reset(
        self,
        *,
        seed: int,
    ) -> np.ndarray:
        self.step_index = 0
        return super().reset(seed=seed)

    def step(
        self,
        actions: np.ndarray,
        *,
        active: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        del actions, active
        self.step_index += 1
        sensory_value = float(self.step_index % 2)
        return (
            np.full(
                (self.num_envs, 2),
                sensory_value,
                dtype=np.float32,
            ),
            np.zeros(self.num_envs, dtype=np.float32),
            np.full(
                self.num_envs,
                self.step_index >= 4,
                dtype=np.bool_,
            ),
            np.zeros(self.num_envs, dtype=np.bool_),
            [{"failure_reason": None} for _ in range(self.num_envs)],
        )


def test_terminated_candidates_receive_no_more_controller_calls() -> None:
    spec = ControllerSpec("reactive", input_size=2, hidden_size=1)
    config = EpisodeConfig(seeds=(5,), max_steps=5, record_trace=True)
    session = PopulationEvaluationSession(
        spec,
        config,
        population_size=2,
        environment=_StaggeredTerminationEnvironment(2),
    )
    results = session.evaluate(
        [np.zeros(spec.parameter_count), np.zeros(spec.parameter_count)],
        generation=0,
        mutation_scale=0.3,
    )
    session.close()
    assert [result.episodes[0].steps for result in results] == [1, 3]
    assert [result.fitness for result in results] == [1.0, 3.0]
    assert [result.episodes[0].terminated for result in results] == [True, True]
    assert [result.episodes[0].truncated for result in results] == [False, False]


def test_max_steps_is_reported_as_truncation_not_collision() -> None:
    spec = ControllerSpec("reactive", input_size=2, hidden_size=1)
    config = EpisodeConfig(seeds=(5,), max_steps=3)
    session = PopulationEvaluationSession(
        spec,
        config,
        population_size=2,
        environment=_NeverTerminatingEnvironment(2),
    )
    results = session.evaluate(
        [np.zeros(spec.parameter_count), np.zeros(spec.parameter_count)],
        generation=0,
        mutation_scale=0.3,
    )
    session.close()
    assert all(not result.episodes[0].terminated for result in results)
    assert all(result.episodes[0].truncated for result in results)
    assert all(result.episodes[0].steps == 3 for result in results)


def test_visual_changes_produce_measured_controller_and_action_responses() -> None:
    spec = ControllerSpec("reactive", input_size=2, hidden_size=1)
    genome = np.zeros(spec.parameter_count)
    genome[0:2] = (5.0, 0.0)
    genome[2] = -2.5
    genome[3:6] = (-1.0, 1.0, 0.0)
    config = EpisodeConfig(seeds=(5,), max_steps=5)
    session = PopulationEvaluationSession(
        spec,
        config,
        population_size=1,
        environment=_ChangingSensoryEnvironment(1),
    )
    try:
        result = session.evaluate(
            [genome],
            generation=0,
            mutation_scale=0.3,
        )[0]
    finally:
        session.close()

    episode = result.episodes[0]
    assert episode.observation_changes == 3
    assert episode.score_changes == 3
    assert episode.action_switches == 3
    assert episode.responsive_transitions == 3
    assert episode.observation_variation_l1 > 0
    assert episode.score_variation_l1 > 0

    intervention = sensory_intervention_analysis(
        genome,
        spec,
        ((0.0, 0.0), (1.0, 0.0), (0.0, 0.0)),
    )
    assert intervention["uses_visual_input"] is True
    assert intervention["changes_discrete_actions"] is True
    assert intervention["changed_action_steps"] == 1
