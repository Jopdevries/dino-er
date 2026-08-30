from __future__ import annotations

import numpy as np

from dino_er.controllers import default_controller_spec
from dino_er.environment import DinoArenaEnv
from dino_er.evaluation import EpisodeConfig, evaluate_candidate


def _visual_rollout(
    seed: int,
    actions: list[int],
    *,
    simulation_speed: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    environment = DinoArenaEnv(
        num_envs=1,
        render_mode="rgb_array",
        simulation_speed=simulation_speed,
    )
    try:
        initial = environment.reset(seed=seed)
        frames: list[np.ndarray] = []
        for action in actions:
            environment.step_perceptions([action])
            batch = environment._last_pixel_batch  # noqa: SLF001 - pixel-boundary regression
            assert batch is not None
            frames.append(batch.materialize()[0])
        return initial[0], np.stack(frames)
    finally:
        environment.close()


def test_equal_seeds_and_keyboard_actions_produce_equal_pixels() -> None:
    actions = [0] * 10 + [1] * 8 + [0] * 12 + [2] * 3
    first_observation, first_frames = _visual_rollout(17, actions)
    second_observation, second_frames = _visual_rollout(17, actions)
    np.testing.assert_array_equal(first_observation, second_observation)
    np.testing.assert_array_equal(first_frames, second_frames)


def test_different_seeds_change_rendered_world() -> None:
    actions = [0] * 5
    _, first_frames = _visual_rollout(31, actions)
    _, second_frames = _visual_rollout(32, actions)
    assert not np.array_equal(first_frames, second_frames)


def test_accelerated_execution_preserves_visual_controller_episode() -> None:
    actions = [0] * 90 + [1] * 5 + [0] * 25
    one_observation, one_frames = _visual_rollout(17, actions, simulation_speed=1.0)
    fast_observation, fast_frames = _visual_rollout(17, actions, simulation_speed=100.0)
    np.testing.assert_array_equal(one_observation, fast_observation)
    np.testing.assert_array_equal(one_frames, fast_frames)

    spec = default_controller_spec("reactive", input_size=5)
    genome = np.zeros(spec.parameter_count)
    baseline = evaluate_candidate(
        genome,
        spec,
        EpisodeConfig(seeds=(17,), max_steps=120, record_trace=True, simulation_speed=1.0),
    )
    accelerated = evaluate_candidate(
        genome,
        spec,
        EpisodeConfig(seeds=(17,), max_steps=120, record_trace=True, simulation_speed=100.0),
    )
    assert baseline.fitness == accelerated.fitness
    assert baseline.selection_score == accelerated.selection_score
    assert baseline.episodes[0].actions == accelerated.episodes[0].actions
    assert baseline.episodes[0].action_scores == accelerated.episodes[0].action_scores
    assert baseline.episodes[0].sensory_trace == accelerated.episodes[0].sensory_trace
    assert baseline.episodes[0].terminated == accelerated.episodes[0].terminated
