from __future__ import annotations

import queue
import threading
from collections import deque

import numpy as np
import pytest

from dino_er.browser import BrowserVisualTransport, VisualTransportError
from dino_er.environment import DinoArenaEnv
from dino_er.perception import SENSORY_NAMES


def _assert_tight_atlas_preserves_legacy_visible_pixels(
    environment: DinoArenaEnv,
) -> None:
    """The v4 crops may remove blank margins, never rendered observations."""

    batch = environment._last_pixel_batch  # noqa: SLF001 - protocol regression test
    assert batch is not None
    frame = batch.materialize()[0, :, :, 0]
    shared = batch.shared_world
    assert np.count_nonzero(frame[:, 40:120] != shared[:, 40:120]) == np.count_nonzero(
        frame[:, 50:110] != shared[:, 50:110]
    )
    assert np.count_nonzero(frame[39:58, 198:402] != shared[39:58, 198:402]) == np.count_nonzero(
        frame[42:53, 205:396] != shared[42:53, 205:396]
    )
    assert np.count_nonzero(frame[70:112, 276:324] != shared[70:112, 276:324]) == np.count_nonzero(
        frame[75:107, 284:320] != shared[75:107, 284:320]
    )


def test_size_one_uses_the_population_pixel_boundary() -> None:
    environment = DinoArenaEnv(num_envs=1, render_mode="rgb_array")
    try:
        initial = environment.reset(seed=7)
        batch = environment._last_pixel_batch  # noqa: SLF001 - pixel-boundary regression
        assert batch is not None
        frame = batch.materialize()
        assert frame.shape == (1, 150, 600, 3)
        assert frame.dtype == np.uint8
        _assert_tight_atlas_preserves_legacy_visible_pixels(environment)

        jumped, _, _, _, jump_results = environment.step_perceptions([1])
        assert jumped[0, 4] < initial[0, 4]
        assert jump_results[0].dino_box is not None

        environment.reset(seed=7)
        _, _, _, _, duck_results = environment.step_perceptions([2])
        assert duck_results[0].dino_box is not None
        assert duck_results[0].dino_box.width >= 50

        environment.reset(seed=7)
        terminated = np.asarray([False])
        reward = 0.0
        # The pinned task waits three seconds before the first obstacle; at
        # two 60-Hz physics frames per controller decision, this deterministic
        # seed collides after roughly 270 controller steps.
        for _ in range(300):
            _, rewards, terminated, truncated, _ = environment.step_perceptions([0])
            reward = float(rewards[0])
            assert not truncated
            if terminated[0]:
                break
        assert terminated[0]
        assert reward == pytest.approx(-1.0)
        _assert_tight_atlas_preserves_legacy_visible_pixels(environment)
        with pytest.raises(ValueError):
            environment.step_perceptions([3])
    finally:
        environment.close()


def test_four_instances_have_isolated_actions_and_pixel_frames() -> None:
    environment = DinoArenaEnv(num_envs=4, render_mode="rgb_array")
    try:
        observations = environment.reset(seed=10)
        assert observations.shape == (4, len(SENSORY_NAMES))
        actions = np.asarray([1, 0, 2, 1], dtype=np.int64)
        stepped, rewards, terminated, truncated, step_results = environment.step_perceptions(
            actions
        )
        batch = environment._last_pixel_batch  # noqa: SLF001 - pixel-boundary regression
        assert batch is not None
        frames = batch.materialize()
        assert stepped.shape == (4, len(SENSORY_NAMES))
        assert rewards.shape == (4,)
        assert not terminated.any()
        assert not truncated.any()
        assert frames.shape == (4, 150, 600, 3)
        assert stepped[0, 4] < stepped[1, 4]
        assert step_results[2].dino_box is not None
        assert step_results[2].dino_box.width >= 50
    finally:
        environment.close()


def test_one_browser_supports_one_hundred_instances() -> None:
    environment = DinoArenaEnv(num_envs=100, render_mode="rgb_array")
    try:
        observations = environment.reset(seed=20)
        stepped, rewards, terminated, truncated, _ = environment.step_perceptions(
            np.zeros(100, dtype=np.int64)
        )
        batch = environment._last_pixel_batch  # noqa: SLF001 - pixel-boundary regression
        assert batch is not None
        frames = batch.materialize()
        assert observations.shape == (100, len(SENSORY_NAMES))
        assert stepped.shape == (100, len(SENSORY_NAMES))
        assert rewards.shape == (100,)
        assert not terminated.any()
        assert not truncated.any()
        assert frames.shape == (100, 150, 600, 3)
    finally:
        environment.close()


def test_each_private_frame_matches_the_same_candidate_playing_alone() -> None:
    action_history = np.asarray(
        [
            [1, 0],
            [1, 2],
            [0, 0],
            [0, 1],
            [2, 0],
        ],
        dtype=np.int64,
    )
    arena = DinoArenaEnv(num_envs=2, render_mode="rgb_array")
    try:
        arena.reset(seed=44)
        for actions in action_history:
            arena.step_perceptions(actions)
        batch = arena._last_pixel_batch  # noqa: SLF001 - pixel-boundary regression
        assert batch is not None
        arena_frames = batch.materialize().copy()
    finally:
        arena.close()

    for candidate_id in range(2):
        solo = DinoArenaEnv(num_envs=1, render_mode="rgb_array")
        try:
            solo.reset(seed=44)
            for action in action_history[:, candidate_id]:
                solo.step_perceptions([int(action)])
            batch = solo._last_pixel_batch  # noqa: SLF001 - pixel-boundary regression
            assert batch is not None
            solo_frame = batch.materialize()
            np.testing.assert_array_equal(
                arena_frames[candidate_id],
                solo_frame[0],
            )
        finally:
            solo.close()


def test_disconnected_visual_channel_fails_without_waiting_for_the_timeout() -> None:
    """A dropped WebSocket cannot leave an evolution generation hanging."""

    transport = object.__new__(BrowserVisualTransport)
    transport.timeout_seconds = 60.0
    transport._messages = queue.Queue()
    transport._arena_controls = queue.Queue()
    transport._connection_lost = threading.Event()
    transport._connection_lost.set()
    transport._window_closed = threading.Event()
    transport._connection_failure = "ConnectionClosedError: no close frame received"
    transport._crashed = False
    transport._closed = False
    transport._browser_log = deque()

    with pytest.raises(VisualTransportError, match="visual channel disconnected"):
        transport._wait_message(expected_type="frame_batch")
