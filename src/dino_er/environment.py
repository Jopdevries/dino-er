"""Shared-world Chrome Dino environment behind the visual black-box boundary."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np

from dino_er.browser import BrowserVisualTransport, PopulationPixelBatch
from dino_er.perception import (
    PerceptionResult,
    VisualPerception,
    process_population_pixel_components,
)

RenderMode = Literal["human", "rgb_array"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class DinoArenaEnv:
    """Shared-world population arena with one private clean frame per candidate."""

    def __init__(
        self,
        num_envs: int = 100,
        *,
        render_mode: RenderMode | None = None,
        simulation_speed: float = 1.0,
        timeout_seconds: float = 20.0,
    ) -> None:
        if not 1 <= num_envs <= 100:
            raise ValueError("num_envs must be between 1 and 100")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.num_envs = num_envs
        self.render_mode = render_mode
        self._transport = BrowserVisualTransport(
            distribution_dir=Path(__file__).resolve().parents[2] / "game" / "dist",
            headless=render_mode != "human",
            instances=num_envs,
            simulation_speed=simulation_speed,
            timeout_seconds=timeout_seconds,
        )
        self._perception = [VisualPerception(create_overlay=False) for _ in range(num_envs)]
        self._last_pixel_batch: PopulationPixelBatch | None = None
        self._last_results: list[PerceptionResult] = []
        self._terminated = np.zeros(num_envs, dtype=np.bool_)
        self._metadata: dict[str, Any] = {
            "controllerType": "replay",
            "generation": 0,
            "mutationScale": None,
            "accelerator": "unconfigured",
        }

    def configure_population(
        self,
        *,
        controller_type: Literal["reactive", "proactive", "replay"],
        generation: int,
        mutation_scale: float | None,
        accelerator: str = "cpu - unspecified",
        study_id: str | None = None,
    ) -> None:
        self._metadata = {
            "controllerType": controller_type,
            "generation": generation,
            "mutationScale": mutation_scale,
            "accelerator": accelerator,
            "runKind": "scientific" if study_id is not None else "engineering",
        }

    def reset(
        self,
        *,
        seed: int,
    ) -> np.ndarray:
        for perception in self._perception:
            perception.reset()
        self._last_pixel_batch = self._transport.configure_arena(
            seed=int(seed),
            metadata=self._metadata,
        )
        results = self._process_pixel_batch(self._last_pixel_batch)
        self._last_results = results
        self._terminated.fill(False)
        return np.stack([result.sensory for result in results])

    def step_perceptions(
        self,
        actions: np.ndarray | list[int],
        *,
        active: np.ndarray | list[bool] | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[PerceptionResult, ...],
    ]:
        action_list = np.asarray(actions, dtype=np.int64).tolist()
        if len(action_list) != self.num_envs:
            raise ValueError("Action count must match num_envs")
        active_list = (
            [True] * self.num_envs
            if active is None
            else np.asarray(active, dtype=np.bool_).tolist()
        )
        if len(active_list) != self.num_envs:
            raise ValueError("Active count must match num_envs")
        self._last_pixel_batch = self._transport.act_batch(
            action_list,
            active=active_list,
        )
        results = self._process_pixel_batch(self._last_pixel_batch)
        self._last_results = results
        observations = np.stack([result.sensory for result in results])
        rewards = np.asarray([result.reward for result in results], dtype=np.float32)
        current_terminated = np.asarray(
            [result.terminated for result in results],
            dtype=np.bool_,
        )
        self._terminated |= current_terminated
        truncated = np.zeros(self.num_envs, dtype=np.bool_)
        return (
            observations,
            rewards,
            self._terminated.copy(),
            truncated,
            tuple(results),
        )

    @property
    def last_perception(self) -> tuple[PerceptionResult, ...]:
        return tuple(self._last_results)

    def send_population_diagnostics(self, payload: dict[str, Any]) -> None:
        self._transport.send_arena_diagnostics(_json_safe(payload))

    def send_arena_status(
        self,
        phase: str,
        message: str,
        **fields: Any,
    ) -> None:
        self._transport.send_arena_status(phase, message, **_json_safe(fields))

    def poll_controls(self) -> list[dict[str, Any]]:
        return self._transport.poll_arena_controls()

    def close(self) -> None:
        self._transport.close()

    def wait_until_window_closed(
        self,
        on_poll: Callable[[], None] | None = None,
    ) -> None:
        """Keep the visible population arena open until Chrome is closed."""

        if self.render_mode != "human":
            raise RuntimeError("Waiting for a window requires render_mode='human'")
        self._transport.wait_until_window_closed(on_poll)

    def _process_pixel_batch(
        self,
        batch: PopulationPixelBatch,
    ) -> list[PerceptionResult]:
        return process_population_pixel_components(
            self._perception,
            batch.shared_world,
            batch.dino_patches,
            batch.game_over_text_patches,
            batch.restart_patches,
        )
