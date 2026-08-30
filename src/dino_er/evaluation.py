"""Deterministic multi-seed evaluation through the pixel/keyboard boundary."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from dino_er.controllers import (
    Accelerator,
    ControllerSpec,
    ControllerType,
    PopulationControllerRuntime,
    build_controller,
    genome_hash,
)
from dino_er.environment import DinoArenaEnv
from dino_er.perception import SENSORY_NAMES, PerceptionResult

VISIBLE_DIAGNOSTICS_STRIDE = 3
RESPONSE_CHANGE_EPSILON = 1e-9

StateIntervention = Literal["normal", "reset_each_step"]
SensoryIntervention = Literal["normal", "constant_first_frame", "ablate_sensor"]


@dataclass(frozen=True)
class EpisodeConfig:
    """Fixed episode conditions used for comparable candidate evaluations."""

    seeds: tuple[int, ...]
    max_steps: int = 600
    render_mode: Literal["human", "rgb_array"] | None = None
    state_intervention: StateIntervention = "normal"
    sensory_intervention: SensoryIntervention = "normal"
    ablation_sensor_index: int | None = None
    record_trace: bool = False
    post_episode_hold_seconds: float = 0.0
    wait_for_window_close: bool = False
    accelerator: Accelerator = "cpu"
    study_id: str | None = None
    simulation_speed: float = 1.0

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("At least one episode seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Episode seeds must be unique")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.post_episode_hold_seconds < 0:
            raise ValueError("post_episode_hold_seconds cannot be negative")
        if self.wait_for_window_close and self.render_mode != "human":
            raise ValueError("wait_for_window_close requires render_mode='human'")
        if self.state_intervention not in ("normal", "reset_each_step"):
            raise ValueError("Unsupported state intervention")
        if self.sensory_intervention not in (
            "normal",
            "constant_first_frame",
            "ablate_sensor",
        ):
            raise ValueError("Unsupported sensory intervention")
        if self.sensory_intervention == "ablate_sensor":
            if self.ablation_sensor_index is None or self.ablation_sensor_index < 0:
                raise ValueError("ablate_sensor requires a non-negative sensor index")
        elif self.ablation_sensor_index is not None:
            raise ValueError("ablation_sensor_index requires ablate_sensor")
        if self.accelerator not in ("auto", "cpu", "cuda"):
            raise ValueError("accelerator must be auto, cpu, or cuda")
        if not np.isfinite(self.simulation_speed) or not 0.25 <= self.simulation_speed <= 100:
            raise ValueError("simulation_speed must be between 0.25 and 100")


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    visual_passes: int
    steps: int
    terminated: bool
    truncated: bool
    perception_failures: int
    actions: tuple[int, ...]
    action_scores: tuple[tuple[float, float, float], ...]
    sensory_trace: tuple[tuple[float, ...], ...]
    hidden_trace: tuple[tuple[float, ...], ...]
    obstacle_pass_steps: tuple[int, ...]
    observation_changes: int
    score_changes: int
    action_switches: int
    responsive_transitions: int
    observation_variation_l1: float
    score_variation_l1: float
    action_counts: tuple[int, int, int]
    pterodactyl_visible_steps: int
    duck_on_pterodactyl_steps: int
    cactus_visible_steps: int
    duck_on_cactus_steps: int


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: int
    genome_hash: str
    fitness: float
    selection_score: float
    mean_survival_fraction: float
    observation_changes: int
    score_changes: int
    action_switches: int
    responsive_transitions: int
    response_rate: float | None
    action_response_rate: float | None
    duration_seconds: float
    episodes: tuple[EpisodeResult, ...]
    action_counts: tuple[int, int, int] = (0, 0, 0)
    pterodactyl_visible_steps: int = 0
    pterodactyl_duck_rate: float | None = None
    cactus_visible_steps: int = 0
    cactus_duck_rate: float | None = None


class PopulationEvaluationCancelled(RuntimeError):
    """Raised when the arena requests cancellation before generation selection."""


@dataclass(frozen=True)
class PopulationRunRestartRequested(RuntimeError):
    """Request a fresh evolution/browser run with GUI-selected settings."""

    parent_count: int
    offspring_count: int
    controller_type: ControllerType
    accelerator: Accelerator
    continuous: bool
    generations: int
    max_steps: int
    mutation_scale: float
    evolution_seed: int
    simulation_speed: float
    training_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    final_test_seeds: tuple[int, ...]
    study_id: str | None

    def __post_init__(self) -> None:
        RuntimeError.__init__(
            self,
            f"new GUI run requested: mu={self.parent_count}, lambda={self.offspring_count}, "
            f"controller={self.controller_type}, accelerator={self.accelerator}",
        )

    @property
    def population_size(self) -> int:
        return self.parent_count + self.offspring_count


def _control_seeds(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    if any(not isinstance(seed, int) for seed in value):
        return None
    seeds = tuple(value)
    return seeds if len(set(seeds)) == len(seeds) else None


def evaluate_candidate(
    genome: np.ndarray | list[float],
    spec: ControllerSpec,
    config: EpisodeConfig,
    *,
    generation: int = 0,
    mutation_scale: float = 0.0,
) -> CandidateEvaluation:
    """Evaluate one controller through the same population runtime used in training."""

    session = PopulationEvaluationSession(spec, config, population_size=1)
    try:
        result = session.evaluate(
            [genome],
            generation=generation,
            mutation_scale=mutation_scale,
        )[0]
        if config.render_mode == "human" and config.post_episode_hold_seconds > 0:
            time.sleep(config.post_episode_hold_seconds)
        if config.wait_for_window_close:
            session.wait_until_window_closed()
        return result
    finally:
        session.close()


def evaluate_population_batch(
    genomes: Sequence[np.ndarray | list[float]] | np.ndarray,
    spec: ControllerSpec,
    config: EpisodeConfig,
    *,
    generation: int,
    mutation_scale: float,
    session: PopulationEvaluationSession | None = None,
    generation_history: Sequence[dict[str, float | int]] = (),
) -> list[CandidateEvaluation]:
    """Evaluate one complete 1--100 Dino generation in one shared world."""

    if not genomes:
        raise ValueError("Population cannot be empty")
    if session is not None and not 1 <= session.population_size <= 100:
        raise ValueError("A supplied session must be a 1-100 Dino visual arena")
    if len(genomes) > 100:
        raise ValueError("One scientific generation may contain at most 100 candidates")
    owns_session = session is None
    arena_session = session or PopulationEvaluationSession(
        spec,
        config,
        population_size=len(genomes),
    )
    try:
        return arena_session.evaluate(
            genomes,
            generation=generation,
            mutation_scale=mutation_scale,
            generation_history=generation_history,
        )
    finally:
        if owns_session:
            arena_session.close()


class PopulationEvaluationSession:
    """Persistent one-app evaluator for synchronous full-population generations."""

    def __init__(
        self,
        spec: ControllerSpec,
        config: EpisodeConfig,
        *,
        population_size: int,
        environment: Any | None = None,
    ) -> None:
        if not 1 <= population_size <= 100:
            raise ValueError("population_size must be between 1 and 100")
        self.spec = spec
        self.config = config
        self.population_size = population_size
        self.environment = environment or DinoArenaEnv(
            num_envs=population_size,
            render_mode=config.render_mode,
            simulation_speed=config.simulation_speed,
        )
        self.stop_after_generation = False
        self.paused_now = False
        self.simulation_speed = config.simulation_speed
        self._candidate_save_requests: list[int] = []
        self._candidate_ids = tuple(range(population_size))
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.environment.close()

    def wait_until_window_closed(
        self,
        on_poll: Callable[[], None] | None = None,
    ) -> None:
        """Retain the final visible population frame for user inspection."""

        if self.config.render_mode != "human":
            raise RuntimeError("Waiting for a window requires visible rendering")
        self.environment.wait_until_window_closed(on_poll)

    def poll_controls(self) -> None:
        """Apply all pending presentation controls without touching candidate state."""

        for control in self.environment.poll_controls():
            self._apply_control(control)

    def consume_candidate_save_requests(self) -> tuple[int, ...]:
        """Return queued valid candidate IDs exactly once."""

        requests = tuple(self._candidate_save_requests)
        self._candidate_save_requests.clear()
        return requests

    def requeue_candidate_save_requests(self, requests: Sequence[int]) -> None:
        """Defer saves until a generation has complete fitness values."""

        self._candidate_save_requests[:0] = [
            candidate for candidate in requests if candidate in self._candidate_ids
        ]

    def announce_status(
        self,
        phase: str,
        message: str,
        **fields: Any,
    ) -> None:
        if self.config.render_mode == "human":
            self.environment.send_arena_status(phase, message, **fields)

    def evaluate(
        self,
        genomes: Sequence[np.ndarray | list[float]] | np.ndarray,
        *,
        generation: int,
        mutation_scale: float,
        generation_history: Sequence[dict[str, float | int]] = (),
        episode_config: EpisodeConfig | None = None,
    ) -> list[CandidateEvaluation]:
        """Run every genome on every seed and return one ordered result per genome."""

        if self._closed:
            raise RuntimeError("PopulationEvaluationSession is closed")
        if len(genomes) != self.population_size:
            raise ValueError("Every generation must contain the session's complete population")
        config = episode_config or self.config
        if config.render_mode != self.config.render_mode:
            raise ValueError("One browser session cannot change render mode")
        if (
            config.ablation_sensor_index is not None
            and config.ablation_sensor_index >= self.spec.input_size
        ):
            raise ValueError("ablation_sensor_index exceeds the controller input size")
        self._candidate_ids = tuple(range(self.population_size))
        digests = [genome_hash(genome, self.spec) for genome in genomes]
        controller_runtime = PopulationControllerRuntime(
            genomes,
            self.spec,
            config.accelerator,
        )
        episode_results: list[list[EpisodeResult]] = [[] for _ in range(self.population_size)]
        started = time.perf_counter()

        for seed in config.seeds:
            self.environment.configure_population(
                controller_type=self.spec.controller_type,
                generation=generation,
                mutation_scale=mutation_scale,
                accelerator=controller_runtime.status.label,
                study_id=config.study_id,
            )
            sensory = self.environment.reset(seed=seed)
            current_perceptions = self.environment.last_perception
            first_sensory: np.ndarray | None = None
            last_controller_sensory = np.asarray(sensory, dtype=np.float64).copy()
            controller_runtime.reset(seed)

            passes = np.zeros(self.population_size, dtype=np.int64)
            controller_steps = np.zeros(self.population_size, dtype=np.int64)
            terminated = np.zeros(self.population_size, dtype=np.bool_)
            truncated = np.zeros(self.population_size, dtype=np.bool_)
            perception_failures = np.asarray(
                [int(result.failure_reason is not None) for result in current_perceptions],
                dtype=np.int64,
            )
            pass_steps: list[list[int]] = [[] for _ in range(self.population_size)]
            action_traces: list[list[int]] = [[] for _ in range(self.population_size)]
            score_traces: list[list[tuple[float, float, float]]] = [
                [] for _ in range(self.population_size)
            ]
            last_scores = np.zeros(
                (self.population_size, self.spec.action_size),
                dtype=np.float64,
            )
            sensory_traces: list[list[tuple[float, ...]]] = [
                [] for _ in range(self.population_size)
            ]
            hidden_traces: list[list[tuple[float, ...]]] = [[] for _ in range(self.population_size)]
            seed_started = time.perf_counter()
            shared_step = 0
            last_actions = np.zeros(self.population_size, dtype=np.int64)
            previous_sensory = np.zeros_like(sensory, dtype=np.float64)
            previous_scores = np.zeros(
                (self.population_size, self.spec.action_size),
                dtype=np.float64,
            )
            previous_actions = np.zeros(self.population_size, dtype=np.int64)
            has_previous_response = np.zeros(
                self.population_size,
                dtype=np.bool_,
            )
            observation_changes = np.zeros(
                self.population_size,
                dtype=np.int64,
            )
            score_changes = np.zeros(self.population_size, dtype=np.int64)
            action_switches = np.zeros(self.population_size, dtype=np.int64)
            responsive_transitions = np.zeros(
                self.population_size,
                dtype=np.int64,
            )
            observation_variation_l1 = np.zeros(
                self.population_size,
                dtype=np.float64,
            )
            score_variation_l1 = np.zeros(
                self.population_size,
                dtype=np.float64,
            )
            action_counts = np.zeros((self.population_size, 3), dtype=np.int64)
            pterodactyl_visible_steps = np.zeros(
                self.population_size,
                dtype=np.int64,
            )
            duck_on_pterodactyl_steps = np.zeros(
                self.population_size,
                dtype=np.int64,
            )
            cactus_visible_steps = np.zeros(
                self.population_size,
                dtype=np.int64,
            )
            duck_on_cactus_steps = np.zeros(
                self.population_size,
                dtype=np.int64,
            )

            for shared_step in range(1, config.max_steps + 1):
                self._handle_controls()
                self._wait_while_paused(shared_step)
                active = ~(terminated | truncated)
                if not bool(np.any(active)):
                    break
                controller_sensory = np.asarray(sensory, dtype=np.float64).copy()
                if config.sensory_intervention == "constant_first_frame":
                    if first_sensory is None:
                        first_sensory = controller_sensory.copy()
                    controller_sensory = first_sensory.copy()
                elif config.sensory_intervention == "ablate_sensor":
                    controller_sensory[:, config.ablation_sensor_index] = 0.0
                last_controller_sensory = controller_sensory
                actions, scores_matrix, hidden_states = controller_runtime.act_with_scores(
                    controller_sensory,
                    active,
                    reset_state_each_step=(
                        config.state_intervention == "reset_each_step"
                        and self.spec.controller_type == "proactive"
                    ),
                )
                last_scores[active] = scores_matrix[active]
                comparable = active & has_previous_response
                sensory_delta = np.sum(
                    np.abs(controller_sensory - previous_sensory),
                    axis=1,
                )
                score_delta = np.sum(
                    np.abs(scores_matrix - previous_scores),
                    axis=1,
                )
                observation_changed = comparable & (sensory_delta > RESPONSE_CHANGE_EPSILON)
                score_changed = comparable & (score_delta > RESPONSE_CHANGE_EPSILON)
                action_switched = comparable & (actions != previous_actions)
                observation_changes += observation_changed.astype(np.int64)
                score_changes += score_changed.astype(np.int64)
                action_switches += action_switched.astype(np.int64)
                responsive_transitions += (
                    observation_changed & (score_changed | action_switched)
                ).astype(np.int64)
                observation_variation_l1 += np.where(
                    comparable,
                    sensory_delta,
                    0.0,
                )
                score_variation_l1 += np.where(
                    comparable,
                    score_delta,
                    0.0,
                )
                previous_sensory[active] = controller_sensory[active]
                previous_scores[active] = scores_matrix[active]
                previous_actions[active] = actions[active]
                has_previous_response[active] = True
                controller_steps += active.astype(np.int64)
                active_indices = np.flatnonzero(active)
                np.add.at(
                    action_counts,
                    (active_indices, actions[active_indices]),
                    1,
                )
                obstacle_classes = np.asarray(
                    [result.estimate.obstacle_class for result in current_perceptions],
                    dtype=object,
                )
                pterodactyl_visible = active & (obstacle_classes == "pterodactyl")
                cactus_visible = active & np.isin(
                    obstacle_classes,
                    ("small_cactus", "large_cactus"),
                )
                pterodactyl_visible_steps += pterodactyl_visible.astype(np.int64)
                duck_on_pterodactyl_steps += (pterodactyl_visible & (actions == 2)).astype(np.int64)
                cactus_visible_steps += cactus_visible.astype(np.int64)
                duck_on_cactus_steps += (cactus_visible & (actions == 2)).astype(np.int64)
                if config.record_trace:
                    for raw_index in active_indices:
                        index = int(raw_index)
                        action = int(actions[index])
                        scores = scores_matrix[index]
                        action_traces[index].append(action)
                        score_traces[index].append(
                            (float(scores[0]), float(scores[1]), float(scores[2]))
                        )
                        sensory_traces[index].append(
                            tuple(float(value) for value in controller_sensory[index])
                        )
                        if hidden_states is not None:
                            hidden_traces[index].append(
                                tuple(float(value) for value in hidden_states[index])
                            )
                (
                    sensory,
                    rewards,
                    current_terminated,
                    _,
                    current_perceptions,
                ) = self.environment.step_perceptions(actions, active=active)
                passed = rewards > 0
                passes += passed
                for raw_index in np.flatnonzero(passed):
                    pass_steps[int(raw_index)].append(shared_step)
                perception_failures += np.fromiter(
                    (int(result.failure_reason is not None) for result in current_perceptions),
                    dtype=np.int64,
                    count=self.population_size,
                )
                terminated |= current_terminated
                last_actions = actions
                diagnostics_stride = max(
                    1,
                    round(VISIBLE_DIAGNOSTICS_STRIDE * self.simulation_speed),
                )
                if config.render_mode == "human" and (
                    shared_step == 1 or shared_step % diagnostics_stride == 0
                ):
                    self._publish_diagnostics(
                        step=shared_step,
                        started=seed_started,
                        actions=actions,
                        action_scores=scores_matrix,
                        sensory=controller_sensory,
                        terminated=terminated,
                        truncated=truncated,
                        generation_history=generation_history,
                        accelerator=controller_runtime.status.label,
                        controller_steps=controller_steps,
                        genome_hashes=digests,
                        visual_results=current_perceptions,
                    )

            remaining = ~(terminated | truncated)
            if bool(np.any(remaining)):
                truncated |= remaining
            self._publish_diagnostics(
                step=shared_step,
                started=seed_started,
                actions=last_actions,
                action_scores=last_scores,
                sensory=last_controller_sensory,
                terminated=terminated,
                truncated=truncated,
                generation_history=generation_history,
                accelerator=controller_runtime.status.label,
                controller_steps=controller_steps,
                genome_hashes=digests,
                visual_results=current_perceptions,
            )
            self._handle_controls()

            for index in range(self.population_size):
                episode_results[index].append(
                    EpisodeResult(
                        seed=seed,
                        visual_passes=int(passes[index]),
                        steps=int(controller_steps[index]),
                        terminated=bool(terminated[index]),
                        truncated=bool(truncated[index]),
                        perception_failures=int(perception_failures[index]),
                        actions=tuple(action_traces[index]),
                        action_scores=tuple(score_traces[index]),
                        sensory_trace=tuple(sensory_traces[index]),
                        hidden_trace=tuple(hidden_traces[index]),
                        obstacle_pass_steps=tuple(pass_steps[index]),
                        observation_changes=int(observation_changes[index]),
                        score_changes=int(score_changes[index]),
                        action_switches=int(action_switches[index]),
                        responsive_transitions=int(responsive_transitions[index]),
                        observation_variation_l1=float(observation_variation_l1[index]),
                        score_variation_l1=float(score_variation_l1[index]),
                        action_counts=(
                            int(action_counts[index, 0]),
                            int(action_counts[index, 1]),
                            int(action_counts[index, 2]),
                        ),
                        pterodactyl_visible_steps=int(pterodactyl_visible_steps[index]),
                        duck_on_pterodactyl_steps=int(duck_on_pterodactyl_steps[index]),
                        cactus_visible_steps=int(cactus_visible_steps[index]),
                        duck_on_cactus_steps=int(duck_on_cactus_steps[index]),
                    )
                )

        duration = time.perf_counter() - started
        evaluations = [
            _candidate_evaluation(
                candidate_id=index,
                digest=digests[index],
                episodes=tuple(episode_results[index]),
                max_steps=config.max_steps,
                duration_seconds=duration,
            )
            for index in range(self.population_size)
        ]
        _validate_complete_population(evaluations, digests)
        return evaluations

    def _handle_controls(self) -> None:
        for control in self.environment.poll_controls():
            action = control.get("action")
            if action != "focus_changed":
                self._apply_control(control)

    def _apply_control(self, control: dict[str, Any]) -> None:
        action = control.get("action")
        if action == "cancel":
            raise PopulationEvaluationCancelled(
                "Population evaluation cancelled before generation selection"
            )
        if action == "start_new_run":
            parent_count = control.get("parentCount")
            offspring_count = control.get("offspringCount")
            controller_type = control.get("controllerType")
            accelerator = control.get("accelerator")
            mode = control.get("mode")
            generations = control.get("generations")
            max_steps = control.get("maxSteps")
            mutation_scale = control.get("mutationScale")
            evolution_seed = control.get("evolutionSeed")
            simulation_speed = control.get("simulationSpeed")
            training_seeds = _control_seeds(control.get("trainingSeeds"))
            validation_seeds = _control_seeds(control.get("validationSeeds"))
            final_test_seeds = _control_seeds(control.get("finalTestSeeds"))
            study_id = control.get("studyId")
            scientific = mode == "scientific"
            if (
                not isinstance(parent_count, int)
                or not 1 <= parent_count <= 99
                or not isinstance(offspring_count, int)
                or not 1 <= offspring_count <= 99
                or parent_count + offspring_count > 100
                or controller_type not in ("reactive", "proactive")
                or accelerator not in ("auto", "cpu", "cuda")
                or mode not in ("continuous", "scientific")
                or not isinstance(generations, int)
                or not 1 <= generations <= 10_000
                or not isinstance(max_steps, int)
                or not 30 <= max_steps <= 1_000_000
                or not isinstance(mutation_scale, (float, int))
                or not 0 < float(mutation_scale) <= 5
                or not isinstance(evolution_seed, int)
                or not isinstance(simulation_speed, (float, int))
                or not 0.25 <= float(simulation_speed) <= 100
                or training_seeds is None
                or not training_seeds
                or (
                    scientific
                    and (
                        not isinstance(study_id, str)
                        or re.fullmatch(
                            r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",
                            study_id,
                        )
                        is None
                        or not validation_seeds
                        or not final_test_seeds
                    )
                )
            ):
                self.announce_status(
                    "restarting",
                    "new-run request rejected: check run mode and all experiment fields",
                )
                return
            if scientific:
                partitions = (
                    set(training_seeds),
                    set(validation_seeds or ()),
                    set(final_test_seeds or ()),
                )
                if any(
                    left & right
                    for index, left in enumerate(partitions)
                    for right in partitions[index + 1 :]
                ):
                    self.announce_status(
                        "restarting",
                        "new-run request rejected: seed partitions must be disjoint",
                    )
                    return
            raise PopulationRunRestartRequested(
                parent_count,
                offspring_count,
                cast(ControllerType, controller_type),
                cast(Accelerator, accelerator),
                not scientific,
                generations,
                max_steps,
                float(mutation_scale),
                evolution_seed,
                float(simulation_speed),
                training_seeds,
                validation_seeds or (),
                final_test_seeds or (),
                study_id if scientific else None,
            )
        if action == "set_simulation_speed":
            speed = control.get("simulationSpeed")
            if isinstance(speed, (float, int)) and 0.25 <= float(speed) <= 100:
                self.simulation_speed = float(speed)
                self.announce_status(
                    "evaluating",
                    f"simulation speed set to {self.simulation_speed:.2f}x; "
                    "fixed-step dynamics are unchanged",
                )
            return
        if action == "save_candidate":
            candidate = control.get("candidateId")
            if isinstance(candidate, int) and candidate in self._candidate_ids:
                self._candidate_save_requests.append(candidate)
                self.announce_status(
                    "evaluating",
                    f"candidate {candidate} save queued for the complete generation",
                )
            return
        if action == "stop_after_generation":
            self.stop_after_generation = True
        elif action == "pause_now":
            self.paused_now = True
        elif action == "next_generation":
            self.paused_now = False

    def _wait_while_paused(
        self,
        shared_step: int,
    ) -> None:
        if not self.paused_now:
            return
        self.announce_status(
            "paused",
            f"paused at control step {shared_step}; use Start / next to continue",
        )
        while self.paused_now:
            self._handle_controls()
            time.sleep(0.05)

    def _publish_diagnostics(
        self,
        *,
        step: int,
        started: float,
        actions: np.ndarray,
        action_scores: np.ndarray,
        sensory: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        generation_history: Sequence[dict[str, float | int]],
        accelerator: str,
        controller_steps: np.ndarray,
        genome_hashes: Sequence[str],
        visual_results: Sequence[PerceptionResult],
    ) -> None:
        if self.config.render_mode != "human":
            return
        candidates: list[dict[str, Any]] = []
        for index in range(self.population_size):
            candidates.append(
                {
                    "id": self._candidate_ids[index],
                    "alive": not bool(terminated[index] or truncated[index]),
                    "action": int(actions[index]),
                    "actionScores": [float(value) for value in action_scores[index]],
                    "genomeHash": genome_hashes[index],
                    "sensory": [float(value) for value in sensory[index]],
                    "sensoryNames": list(SENSORY_NAMES),
                    "currentFitness": float(controller_steps[index]),
                    "obstacleClass": visual_results[index].estimate.obstacle_class,
                }
            )
        self.environment.send_population_diagnostics(
            {
                "alive": int(np.count_nonzero(~(terminated | truncated))),
                "candidates": candidates,
                "generationHistory": [dict(point) for point in generation_history],
                "accelerator": accelerator,
                "controlTicksPerSecond": (step / max(time.perf_counter() - started, 1e-9)),
                "simulationSpeed": self.simulation_speed,
            }
        )


def _candidate_evaluation(
    *,
    candidate_id: int,
    digest: str,
    episodes: tuple[EpisodeResult, ...],
    max_steps: int,
    duration_seconds: float,
) -> CandidateEvaluation:
    """Aggregate the deliberately simple survival-time objective."""

    seed_count = len(episodes)
    if seed_count == 0:
        raise ValueError("Candidate evaluation requires at least one episode")
    # Survival until the visually detected game-over state is the dense
    # control-step objective used for selection. It is intentionally distinct
    # from Chromium's five-digit distance score and preserves the black-box
    # boundary and distinguishes candidates that crash before passing their
    # first obstacle.
    fitness = float(np.mean([episode.steps for episode in episodes]))
    survival = float(
        np.mean(
            [
                (
                    1.0
                    if episode.truncated and not episode.terminated
                    else min(1.0, episode.steps / max_steps)
                )
                for episode in episodes
            ]
        )
    )
    # There is no action penalty, novelty term or obstacle-count tie-breaker.
    # Evolution ranks exactly the same mean survival time that the UI reports.
    selection_score = fitness
    observation_changes = sum(episode.observation_changes for episode in episodes)
    score_changes = sum(episode.score_changes for episode in episodes)
    action_switches = sum(episode.action_switches for episode in episodes)
    responsive_transitions = sum(episode.responsive_transitions for episode in episodes)
    action_counts = tuple(
        sum(episode.action_counts[action] for episode in episodes) for action in range(3)
    )
    pterodactyl_visible_steps = sum(episode.pterodactyl_visible_steps for episode in episodes)
    duck_on_pterodactyl_steps = sum(episode.duck_on_pterodactyl_steps for episode in episodes)
    cactus_visible_steps = sum(episode.cactus_visible_steps for episode in episodes)
    duck_on_cactus_steps = sum(episode.duck_on_cactus_steps for episode in episodes)
    return CandidateEvaluation(
        candidate_id=candidate_id,
        genome_hash=digest,
        fitness=fitness,
        selection_score=selection_score,
        mean_survival_fraction=survival,
        observation_changes=observation_changes,
        score_changes=score_changes,
        action_switches=action_switches,
        responsive_transitions=responsive_transitions,
        response_rate=(
            responsive_transitions / observation_changes if observation_changes > 0 else None
        ),
        action_response_rate=(
            action_switches / observation_changes if observation_changes > 0 else None
        ),
        duration_seconds=duration_seconds,
        episodes=episodes,
        action_counts=(
            int(action_counts[0]),
            int(action_counts[1]),
            int(action_counts[2]),
        ),
        pterodactyl_visible_steps=pterodactyl_visible_steps,
        pterodactyl_duck_rate=(
            duck_on_pterodactyl_steps / pterodactyl_visible_steps
            if pterodactyl_visible_steps > 0
            else None
        ),
        cactus_visible_steps=cactus_visible_steps,
        cactus_duck_rate=(
            duck_on_cactus_steps / cactus_visible_steps if cactus_visible_steps > 0 else None
        ),
    )


def _validate_complete_population(
    evaluations: Sequence[CandidateEvaluation],
    expected_hashes: Sequence[str],
) -> None:
    if len(evaluations) != len(expected_hashes):
        raise RuntimeError("Evaluator did not return the complete population")
    for index, (evaluation, expected_hash) in enumerate(
        zip(evaluations, expected_hashes, strict=True)
    ):
        if evaluation.candidate_id != index or evaluation.genome_hash != expected_hash:
            raise RuntimeError("Evaluator changed candidate order or identity")
        if not np.isfinite(evaluation.fitness):
            raise RuntimeError("Evaluator returned a non-finite fitness")
        if not np.isfinite(evaluation.selection_score):
            raise RuntimeError("Evaluator returned a non-finite selection score")


def sensory_intervention_analysis(
    genome: np.ndarray | list[float],
    spec: ControllerSpec,
    sensory_trace: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Compare observed visual inputs with a constant-input counterfactual."""

    if not sensory_trace:
        raise ValueError("Sensory intervention requires a non-empty trace")
    observed_controller = build_controller(genome, spec)
    constant_controller = build_controller(genome, spec)
    observed_controller.reset(0)
    constant_controller.reset(0)
    constant = np.asarray(sensory_trace[0], dtype=np.float64)
    observed_actions: list[int] = []
    constant_actions: list[int] = []
    changed_score_steps = 0
    score_l1_difference = 0.0
    for sensory_values in sensory_trace:
        sensory = np.asarray(sensory_values, dtype=np.float64)
        observed_action, observed_scores = observed_controller.act_with_scores(sensory)
        constant_action, constant_scores = constant_controller.act_with_scores(constant)
        observed_actions.append(observed_action)
        constant_actions.append(constant_action)
        difference = float(np.sum(np.abs(observed_scores - constant_scores)))
        score_l1_difference += difference
        changed_score_steps += int(difference > RESPONSE_CHANGE_EPSILON)
    changed_action_steps = sum(
        observed != constant_action
        for observed, constant_action in zip(
            observed_actions,
            constant_actions,
            strict=True,
        )
    )
    return {
        "intervention": "replace every visual sensory vector by the first vector",
        "steps": len(observed_actions),
        "changed_action_steps": changed_action_steps,
        "changed_score_steps": changed_score_steps,
        "score_l1_difference": score_l1_difference,
        "observed_action_counts": {
            str(action): observed_actions.count(action) for action in range(3)
        },
        "constant_input_action_counts": {
            str(action): constant_actions.count(action) for action in range(3)
        },
        "uses_visual_input": changed_score_steps > 0,
        "changes_discrete_actions": changed_action_steps > 0,
        "interpretation": (
            "controller-level causal input intervention; it tests sensory "
            "dependence, not whether the learned policy is effective"
        ),
    }
