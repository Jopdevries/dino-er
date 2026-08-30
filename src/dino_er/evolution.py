"""Elitist (mu + lambda) evolution, logging, and checkpoint/resume."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from dino_er.controllers import (
    Accelerator,
    ControllerSpec,
    ControllerType,
    default_controller_spec,
    genome_hash,
    validate_genome,
)
from dino_er.evaluation import (
    CandidateEvaluation,
    EpisodeConfig,
    PopulationEvaluationCancelled,
    PopulationEvaluationSession,
    evaluate_population_batch,
    sensory_intervention_analysis,
)
from dino_er.perception import SENSORY_NAMES
from dino_er.scientific import (
    ParquetResultStore,
    figure_data_urls,
    plot_behaviour_trace,
)

CHECKPOINT_FORMAT_VERSION = 6
SAVED_MODEL_FORMAT_VERSION = 2
EVOLUTION_METHOD = "elitist_mu_plus_lambda_rank_biased_plateau_aware_sigma_es"
MAX_POPULATION_SIZE = 100
SIGMA_SUCCESS_TARGET = 0.2
SIGMA_ADAPTATION_FACTOR = 0.85
SIGMA_MIN_FACTOR = 0.1
SIGMA_MAX_FACTOR = 1.0
SIGMA_PLATEAU_MAX_FACTOR = 2.0


@dataclass(frozen=True)
class EvolutionConfig:
    """Reproducible configuration for the single project evolution strategy."""

    controller_type: ControllerType
    training_seeds: tuple[int, ...]
    generations: int
    parent_count: int
    offspring_count: int
    mutation_scale: float
    evolution_seed: int
    max_steps: int
    run_id: str
    output_dir: Path
    adaptive_sigma: bool = True
    results_root: Path = Path("results")
    model_version: int = 1
    # Visible 1x is the Chromium-rate baseline: one controller update per
    # fixed 60-Hz physics frame, per wall-clock second.
    simulation_speed: float = 1.0
    render_mode: Literal["human", "rgb_array"] | None = None
    continuous: bool = False
    accelerator: Accelerator = "cpu"
    study_id: str | None = None
    scientific_study_dir: Path | None = None
    validation_seeds: tuple[int, ...] = ()
    final_test_seeds: tuple[int, ...] = ()
    evaluate_final_test: bool = False
    campaign_timestamp: str | None = None
    result_model_hash: str | None = None

    @property
    def population_size(self) -> int:
        """The unique ES candidates evaluated together in one shared arena."""

        return self.parent_count + self.offspring_count

    def __post_init__(self) -> None:
        if self.controller_type not in ("reactive", "proactive"):
            raise ValueError("controller_type must be reactive or proactive")
        if not self.training_seeds:
            raise ValueError("training_seeds cannot be empty")
        if self.generations <= 0:
            raise ValueError("generations must be positive")
        if self.parent_count < 1:
            raise ValueError("parent_count (mu) must be at least 1")
        if self.offspring_count < 1:
            raise ValueError("offspring_count (lambda) must be at least 1")
        if self.offspring_count > MAX_POPULATION_SIZE - 1:
            raise ValueError(f"offspring_count (lambda) must be at most {MAX_POPULATION_SIZE - 1}")
        if self.population_size > MAX_POPULATION_SIZE:
            raise ValueError(
                f"parent_count + offspring_count must be at most {MAX_POPULATION_SIZE}"
            )
        if self.mutation_scale <= 0:
            raise ValueError("mutation_scale must be positive")
        if not isinstance(self.adaptive_sigma, bool):
            raise ValueError("adaptive_sigma must be boolean")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not np.isfinite(self.simulation_speed) or not 0.25 <= self.simulation_speed <= 100:
            raise ValueError("simulation_speed must be between 0.25 and 100")
        if self.model_version < 1:
            raise ValueError("model_version must be at least 1")
        if self.render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, human, or rgb_array")
        if self.accelerator not in ("auto", "cpu", "cuda"):
            raise ValueError("accelerator must be auto, cpu, or cuda")
        if self.campaign_timestamp is not None and (
            not self.campaign_timestamp
            or any(
                character
                not in ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._+-")
                for character in self.campaign_timestamp
            )
        ):
            raise ValueError("campaign_timestamp must be a filesystem-safe identifier")
        if self.result_model_hash is not None and (
            len(self.result_model_hash) != 12
            or any(character not in "0123456789abcdef" for character in self.result_model_hash)
        ):
            raise ValueError("result_model_hash must be a 12-character lowercase hexadecimal hash")
        scientific_fields = (
            self.study_id is not None,
            self.scientific_study_dir is not None,
        )
        if any(scientific_fields) and not all(scientific_fields):
            raise ValueError("study_id and scientific_study_dir must be configured together")
        if self.study_id is not None:
            if self.continuous:
                raise ValueError("scientific study runs require a finite generation budget")
            if not self.validation_seeds or not self.final_test_seeds:
                raise ValueError("scientific study runs require validation and final-test seeds")
            partitions = (
                set(self.training_seeds),
                set(self.validation_seeds),
                set(self.final_test_seeds),
            )
            if any(
                left & right
                for index, left in enumerate(partitions)
                for right in partitions[index + 1 :]
            ):
                raise ValueError("training, validation and final-test seeds must be disjoint")
        elif self.evaluate_final_test:
            raise ValueError("evaluate_final_test requires a scientific study configuration")


@dataclass(frozen=True)
class GenerationRecord:
    generation: int
    evaluations: int
    seed_set: tuple[int, ...]
    mutation_scale: float
    mutation_scale_next: float
    mutation_scale_reason: str
    offspring_success_rate: float
    duration_seconds: float
    fitness_min: float
    fitness_mean: float
    fitness_max: float
    parent_fitness_mean: float
    offspring_fitness_mean: float
    selected_parent_fitness_mean: float
    selected_offspring_count: int
    selection_score_min: float
    selection_score_mean: float
    selection_score_max: float
    mean_survival_fraction: float
    distinct_primary_fitness_values: int
    best_genome_hash: str
    selected_parent_hashes: tuple[str, ...]
    candidate_fitness: tuple[float, ...]
    candidate_selection_score: tuple[float, ...]
    candidate_survival_fraction: tuple[float, ...]
    candidate_response_rate: tuple[float | None, ...]
    candidate_action_response_rate: tuple[float | None, ...]
    candidate_action_counts: tuple[tuple[int, int, int], ...]
    candidate_pterodactyl_visible_steps: tuple[int, ...]
    candidate_pterodactyl_duck_rate: tuple[float | None, ...]
    candidate_cactus_visible_steps: tuple[int, ...]
    candidate_cactus_duck_rate: tuple[float | None, ...]


@dataclass(frozen=True)
class EvolutionResult:
    config: EvolutionConfig
    controller_spec: ControllerSpec
    records: tuple[GenerationRecord, ...]
    best_genome: np.ndarray
    best_genome_hash: str
    best_fitness: float
    best_selection_score: float
    checkpoint_path: Path
    saved_candidate_paths: tuple[str, ...] = ()
    scientific_output_path: str | None = None
    aggregate_output_path: str | None = None
    scientific_evaluation: dict[str, Any] | None = None


def initial_parent_population(
    parameter_count: int,
    parent_count: int,
    mutation_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the initial mu parents from a zero-centred isotropic Gaussian."""

    if parameter_count < 1 or parent_count < 1 or mutation_scale <= 0:
        raise ValueError("parameter_count, parent_count and mutation_scale must be positive")
    return np.ascontiguousarray(
        rng.normal(0.0, mutation_scale, size=(parent_count, parameter_count)),
        dtype=np.float64,
    )


def mutate_offspring(
    parents: np.ndarray,
    offspring_count: int,
    mutation_scale: float,
    rng: np.random.Generator,
    parent_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Create lambda offspring using theta' = theta_parent + sigma * epsilon."""

    canonical = np.asarray(parents, dtype=np.float64)
    if canonical.ndim != 2 or canonical.shape[0] < 1 or canonical.shape[1] < 1:
        raise ValueError("parents must be a non-empty two-dimensional array")
    if not np.isfinite(canonical).all():
        raise ValueError("parents contain non-finite values")
    if offspring_count < 1 or mutation_scale <= 0:
        raise ValueError("offspring_count and mutation_scale must be positive")
    if parent_indices is None:
        indices = np.arange(offspring_count, dtype=np.int64) % canonical.shape[0]
    else:
        indices = np.asarray(parent_indices, dtype=np.int64)
        if (
            indices.shape != (offspring_count,)
            or np.any(indices < 0)
            or np.any(indices >= canonical.shape[0])
        ):
            raise ValueError("parent_indices must select one valid parent per offspring")
    epsilon = rng.standard_normal((offspring_count, canonical.shape[1]))
    return np.ascontiguousarray(
        canonical[indices] + mutation_scale * epsilon,
        dtype=np.float64,
    )


def rank_biased_offspring_parents(
    parent_selection_scores: Sequence[float] | np.ndarray,
    offspring_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Allocate offspring to every elite once, then favour higher ranked parents.

    Equal parent scores receive exactly equal probability.  This preserves
    diversity during a tie plateau, while successful parents receive more local
    Gaussian samples once selection has supplied a genuine ordering.
    """

    scores = np.asarray(parent_selection_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size < 1 or offspring_count < 1:
        raise ValueError("parent_selection_scores and offspring_count must be non-empty")
    parent_count = scores.size
    if not np.isfinite(scores).all():
        return np.arange(offspring_count, dtype=np.int64) % parent_count
    ranked = np.argsort(-scores, kind="stable")
    weights = np.empty(parent_count, dtype=np.float64)
    start = 0
    while start < parent_count:
        end = start + 1
        score = scores[ranked[start]]
        while end < parent_count and scores[ranked[end]] == score:
            end += 1
        average_rank_weight = parent_count - ((start + end - 1) / 2.0)
        weights[ranked[start:end]] = average_rank_weight
        start = end
    probabilities = weights / np.sum(weights)
    protected = np.arange(min(parent_count, offspring_count), dtype=np.int64)
    remaining = offspring_count - protected.size
    if remaining == 0:
        return protected
    sampled = rng.choice(parent_count, size=remaining, replace=True, p=probabilities)
    return np.concatenate((protected, np.asarray(sampled, dtype=np.int64)))


def adapt_mutation_scale(
    mutation_scale: float,
    parent_selection_scores: Sequence[float] | np.ndarray,
    offspring_selection_scores: Sequence[float] | np.ndarray,
    offspring_parent_indices: Sequence[int] | np.ndarray,
    *,
    initial_mutation_scale: float,
) -> tuple[float, float, str]:
    """Apply a plateau-aware deterministic one-fifth rule to Gaussian sigma.

    A mutation succeeds only when its offspring strictly outperforms its own
    source parent on the same common-random-number training seeds. Ties are not
    successes. On a flat retained-parent plateau with no successful offspring,
    contracting sigma would remove exploration despite the absence of local
    improvement information. That case expands sigma up to twice its initial
    value. The returned tuple is ``(success_rate, next_sigma, reason)``.
    """

    parents = np.asarray(parent_selection_scores, dtype=np.float64)
    offspring = np.asarray(offspring_selection_scores, dtype=np.float64)
    source_indices = np.asarray(offspring_parent_indices, dtype=np.int64)
    if (
        mutation_scale <= 0
        or initial_mutation_scale <= 0
        or parents.ndim != 1
        or offspring.ndim != 1
        or source_indices.shape != offspring.shape
        or offspring.size < 1
        or np.any(source_indices < 0)
        or np.any(source_indices >= parents.size)
        or not np.isfinite(parents).all()
        or not np.isfinite(offspring).all()
    ):
        raise ValueError("Adaptive-sigma inputs must be finite and shape-compatible")
    success_rate = float(np.mean(offspring > parents[source_indices]))
    flat_unsuccessful_plateau = bool(
        success_rate == 0.0
        and np.ptp(parents) == 0.0
        and float(np.max(offspring)) <= float(parents[0])
    )
    upper_factor = SIGMA_MAX_FACTOR
    if flat_unsuccessful_plateau:
        proposed = initial_mutation_scale * SIGMA_PLATEAU_MAX_FACTOR
        upper_factor = SIGMA_PLATEAU_MAX_FACTOR
        reason = "flat_plateau_expand"
    elif success_rate > SIGMA_SUCCESS_TARGET:
        proposed = mutation_scale / SIGMA_ADAPTATION_FACTOR
        reason = "success_above_target_expand_or_cap"
    elif success_rate < SIGMA_SUCCESS_TARGET:
        proposed = mutation_scale * SIGMA_ADAPTATION_FACTOR
        reason = "success_below_target_contract"
    else:
        proposed = mutation_scale
        reason = "success_at_target_hold"
    lower = initial_mutation_scale * SIGMA_MIN_FACTOR
    upper = initial_mutation_scale * upper_factor
    return success_rate, float(np.clip(proposed, lower, upper)), reason


def select_parent_indices(
    selection_scores: Sequence[float] | np.ndarray,
    parent_count: int,
) -> np.ndarray:
    """Return the stable top-mu ranking for maximised selection scores.

    Existing parents are placed before offspring by the runner. Stable sorting
    therefore keeps the unchanged parent when scores tie, which implements
    strict elitism without an additional archive injection.
    """

    scores = np.asarray(selection_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size < 1 or not np.isfinite(scores).all():
        raise ValueError("selection_scores must be a non-empty finite vector")
    if not 1 <= parent_count <= scores.size:
        raise ValueError("parent_count must be between 1 and the number of scores")
    return np.argsort(-scores, kind="stable")[:parent_count]


class EvolutionRunner:
    """Explicit deterministic (mu + lambda)-ES with strict elitism."""

    def __init__(
        self,
        config: EvolutionConfig,
        *,
        checkpoint_path: Path | None = None,
        progress: Callable[[str], None] | None = None,
        keep_open_after_run: bool = False,
    ) -> None:
        self.config = config
        self._progress = progress
        self._keep_open_after_run = keep_open_after_run
        if keep_open_after_run and config.render_mode != "human":
            raise ValueError("keep_open_after_run requires visible rendering")
        self.spec = default_controller_spec(config.controller_type, len(SENSORY_NAMES))
        self.checkpoint_path = checkpoint_path or (
            config.output_dir / f"{config.controller_type}-checkpoint.pkl"
        )
        self.records: list[GenerationRecord] = []
        self.best_genome = np.zeros(self.spec.parameter_count, dtype=np.float64)
        self.best_fitness = -float("inf")
        self.best_selection_score = -float("inf")
        self.evaluations_completed = 0
        self.rng = np.random.default_rng(config.evolution_seed)
        self.mutation_scale = float(config.mutation_scale)
        self.parents = initial_parent_population(
            self.spec.parameter_count,
            config.parent_count,
            config.mutation_scale,
            self.rng,
        )
        self.parent_fitness = np.full(config.parent_count, np.nan, dtype=np.float64)
        self.parent_selection_scores = np.full(
            config.parent_count,
            np.nan,
            dtype=np.float64,
        )
        self._latest_generation = -1
        self._latest_solutions: tuple[np.ndarray, ...] = ()
        self._latest_fitness: tuple[float, ...] | None = None
        self._saved_candidate_paths: list[str] = []
        self._scientific_evaluation: dict[str, Any] | None = None
        self._result_store = ParquetResultStore(config)
        self._scientific_output_path = str(self._result_store.location.run_dir.resolve())
        self._aggregate_output_path = str(
            (self._result_store.location.run_dir.parents[1] / "figures").resolve()
        )
        if self.checkpoint_path.exists():
            self._restore_checkpoint(self.checkpoint_path)

    def run(self) -> EvolutionResult:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        episode_config = EpisodeConfig(
            seeds=self.config.training_seeds,
            max_steps=self.config.max_steps,
            render_mode=self.config.render_mode,
            accelerator=self.config.accelerator,
            study_id=self.config.study_id,
            simulation_speed=self.config.simulation_speed,
        )
        session = PopulationEvaluationSession(
            self.spec,
            episode_config,
            population_size=self.config.population_size,
        )
        stopped_by_user = False
        completed_normally = False
        failure: Exception | None = None
        try:
            while self.config.continuous or len(self.records) < self.config.generations:
                generation = len(self.records)
                session.poll_controls()
                self._flush_candidate_save_requests(session)
                session.announce_status(
                    "preparing",
                    f"preparing generation {generation + 1}",
                )
                started = time.perf_counter()
                parent_genome_ids = tuple(genome_hash(parent, self.spec) for parent in self.parents)
                offspring_parent_indices = rank_biased_offspring_parents(
                    self.parent_selection_scores,
                    self.config.offspring_count,
                    self.rng,
                )
                offspring = mutate_offspring(
                    self.parents,
                    self.config.offspring_count,
                    self.mutation_scale,
                    self.rng,
                    parent_indices=offspring_parent_indices,
                )
                candidate_matrix = np.vstack((self.parents, offspring))
                self._latest_generation = generation
                self._latest_solutions = tuple(candidate_matrix)
                self._latest_fitness = None
                if self._progress is not None:
                    app_mode = (
                        "one visible shared-world arena"
                        if self.config.render_mode == "human"
                        else "one headless shared-world arena"
                    )
                    label = (
                        f"generation {generation + 1} (continuous)"
                        if self.config.continuous
                        else f"generation {generation + 1}/{self.config.generations}"
                    )
                    self._progress(
                        f"{label}: evaluating {self.config.parent_count} parents + "
                        f"{self.config.offspring_count} offspring in {app_mode}; "
                    )
                evaluations = evaluate_population_batch(
                    candidate_matrix,
                    self.spec,
                    episode_config,
                    generation=generation,
                    mutation_scale=self.mutation_scale,
                    session=session,
                    generation_history=self._generation_history(),
                )
                session.announce_status(
                    "evaluating",
                    f"generation {generation + 1} fitness complete; selecting top mu",
                )
                fitness = np.asarray(
                    [evaluation.fitness for evaluation in evaluations],
                    dtype=np.float64,
                )
                selection_scores = np.asarray(
                    [evaluation.selection_score for evaluation in evaluations],
                    dtype=np.float64,
                )
                generation_mutation_scale = self.mutation_scale
                (
                    offspring_success_rate,
                    adapted_mutation_scale,
                    mutation_scale_reason,
                ) = adapt_mutation_scale(
                    generation_mutation_scale,
                    selection_scores[: self.config.parent_count],
                    selection_scores[self.config.parent_count :],
                    offspring_parent_indices,
                    initial_mutation_scale=self.config.mutation_scale,
                )
                next_mutation_scale = (
                    adapted_mutation_scale
                    if self.config.adaptive_sigma
                    else generation_mutation_scale
                )
                if not self.config.adaptive_sigma:
                    mutation_scale_reason = "fixed_by_experiment_design"
                self._latest_fitness = tuple(float(value) for value in fitness)
                self._flush_candidate_save_requests(session)
                selected = select_parent_indices(
                    selection_scores,
                    self.config.parent_count,
                )
                self.parents = np.ascontiguousarray(candidate_matrix[selected].copy())
                self.parent_fitness = np.ascontiguousarray(fitness[selected].copy())
                self.parent_selection_scores = np.ascontiguousarray(
                    selection_scores[selected].copy()
                )
                self.evaluations_completed += self.config.population_size
                best_index = int(selected[0])
                archive_updated = False
                if selection_scores[best_index] > self.best_selection_score:
                    self.best_selection_score = float(selection_scores[best_index])
                    self.best_fitness = float(fitness[best_index])
                    self.best_genome = candidate_matrix[best_index].copy()
                    archive_updated = True
                record = GenerationRecord(
                    generation=generation,
                    evaluations=self.evaluations_completed,
                    seed_set=self.config.training_seeds,
                    mutation_scale=generation_mutation_scale,
                    mutation_scale_next=next_mutation_scale,
                    mutation_scale_reason=mutation_scale_reason,
                    offspring_success_rate=offspring_success_rate,
                    duration_seconds=time.perf_counter() - started,
                    fitness_min=float(np.min(fitness)),
                    fitness_mean=float(np.mean(fitness)),
                    fitness_max=float(np.max(fitness)),
                    parent_fitness_mean=float(np.mean(fitness[: self.config.parent_count])),
                    offspring_fitness_mean=float(np.mean(fitness[self.config.parent_count :])),
                    selected_parent_fitness_mean=float(np.mean(fitness[selected])),
                    selected_offspring_count=int(
                        np.count_nonzero(selected >= self.config.parent_count)
                    ),
                    selection_score_min=float(np.min(selection_scores)),
                    selection_score_mean=float(np.mean(selection_scores)),
                    selection_score_max=float(np.max(selection_scores)),
                    mean_survival_fraction=float(
                        np.mean([evaluation.mean_survival_fraction for evaluation in evaluations])
                    ),
                    distinct_primary_fitness_values=int(np.unique(fitness).size),
                    best_genome_hash=genome_hash(candidate_matrix[best_index], self.spec),
                    selected_parent_hashes=tuple(
                        genome_hash(candidate_matrix[int(index)], self.spec) for index in selected
                    ),
                    candidate_fitness=tuple(float(value) for value in fitness),
                    candidate_selection_score=tuple(float(value) for value in selection_scores),
                    candidate_survival_fraction=tuple(
                        evaluation.mean_survival_fraction for evaluation in evaluations
                    ),
                    candidate_response_rate=tuple(
                        evaluation.response_rate for evaluation in evaluations
                    ),
                    candidate_action_response_rate=tuple(
                        evaluation.action_response_rate for evaluation in evaluations
                    ),
                    candidate_action_counts=tuple(
                        evaluation.action_counts for evaluation in evaluations
                    ),
                    candidate_pterodactyl_visible_steps=tuple(
                        evaluation.pterodactyl_visible_steps for evaluation in evaluations
                    ),
                    candidate_pterodactyl_duck_rate=tuple(
                        evaluation.pterodactyl_duck_rate for evaluation in evaluations
                    ),
                    candidate_cactus_visible_steps=tuple(
                        evaluation.cactus_visible_steps for evaluation in evaluations
                    ),
                    candidate_cactus_duck_rate=tuple(
                        evaluation.cactus_duck_rate for evaluation in evaluations
                    ),
                )
                self.records.append(record)
                self.mutation_scale = next_mutation_scale
                candidate_parent_ids: tuple[str | None, ...] = tuple(parent_genome_ids) + tuple(
                    parent_genome_ids[int(index)] for index in offspring_parent_indices
                )
                self._result_store.record_generation(
                    record,
                    evaluations,
                    candidate_parent_ids,
                )
                if archive_updated and evaluations[best_index].episodes:
                    self._result_store.record_held_out(
                        "archive_training",
                        generation,
                        evaluations[best_index],
                    )
                scientific_figures = (
                    figure_data_urls(self._result_store.location)
                    if self.config.render_mode == "human"
                    else {}
                )
                should_stop = session.stop_after_generation
                self._save_checkpoint()
                self._save_best_candidate()
                session.announce_status(
                    "generation_complete",
                    (
                        f"generation {generation + 1} complete; "
                        f"mean {record.fitness_mean:.3f}, "
                        f"generation best {record.fitness_max:.3f}, "
                        f"best ever {max(item.fitness_max for item in self.records):.3f}"
                    ),
                    generationHistory=self._generation_history(),
                    totalEvaluations=self.evaluations_completed,
                    mutationScale=self.mutation_scale,
                    scientificOutputPath=self._scientific_output_path,
                    aggregateOutputPath=self._aggregate_output_path,
                    resultDirectory=str(self._result_store.location.run_dir.resolve()),
                    scientificFigures=scientific_figures,
                )
                if self._progress is not None:
                    self._progress(
                        f"generation {generation + 1} complete in "
                        f"{record.duration_seconds:.1f}s; "
                        f"fitness mean={record.fitness_mean:.3f}, "
                        f"generation best={record.fitness_max:.3f}, "
                        f"best ever={max(item.fitness_max for item in self.records):.3f}, "
                        "selected-parent mean="
                        f"{record.selected_parent_fitness_mean:.3f}, "
                        f"offspring success={record.offspring_success_rate:.3f}, "
                        f"sigma {record.mutation_scale:.5f}->{record.mutation_scale_next:.5f}"
                    )
                if (
                    not should_stop
                    and self.config.render_mode != "human"
                    and (self.config.continuous or len(self.records) < self.config.generations)
                ):
                    # Chromium/V8 retains sizeable temporary Canvas and
                    # compression allocations during long headless runs.
                    # A new browser between deterministic generations keeps
                    # memory bounded without changing any seed, physics tick,
                    # controller decision, fitness, or checkpoint state.
                    session.close()
                    session = PopulationEvaluationSession(
                        self.spec,
                        episode_config,
                        population_size=self.config.population_size,
                    )
                if should_stop:
                    stopped_by_user = True
                    break
            if self.config.study_id is not None and not stopped_by_user:
                if self.config.render_mode != "human":
                    # Validation and final test evaluate one frozen archive,
                    # not a population. One private Dino is exactly equivalent
                    # to 100 identical clones in the same deterministic world.
                    session.close()
                    session = PopulationEvaluationSession(
                        self.spec,
                        episode_config,
                        population_size=1,
                    )
                self._evaluate_held_out(session, len(self.records))
            if self._keep_open_after_run:
                phase = "stopped" if stopped_by_user else "run_complete"
                message = (
                    "evolution stopped after the completed generation; close this window to finish"
                    if stopped_by_user
                    else "configured generations complete; close this window to finish"
                )
                session.announce_status(
                    phase,
                    message,
                    generationHistory=self._generation_history(),
                    totalEvaluations=self.evaluations_completed,
                    mutationScale=self.mutation_scale,
                    scientificOutputPath=self._scientific_output_path,
                    scientificFigures=(
                        figure_data_urls(self._result_store.location)
                        if self.config.render_mode == "human"
                        else {}
                    ),
                )
                if self._progress is not None:
                    self._progress(
                        "The final game remains open. Close its Chrome window "
                        "or press Ctrl+C in this terminal to finish."
                    )

                def poll() -> None:
                    session.poll_controls()
                    self._flush_candidate_save_requests(session)

                try:
                    session.wait_until_window_closed(on_poll=poll)
                except KeyboardInterrupt:
                    if self._progress is not None:
                        self._progress("Closing the completed game and finalising saved outputs.")
            completed_normally = True
        except PopulationEvaluationCancelled:
            stopped_by_user = True
            raise
        except Exception as error:
            failure = error
            raise
        finally:
            session.close()
            if failure is not None:
                self._result_store.mark_failed(failure)
            else:
                self._result_store.mark_complete(
                    "complete"
                    if completed_normally
                    else "stopped"
                    if stopped_by_user
                    else "interrupted"
                )
        return EvolutionResult(
            config=self.config,
            controller_spec=self.spec,
            records=tuple(self.records),
            best_genome=self.best_genome.copy(),
            best_genome_hash=genome_hash(self.best_genome, self.spec),
            best_fitness=self.best_fitness,
            best_selection_score=self.best_selection_score,
            checkpoint_path=self.checkpoint_path,
            saved_candidate_paths=tuple(self._saved_candidate_paths),
            scientific_output_path=self._scientific_output_path,
            aggregate_output_path=self._aggregate_output_path,
            scientific_evaluation=self._scientific_evaluation,
        )

    def _generation_history(self) -> list[dict[str, float | int]]:
        history: list[dict[str, float | int]] = []
        best_ever = -float("inf")
        for record in self.records:
            best_ever = max(best_ever, record.fitness_max)
            history.append(
                {
                    "generation": record.generation,
                    "best": record.fitness_max,
                    "mean": record.selected_parent_fitness_mean,
                    "bestEver": best_ever,
                    "selectionMean": record.selection_score_mean,
                    "mutationScale": record.mutation_scale,
                    "offspringSuccessRate": record.offspring_success_rate,
                }
            )
        return history

    def _evaluate_held_out(
        self,
        session: PopulationEvaluationSession,
        generation: int,
    ) -> None:
        evaluation_session = session
        validation: CandidateEvaluation | None = None
        if not self._result_store.held_out_recorded("validation"):
            evaluation_session.announce_status(
                "evaluating",
                f"evaluating archived best on validation seeds after generation {generation}",
            )
            held_out_config = EpisodeConfig(
                seeds=self.config.validation_seeds,
                max_steps=self.config.max_steps,
                render_mode=self.config.render_mode,
                record_trace=True,
                accelerator=self.config.accelerator,
                study_id=self.config.study_id,
                simulation_speed=self.config.simulation_speed,
            )
            validation_genomes = [self.best_genome] * (
                self.config.population_size if self.config.render_mode == "human" else 1
            )
            validation_population = evaluation_session.evaluate(
                validation_genomes,
                generation=generation,
                mutation_scale=self.mutation_scale,
                episode_config=held_out_config,
            )
            validation = validation_population[0]
            self._result_store.record_held_out("validation", generation, validation)
            self._latest_generation = generation
            self._latest_solutions = tuple(validation_genomes)
            self._latest_fitness = tuple(evaluation.fitness for evaluation in validation_population)
            if validation.episodes:
                trace_path = self._result_store.record_trace(
                    trace_id=f"rq3-validation-best-generation-{generation:04d}",
                    evaluation=validation,
                    controller_type=self.config.controller_type,
                    intervention="normal validation replay",
                )
                if trace_path.is_file():
                    plot_behaviour_trace(trace_path)
        self._scientific_evaluation = {
            "validation": (
                asdict(validation) if validation is not None else {"status": "already_recorded"}
            ),
            "validation_generation": generation,
            "sensory_interventions": (
                [
                    {
                        "seed": episode.seed,
                        **sensory_intervention_analysis(
                            self.best_genome,
                            self.spec,
                            episode.sensory_trace,
                        ),
                    }
                    for episode in validation.episodes
                ]
                if validation is not None
                else []
            ),
            "final_test": self._final_test_evaluation(evaluation_session, generation),
        }

    def _final_test_evaluation(
        self,
        session: PopulationEvaluationSession,
        generation: int,
    ) -> dict[str, Any]:
        if not self.config.evaluate_final_test:
            return {
                "status": "reserved_not_run",
                "seeds": list(self.config.final_test_seeds),
                "reason": (
                    "final-test seeds remain locked until one method/model is selected "
                    "without consulting them"
                ),
            }
        if self._result_store.held_out_recorded("final_test"):
            return {"status": "already_recorded"}
        session.announce_status(
            "evaluating",
            "locked final-test evaluation of the already selected archived best",
        )
        population = session.evaluate(
            [self.best_genome] * session.population_size,
            generation=generation,
            mutation_scale=self.mutation_scale,
            episode_config=EpisodeConfig(
                seeds=self.config.final_test_seeds,
                max_steps=self.config.max_steps,
                render_mode=self.config.render_mode,
                record_trace=True,
                accelerator=self.config.accelerator,
                study_id=self.config.study_id,
                simulation_speed=self.config.simulation_speed,
            ),
        )
        final_test = population[0]
        self._result_store.record_held_out(
            "final_test",
            generation,
            final_test,
        )
        return asdict(final_test)

    def _flush_candidate_save_requests(
        self,
        session: PopulationEvaluationSession,
    ) -> None:
        requested = session.consume_candidate_save_requests()
        if not requested:
            return
        if not self._latest_solutions or self._latest_fitness is None:
            session.requeue_candidate_save_requests(requested)
            return
        for candidate_id in dict.fromkeys(requested):
            if not 0 <= candidate_id < len(self._latest_solutions):
                continue
            path = _available_saved_model_path(
                self.config.output_dir
                / "saved-models"
                / (f"generation-{self._latest_generation:04d}-candidate-{candidate_id:03d}.npz")
            )
            save_candidate_model(
                path,
                self._latest_solutions[candidate_id],
                self.spec,
                metadata={
                    "label": "SMOKE / NOT SCIENTIFIC RESULTS",
                    "run_id": self.config.run_id,
                    "generation": self._latest_generation,
                    "candidate_id": candidate_id,
                    "fitness": self._latest_fitness[candidate_id],
                    "training_seeds": list(self.config.training_seeds),
                    "evolution_seed": self.config.evolution_seed,
                    "source_sha256": _source_fingerprint(),
                },
            )
            resolved = str(path.resolve())
            self._saved_candidate_paths.append(resolved)
            session.announce_status(
                "candidate_saved",
                f"candidate {candidate_id} saved to {path.as_posix()}",
                savedModelPath=resolved,
            )

    def _save_checkpoint(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint_path.with_suffix(
            self.checkpoint_path.suffix + f".{os.getpid()}.tmp"
        )
        with temporary.open("wb") as stream:
            pickle.dump(
                {
                    "format_version": CHECKPOINT_FORMAT_VERSION,
                    "evolution_method": EVOLUTION_METHOD,
                    "numpy_version": np.__version__,
                    "git_commit": _git_commit(),
                    "source_sha256": _source_fingerprint(),
                    "config": _serialise_config(self.config),
                    "controller_spec": asdict(self.spec),
                    "sensory_names": SENSORY_NAMES,
                    "rng_state": self.rng.bit_generator.state,
                    "records": self.records,
                    "parents": self.parents,
                    "parent_fitness": self.parent_fitness,
                    "parent_selection_scores": self.parent_selection_scores,
                    "current_mutation_scale": self.mutation_scale,
                    "evaluations_completed": self.evaluations_completed,
                    "best_genome": self.best_genome,
                    "best_fitness": self.best_fitness,
                    "best_selection_score": self.best_selection_score,
                    "best_genome_hash": genome_hash(self.best_genome, self.spec),
                },
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.checkpoint_path)

    def _save_best_candidate(self) -> None:
        save_candidate_model(
            self.config.output_dir / "best-candidate.npz",
            self.best_genome,
            self.spec,
            metadata={
                "label": "CURRENT BEST",
                "run_id": self.config.run_id,
                "generation": self._latest_generation,
                "fitness": self.best_fitness,
                "selection_score": self.best_selection_score,
                "training_seeds": list(self.config.training_seeds),
                "evolution_seed": self.config.evolution_seed,
                "source_sha256": _source_fingerprint(),
            },
        )

    def _restore_checkpoint(self, path: Path) -> None:
        with path.open("rb") as stream:
            payload = pickle.load(stream)  # noqa: S301 - trusted local checkpoint
        if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("Unsupported checkpoint format; start a new evolution-strategy run")
        if payload.get("evolution_method") != EVOLUTION_METHOD:
            raise ValueError("Checkpoint evolution method does not match")
        _validate_resume_config(payload.get("config"), self.config)
        saved_spec = ControllerSpec(**payload["controller_spec"])
        if saved_spec != self.spec:
            raise ValueError("Checkpoint controller architecture does not match")
        if tuple(payload.get("sensory_names", ())) != SENSORY_NAMES:
            raise ValueError("Checkpoint sensory schema does not match")
        parents = np.asarray(payload["parents"], dtype=np.float64)
        expected_shape = (self.config.parent_count, self.spec.parameter_count)
        if parents.shape != expected_shape or not np.isfinite(parents).all():
            raise ValueError("Checkpoint parent population is invalid")
        self.parents = np.ascontiguousarray(parents.copy())
        self.parent_fitness = _checkpoint_vector(
            payload["parent_fitness"],
            self.config.parent_count,
            "parent_fitness",
        )
        self.parent_selection_scores = _checkpoint_vector(
            payload["parent_selection_scores"],
            self.config.parent_count,
            "parent_selection_scores",
        )
        self.mutation_scale = float(payload["current_mutation_scale"])
        lower_sigma = self.config.mutation_scale * SIGMA_MIN_FACTOR
        upper_sigma = self.config.mutation_scale * SIGMA_MAX_FACTOR
        if (
            not np.isfinite(self.mutation_scale)
            or not lower_sigma <= self.mutation_scale <= upper_sigma
        ):
            raise ValueError("Checkpoint mutation scale is invalid")
        self.rng.bit_generator.state = payload["rng_state"]
        self.records = list(payload["records"])
        self.evaluations_completed = int(payload["evaluations_completed"])
        if self.evaluations_completed != len(self.records) * self.config.population_size:
            raise ValueError("Checkpoint evaluation count is inconsistent")
        if self.records and self.mutation_scale != self.records[-1].mutation_scale_next:
            raise ValueError("Checkpoint mutation scale disagrees with history")
        stored_generations = self._result_store.completed_generation_count()
        if stored_generations != len(self.records):
            raise ValueError(
                "Checkpoint and Parquet progress disagree: "
                f"checkpoint={len(self.records)}, parquet={stored_generations}"
            )
        self.best_genome = validate_genome(payload["best_genome"], self.spec)
        self.best_fitness = float(payload["best_fitness"])
        self.best_selection_score = float(payload["best_selection_score"])
        expected_hash = genome_hash(self.best_genome, self.spec)
        if payload.get("best_genome_hash") != expected_hash:
            raise ValueError("Checkpoint best-genome hash does not match its bytes")


def load_best_from_checkpoint(
    path: Path,
) -> tuple[np.ndarray, ControllerSpec, dict[str, Any]]:
    """Load and validate the best controller from a trusted local checkpoint."""

    with path.open("rb") as stream:
        payload = pickle.load(stream)  # noqa: S301 - trusted local checkpoint
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format")
    if payload.get("evolution_method") != EVOLUTION_METHOD:
        raise ValueError("Checkpoint evolution method does not match")
    spec = ControllerSpec(**payload["controller_spec"])
    genome = validate_genome(payload["best_genome"], spec)
    if genome_hash(genome, spec) != payload["best_genome_hash"]:
        raise ValueError("Checkpoint best-genome hash mismatch")
    metadata = {
        "evolution_method": payload["evolution_method"],
        "numpy_version": payload["numpy_version"],
        "git_commit": payload["git_commit"],
        "source_sha256": payload.get("source_sha256"),
        "best_fitness": float(payload["best_fitness"]),
        "best_selection_score": float(payload["best_selection_score"]),
        "config": payload["config"],
        "records": [dict(vars(record)) for record in payload["records"]],
    }
    return genome, spec, metadata


def save_candidate_model(
    path: Path,
    genome: np.ndarray | list[float],
    spec: ControllerSpec,
    *,
    metadata: dict[str, Any],
) -> Path:
    """Atomically save one reusable controller selected in the population GUI."""

    canonical = validate_genome(genome, spec)
    payload = {
        **metadata,
        "format_version": SAVED_MODEL_FORMAT_VERSION,
        "controller_type": spec.controller_type,
        "input_size": spec.input_size,
        "hidden_size": spec.hidden_size,
        "action_size": spec.action_size,
        "dt": spec.dt,
        "tau": spec.tau,
        "genome_hash": genome_hash(canonical, spec),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            genome=canonical,
            metadata_json=np.asarray(json.dumps(payload, sort_keys=True)),
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return path


def load_candidate_model(
    path: Path,
) -> tuple[np.ndarray, ControllerSpec, dict[str, Any]]:
    """Load a GUI-saved candidate without permitting pickled array objects."""

    with np.load(path, allow_pickle=False) as archive:
        genome = np.asarray(archive["genome"], dtype=np.float64)
        metadata = json.loads(str(archive["metadata_json"].item()))
    if metadata.get("format_version") != SAVED_MODEL_FORMAT_VERSION:
        raise ValueError("Unsupported saved-model format")
    spec = ControllerSpec(
        controller_type=metadata["controller_type"],
        input_size=int(metadata["input_size"]),
        hidden_size=int(metadata["hidden_size"]),
        action_size=int(metadata["action_size"]),
        dt=float(metadata["dt"]),
        tau=float(metadata["tau"]),
    )
    canonical = validate_genome(genome, spec)
    if metadata.get("genome_hash") != genome_hash(canonical, spec):
        raise ValueError("Saved-model genome hash mismatch")
    return canonical, spec, metadata


def _checkpoint_vector(value: Any, size: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"Checkpoint {label} is invalid")
    return np.ascontiguousarray(vector.copy())


def _available_saved_model_path(requested: Path) -> Path:
    if not requested.exists():
        return requested
    for number in range(2, 10_000):
        candidate = requested.with_name(f"{requested.stem}-{number:03d}{requested.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available saved-model path beside {requested}")


def _serialise_config(config: EvolutionConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir.resolve())
    payload["results_root"] = str(config.results_root.resolve())
    payload["population_size"] = config.population_size
    return payload


def _validate_resume_config(
    saved: dict[str, Any] | None,
    requested: EvolutionConfig,
) -> None:
    if saved is None:
        raise ValueError("Checkpoint has no run configuration")
    current = _serialise_config(requested)
    mutable_fields = {
        "continuous",
        "simulation_speed",
        "evaluate_final_test",
    }
    for key, value in current.items():
        if key not in mutable_fields and saved.get(key) != value:
            raise ValueError(
                f"Checkpoint config mismatch for {key}: {saved.get(key)!r} != {value!r}"
            )


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _source_fingerprint() -> str:
    """Hash the complete local dino_er source when no commit exists yet."""

    source_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(source_root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()
