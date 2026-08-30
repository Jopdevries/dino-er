"""Reproducible scientific-output exports from genuine evolution records."""

from __future__ import annotations

import ast
import base64
import csv
import hashlib
import io
import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

MODEL_VERSION = 2
CONTROL_HZ = 60.0
NO_ACTION_SURVIVAL_REFERENCE_STEPS = 270.0
NUMERICAL_CHANGE_EPSILON = 1e-9
RQ1_COLOURS = {"reactive": "#0072B2", "proactive": "#D55E00"}
RQ1_LINE_STYLES = {"reactive": "-", "proactive": "--"}
RQ1_MARKERS = {"reactive": "o", "proactive": "s"}
SENSOR_DISPLAY_NAMES = (
    "Obstacle distance x",
    "Obstacle offset y",
    "Obstacle width",
    "Obstacle height",
    "Dino height y",
)
ACTION_DISPLAY_NAMES = ("No action", "Jump", "Duck / fast drop")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class _StripDocstrings(ast.NodeTransformer):
    """Exclude explanatory prose from the behavioural-model hash."""

    def _strip(self, node: ast.AST) -> ast.AST:
        if not isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            return self.generic_visit(node)
        if node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body = node.body[1:]
        return self.generic_visit(node)

    visit_Module = _strip
    visit_ClassDef = _strip
    visit_FunctionDef = _strip


def _named_ast_dump(
    path: Path,
    names: set[str],
    class_methods: Mapping[str, set[str]] | None = None,
) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected_methods = class_methods or {}
    nodes: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in selected_methods:
            requested = selected_methods[node.name]
            nodes.extend(
                child
                for child in node.body
                if isinstance(child, ast.FunctionDef) and child.name in requested
            )
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.Assign)) and (
            getattr(node, "name", None) in names
            or any(
                isinstance(target, ast.Name) and target.id in names
                for target in getattr(node, "targets", ())
            )
        ):
            nodes.append(node)
    if not nodes:
        raise RuntimeError(f"Could not find model definitions in {path}")
    normaliser = _StripDocstrings()
    return "\n".join(ast.dump(normaliser.visit(node), include_attributes=False) for node in nodes)


def _comment_insensitive_typescript(path: Path) -> str:
    # These local TypeScript files contain no URL literals, so this deliberately
    # small stripper is enough and avoids a second parser solely for hashing.
    source = path.read_text(encoding="utf-8")
    without_blocks = re.sub(r"/\\*.*?\\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\\r\\n]*", "", without_blocks)


def model_hash() -> str:
    """Hash only definitions that can change the implemented scientific model.

    Comments, GUI layout, plotting and timestamps are deliberately excluded.  A
    meaningful model change should also increment ``MODEL_VERSION`` manually.
    """

    source_root = Path(__file__).resolve().parent
    project_root = source_root.parents[1]
    components: dict[str, tuple[set[str], Mapping[str, set[str]]]] = {
        "perception.py": (
            {"SENSORY_SCHEMA", "_boxes_from_pixels", "process_population_pixel_components"},
            {
                "VisualPerception": {
                    "process",
                    "_pose",
                    "_obstacle_class",
                    "_detect_game_over",
                    "_normalise",
                }
            },
        ),
        "controllers.py": (
            {
                "action_from_scores",
                "ControllerSpec",
                "default_controller_spec",
                "unflatten_genome",
                "ReactiveController",
                "CTRNNController",
            },
            {},
        ),
        "evaluation.py": ({"_candidate_evaluation", "evaluate_population_batch"}, {}),
        "evolution.py": (
            {
                "MAX_POPULATION_SIZE",
                "SIGMA_SUCCESS_TARGET",
                "SIGMA_ADAPTATION_FACTOR",
                "SIGMA_MIN_FACTOR",
                "SIGMA_MAX_FACTOR",
                "SIGMA_PLATEAU_MAX_FACTOR",
                "initial_parent_population",
                "mutate_offspring",
                "rank_biased_offspring_parents",
                "adapt_mutation_scale",
                "select_parent_indices",
            },
            {},
        ),
    }
    digest = hashlib.sha256()
    for filename, (names, class_methods) in components.items():
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_named_ast_dump(source_root / filename, names, class_methods).encode("utf-8"))
        digest.update(b"\0")
    # Dynamics, held-key conversion, and private rendering can change a
    # controller's task or its pixel observations; GUI layout cannot.
    for relative_path in (
        "game/src/engine.ts",
        "game/src/input.ts",
        "game/src/renderer.ts",
    ):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_comment_insensitive_typescript(project_root / relative_path).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def experiment_hash(config: Any, *, current_model_hash: str) -> str:
    """Hash a planned condition, including its optimizer seed and budget."""

    settings = {
        "model_hash": current_model_hash,
        "controller_type": config.controller_type,
        "evolution_seed": config.evolution_seed,
        "training_seeds": list(config.training_seeds),
        "validation_seeds": list(config.validation_seeds),
        "final_test_seeds": list(config.final_test_seeds),
        "parent_count": config.parent_count,
        "offspring_count": config.offspring_count,
        "mutation_scale": config.mutation_scale,
        "adaptive_sigma": config.adaptive_sigma,
        "generations": config.generations,
        "max_steps": config.max_steps,
    }
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    pq.write_table(pa.Table.from_pylist(list(rows)), temporary, compression="zstd")
    temporary.replace(path)


def read_parquet_parts(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _episode_row(
    common: Mapping[str, Any],
    episode: Any,
    *,
    phase: str,
    generation: int,
    candidate_id: int,
    candidate_role: str,
    genome_id: str,
    parent_genome_id: str | None,
    selected_as_parent: bool,
    candidate_fitness: float,
    selection_score: float,
    intervention: str | None = None,
) -> dict[str, Any]:
    row = {**common, "phase": phase}
    if intervention is not None:
        row["intervention"] = intervention
    row.update(
        {
            "generation": generation,
            "candidate_id": candidate_id,
            "candidate_role": candidate_role,
            "genome_id": genome_id,
            "parent_genome_id": parent_genome_id,
            "selected_as_parent": selected_as_parent,
            "candidate_fitness": candidate_fitness,
            "selection_score": selection_score,
            "world_seed": int(episode.seed),
            "survival_steps": int(episode.steps),
            "terminated": bool(episode.terminated),
            "truncated": bool(episode.truncated),
            "no_action_steps": int(episode.action_counts[0]),
            "jump_steps": int(episode.action_counts[1]),
            "duck_steps": int(episode.action_counts[2]),
            "action_switches": int(episode.action_switches),
            "pterodactyl_visible_steps": int(episode.pterodactyl_visible_steps),
            "duck_on_pterodactyl_steps": int(episode.duck_on_pterodactyl_steps),
            "cactus_visible_steps": int(episode.cactus_visible_steps),
            "duck_on_cactus_steps": int(episode.duck_on_cactus_steps),
        }
    )
    return row


def _trace_rows(
    identity: Mapping[str, Any],
    evaluation: Any,
    *,
    controller_type: str,
    intervention: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in evaluation.episodes:
        for timestep, (sensory, scores, action) in enumerate(
            zip(
                episode.sensory_trace,
                episode.action_scores,
                episode.actions,
                strict=True,
            )
        ):
            row: dict[str, Any] = {
                **identity,
                "intervention": intervention,
                "controller_type": controller_type,
                "world_seed": int(episode.seed),
                "timestep": timestep,
                "control_time_seconds": timestep / CONTROL_HZ,
                **{f"sensor_{index}": float(sensory[index]) for index in range(5)},
                **{f"score_{index}": float(scores[index]) for index in range(3)},
                "action": int(action),
            }
            if controller_type == "proactive" and timestep < len(episode.hidden_trace):
                row.update(
                    {
                        f"hidden_{index}": float(value)
                        for index, value in enumerate(episode.hidden_trace[timestep])
                    }
                )
            rows.append(row)
    return rows


@dataclass(frozen=True)
class ResultLocation:
    """Stable result hierarchy for one model/experiment/run triple."""

    model_version: int
    model_hash: str
    experiment_hash: str
    run_id: str
    root: Path
    campaign_timestamp: str | None = None

    @property
    def run_dir(self) -> Path:
        directory_name = (
            f"{self.campaign_timestamp}__{self.run_id}" if self.campaign_timestamp else self.run_id
        )
        return (
            self.root
            / f"model_v{self.model_version:03d}_{self.model_hash}"
            / self.experiment_hash
            / directory_name
        )


@dataclass(frozen=True)
class RunInspection:
    """Persisted state of one planned optimisation run.

    A directory alone is never evidence that a run is complete.  This small
    record is used by the overnight runner to decide whether to skip, resume,
    retry, or surface a failed condition.
    """

    status: str
    completed_generations: int
    message: str


def inspect_run(config: Any, *, require_final_test: bool = False) -> RunInspection:
    """Validate a planned run from its manifest and immutable Parquet parts."""

    location = result_location(config)
    manifest_path = location.run_dir / "run-manifest.json"
    if not manifest_path.is_file():
        return RunInspection("not_started", 0, "no manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return RunInspection("invalid", 0, f"invalid manifest: {error}")
    if (
        manifest.get("model_hash") != location.model_hash
        or manifest.get("experiment_hash") != location.experiment_hash
        or manifest.get("run_id") != config.run_id
        or manifest.get("campaign_timestamp") != location.campaign_timestamp
    ):
        return RunInspection("invalid", 0, "manifest identity does not match planned run")
    expected = int(config.generations)
    generation_dir = location.run_dir / "generation-results"
    episode_dir = location.run_dir / "episode-results"
    completed = 0
    for generation in range(expected):
        generation_path = generation_dir / f"generation-{generation:04d}.parquet"
        episode_path = episode_dir / f"generation-{generation:04d}.parquet"
        if not generation_path.is_file() and not episode_path.is_file():
            break
        if not generation_path.is_file() or not episode_path.is_file():
            return RunInspection("invalid", completed, "generation and episode parts disagree")
        try:
            if pq.ParquetFile(generation_path).metadata.num_rows != 1:
                return RunInspection("invalid", completed, "generation part does not have one row")
            if pq.ParquetFile(episode_path).metadata.num_rows < 1:
                return RunInspection("invalid", completed, "episode part is empty")
        except (OSError, pa.ArrowException) as error:
            return RunInspection("invalid", completed, f"unreadable Parquet part: {error}")
        completed += 1
    if any(
        (generation_dir / f"generation-{generation:04d}.parquet").is_file()
        for generation in range(completed + 1, expected)
    ):
        return RunInspection("invalid", completed, "generation parts are not contiguous")
    if manifest.get("status") == "failed":
        return RunInspection("failed", completed, str(manifest.get("failure", "worker failed")))
    if completed < expected:
        return RunInspection("partial", completed, f"{completed}/{expected} complete generations")
    if config.study_id is not None and not any(
        episode_dir.glob("archive_training-generation-*.parquet")
    ):
        return RunInspection("partial", completed, "training complete; archive record missing")
    if config.study_id is not None and not (episode_dir / "validation.parquet").is_file():
        return RunInspection("partial", completed, "training complete; validation missing")
    if require_final_test and not (episode_dir / "final_test.parquet").is_file():
        return RunInspection("partial", completed, "final test missing")
    if manifest.get("status") != "complete":
        return RunInspection("partial", completed, f"manifest status is {manifest.get('status')!r}")
    return RunInspection("complete", completed, f"{completed}/{expected} complete generations")


def result_location(config: Any) -> ResultLocation:
    """Calculate a run location without creating or mutating result files."""

    current_hash = getattr(config, "result_model_hash", None) or model_hash()
    return ResultLocation(
        model_version=config.model_version,
        model_hash=current_hash,
        experiment_hash=experiment_hash(config, current_model_hash=current_hash),
        run_id=config.run_id,
        root=config.results_root,
        campaign_timestamp=config.campaign_timestamp,
    )


class ParquetResultStore:
    """Small append-only result writer for one evolution run.

    It stores only report-relevant tables.  Each completed generation becomes
    two immutable files, so an interruption cannot erase earlier evidence.
    """

    def __init__(self, config: Any) -> None:
        self.location = result_location(config)
        self.config = config
        self.location.run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.location.run_dir / "run-manifest.json"
        self._started_at_utc = _utc_now()
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(manifest.get("started_at_utc"), str):
                    self._started_at_utc = str(manifest["started_at_utc"])
            except (OSError, json.JSONDecodeError):
                pass
        self._write_manifest("running")

    @property
    def run_dir(self) -> Path:
        return self.location.run_dir

    def completed_generation_count(self) -> int:
        """Count the contiguous immutable generations available for checkpoint resume."""

        completed = 0
        while (
            self.location.run_dir / "generation-results" / f"generation-{completed:04d}.parquet"
        ).is_file():
            completed += 1
        return completed

    def held_out_recorded(self, phase: str) -> bool:
        if phase not in ("validation", "final_test"):
            raise ValueError("Held-out phase must be validation or final_test")
        return (self.location.run_dir / "episode-results" / f"{phase}.parquet").is_file()

    def record_generation(
        self,
        record: Any,
        evaluations: Sequence[Any],
        parent_genome_ids: Sequence[str | None],
    ) -> None:
        """Persist one selected generation and all seed-level evaluations."""

        if len(evaluations) != len(parent_genome_ids):
            raise ValueError("Each candidate needs exactly one parent genome identifier")
        generation = int(record.generation)
        common = self._common_fields()
        generation_row = {
            **common,
            "generation": generation,
            "cumulative_candidate_evaluations": int(record.evaluations),
            "fitness_best": float(record.fitness_max),
            "fitness_mean": float(record.fitness_mean),
            "parent_fitness_mean": float(record.parent_fitness_mean),
            "offspring_fitness_mean": float(record.offspring_fitness_mean),
            "selected_parent_fitness_mean": float(record.selected_parent_fitness_mean),
            "no_action_survival_reference_steps": NO_ACTION_SURVIVAL_REFERENCE_STEPS,
            "all_candidate_excess_survival_mean": max(
                0.0,
                float(record.fitness_mean) - NO_ACTION_SURVIVAL_REFERENCE_STEPS,
            ),
            "selected_parent_excess_survival_mean": max(
                0.0,
                float(record.selected_parent_fitness_mean) - NO_ACTION_SURVIVAL_REFERENCE_STEPS,
            ),
            "selected_offspring_count": int(record.selected_offspring_count),
            "fitness_median": float(np.median(record.candidate_fitness)),
            "fitness_standard_deviation": float(np.std(record.candidate_fitness)),
            "best_genome_id": str(record.best_genome_hash),
            "parent_count": int(self.config.parent_count),
            "offspring_count": int(self.config.offspring_count),
            "mutation_scale": float(self.config.mutation_scale),
            "adaptive_sigma": bool(self.config.adaptive_sigma),
            "generation_mutation_scale": float(record.mutation_scale),
            "next_generation_mutation_scale": float(record.mutation_scale_next),
            "mutation_scale_reason": str(record.mutation_scale_reason),
            "offspring_success_rate": float(record.offspring_success_rate),
        }
        _atomic_parquet(
            self.location.run_dir / "generation-results" / f"generation-{generation:04d}.parquet",
            [generation_row],
        )
        episode_rows: list[dict[str, Any]] = []
        selected_genomes = set(record.selected_parent_hashes)
        for candidate_id, (evaluation, parent_id) in enumerate(
            zip(evaluations, parent_genome_ids, strict=True)
        ):
            for episode in evaluation.episodes:
                episode_rows.append(
                    _episode_row(
                        common,
                        episode,
                        phase="training",
                        generation=generation,
                        candidate_id=candidate_id,
                        candidate_role=(
                            "parent" if candidate_id < self.config.parent_count else "offspring"
                        ),
                        genome_id=str(evaluation.genome_hash),
                        parent_genome_id=parent_id,
                        selected_as_parent=str(evaluation.genome_hash) in selected_genomes,
                        candidate_fitness=float(evaluation.fitness),
                        selection_score=float(evaluation.selection_score),
                    )
                )
        _atomic_parquet(
            self.location.run_dir / "episode-results" / f"generation-{generation:04d}.parquet",
            episode_rows,
        )
        self._write_manifest("running")

    def record_held_out(
        self,
        phase: str,
        generation: int,
        evaluation: Any,
    ) -> None:
        """Persist archive-training, validation, or final-test episodes."""

        if phase not in ("archive_training", "validation", "final_test"):
            raise ValueError(
                "Archive evaluation phase must be archive_training, validation, or final_test"
            )
        common = self._common_fields()
        rows = [
            _episode_row(
                common,
                episode,
                phase=phase,
                generation=generation,
                candidate_id=0,
                candidate_role="selected_archive",
                genome_id=str(evaluation.genome_hash),
                parent_genome_id=None,
                selected_as_parent=True,
                candidate_fitness=float(evaluation.fitness),
                selection_score=float(evaluation.selection_score),
            )
            for episode in evaluation.episodes
        ]
        filename = (
            f"archive_training-generation-{generation:04d}.parquet"
            if phase == "archive_training"
            else f"{phase}.parquet"
        )
        _atomic_parquet(self.location.run_dir / "episode-results" / filename, rows)

    def record_trace(
        self,
        *,
        trace_id: str,
        evaluation: Any,
        controller_type: str,
        intervention: str,
    ) -> Path:
        """Persist an intentionally selected RQ3 trace, never all training frames."""

        rows = _trace_rows(
            {**self._common_fields(), "trace_id": trace_id},
            evaluation,
            controller_type=controller_type,
            intervention=intervention,
        )
        path = self.location.run_dir / "behavioural-traces" / f"{trace_id}.parquet"
        _atomic_parquet(path, rows)
        return path

    def record_intervention(
        self,
        *,
        intervention: str,
        generation: int,
        evaluation: Any,
    ) -> Path:
        """Store seed-level RQ3 outcomes without treating them as retraining."""

        rows = [
            _episode_row(
                self._common_fields(),
                episode,
                phase="intervention",
                intervention=intervention,
                generation=generation,
                candidate_id=0,
                candidate_role="selected_model",
                genome_id=str(evaluation.genome_hash),
                parent_genome_id=None,
                selected_as_parent=True,
                candidate_fitness=float(evaluation.fitness),
                selection_score=float(evaluation.selection_score),
            )
            for episode in evaluation.episodes
        ]
        filename = _safe_filename(intervention)
        path = self.location.run_dir / "episode-results" / f"intervention-{filename}.parquet"
        _atomic_parquet(path, rows)
        return path

    def mark_complete(self, status: str) -> None:
        self._write_manifest(status)

    def mark_failed(self, error: BaseException) -> None:
        """Persist a bounded worker failure while retaining completed parts."""

        self._write_manifest("failed", failure=f"{type(error).__name__}: {error}")

    def _common_fields(self) -> dict[str, Any]:
        return {
            "model_version": self.location.model_version,
            "model_hash": self.location.model_hash,
            "experiment_hash": self.location.experiment_hash,
            "run_id": self.location.run_id,
            "controller_type": self.config.controller_type,
            "optimizer_seed": int(self.config.evolution_seed),
            "experiment_group": self.config.study_id or "engineering",
            "parent_count": int(self.config.parent_count),
            "offspring_count": int(self.config.offspring_count),
            "mutation_scale": float(self.config.mutation_scale),
            "adaptive_sigma": bool(self.config.adaptive_sigma),
            "campaign_timestamp": self.location.campaign_timestamp,
            "run_started_at_utc": self._started_at_utc,
            "recorded_at_utc": _utc_now(),
        }

    def _write_manifest(self, status: str, *, failure: str | None = None) -> None:
        generation_parts = sorted((self.location.run_dir / "generation-results").glob("*.parquet"))
        payload = {
            **self._common_fields(),
            "status": status,
            "started_at_utc": self._started_at_utc,
            "updated_at_utc": _utc_now(),
            "training_seeds": list(self.config.training_seeds),
            "validation_seeds": list(self.config.validation_seeds),
            "final_test_seeds": list(self.config.final_test_seeds),
            "parent_count": self.config.parent_count,
            "offspring_count": self.config.offspring_count,
            "mutation_scale": self.config.mutation_scale,
            "adaptive_sigma": self.config.adaptive_sigma,
            "generations": self.config.generations,
            "max_steps": self.config.max_steps,
            "evaluate_final_test": self.config.evaluate_final_test,
            "fitness": "mean survival steps only; no shaping or auxiliary components",
            "simulation_speed_scope": "wall-clock pacing and redraw only",
            "completed_generations": len(generation_parts),
            "validation_recorded": (
                self.location.run_dir / "episode-results" / "validation.parquet"
            ).is_file(),
            "final_test_recorded": (
                self.location.run_dir / "episode-results" / "final_test.parquet"
            ).is_file(),
        }
        if status in {"complete", "failed", "stopped", "interrupted"}:
            payload["finished_at_utc"] = _utc_now()
        if failure is not None:
            payload["failure"] = failure[:2_000]
        _atomic_text(
            self.location.run_dir / "run-manifest.json",
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _safe_filename(value: str) -> str:
    """Keep one human-readable intervention identifier inside its run folder."""

    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    if not result:
        raise ValueError("intervention name cannot be empty")
    return result


def _atomic_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    figure.savefig(
        temporary,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    temporary.replace(path)


def _group_evidence(
    location: ResultLocation,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    current_group: str | None = None
    manifest_path = location.run_dir / "run-manifest.json"
    if manifest_path.is_file():
        current_group = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "experiment_group"
        )
    generations: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    traces: list[Path] = []
    for run_dir, manifest in _manifest_rows(
        location.run_dir.parents[1],
        campaign_timestamp=location.campaign_timestamp,
    ):
        if current_group is not None and manifest.get("experiment_group") != current_group:
            continue
        generations.extend(read_parquet_parts(run_dir / "generation-results"))
        episodes.extend(read_parquet_parts(run_dir / "episode-results"))
        traces.extend(sorted((run_dir / "behavioural-traces").glob("*.parquet")))
    return generations, episodes, traces


# Two-sided 95% Student-t critical values for df=1..30.  The primary study has
# only three independent optimiser runs, for which a normal 1.96 multiplier
# would materially understate uncertainty.  Values above 30 use the normal
# limit, which is adequate at that sample size for a descriptive interval.
_T975_BY_DF: tuple[float, ...] = (
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


def _student_t_95_half_width(values: Sequence[float]) -> float | None:
    """Return a descriptive two-sided 95% Student-t interval half-width.

    The independent unit is an optimiser run (or, for the frozen final model,
    a locked environment seed).  A singleton has no estimable uncertainty and
    therefore returns ``None`` rather than a misleading zero-width interval.
    """

    count = len(values)
    if count < 2:
        return None
    standard_error = float(np.std(values, ddof=1)) / math.sqrt(count)
    degrees_of_freedom = count - 1
    multiplier = (
        _T975_BY_DF[degrees_of_freedom - 1] if degrees_of_freedom <= len(_T975_BY_DF) else 1.96
    )
    return multiplier * standard_error


def _central_curve(
    rows: Sequence[Mapping[str, Any]],
    required_run_ids: set[str] | None = None,
    *,
    x_field: str = "cumulative_candidate_evaluations",
    value_field: str | None = None,
    x_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        point = int(row[x_field]) + x_offset
        value = _survivor_learning_mean(row) if value_field is None else float(row[value_field])
        grouped[point][str(row["run_id"])] = value
    required = required_run_ids or {str(row["run_id"]) for row in rows}
    complete_points = [
        point for point, run_values in grouped.items() if required and required.issubset(run_values)
    ]
    x = np.asarray(sorted(complete_points), dtype=np.int64)
    samples = [[grouped[int(point)][run_id] for run_id in sorted(required)] for point in x]
    y = np.asarray([np.mean(values) for values in samples], dtype=np.float64)
    ci = np.asarray(
        [_student_t_95_half_width(values) or 0.0 for values in samples],
        dtype=np.float64,
    )
    return x, y, ci


def _rq1_generation_fitness_figure(
    rows: Sequence[Mapping[str, Any]],
    required_run_ids: Mapping[str, set[str]] | None = None,
) -> Any:
    figure = _rq1_single_generation_figure(
        rows,
        required_run_ids,
        value_field=None,
        title="Selected-parent learning over the evaluation budget",
    )
    axis = figure.axes[0]
    per_generation = {
        int(row["cumulative_candidate_evaluations"]) // (int(row["generation"]) + 1)
        for row in rows
        if int(row["generation"]) >= 0
    }
    if len(per_generation) == 1:
        candidates = float(next(iter(per_generation)))
        budget_axis = axis.secondary_xaxis(
            "top",
            functions=(
                lambda generation: generation * candidates,
                lambda budget: budget / candidates,
            ),
        )
        budget_axis.set_xlabel("Cumulative candidate evaluations", fontsize=8.3, labelpad=6)
        budget_axis.tick_params(labelsize=7.8, width=0.8)
        budget_axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, integer=True))
    return figure


def _rq1_single_generation_figure(
    rows: Sequence[Mapping[str, Any]],
    required_run_ids: Mapping[str, set[str]] | None,
    *,
    value_field: str | None,
    title: str,
) -> Any:
    grouped_runs: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_controller: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        controller = str(row["controller_type"])
        grouped_runs[(controller, str(row["run_id"]))].append(row)
        by_controller[controller].append(row)

    figure, axis = plt.subplots(figsize=(7.2, 4.15), constrained_layout=True)
    plotted_values: list[float] = [NO_ACTION_SURVIVAL_REFERENCE_STEPS]
    central_endpoints: list[tuple[str, float, float]] = []
    for (controller, _run_id), run_rows in sorted(grouped_runs.items()):
        ordered = sorted(run_rows, key=lambda row: int(row["generation"]))
        values = [
            _survivor_learning_mean(row) if value_field is None else float(row[value_field])
            for row in ordered
        ]
        plotted_values.extend(values)
        axis.plot(
            [int(row["generation"]) + 1 for row in ordered],
            values,
            color=RQ1_COLOURS.get(controller, "#777777"),
            linestyle=RQ1_LINE_STYLES.get(controller, "-"),
            linewidth=0.9,
            alpha=0.38,
            zorder=2,
        )
    for controller, controller_rows in sorted(by_controller.items()):
        observed_run_ids = {str(row["run_id"]) for row in controller_rows}
        required = (
            required_run_ids.get(controller, observed_run_ids)
            if required_run_ids is not None
            else observed_run_ids
        )
        x, y, _ci = _central_curve(
            controller_rows,
            set(required),
            x_field="generation",
            value_field=value_field,
            x_offset=1,
        )
        if x.size == 0:
            continue
        axis.plot(
            x,
            y,
            color=RQ1_COLOURS.get(controller, "#777777"),
            linestyle=RQ1_LINE_STYLES.get(controller, "-"),
            marker=RQ1_MARKERS.get(controller, "o"),
            markersize=4.2,
            linewidth=2.2,
            label=controller.title(),
            zorder=4,
        )
        central_endpoints.append((controller, float(x[-1]), float(y[-1])))
    axis.axhline(
        NO_ACTION_SURVIVAL_REFERENCE_STEPS,
        color="#333333",
        linestyle=":",
        linewidth=1.2,
        zorder=1,
    )
    axis.set_ylabel("Selected-parent mean training survival [control steps]")
    axis.set_xlabel("Generation")
    # Survival is right-censored at the evaluation horizon.  Keep the complete
    # physical range visible so ceiling hits cannot be mistaken for convergence.
    axis.set_ylim(0.0, 3600.0)
    axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axis.margins(x=0.025)
    _style_journal_axis(axis)
    axis.set_title(title, loc="left", fontsize=10.5, fontweight="normal", pad=34)
    for controller, x_value, y_value in central_endpoints:
        axis.annotate(
            controller.title(),
            (x_value, y_value),
            xytext=(6, 0),
            textcoords="offset points",
            color=RQ1_COLOURS.get(controller, "#555555"),
            fontsize=8.2,
            fontweight="bold",
            va="center",
        )
    axis.annotate(
        "No-action reference",
        (0.99, NO_ACTION_SURVIVAL_REFERENCE_STEPS),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#333333",
    )
    axis.axhline(3600.0, color="#999999", linestyle="--", linewidth=0.8, zorder=1)
    axis.annotate(
        "Evaluation ceiling (3600 control steps)",
        (0.99, 3600.0),
        xycoords=("axes fraction", "data"),
        xytext=(0, -4),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=7.2,
        color="#666666",
    )
    return figure


def _journal_y_limits(values: Sequence[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return 0.0, 1.0
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    span = max(upper - lower, 10.0)
    padding = max(3.0, 0.08 * span)
    return max(0.0, lower - padding), upper + padding


def _style_journal_axis(axis: Any) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.8)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(axis="both", labelsize=8.3, width=0.8)


def _survivor_learning_mean(row: Mapping[str, Any]) -> float:
    return float(row.get("selected_parent_fitness_mean", row["fitness_mean"]))


def export_parquet_figures(location: ResultLocation) -> dict[str, Path]:
    """Create the small RQ1--RQ3 figure set solely from Parquet result parts."""

    generation_rows, episode_rows, trace_paths = _group_evidence(location)
    figures: dict[str, Path] = {}
    if not generation_rows:
        for trace_path in trace_paths:
            trace_label = trace_path.stem.replace("-", " ")
            figures[f"RQ3 behavioural trace ({trace_label})"] = plot_behaviour_trace(trace_path)
        return figures
    figures_dir = (
        location.run_dir.parents[1] / "figures" / (location.campaign_timestamp or "legacy-undated")
    )
    generation_figure = _rq1_generation_fitness_figure(generation_rows)
    generation_path = figures_dir / "rq1-learning.png"
    _atomic_figure(generation_figure, generation_path)
    plt.close(generation_figure)
    figures["RQ1 learning"] = generation_path

    held_out: dict[tuple[str, str], list[float]] = defaultdict(list)
    held_out_phase = (
        "final_test"
        if any(row.get("phase") == "final_test" for row in episode_rows)
        else "validation"
    )
    for row in episode_rows:
        if row.get("phase") == held_out_phase:
            held_out[(str(row["controller_type"]), str(row["run_id"]))].append(
                float(row["survival_steps"])
            )
    if held_out:
        figure, axis = plt.subplots(figsize=(6.3, 4.2), constrained_layout=True)
        controllers = sorted({controller for controller, _ in held_out})
        samples = []
        for controller in controllers:
            samples.append(
                [
                    float(np.mean(held_out[(controller, run_id)]))
                    for current, run_id in held_out
                    if current == controller
                ]
            )
        axis.boxplot(samples, tick_labels=controllers, showmeans=True)
        for index, values in enumerate(samples, start=1):
            axis.scatter(np.full(len(values), index), values, color="#202124", alpha=0.7, zorder=3)
        axis.set(
            title=f"RQ1 held-out {held_out_phase.replace('_', ' ')} survival",
            ylabel="Mean survival fitness (control steps per run)",
            xlabel="Controller",
        )
        axis.grid(axis="y", alpha=0.25)
        heldout_path = figures_dir / "rq1-held-out-survival.png"
        _atomic_figure(figure, heldout_path)
        plt.close(figure)
        figures["RQ1 held-out survival"] = heldout_path

    sensitivity: dict[tuple[str, int, float, str], list[float]] = defaultdict(list)
    for row in episode_rows:
        if row.get("phase") == "validation":
            sensitivity[
                (
                    str(row["controller_type"]),
                    int(row["parent_count"]),
                    float(row["mutation_scale"]),
                    str(row["run_id"]),
                )
            ].append(float(row["survival_steps"]))
    cells: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for (controller, parents, sigma, _run_id), values in sensitivity.items():
        cells[(controller, parents, sigma)].append(float(np.mean(values)))
    observed_conditions = {(parents, sigma) for _controller, parents, sigma in cells}
    if len(observed_conditions) > 1:
        figure = _rq2_ofat_figure(cells, observed_conditions)
        sensitivity_path = figures_dir / "rq2-sensitivity.png"
        _atomic_figure(figure, sensitivity_path)
        plt.close(figure)
        figures["RQ2 sensitivity"] = sensitivity_path
    for trace_path in trace_paths:
        trace_label = trace_path.stem.replace("-", " ")
        figures[f"RQ3 behavioural trace ({trace_label})"] = plot_behaviour_trace(trace_path)
    return figures


def figure_data_urls(location: ResultLocation) -> dict[str, str]:
    """Return browser-display/export data URLs for PNGs generated from Parquet."""

    return {
        label: "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        for label, path in export_parquet_figures(location).items()
    }


def model_result_directory(
    results_root: Path,
    model_version: int,
    selected_model_hash: str | None = None,
) -> Path:
    """Return one model directory without creating it."""

    selected_hash = selected_model_hash or model_hash()
    return results_root / f"model_v{model_version:03d}_{selected_hash}"


def _write_summary_table(
    directory: Path,
    name: str,
    rows: Sequence[Mapping[str, Any]],
) -> Path | None:
    if not rows:
        return None
    headers = list(dict.fromkeys(key for row in rows for key in row))
    normalized_rows = [{header: row.get(header) for header in headers} for row in rows]
    parquet_path = directory / f"{name}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = parquet_path.with_name(
        f"{parquet_path.stem}.{os.getpid()}.tmp{parquet_path.suffix}"
    )
    pq.write_table(pa.Table.from_pylist(normalized_rows), temporary, compression="zstd")
    temporary.replace(parquet_path)
    csv_output = io.StringIO(newline="")
    writer = csv.writer(csv_output)
    writer.writerow(headers)
    for row in normalized_rows:
        writer.writerow(
            [
                (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else ""
                    if value is None
                    else value
                )
                for header in headers
                for value in (row.get(header),)
            ]
        )
    _atomic_text(directory / f"{name}.csv", csv_output.getvalue())
    return parquet_path


def _mean_ci(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean_value = float(np.mean(values))
    return mean_value, _student_t_95_half_width(values)


def _rq1_heldout_figure(
    archive_training_means: Mapping[tuple[str, str], float],
    validation_means: Mapping[tuple[str, str], float],
    paired_effects: Sequence[Mapping[str, Any]] | None = None,
) -> Any:
    controllers = [
        controller
        for controller in ("reactive", "proactive")
        if any(current == controller for current, _run_id in validation_means)
    ]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.55), constrained_layout=True)
    validation_axis, difference_axis = axes
    if paired_effects:
        validation_axis.clear()
        difference_axis.clear()
        for index, row in enumerate(
            sorted(paired_effects, key=lambda item: int(item["optimizer_seed"]))
        ):
            validation_axis.scatter(
                0,
                row["reactive_validation"],
                color=RQ1_COLOURS["reactive"],
                marker=RQ1_MARKERS["reactive"],
                s=28,
            )
            validation_axis.scatter(
                1,
                row["proactive_validation"],
                color=RQ1_COLOURS["proactive"],
                marker=RQ1_MARKERS["proactive"],
                s=28,
            )
            validation_axis.plot(
                [0, 1],
                [row["reactive_validation"], row["proactive_validation"]],
                color="#999999",
                linewidth=0.75,
            )
            difference_axis.scatter(
                index,
                row["proactive_minus_reactive"],
                color=RQ1_COLOURS["proactive"],
                marker=RQ1_MARKERS["proactive"],
                s=28,
            )
        for index, controller in enumerate(("reactive", "proactive")):
            values = [float(row[f"{controller}_validation"]) for row in paired_effects]
            mean_architecture, ci_architecture = _mean_ci(values)
            if mean_architecture is not None:
                validation_axis.errorbar(
                    index,
                    mean_architecture,
                    yerr=ci_architecture,
                    color=RQ1_COLOURS[controller],
                    marker=RQ1_MARKERS[controller],
                    capsize=3,
                    linewidth=1.5,
                    zorder=4,
                )
        differences = [float(row["proactive_minus_reactive"]) for row in paired_effects]
        mean_value, interval = _mean_ci(differences)
        if mean_value is not None:
            difference_axis.errorbar(
                (len(differences) - 1) / 2,
                mean_value,
                yerr=interval,
                color=RQ1_COLOURS["proactive"],
                marker=RQ1_MARKERS["proactive"],
                capsize=3,
                linewidth=1.5,
            )
        validation_axis.set(
            title="A  Matched validation survival",
            ylabel="Validation survival [control steps]",
            xticks=(0, 1),
            xticklabels=("Reactive", "Proactive"),
        )
        difference_axis.axhline(0.0, color="#333333", linestyle=":", linewidth=1.0)
        difference_axis.set(
            title="B  Paired architecture difference",
            ylabel="Proactive minus reactive [control steps]",
            xlabel="Positive = proactive higher; negative = reactive higher",
        )
        for axis in axes:
            _style_journal_axis(axis)
        return figure
    run_sets = {
        controller: {run_id for current, run_id in validation_means if current == controller}
        for controller in controllers
    }
    paired_runs = sorted(run_sets.get("reactive", set()) & run_sets.get("proactive", set()))
    # Neutral connectors make the optimizer-seed pairing explicit.
    for run_id in paired_runs:
        for index, controller in enumerate(("reactive", "proactive")):
            validation_axis.scatter(
                index,
                validation_means[(controller, run_id)],
                color=RQ1_COLOURS[controller],
                marker=RQ1_MARKERS[controller],
                s=28,
                zorder=3,
            )
        validation_axis.plot(
            [0, 1],
            [validation_means[("reactive", run_id)], validation_means[("proactive", run_id)]],
            color="#999999",
            linewidth=0.75,
            zorder=1,
        )
    for index, controller in enumerate(controllers):
        values = [validation_means[(controller, run_id)] for run_id in sorted(run_sets[controller])]
        mean_value, interval = _mean_ci(values)
        if mean_value is not None:
            validation_axis.errorbar(
                index,
                mean_value,
                yerr=interval,
                color=RQ1_COLOURS[controller],
                marker=RQ1_MARKERS[controller],
                markersize=5.0,
                capsize=3.0,
                linewidth=1.5,
                zorder=4,
            )
    differences = [
        validation_means[("proactive", run_id)] - validation_means[("reactive", run_id)]
        for run_id in paired_runs
    ]
    if differences:
        x = np.arange(len(differences), dtype=float)
        difference_axis.scatter(
            x,
            differences,
            color=RQ1_COLOURS["proactive"],
            marker=RQ1_MARKERS["proactive"],
            s=28,
            zorder=3,
        )
        mean_value, interval = _mean_ci(differences)
        difference_axis.errorbar(
            float(np.mean(x)),
            mean_value,
            yerr=interval,
            color=RQ1_COLOURS["proactive"],
            marker=RQ1_MARKERS["proactive"],
            markersize=5.0,
            capsize=3.0,
            linewidth=1.5,
            zorder=4,
        )
    validation_axis.set(
        title="A  Matched validation survival",
        ylabel="Validation survival [control steps]",
        xticks=range(len(controllers)),
        xticklabels=[c.title() for c in controllers],
    )
    difference_axis.axhline(0.0, color="#333333", linestyle=":", linewidth=1.0)
    difference_axis.set(
        title="B  Paired architecture difference",
        ylabel="Proactive minus reactive [control steps]",
        xlabel="Positive = proactive higher; negative = reactive higher",
    )
    for axis in axes:
        _style_journal_axis(axis)
        axis.margins(x=0.25)
    return figure


def _plot_ofat_panel(
    axis: Any,
    *,
    cells: Mapping[tuple[str, int, float], Sequence[float]],
    conditions: Sequence[tuple[int, float]],
    labels: Sequence[str],
    title: str,
    xlabel: str,
) -> None:
    x = np.arange(len(conditions), dtype=np.float64)
    for controller in ("reactive", "proactive"):
        means: list[float] = []
        mean_x: list[float] = []
        intervals: list[float] = []
        for index, (parents, sigma) in enumerate(conditions):
            values = list(cells.get((controller, parents, sigma), ()))
            if not values:
                continue
            offsets = np.linspace(-0.045, 0.045, len(values)) if len(values) > 1 else [0.0]
            axis.scatter(
                x[index] + np.asarray(offsets),
                values,
                color=RQ1_COLOURS[controller],
                facecolors="white",
                linewidths=0.9,
                s=20,
                alpha=0.9,
                zorder=3,
            )
            mean_value, interval = _mean_ci(values)
            if mean_value is not None:
                mean_x.append(x[index])
                means.append(mean_value)
                intervals.append(interval or 0.0)
        if means:
            axis.errorbar(
                mean_x,
                means,
                yerr=intervals,
                color=RQ1_COLOURS[controller],
                linestyle=RQ1_LINE_STYLES[controller],
                marker=RQ1_MARKERS[controller],
                markersize=4.5,
                capsize=2.5,
                linewidth=1.8,
                label=controller.title(),
                zorder=4,
            )
    axis.set(title=title, xlabel=xlabel, xticks=x, xticklabels=labels)
    _style_journal_axis(axis)


def _rq2_ofat_figure(
    cells: Mapping[tuple[str, int, float], Sequence[float]],
    planned_conditions: set[tuple[int, float]],
    paired_differences: Mapping[tuple[int, float], Sequence[float]] | None = None,
) -> Any:
    sigma_conditions = sorted(
        (parents, sigma) for parents, sigma in planned_conditions if parents == 20
    )
    allocation_conditions = sorted(
        ((parents, sigma) for parents, sigma in planned_conditions if sigma == 0.3),
        key=lambda condition: condition[0],
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.55),
        constrained_layout=True,
        sharey=True,
    )
    paired_differences = paired_differences or {}
    for axis, conditions, labels, title, xlabel in (
        (
            axes[0],
            sigma_conditions,
            [f"{s:g}" for _p, s in sigma_conditions],
            "A  Mutation scale",
            "Mutation scale",
        ),
        (
            axes[1],
            allocation_conditions,
            [f"{p}/{100 - p}" for p, _s in allocation_conditions],
            "B  Parent--offspring split",
            "Parent--offspring split",
        ),
    ):
        x = np.arange(len(conditions), dtype=float)
        for i, condition in enumerate(conditions):
            values = list(paired_differences.get(condition, ()))
            if not values:
                continue
            offsets = np.linspace(-0.06, 0.06, len(values)) if len(values) > 1 else [0.0]
            axis.scatter(
                x[i] + np.asarray(offsets),
                values,
                color=RQ1_COLOURS["proactive"],
                marker=RQ1_MARKERS["proactive"],
                s=24,
                zorder=3,
            )
            mean_value, interval = _mean_ci(values)
            axis.errorbar(
                x[i],
                mean_value,
                yerr=interval,
                color=RQ1_COLOURS["proactive"],
                marker=RQ1_MARKERS["proactive"],
                capsize=2.5,
                linewidth=1.5,
                zorder=4,
            )
        axis.axhline(0.0, color="#333333", linestyle=":", linewidth=1.0)
        axis.set(
            title=title,
            xlabel=xlabel,
            xticks=x,
            xticklabels=labels,
            ylabel="Proactive minus reactive [control steps]",
        )
        _style_journal_axis(axis)
    return figure


def _rq2_controller_heatmaps(
    cells: Mapping[tuple[str, int, float], Sequence[float]],
    planned_conditions: set[tuple[int, float]],
) -> Any:
    """Show controller-specific OFAT means without inventing factorial cells."""

    parent_counts = sorted({parents for parents, _sigma in planned_conditions})
    mutation_scales = sorted({sigma for _parents, sigma in planned_conditions})
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), constrained_layout=True)
    colour_map = matplotlib.colormaps["Blues"].with_extremes(bad="#E7EAED")
    image = None
    for panel, (axis, controller) in enumerate(zip(axes, ("reactive", "proactive"), strict=True)):
        matrix = np.full((len(parent_counts), len(mutation_scales)), np.nan)
        counts = np.zeros_like(matrix, dtype=np.int64)
        for row_index, parent_count in enumerate(parent_counts):
            for column_index, mutation_scale in enumerate(mutation_scales):
                values = list(cells.get((controller, parent_count, mutation_scale), ()))
                if values:
                    matrix[row_index, column_index] = float(np.mean(values))
                    counts[row_index, column_index] = len(values)
        image = axis.imshow(
            np.ma.masked_invalid(matrix),
            cmap=colour_map,
            vmin=0.0,
            vmax=3600.0,
            aspect="auto",
        )
        for row_index in range(len(parent_counts)):
            for column_index in range(len(mutation_scales)):
                value = matrix[row_index, column_index]
                if np.isnan(value):
                    annotation = "not tested"
                    colour = "#5F6368"
                else:
                    annotation = f"{value:.0f}\n(n={counts[row_index, column_index]})"
                    colour = "white" if value >= 2100 else "#111111"
                axis.text(
                    column_index,
                    row_index,
                    annotation,
                    ha="center",
                    va="center",
                    color=colour,
                    fontsize=7.2,
                )
        axis.set(
            title=f"{'AB'[panel]}  {controller.title()}",
            xlabel="Fixed mutation scale",
            xticks=np.arange(len(mutation_scales)),
            xticklabels=[f"{value:g}" for value in mutation_scales],
            yticks=np.arange(len(parent_counts)),
            yticklabels=[f"{value}/{100 - value}" for value in parent_counts],
        )
        axis.set_ylabel("Parent--offspring split" if panel == 0 else "")
        axis.tick_params(length=0)
    if image is not None:
        colour_bar = figure.colorbar(image, ax=axes, shrink=0.86, pad=0.025)
        colour_bar.set_label("Mean held-out survival [control steps]")
    return figure


def _paired_intervention_effects(
    rows: Sequence[Mapping[str, Any]],
    intervention: str,
    *,
    controller: str | None = None,
) -> dict[str, list[float]]:
    values: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        current_controller = str(row["controller_type"])
        if controller is not None and current_controller != controller:
            continue
        label = str(row.get("intervention", "unknown"))
        if label not in {"normal", intervention}:
            continue
        key = (current_controller, str(row["run_id"]), int(row["world_seed"]))
        values[key][label] = float(row["survival_steps"])
    effects: dict[str, list[float]] = defaultdict(list)
    for (current_controller, _run_id, _seed), pair in values.items():
        if {"normal", intervention}.issubset(pair):
            effects[current_controller].append(pair["normal"] - pair[intervention])
    return effects


def _rq3_strategy_figure(rows: Sequence[Mapping[str, Any]]) -> Any:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(10.2, 3.55),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.0, 1.55, 0.95)},
    )
    occupancy_axis, visual_axis, state_axis = axes
    normal_rows = [row for row in rows if row.get("intervention") == "normal"]
    totals: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    for row in normal_rows:
        totals[str(row["controller_type"])] += np.asarray(
            [row["no_action_steps"], row["jump_steps"], row["duck_steps"]],
            dtype=np.float64,
        )
    controllers = [controller for controller in ("reactive", "proactive") if controller in totals]
    x = np.arange(3, dtype=np.float64)
    width = 0.34
    for index, controller in enumerate(controllers):
        proportions = totals[controller] / max(float(np.sum(totals[controller])), 1.0)
        occupancy_axis.bar(
            x + (index - (len(controllers) - 1) / 2) * width,
            proportions,
            width=width,
            color=RQ1_COLOURS[controller],
            alpha=0.82,
            label=controller.title(),
        )
    occupancy_axis.set(
        title="A  Normal-play strategy",
        ylabel="Fraction of control steps",
        xticks=x,
        xticklabels=("No action", "Jump", "Duck /\ndrop"),
        ylim=(0.0, 1.0),
    )
    occupancy_axis.legend(frameon=False, fontsize=7.8)
    _style_journal_axis(occupancy_axis)

    visual_interventions = ("constant_first_frame", *(f"ablate_sensor_{i}" for i in range(1, 6)))
    for index, intervention in enumerate(visual_interventions):
        effects = _paired_intervention_effects(rows, intervention)
        for controller in ("reactive", "proactive"):
            values = effects.get(controller, [])
            if not values:
                continue
            offsets = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0.0]
            visual_axis.scatter(
                index + np.asarray(offsets),
                values,
                color=RQ1_COLOURS[controller],
                marker=RQ1_MARKERS[controller],
                s=20,
                zorder=3,
            )
            mean_value, interval = _mean_ci(values)
            visual_axis.errorbar(
                index - 0.08 + (0.16 if controller == "proactive" else 0),
                mean_value,
                yerr=interval,
                color=RQ1_COLOURS[controller],
                marker=RQ1_MARKERS[controller],
                markersize=4.2,
                capsize=2,
                linewidth=1.2,
                zorder=4,
            )
    visual_axis.axhline(0.0, color="#333333", linestyle=":", linewidth=1.0)
    visual_axis.set(
        title="B  Visual-input dependence",
        ylabel="Normal minus intervention [control steps]",
        xticks=range(len(visual_interventions)),
        xticklabels=(
            "Constant\nframe",
            "Obstacle\nx",
            "Obstacle\ny",
            "Obstacle\nwidth",
            "Obstacle\nheight",
            "Dino\ny",
        ),
    )
    visual_axis.tick_params(axis="x", labelsize=6.9, pad=3)
    visual_axis.margins(x=0.06)
    visual_axis.legend(
        handles=[
            plt.Line2D(
                [], [], color=RQ1_COLOURS[c], marker=RQ1_MARKERS[c], linestyle="", label=c.title()
            )
            for c in ("reactive", "proactive")
        ],
        frameon=False,
        fontsize=7.5,
    )
    _style_journal_axis(visual_axis)

    state_effects = _paired_intervention_effects(
        rows,
        "reset_each_step",
        controller="proactive",
    ).get("proactive", [])
    if state_effects:
        offsets = np.linspace(-0.06, 0.06, len(state_effects)) if len(state_effects) > 1 else [0.0]
        state_axis.scatter(
            np.asarray(offsets),
            state_effects,
            color=RQ1_COLOURS["proactive"],
            facecolors="white",
            linewidths=0.9,
            s=22,
            zorder=3,
        )
        mean_value, interval = _mean_ci(state_effects)
        state_axis.errorbar(
            0,
            mean_value,
            yerr=interval,
            color=RQ1_COLOURS["proactive"],
            marker=RQ1_MARKERS["proactive"],
            markersize=4.8,
            capsize=2.5,
            linewidth=1.4,
            zorder=4,
        )
    else:
        state_axis.text(
            0.5,
            0.5,
            "Awaiting paired\nstate-reset evidence",
            ha="center",
            va="center",
            transform=state_axis.transAxes,
            fontsize=8.2,
        )
    state_axis.axhline(0.0, color="#333333", linestyle=":", linewidth=1.0)
    state_axis.set(
        title="C  State dependence",
        ylabel="Normal minus state reset\n[control steps]",
        xticks=(0,),
        xticklabels=("Proactive",),
    )
    _style_journal_axis(state_axis)
    return figure


def _mean_jump_hold_steps(action_sequences: Sequence[Sequence[int]]) -> float | None:
    holds: list[int] = []
    for actions in action_sequences:
        current = 0
        for action in actions:
            if action == 1:
                current += 1
            elif current:
                holds.append(current)
                current = 0
        if current:
            holds.append(current)
    return float(np.mean(holds)) if holds else None


def _paired_trace_effect(
    normal_actions: Sequence[int],
    intervention_actions: Sequence[int],
    normal_scores: Sequence[Sequence[float]],
    intervention_scores: Sequence[Sequence[float]],
) -> tuple[float | None, float | None, int]:
    shared_steps = min(
        len(normal_actions),
        len(intervention_actions),
        len(normal_scores),
        len(intervention_scores),
    )
    if shared_steps == 0:
        return None, None, 0
    action_disagreement = float(
        np.mean(
            np.asarray(normal_actions[:shared_steps], dtype=np.int64)
            != np.asarray(intervention_actions[:shared_steps], dtype=np.int64)
        )
    )
    normal = np.asarray(normal_scores[:shared_steps], dtype=np.float64)
    intervention = np.asarray(intervention_scores[:shared_steps], dtype=np.float64)
    score_l1_per_step = float(np.mean(np.sum(np.abs(normal - intervention), axis=1)))
    return action_disagreement, score_l1_per_step, shared_steps


def _manifest_rows(
    model_dir: Path,
    *,
    campaign_timestamp: str | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(model_dir.glob("*/*/run-manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            campaign_timestamp is not None
            and manifest.get("campaign_timestamp") != campaign_timestamp
        ):
            continue
        manifests.append((path.parent, manifest))
    return manifests


def _run_means(
    rows: Sequence[Mapping[str, Any]],
    phase: str,
) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("phase") == phase:
            values[(str(row["controller_type"]), str(row["run_id"]))].append(
                float(row["survival_steps"])
            )
    return {key: float(np.mean(value)) for key, value in values.items() if value}


def _latest_archive_training_means(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("phase") == "archive_training":
            grouped[(str(row["controller_type"]), str(row["run_id"]))].append(row)
    result: dict[tuple[str, str], float] = {}
    for key, candidates in grouped.items():
        latest_generation = max(int(row["generation"]) for row in candidates)
        latest = [
            float(row["survival_steps"])
            for row in candidates
            if int(row["generation"]) == latest_generation
        ]
        if latest:
            result[key] = float(np.mean(latest))
    return result


def aggregate_study_results(
    results_root: Path,
    *,
    model_version: int,
    planned_runs: Sequence[Mapping[str, Any]],
    selected_model_hash: str | None = None,
) -> dict[str, Any]:
    """Reconstruct report figures, tables, and GUI index solely from Parquet.

    This is deliberately project-specific: it aggregates the fixed RQ1/RQ2/RQ3
    study rather than exposing a generic analytics framework.
    """

    selected_hash = selected_model_hash or model_hash()
    model_dir = results_root / f"model_v{model_version:03d}_{selected_hash}"
    model_dir.mkdir(parents=True, exist_ok=True)
    pipeline_manifest: Mapping[str, Any] = {}
    pipeline_manifest_path = model_dir / "results-manifest.json"
    if pipeline_manifest_path.is_file():
        try:
            parsed_manifest = json.loads(pipeline_manifest_path.read_text(encoding="utf-8"))
            if isinstance(parsed_manifest, dict):
                pipeline_manifest = parsed_manifest
        except (OSError, json.JSONDecodeError):
            # The results GUI can still reconstruct all durable Parquet data
            # while a concurrent runner is atomically replacing its manifest.
            pipeline_manifest = {}
    campaign_timestamp = pipeline_manifest.get("campaign_timestamp")
    if not isinstance(campaign_timestamp, str):
        campaign_timestamp = None
    manifests = _manifest_rows(model_dir, campaign_timestamp=campaign_timestamp)
    manifest_by_run_id = {
        str(manifest["run_id"]): manifest
        for _run_dir, manifest in manifests
        if isinstance(manifest.get("run_id"), str)
    }
    frozen_selection = pipeline_manifest.get("frozen_selection")
    selected_trace_run_ids = {
        str(item["run_id"])
        for item in (
            frozen_selection.get("selected_models", [])
            if isinstance(frozen_selection, Mapping)
            else []
        )
        if isinstance(item, Mapping) and isinstance(item.get("run_id"), str)
    }
    planned_run_ids = {
        str(run["run_id"]) for run in planned_runs if isinstance(run.get("run_id"), str)
    }
    generation_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    trace_paths: list[Path] = []
    for run_dir, manifest in manifests:
        if planned_run_ids and str(manifest.get("run_id")) not in planned_run_ids:
            continue
        generation_rows.extend(read_parquet_parts(run_dir / "generation-results"))
        episode_rows.extend(read_parquet_parts(run_dir / "episode-results"))
        if not selected_trace_run_ids or str(manifest.get("run_id")) in selected_trace_run_ids:
            trace_paths.extend(sorted((run_dir / "behavioural-traces").glob("*.parquet")))

    export_suffix = campaign_timestamp or "legacy-undated"
    figures_dir = model_dir / "final_figures" / export_suffix
    summaries_dir = model_dir / "summary_tables" / export_suffix
    figures: dict[str, Path] = {}
    tables: dict[str, Path] = {}
    rq3_traces: list[dict[str, Any]] = []
    trace_rows_by_key: dict[str, list[dict[str, Any]]] = {}
    trace_actions: defaultdict[tuple[str, str], list[tuple[int, ...]]] = defaultdict(list)
    trace_actions_by_condition: dict[tuple[str, str, str, int], tuple[int, ...]] = {}
    trace_scores_by_condition: dict[
        tuple[str, str, str, int], tuple[tuple[float, float, float], ...]
    ] = {}
    for trace_path in trace_paths:
        try:
            rows = pq.read_table(trace_path).to_pylist()
        except (OSError, pa.ArrowException):
            continue
        if not rows:
            continue
        relative_trace = trace_path.relative_to(model_dir).as_posix()
        for world_seed in sorted({int(row["world_seed"]) for row in rows}):
            trace_rows = [row for row in rows if int(row["world_seed"]) == world_seed]
            if not trace_rows:
                continue
            first = trace_rows[0]
            controller = str(first.get("controller_type", "unknown"))
            intervention = str(first.get("intervention", trace_path.stem))
            run_id = str(first.get("run_id", trace_path.parent.parent.name))
            trace_id = str(first.get("trace_id", trace_path.stem))
            key = hashlib.sha256(f"{relative_trace}|{world_seed}".encode()).hexdigest()[:12]
            figure_path = (
                figures_dir
                / "rq3_traces"
                / (
                    f"rq3_trace_{_safe_filename(controller)}_{_safe_filename(run_id)}_"
                    f"world_{world_seed}_{_safe_filename(intervention)}.png"
                )
            )
            if (
                not figure_path.is_file()
                or figure_path.stat().st_mtime_ns < trace_path.stat().st_mtime_ns
            ):
                plot_behaviour_trace(
                    trace_path,
                    output_path=figure_path,
                    world_seed=world_seed,
                )
            trace_rows_by_key[key] = trace_rows
            actions = tuple(int(row["action"]) for row in trace_rows)
            trace_condition = (controller, run_id, intervention, world_seed)
            trace_actions[(controller, intervention)].append(actions)
            trace_actions_by_condition[trace_condition] = actions
            trace_scores_by_condition[trace_condition] = tuple(
                (
                    float(row["score_0"]),
                    float(row["score_1"]),
                    float(row["score_2"]),
                )
                for row in trace_rows
            )
            rq3_traces.append(
                {
                    "id": key,
                    "controller": controller,
                    "run_id": run_id,
                    "world_seed": world_seed,
                    "intervention": intervention,
                    "trace_id": trace_id,
                    "label": " · ".join(
                        (
                            controller.title(),
                            (
                                f"optimiser seed {match.group(1)}"
                                if (match := re.search(r"seed-(\d+)", run_id))
                                else "archived run"
                            ),
                            f"world seed {world_seed}",
                            intervention.replace("_", " "),
                        )
                    ),
                    "figure": str(figure_path.relative_to(model_dir).as_posix()),
                    "data": relative_trace,
                }
            )
    planned_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for run in planned_runs:
        planned_by_group[str(run["experiment_group"])].append(run)

    rq1_generation = [row for row in generation_rows if row.get("experiment_group") == "rq1-main"]
    rq1_episode = [row for row in episode_rows if row.get("experiment_group") == "rq1-main"]
    archive_training_means = _latest_archive_training_means(rq1_episode)
    validation_means = _run_means(rq1_episode, "validation")
    if rq1_generation:
        rq1_controllers = {str(row["controller_type"]) for row in rq1_generation}
        required_rq1_run_ids = {
            controller: {
                str(run["run_id"])
                for run in planned_by_group["rq1-main"]
                if run.get("controller_type") == controller
            }
            for controller in rq1_controllers
        }
        generation_figure = _rq1_generation_fitness_figure(
            rq1_generation,
            required_rq1_run_ids,
        )
        path = figures_dir / "rq1_learning.png"
        _atomic_figure(generation_figure, path)
        plt.close(generation_figure)
        figures["RQ1 learning"] = path
        learning_rows: list[dict[str, Any]] = []
        for row in sorted(
            rq1_generation,
            key=lambda item: (
                str(item.get("controller_type")),
                str(item.get("run_id")),
                int(item.get("generation", 0)),
            ),
        ):
            learning_rows.append(
                {
                    "controller": str(row["controller_type"]),
                    "optimizer_seed": row.get("optimizer_seed"),
                    "run_id": str(row["run_id"]),
                    "generation": int(row["generation"]) + 1,
                    "cumulative_candidate_evaluations": int(
                        row["cumulative_candidate_evaluations"]
                    ),
                    "selected_parent_mean_survival": _survivor_learning_mean(row),
                    "series": "raw_run",
                }
            )
        grouped_learning: defaultdict[tuple[str, int], dict[str, float]] = defaultdict(dict)
        for row in learning_rows:
            grouped_learning[(str(row["controller"]), int(row["generation"]))][
                str(row["run_id"])
            ] = float(row["selected_parent_mean_survival"])
        for (controller, generation), run_values in sorted(grouped_learning.items()):
            required = required_rq1_run_ids.get(controller, set(run_values))
            if not required or not required.issubset(run_values):
                continue
            source = next(
                item
                for item in learning_rows
                if item["controller"] == controller and item["generation"] == generation
            )
            learning_rows.append(
                {
                    **{
                        key: source[key]
                        for key in ("controller", "generation", "cumulative_candidate_evaluations")
                    },
                    "optimizer_seed": None,
                    "run_id": "mean",
                    "selected_parent_mean_survival": float(
                        np.mean([run_values[run_id] for run_id in sorted(required)])
                    ),
                    "series": "arithmetic_mean",
                }
            )
        table_path = _write_summary_table(summaries_dir, "rq1_learning_data", learning_rows)
        if table_path is not None:
            tables["RQ1 learning data"] = table_path

    rq1_summary: list[dict[str, Any]] = []
    for controller in ("reactive", "proactive"):
        paired = sorted(
            {run_id for current, run_id in archive_training_means if current == controller}
            & {run_id for current, run_id in validation_means if current == controller}
        )
        train_values = [archive_training_means[(controller, run_id)] for run_id in paired]
        validation_values = [validation_means[(controller, run_id)] for run_id in paired]
        train_mean, train_ci = _mean_ci(train_values)
        validation_mean, validation_ci = _mean_ci(validation_values)
        gap_values = [
            archive_training_means[(controller, run_id)] - validation_means[(controller, run_id)]
            for run_id in paired
        ]
        gap_mean, gap_ci = _mean_ci(gap_values)
        archive_rows_for_controller = [
            row
            for row in rq1_episode
            if row.get("phase") == "archive_training"
            and str(row.get("controller_type")) == controller
            and str(row.get("run_id")) in paired
        ]
        planned_count = sum(
            1 for run in planned_by_group["rq1-main"] if run.get("controller_type") == controller
        )
        rq1_summary.append(
            {
                "controller": controller,
                "completed_runs": len(paired),
                "planned_runs": planned_count,
                "archive_training_mean": train_mean,
                "archive_training_95ci": train_ci,
                "archive_training_ceiling_fraction": (
                    sum(float(row["survival_steps"]) >= 3600 for row in archive_rows_for_controller)
                    / len(archive_rows_for_controller)
                    if archive_rows_for_controller
                    else None
                ),
                "validation_mean": validation_mean,
                "validation_95ci": validation_ci,
                "training_minus_validation_gap": gap_mean,
                "training_minus_validation_gap_95ci": gap_ci,
                "gap_sign_convention": "positive means validation survival is lower",
                "paired_archive_validation_runs": len(paired),
                "best_validation_run": max(validation_values) if validation_values else None,
                "candidate_evaluations_per_run": max(
                    [
                        int(row["cumulative_candidate_evaluations"])
                        for row in rq1_generation
                        if row["controller_type"] == controller
                    ],
                    default=0,
                ),
                "uncertainty_definition": (
                    "two-sided 95% Student-t CI across independent optimiser runs; "
                    "undefined for fewer than two runs"
                ),
            }
        )
    table_path = _write_summary_table(summaries_dir, "rq1_summary", rq1_summary)
    if table_path is not None:
        tables["RQ1 summary"] = table_path
    rq1_heldout_rows: list[dict[str, Any]] = []
    for controller in ("reactive", "proactive"):
        run_ids = sorted({run_id for current, run_id in validation_means if current == controller})
        for run_id in run_ids:
            validation_rows = [
                row
                for row in rq1_episode
                if row.get("phase") == "validation"
                and str(row.get("controller_type")) == controller
                and str(row.get("run_id")) == run_id
            ]
            archive_rows = [
                row
                for row in rq1_episode
                if row.get("phase") == "archive_training"
                and str(row.get("controller_type")) == controller
                and str(row.get("run_id")) == run_id
            ]
            if not validation_rows:
                continue
            rq1_heldout_rows.append(
                {
                    "controller": controller,
                    "run_id": run_id,
                    "optimizer_seed": validation_rows[0].get("optimizer_seed"),
                    "archive_training_mean": archive_training_means.get((controller, run_id)),
                    "archive_training_ceiling_fraction": (
                        sum(float(r["survival_steps"]) >= 3600 for r in archive_rows)
                        / len(archive_rows)
                        if archive_rows
                        else None
                    ),
                    "validation_survival": validation_means.get((controller, run_id)),
                    "validation_ceiling_fraction": sum(
                        float(r["survival_steps"]) >= 3600 for r in validation_rows
                    )
                    / len(validation_rows),
                }
            )
    table_path = _write_summary_table(summaries_dir, "rq1_heldout_data", rq1_heldout_rows)
    if table_path is not None:
        tables["RQ1 held-out performance data"] = table_path

    rq1_run_seed: dict[tuple[str, str], int] = {}
    for row in rq1_episode:
        if row.get("phase") == "validation":
            rq1_run_seed[(str(row["controller_type"]), str(row["run_id"]))] = int(
                row["optimizer_seed"]
            )
    rq1_by_optimizer_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for run_key, value in validation_means.items():
        optimizer_seed = rq1_run_seed.get(run_key)
        if optimizer_seed is not None:
            rq1_by_optimizer_seed[optimizer_seed][run_key[0]] = value
    rq1_paired_effects = [
        {
            "optimizer_seed": seed,
            "reactive_validation": values["reactive"],
            "proactive_validation": values["proactive"],
            "proactive_minus_reactive": values["proactive"] - values["reactive"],
        }
        for seed, values in sorted(rq1_by_optimizer_seed.items())
        if {"reactive", "proactive"}.issubset(values)
    ]
    if rq1_paired_effects:
        paired_differences = [float(row["proactive_minus_reactive"]) for row in rq1_paired_effects]
        paired_mean, paired_ci = _mean_ci(paired_differences)
        architecture_summary = [
            {
                "effect": "proactive_minus_reactive_validation_survival",
                "mean_effect": paired_mean,
                "effect_95ci": paired_ci,
                "matched_optimizer_runs": len(paired_differences),
                "uncertainty_definition": (
                    "two-sided 95% Student-t CI across paired independent optimiser runs; "
                    "descriptive, not a significance test"
                ),
            }
        ]
        table_path = _write_summary_table(
            summaries_dir,
            "rq1_architecture_comparison",
            architecture_summary,
        )
        if table_path is not None:
            tables["RQ1 paired architecture comparison"] = table_path
        table_path = _write_summary_table(summaries_dir, "rq1_paired_effects", rq1_paired_effects)
        if table_path is not None:
            tables["RQ1 paired effects"] = table_path
        by_seed_controller = {
            (int(row["optimizer_seed"]), str(row["controller"])): row
            for row in rq1_heldout_rows
            if row.get("optimizer_seed") is not None
        }
        paired_rows = []
        for effect in rq1_paired_effects:
            seed = int(effect["optimizer_seed"])
            reactive_row = by_seed_controller.get((seed, "reactive"), {})
            proactive_row = by_seed_controller.get((seed, "proactive"), {})
            paired_rows.append(
                {
                    **effect,
                    "reactive_archive_training_mean": reactive_row.get("archive_training_mean"),
                    "proactive_archive_training_mean": proactive_row.get("archive_training_mean"),
                    "reactive_archive_training_ceiling_fraction": reactive_row.get(
                        "archive_training_ceiling_fraction"
                    ),
                    "proactive_archive_training_ceiling_fraction": proactive_row.get(
                        "archive_training_ceiling_fraction"
                    ),
                    "reactive_validation_ceiling_fraction": reactive_row.get(
                        "validation_ceiling_fraction"
                    ),
                    "proactive_validation_ceiling_fraction": proactive_row.get(
                        "validation_ceiling_fraction"
                    ),
                }
            )
        table_path = _write_summary_table(summaries_dir, "rq1_heldout_data", paired_rows)
        if table_path is not None:
            tables["RQ1 held-out performance data"] = table_path

    planned_rq1_keys = {
        (str(run["controller_type"]), str(run["run_id"]))
        for run in planned_by_group["rq1-main"]
        if isinstance(run.get("controller_type"), str)
    }
    complete_rq1_keys = set(archive_training_means) & set(validation_means)
    if planned_rq1_keys and planned_rq1_keys.issubset(complete_rq1_keys):
        figure = _rq1_heldout_figure(archive_training_means, validation_means, rq1_paired_effects)
        path = figures_dir / "rq1_heldout_performance.png"
        _atomic_figure(figure, path)
        plt.close(figure)
        figures["RQ1 held-out performance"] = path

    planned_rq2 = planned_by_group["rq2-sensitivity"]
    if any(run.get("adaptive_sigma") is not False for run in planned_rq2):
        raise ValueError("RQ2 sensitivity analysis requires adaptive_sigma=false")
    rq2_group_episode = [
        row for row in episode_rows if row.get("experiment_group") == "rq2-sensitivity"
    ]
    if any(row.get("adaptive_sigma") is not False for row in rq2_group_episode):
        raise ValueError("RQ2 Parquet evidence contains an adaptive mutation scale")
    rq2_episode = rq2_group_episode
    rq2_validation = _run_means(rq2_episode, "validation")
    rq2_cells: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    run_lookup: dict[tuple[str, str], tuple[int, float]] = {}
    rq2_run_seed: dict[tuple[str, str], int] = {}
    for row in rq2_episode:
        if row.get("phase") == "validation":
            run_key = (str(row["controller_type"]), str(row["run_id"]))
            run_lookup[run_key] = (
                int(row["parent_count"]),
                float(row["mutation_scale"]),
            )
            rq2_run_seed[run_key] = int(row["optimizer_seed"])
    for run_key, value in rq2_validation.items():
        condition = run_lookup.get(run_key)
        if condition is not None:
            rq2_cells[(run_key[0], *condition)].append(value)
    rq2_by_optimizer_seed: dict[tuple[int, float, int], dict[str, float]] = defaultdict(dict)
    for run_key, value in rq2_validation.items():
        condition = run_lookup.get(run_key)
        optimizer_seed = rq2_run_seed.get(run_key)
        if condition is not None and optimizer_seed is not None:
            rq2_by_optimizer_seed[(*condition, optimizer_seed)][run_key[0]] = value
    rq2_paired_differences: dict[tuple[int, float], list[float]] = defaultdict(list)
    for (
        parent_count,
        mutation_scale,
        _optimizer_seed,
    ), controller_values in rq2_by_optimizer_seed.items():
        if {"reactive", "proactive"}.issubset(controller_values):
            rq2_paired_differences[(parent_count, mutation_scale)].append(
                controller_values["proactive"] - controller_values["reactive"]
            )
    planned_conditions = {
        (int(run["parent_count"]), float(run["mutation_scale"])) for run in planned_rq2
    }
    planned_cells: dict[tuple[str, int, float], int] = defaultdict(int)
    for run in planned_rq2:
        planned_cells[
            (str(run["controller_type"]), int(run["parent_count"]), float(run["mutation_scale"]))
        ] += 1
    rq2_summary: list[dict[str, Any]] = []
    for parent_count, mutation_scale in sorted(planned_conditions):
        reactive = rq2_cells.get(("reactive", parent_count, mutation_scale), [])
        proactive = rq2_cells.get(("proactive", parent_count, mutation_scale), [])
        reactive_mean, reactive_ci = _mean_ci(reactive)
        proactive_mean, proactive_ci = _mean_ci(proactive)
        paired_effects = rq2_paired_differences[(parent_count, mutation_scale)]
        paired_effect, paired_effect_ci = _mean_ci(paired_effects)
        rq2_summary.append(
            {
                "mutation_scale": mutation_scale,
                "parent_count": parent_count,
                "offspring_count": 100 - parent_count,
                "reactive_validation_mean": reactive_mean,
                "reactive_validation_95ci": reactive_ci,
                "proactive_validation_mean": proactive_mean,
                "proactive_validation_95ci": proactive_ci,
                "proactive_minus_reactive": paired_effect,
                "proactive_minus_reactive_95ci": paired_effect_ci,
                "paired_optimizer_runs": len(paired_effects),
                "paired_difference_1": paired_effects[0] if len(paired_effects) > 0 else None,
                "paired_difference_2": paired_effects[1] if len(paired_effects) > 1 else None,
                "paired_difference_3": paired_effects[2] if len(paired_effects) > 2 else None,
                "reactive_runs": len(reactive),
                "proactive_runs": len(proactive),
                "planned_runs_per_architecture": planned_cells[
                    ("reactive", parent_count, mutation_scale)
                ],
                "evidence_source": "independent fixed-sigma RQ2 OFAT runs",
                "equal_candidate_budget": True,
                "uncertainty_definition": (
                    "two-sided 95% Student-t CI across independent optimiser runs; "
                    "paired architecture difference uses matched optimiser seeds"
                ),
            }
        )
    table_path = _write_summary_table(summaries_dir, "rq2_summary", rq2_summary)
    if table_path is not None:
        tables["RQ2 summary"] = table_path
    rq2_raw_rows = []
    for (parent_count, mutation_scale, optimizer_seed), controller_values in sorted(
        rq2_by_optimizer_seed.items()
    ):
        if not {"reactive", "proactive"}.issubset(controller_values):
            continue
        difference = controller_values["proactive"] - controller_values["reactive"]
        for index, difference_value in enumerate([difference], 1):
            rq2_raw_rows.append(
                {
                    "parent_count": parent_count,
                    "offspring_count": 100 - parent_count,
                    "sigma": mutation_scale,
                    "paired_index": index,
                    "optimizer_seed": optimizer_seed,
                    "reactive_validation": controller_values["reactive"],
                    "proactive_validation": controller_values["proactive"],
                    "paired_proactive_minus_reactive": difference_value,
                    "reactive_mean_validation": _mean_ci(
                        rq2_cells.get(("reactive", parent_count, mutation_scale), [])
                    )[0],
                    "proactive_mean_validation": _mean_ci(
                        rq2_cells.get(("proactive", parent_count, mutation_scale), [])
                    )[0],
                }
            )
    table_path = _write_summary_table(summaries_dir, "rq2_paired_data", rq2_raw_rows)
    if table_path is not None:
        tables["RQ2 OFAT sensitivity data"] = table_path
        tables["RQ2 controller heatmaps data"] = table_path
    if planned_conditions and rq2_cells:
        figure = _rq2_ofat_figure(rq2_cells, planned_conditions, rq2_paired_differences)
        path = figures_dir / "rq2_ofat_sensitivity.png"
        _atomic_figure(figure, path)
        plt.close(figure)
        figures["RQ2 OFAT sensitivity"] = path
        figure = _rq2_controller_heatmaps(rq2_cells, planned_conditions)
        path = figures_dir / "rq2_controller_heatmaps.png"
        _atomic_figure(figure, path)
        plt.close(figure)
        figures["RQ2 controller heatmaps"] = path

    intervention_rows = [row for row in episode_rows if row.get("phase") == "intervention"]
    rq3_summary: list[dict[str, Any]] = []
    intervention_means: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in intervention_rows:
        intervention_means[
            (str(row["controller_type"]), str(row.get("intervention", "unknown")))
        ].append(float(row["survival_steps"]))
    for (controller, intervention), intervention_values in sorted(intervention_means.items()):
        intervention_mean: float | None
        intervention_ci: float | None
        intervention_mean, intervention_ci = _mean_ci(intervention_values)
        action_totals = np.sum(
            np.asarray(
                [
                    [
                        int(row.get("no_action_steps", 0)),
                        int(row.get("jump_steps", 0)),
                        int(row.get("duck_steps", 0)),
                    ]
                    for row in intervention_rows
                    if str(row["controller_type"]) == controller
                    and str(row.get("intervention", "unknown")) == intervention
                ],
                dtype=np.float64,
            ),
            axis=0,
        )
        action_total = float(np.sum(action_totals))
        rq3_summary.append(
            {
                "controller": controller,
                "intervention": intervention,
                "mean_survival": intervention_mean,
                "survival_95ci": intervention_ci,
                "episodes": len(intervention_values),
                "mean_action_switches": float(
                    np.mean(
                        [
                            int(row.get("action_switches", 0))
                            for row in intervention_rows
                            if str(row["controller_type"]) == controller
                            and str(row.get("intervention", "unknown")) == intervention
                        ]
                    )
                ),
                "no_action_fraction": (
                    float(action_totals[0] / action_total) if action_total else None
                ),
                "jump_held_fraction": (
                    float(action_totals[1] / action_total) if action_total else None
                ),
                "duck_fast_drop_fraction": (
                    float(action_totals[2] / action_total) if action_total else None
                ),
                "mean_jump_hold_steps": _mean_jump_hold_steps(
                    trace_actions[(controller, intervention)]
                ),
                "uncertainty_definition": (
                    "two-sided 95% Student-t CI across matched replay world seeds for one "
                    "frozen policy"
                ),
            }
        )
    table_path = _write_summary_table(summaries_dir, "rq3_summary", rq3_summary)
    if table_path is not None:
        tables["RQ3 summary"] = table_path
    rq3_effect_rows: list[dict[str, Any]] = []
    for row in intervention_rows:
        if str(row.get("intervention")) == "normal":
            rq3_effect_rows.append(
                {
                    "record_type": "normal_occupancy",
                    "controller": row.get("controller_type"),
                    "run_id": row.get("run_id"),
                    "world_seed": row.get("world_seed"),
                    "intervention": "normal",
                    "no_action_steps": row.get("no_action_steps", 0),
                    "jump_steps": row.get("jump_steps", 0),
                    "duck_steps": row.get("duck_steps", 0),
                    "pterodactyl_visible_steps": row.get("pterodactyl_visible_steps", 0),
                    "pterodactyl_exposure": int(row.get("pterodactyl_visible_steps", 0)) > 0,
                }
            )
    causal_interventions = (
        "constant_first_frame",
        *(f"ablate_sensor_{index}" for index in range(1, 6)),
        "reset_each_step",
    )
    effect_records: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for intervention in causal_interventions:
        survival_pairs: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
        for row in intervention_rows:
            effect_key = (
                str(row["controller_type"]),
                str(row["run_id"]),
                int(row["world_seed"]),
            )
            label = str(row.get("intervention"))
            if label in {"normal", intervention}:
                survival_pairs[effect_key][label] = float(row["survival_steps"])
        for (controller, run_id, world_seed), pair in sorted(survival_pairs.items()):
            if "normal" in pair and intervention in pair:
                effect_records[(controller, run_id, world_seed, intervention)] = {
                    "record_type": "paired_causal_effect",
                    "controller": controller,
                    "run_id": run_id,
                    "world_seed": world_seed,
                    "intervention": intervention,
                    "normal_minus_intervention_survival": pair["normal"] - pair[intervention],
                }
    for (
        controller,
        run_id,
        intervention,
        world_seed,
    ), intervention_actions in trace_actions_by_condition.items():
        if intervention not in causal_interventions:
            continue
        normal_key = (controller, run_id, "normal", world_seed)
        if normal_key not in trace_actions_by_condition:
            continue
        action_effect, score_effect, shared_steps = _paired_trace_effect(
            trace_actions_by_condition[normal_key],
            intervention_actions,
            trace_scores_by_condition[normal_key],
            trace_scores_by_condition[(controller, run_id, intervention, world_seed)],
        )
        record = effect_records.setdefault(
            (controller, run_id, world_seed, intervention),
            {
                "record_type": "paired_causal_effect",
                "controller": controller,
                "run_id": run_id,
                "world_seed": world_seed,
                "intervention": intervention,
            },
        )
        record.update(
            {
                "shared_prefix_steps": shared_steps,
                "action_disagreement_fraction": action_effect,
                "output_score_l1_per_step": score_effect,
            }
        )
    rq3_effect_rows.extend(effect_records[key] for key in sorted(effect_records))

    rq3_causal_summary: list[dict[str, Any]] = []
    summary_keys = sorted({(key[0], key[1], key[3]) for key in effect_records})
    for controller, run_id, intervention in summary_keys:
        records = [
            record
            for (
                current,
                current_run,
                _seed,
                current_intervention,
            ), record in effect_records.items()
            if (current, current_run, current_intervention) == (controller, run_id, intervention)
        ]
        survival_effects = [
            float(record["normal_minus_intervention_survival"])
            for record in records
            if record.get("normal_minus_intervention_survival") is not None
        ]
        action_effects = [
            float(record["action_disagreement_fraction"])
            for record in records
            if record.get("action_disagreement_fraction") is not None
        ]
        score_effects = [
            float(record["output_score_l1_per_step"])
            for record in records
            if record.get("output_score_l1_per_step") is not None
        ]
        normal_rows_for_run = [
            row
            for row in intervention_rows
            if str(row["controller_type"]) == controller
            and str(row["run_id"]) == run_id
            and str(row.get("intervention")) == "normal"
        ]
        mean_survival_effect = float(np.mean(survival_effects)) if survival_effects else None
        shared_prefix_steps = sum(int(record.get("shared_prefix_steps", 0)) for record in records)
        rq3_causal_summary.append(
            {
                "controller": controller,
                "run_id": run_id,
                "intervention": intervention,
                "n_matched_worlds": len(survival_effects),
                "paired_survival_seeds": len(survival_effects),
                "paired_trace_seeds": len(action_effects),
                "shared_prefix_steps": shared_prefix_steps,
                "shared_trace_steps": shared_prefix_steps,
                "normal_minus_intervention_survival": mean_survival_effect,
                "mean_normal_minus_constant_survival": (
                    mean_survival_effect if intervention == "constant_first_frame" else None
                ),
                "delta_t_visual": (
                    mean_survival_effect if intervention != "reset_each_step" else None
                ),
                "delta_t_state": (
                    mean_survival_effect if intervention == "reset_each_step" else None
                ),
                "mean_action_disagreement_fraction": (
                    float(np.mean(action_effects)) if action_effects else None
                ),
                "mean_output_score_l1_difference_per_step": (
                    float(np.mean(score_effects)) if score_effects else None
                ),
                "discrete_actions_depend_on_visual_input": (
                    any(effect > 0 for effect in action_effects) if action_effects else None
                ),
                "output_scores_depend_on_visual_input": (
                    any(effect > NUMERICAL_CHANGE_EPSILON for effect in score_effects)
                    if score_effects
                    else None
                ),
                "normal_pterodactyl_visible_steps": sum(
                    int(row.get("pterodactyl_visible_steps", 0)) for row in normal_rows_for_run
                ),
                "normal_pterodactyl_exposure_episodes": sum(
                    int(row.get("pterodactyl_visible_steps", 0)) > 0 for row in normal_rows_for_run
                ),
                "interpretation": (
                    "paired causal intervention on matched frozen-policy worlds; null effects "
                    "remain reportable"
                ),
            }
        )
    table_path = _write_summary_table(
        summaries_dir,
        "rq3_causal_sensor_dependence",
        rq3_causal_summary,
    )
    if table_path is not None:
        tables["RQ3 causal sensor dependence"] = table_path
    table_path = _write_summary_table(summaries_dir, "rq3_paired_effect_data", rq3_effect_rows)
    if table_path is not None:
        tables["RQ3 strategy and causal interventions data"] = table_path
    selected_models_for_rq3 = (
        pipeline_manifest.get("frozen_selection", {}).get("selected_models", [])
        if isinstance(pipeline_manifest.get("frozen_selection"), Mapping)
        else []
    )
    selected_rq3_run_ids = {
        str(item.get("run_id"))
        for item in selected_models_for_rq3
        if isinstance(item, Mapping) and item.get("run_id") is not None
    }
    visual_interventions = {"normal", "constant_first_frame"} | {
        f"ablate_sensor_{index}" for index in range(1, 6)
    }
    rq3_primary_complete = bool(selected_rq3_run_ids)
    for run_id in selected_rq3_run_ids:
        run_interventions = {
            str(row.get("intervention"))
            for row in intervention_rows
            if str(row.get("run_id")) == run_id
        }
        planned_rq3 = next((run for run in planned_runs if str(run.get("run_id")) == run_id), {})
        expected_worlds = {
            int(seed)
            for seed in (
                planned_rq3.get("validation_seeds")
                or manifest_by_run_id.get(run_id, {}).get("validation_seeds", [])
            )
        }
        for intervention in visual_interventions:
            observed_worlds = {
                int(row["world_seed"])
                for row in intervention_rows
                if str(row.get("run_id")) == run_id and str(row.get("intervention")) == intervention
            }
            rq3_primary_complete &= observed_worlds == expected_worlds
        rq3_primary_complete &= visual_interventions.issubset(run_interventions)
        if any(
            str(row.get("run_id")) == run_id and str(row.get("controller_type")) == "proactive"
            for row in intervention_rows
        ):
            rq3_primary_complete &= "reset_each_step" in run_interventions
            observed_reset = {
                int(row["world_seed"])
                for row in intervention_rows
                if str(row.get("run_id")) == run_id
                and str(row.get("intervention")) == "reset_each_step"
            }
            rq3_primary_complete &= observed_reset == expected_worlds
    if intervention_rows and rq3_primary_complete:
        figure = _rq3_strategy_figure(intervention_rows)
        path = figures_dir / "rq3_strategy_and_causal_interventions.png"
        _atomic_figure(figure, path)
        plt.close(figure)
        figures["RQ3 strategy and causal interventions"] = path

    final_test_by_controller: dict[str, dict[int, float]] = defaultdict(dict)
    final_test_seen: set[tuple[str, int, str]] = set()
    final_test_representatives: dict[str, set[str]] = defaultdict(set)
    for row in rq1_episode:
        if row.get("phase") == "final_test":
            controller = str(row["controller_type"])
            world_seed = int(row["world_seed"])
            run_id = str(row["run_id"])
            identity = (controller, world_seed, run_id)
            if identity in final_test_seen:
                raise ValueError(
                    f"Duplicate final-test record for {controller}, run {run_id}, "
                    f"world {world_seed}"
                )
            final_test_seen.add(identity)
            if world_seed in final_test_by_controller[controller]:
                raise ValueError(
                    f"Multiple final-test representatives for {controller}, world {world_seed}"
                )
            final_test_by_controller[controller][world_seed] = float(row["survival_steps"])
            final_test_representatives[controller].add(str(row["run_id"]))
    final_test_summary: list[dict[str, Any]] = []
    selected_final_run_ids = {
        str(item.get("run_id"))
        for item in pipeline_manifest.get("frozen_selection", {}).get("selected_models", [])
        if isinstance(item, Mapping) and item.get("run_id") is not None
    }
    expected_final_seed_sets = [
        {int(seed) for seed in manifest_by_run_id.get(run_id, {}).get("final_test_seeds", [])}
        for run_id in sorted(selected_final_run_ids)
    ]
    expected_final_seeds = expected_final_seed_sets[0] if expected_final_seed_sets else set()
    final_protocol_matches = bool(expected_final_seeds) and all(
        seeds == expected_final_seeds for seeds in expected_final_seed_sets
    )
    final_sets_match = (
        final_protocol_matches
        and set(final_test_by_controller.get("reactive", {})) == expected_final_seeds
        and set(final_test_by_controller.get("proactive", {})) == expected_final_seeds
    )
    if {"reactive", "proactive"}.issubset(final_test_by_controller) and final_sets_match:
        final_reactive = final_test_by_controller["reactive"]
        final_proactive = final_test_by_controller["proactive"]
        seeds = sorted(set(final_reactive) & set(final_proactive))
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.55), constrained_layout=True)
        scatter_axis, difference_axis = axes
        coordinate_counts: defaultdict[tuple[float, float], int] = defaultdict(int)
        for seed in seeds:
            coordinate_counts[(final_proactive[seed], final_reactive[seed])] += 1
        for (proactive_value, reactive_value), count in coordinate_counts.items():
            scatter_axis.scatter(
                proactive_value,
                reactive_value,
                color="#202124",
                s=24 + 12 * (count - 1),
                zorder=3,
            )
            if count > 1:
                scatter_axis.annotate(
                    f"n={count}",
                    (proactive_value, reactive_value),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=5.8,
                    color="#555555",
                )
        scatter_axis.plot([0, 3600], [0, 3600], color="#777777", linestyle=":", linewidth=1.0)
        scatter_axis.axvline(3600, color="#999999", linestyle="--", linewidth=0.7)
        scatter_axis.axhline(3600, color="#999999", linestyle="--", linewidth=0.7)
        scatter_axis.set(
            xlabel="Proactive survival [control steps]",
            ylabel="Reactive survival [control steps]",
            title="A  Paired world performance",
            xlim=(0, 3600),
            ylim=(0, 3600),
            aspect="equal",
        )
        differences = [final_reactive[seed] - final_proactive[seed] for seed in seeds]
        difference_axis.scatter(
            np.arange(len(differences)),
            differences,
            color=RQ1_COLOURS["reactive"],
            marker=RQ1_MARKERS["reactive"],
            s=24,
        )
        mean_value, interval = _mean_ci(differences)
        if mean_value is not None:
            difference_axis.errorbar(
                (len(differences) - 1) / 2,
                mean_value,
                yerr=interval,
                color=RQ1_COLOURS["reactive"],
                marker="o",
                capsize=3,
                linewidth=1.5,
            )
        difference_axis.axhline(0, color="#333333", linestyle=":", linewidth=1)
        difference_axis.set(
            title="B  Paired final-test difference",
            xlabel="Locked world seed",
            ylabel="Reactive minus proactive [control steps]",
            xticks=np.arange(len(seeds)),
            xticklabels=[str(seed) for seed in seeds],
        )
        difference_axis.tick_params(axis="x", labelsize=6.5)
        plt.setp(difference_axis.get_xticklabels(), rotation=90, ha="center", va="top")
        for axis in axes:
            _style_journal_axis(axis)
        for controller in ("reactive", "proactive"):
            values = list(final_test_by_controller[controller].values())
            representative_rows = [
                row
                for row in rq1_episode
                if row.get("phase") == "final_test"
                and str(row.get("controller_type")) == controller
            ]
            representative = representative_rows[0] if representative_rows else {}
            final_test_mean, final_test_ci = _mean_ci(values)
            final_test_summary.append(
                {
                    "controller": controller,
                    "summary_type": "architecture",
                    "selected_run_id": representative.get("run_id"),
                    "selected_optimizer_seed": representative.get("optimizer_seed"),
                    "frozen_representatives": len(final_test_representatives[controller]),
                    "locked_environment_seeds": len(values),
                    "mean_final_test_survival": final_test_mean,
                    "median_final_test_survival": float(np.median(values)) if values else None,
                    "quartiles_final_test_survival": list(np.percentile(values, [25, 75]))
                    if values
                    else [],
                    "q1_final_test_survival": float(np.percentile(values, 25)) if values else None,
                    "q3_final_test_survival": float(np.percentile(values, 75)) if values else None,
                    "minimum_final_test_survival": min(values) if values else None,
                    "maximum_final_test_survival": max(values) if values else None,
                    "ceiling_count": sum(value >= 3600 for value in values),
                    "ceiling_fraction": sum(value >= 3600 for value in values) / len(values)
                    if values
                    else None,
                    "environment_seed_95ci": final_test_ci,
                    "uncertainty_definition": (
                        "two-sided 95% Student-t CI across locked environment seeds for "
                        "one validation-frozen representative"
                    ),
                }
            )
        final_test_summary.append(
            {
                "summary_type": "paired_difference",
                "paired_reactive_minus_proactive_mean": mean_value,
                "paired_reactive_minus_proactive_95ci": interval,
                "paired_reactive_minus_proactive_median": float(np.median(differences))
                if differences
                else None,
                "locked_environment_seeds": len(differences),
                "uncertainty_definition": (
                    "environment-seed variation for two frozen representatives"
                ),
            }
        )
        path = figures_dir / "final_test_comparison.png"
        _atomic_figure(figure, path)
        plt.close(figure)
        figures["Final test comparison"] = path
        table_path = _write_summary_table(summaries_dir, "final_test_summary", final_test_summary)
        if table_path is not None:
            tables["Final test summary"] = table_path
        paired_rows = [
            {
                "world_seed": seed,
                "proactive_survival": final_proactive[seed],
                "reactive_survival": final_reactive[seed],
                "reactive_minus_proactive": final_reactive[seed] - final_proactive[seed],
            }
            for seed in seeds
        ]
        table_path = _write_summary_table(summaries_dir, "final_test_paired_data", paired_rows)
        if table_path is not None:
            tables["Final test comparison data"] = table_path

    expected = {group: len(entries) for group, entries in planned_by_group.items()}
    completed: defaultdict[str, int] = defaultdict(int)
    failed: defaultdict[str, int] = defaultdict(int)
    running: defaultdict[str, int] = defaultdict(int)
    interrupted: defaultdict[str, int] = defaultdict(int)
    manifested: defaultdict[str, int] = defaultdict(int)
    for _run_dir, manifest in manifests:
        group = str(manifest.get("experiment_group", "engineering"))
        if planned_run_ids and str(manifest.get("run_id")) not in planned_run_ids:
            continue
        manifested[group] += 1
        if manifest.get("status") == "complete":
            completed[group] += 1
        elif manifest.get("status") == "failed":
            failed[group] += 1
        elif manifest.get("status") == "running":
            running[group] += 1
        elif manifest.get("status") in {"interrupted", "stopped"}:
            interrupted[group] += 1
    frozen_selection = pipeline_manifest.get("frozen_selection")
    selected_models = (
        frozen_selection.get("selected_models", []) if isinstance(frozen_selection, Mapping) else []
    )
    rq3_reports = pipeline_manifest.get("rq3", [])
    if isinstance(selected_models, list):
        expected["rq3"] = len(selected_models)
    if isinstance(rq3_reports, list):
        completed["rq3"] = sum(
            1
            for report in rq3_reports
            if isinstance(report, dict) and report.get("status") == "complete"
        )
        failed["rq3"] = sum(
            1
            for report in rq3_reports
            if isinstance(report, dict) and report.get("status") == "failed"
        )
    selected_run_ids = {
        str(item["run_id"])
        for item in selected_models
        if isinstance(item, Mapping) and isinstance(item.get("run_id"), str)
    }
    expected["final_test"] = len(selected_run_ids)
    completed["final_test"] = sum(
        1
        for _run_dir, manifest in manifests
        if manifest.get("run_id") in selected_run_ids and manifest.get("final_test_recorded")
    )
    rq1_expected = expected.get("rq1-main", 0)
    rq2_expected = expected.get("rq2-sensitivity", 0)
    final_expected = expected["final_test"]
    evidence_ready = {
        "rq1": bool(rq1_expected)
        and completed["rq1-main"] == rq1_expected
        and all(
            row["archive_training_mean"] is not None and row["validation_mean"] is not None
            for row in rq1_summary
        ),
        "rq2": bool(rq2_expected)
        and completed["rq2-sensitivity"] == rq2_expected
        and all(
            row["reactive_validation_mean"] is not None
            and row["proactive_validation_mean"] is not None
            for row in rq2_summary
        ),
        "rq3": rq3_primary_complete,
        "final": bool(final_expected)
        and completed["final_test"] == final_expected
        and bool(final_test_summary),
    }
    payload = {
        "model_version": model_version,
        "model_hash": selected_hash,
        "campaign_timestamp": campaign_timestamp,
        "generated_at_utc": _utc_now(),
        "progress": {
            group: {
                "completed": completed[group],
                "planned": expected_count,
                "failed": failed[group],
                "running": running[group],
                "interrupted": interrupted[group],
                "not_started": max(
                    0,
                    expected_count
                    - completed[group]
                    - failed[group]
                    - running[group]
                    - interrupted[group],
                ),
            }
            for group, expected_count in sorted(expected.items())
        },
        "figures": {
            label: str(path.relative_to(model_dir).as_posix()) for label, path in figures.items()
        },
        "tables": {
            label: str(path.relative_to(model_dir).as_posix()) for label, path in tables.items()
        },
        "rq1_summary": rq1_summary,
        "rq2_summary": rq2_summary,
        "rq3_summary": rq3_summary,
        "rq3_causal_summary": rq3_causal_summary,
        "rq3_traces": rq3_traces,
        "final_test_summary": final_test_summary,
        "final_test_locked": not bool(selected_run_ids),
        "evidence_ready": evidence_ready,
        "source": "immutable Parquet generation, episode, and behavioural-trace parts",
    }
    _atomic_text(
        model_dir / "scientific-results.json", json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    _atomic_text(
        model_dir / f"scientific-results__{export_suffix}.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


def export_behaviour_trace(
    path: Path,
    *,
    evaluation: Any,
    controller_type: str,
    source_id: str,
    intervention: str,
) -> Path:
    """Export one deliberately selected RQ3 trace as compact Parquet rows."""

    rows = _trace_rows(
        {"source_id": source_id},
        evaluation,
        controller_type=controller_type,
        intervention=intervention,
    )
    _atomic_parquet(path, rows)
    return path


def plot_behaviour_trace(
    trace_path: Path,
    *,
    output_path: Path | None = None,
    world_seed: int | None = None,
) -> Path:
    """Create one report-ready RQ3 trace plot directly from Parquet evidence."""

    rows = pq.read_table(trace_path).to_pylist()
    if world_seed is not None:
        rows = [row for row in rows if int(row["world_seed"]) == world_seed]
    if not rows:
        raise ValueError("Cannot plot an empty behavioural trace")
    control_time = np.asarray(
        [float(row.get("control_time_seconds", int(row["timestep"]) / CONTROL_HZ)) for row in rows],
        dtype=np.float64,
    )
    full_duration = float(control_time[-1] - control_time[0])
    detail_end = float(control_time[0]) + min(12.0, full_duration)
    detail_count = int(np.searchsorted(control_time, detail_end, side="right"))
    rows = rows[:detail_count]
    control_time = control_time[:detail_count]
    hidden_names = sorted(
        (key for key in rows[0] if key.startswith("hidden_")),
        key=lambda name: int(name.split("_")[1]),
    )
    panels = 4 if hidden_names else 3
    figure, axes = plt.subplots(
        panels,
        1,
        figsize=(7.2, 1.35 * panels),
        constrained_layout=True,
        sharex=True,
        gridspec_kw={
            "height_ratios": ([1.2, 1.25, 0.55, 1.15] if hidden_names else [1.2, 1.25, 0.55])
        },
    )
    sensor_axis = axes[0]
    sensor_matrix = np.asarray(
        [[float(row[f"sensor_{index}"]) for row in rows] for index in range(5)],
        dtype=np.float64,
    )
    time_start = float(control_time[0])
    time_end = (
        float(control_time[-1])
        if control_time[-1] > control_time[0]
        else time_start + 1 / CONTROL_HZ
    )
    sensor_image = sensor_axis.imshow(
        sensor_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        extent=(time_start, time_end, 4.5, -0.5),
    )
    sensor_axis.set_yticks(range(5), SENSOR_DISPLAY_NAMES, fontsize=7.2)
    figure.colorbar(sensor_image, ax=sensor_axis, label="Normalised value", pad=0.012)
    controller = str(rows[0].get("controller_type", "controller"))
    intervention = str(rows[0].get("intervention", "trace"))
    seed = int(rows[0]["world_seed"])
    figure.suptitle(
        "Selected episode detail — "
        f"{controller.title()}, {intervention.replace('_', ' ')}, world seed {seed}\n"
        f"first {control_time[-1] - control_time[0]:.1f} s of {full_duration:.1f} s",
        fontsize=10.5,
    )
    sensor_axis.set_title("A  Current-frame visual inputs", loc="left", fontsize=8.8)
    action_axis = axes[1]
    for index in range(3):
        values = [float(row[f"score_{index}"]) for row in rows]
        action_axis.plot(
            control_time,
            values,
            linewidth=1.0,
            label=ACTION_DISPLAY_NAMES[index],
        )
    action_axis.axhline(0.0, color="#777777", linewidth=0.7, linestyle=":")
    action_axis.set_title("B  Controller output scores", loc="left", fontsize=8.8)
    action_axis.set_ylabel("Output score")
    action_axis.legend(frameon=False, ncols=3, fontsize=7.2, loc="upper right")
    _style_journal_axis(action_axis)

    chosen_axis = axes[2]
    chosen_axis.step(
        control_time,
        [int(row["action"]) for row in rows],
        where="post",
        color="#202124",
        linewidth=0.9,
    )
    chosen_axis.set(
        title="C  Chosen held-key action",
        yticks=(0, 1, 2),
        yticklabels=ACTION_DISPLAY_NAMES,
        ylim=(-0.25, 2.25),
    )
    chosen_axis.title.set_fontsize(8.8)
    chosen_axis.title.set_position((0.0, 1.0))
    chosen_axis.title.set_ha("left")
    _style_journal_axis(chosen_axis)
    if hidden_names:
        hidden_axis = axes[3]
        hidden_matrix = np.asarray(
            [[float(row[name]) for row in rows] for name in hidden_names],
            dtype=np.float64,
        )
        hidden_image = hidden_axis.imshow(
            hidden_matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            extent=(time_start, time_end, len(hidden_names) - 0.5, -0.5),
        )
        hidden_axis.set_yticks(
            range(len(hidden_names)),
            [f"State {index + 1}" for index in range(len(hidden_names))],
            fontsize=7.2,
        )
        hidden_axis.set_title("D  Proactive CTRNN state", loc="left", fontsize=8.8)
        hidden_axis.set_xlabel("Control time (s)")
        figure.colorbar(hidden_image, ax=hidden_axis, label="State", pad=0.012)
    else:
        chosen_axis.set_xlabel("Control time (s)")
    output = output_path or trace_path.with_suffix(".png")
    _atomic_figure(figure, output)
    plt.close(figure)
    return output
