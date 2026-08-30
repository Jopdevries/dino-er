"""Run the complete, resumable AE4350 final experiment plan.

This is the single project-specific runner. It parallelises only independent
optimisation runs, then performs frozen selection, the locked final test, and
RQ3 interventions in scientific dependency order. Raw evidence is Parquet.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from collections import defaultdict
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from dino_er.evaluation import EpisodeConfig, PopulationEvaluationSession
from dino_er.evolution import (
    MAX_POPULATION_SIZE,
    EvolutionConfig,
    EvolutionRunner,
    load_candidate_model,
)
from dino_er.scientific import (
    ParquetResultStore,
    aggregate_study_results,
    inspect_run,
    model_hash,
    model_result_directory,
    read_parquet_parts,
    result_location,
)

RQ1 = "rq1-main"
RQ2 = "rq2-sensitivity"
FINAL_SELECTION_FILE = "frozen-selection.json"
PIPELINE_MANIFEST_FILE = "results-manifest.json"
MAX_SAFE_WORKERS = 6


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _integer_tuple(values: Any, label: str) -> tuple[int, ...]:
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, int) for value in values)
    ):
        raise ValueError(f"{label} must be a non-empty integer list")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates")
    return tuple(values)


def _parameter_cells(
    section: Mapping[str, Any],
    identifier: str,
) -> tuple[tuple[int, int, float], ...]:
    raw_cells = section.get("cells")
    if raw_cells is not None:
        if any(key in section for key in ("parent_counts", "offspring_counts", "mutation_scales")):
            raise ValueError(f"{identifier}.cells cannot be combined with count/scale grids")
        if not isinstance(raw_cells, list) or not raw_cells:
            raise ValueError(f"{identifier}.cells must be a non-empty list")
        cells: list[tuple[int, int, float]] = []
        for index, raw_cell in enumerate(raw_cells):
            if not isinstance(raw_cell, Mapping):
                raise ValueError(f"{identifier}.cells[{index}] must be an object")
            parent_count = raw_cell.get("parent_count")
            mutation_scale = raw_cell.get("mutation_scale")
            if (
                not isinstance(parent_count, int)
                or isinstance(parent_count, bool)
                or not isinstance(mutation_scale, (int, float))
                or isinstance(mutation_scale, bool)
            ):
                raise ValueError(
                    f"{identifier}.cells[{index}] requires numeric parent_count and mutation_scale"
                )
            offspring_count = raw_cell.get(
                "offspring_count",
                MAX_POPULATION_SIZE - parent_count,
            )
            if not isinstance(offspring_count, int) or isinstance(offspring_count, bool):
                raise ValueError(f"{identifier}.cells[{index}].offspring_count must be an integer")
            if (
                parent_count <= 0
                or offspring_count <= 0
                or parent_count + offspring_count != MAX_POPULATION_SIZE
                or float(mutation_scale) <= 0
            ):
                raise ValueError(
                    f"{identifier}.cells[{index}] requires positive mu, lambda, sigma "
                    f"and mu + lambda = {MAX_POPULATION_SIZE}"
                )
            cells.append((parent_count, offspring_count, float(mutation_scale)))
        if len(set(cells)) != len(cells):
            raise ValueError(f"{identifier}.cells must not contain duplicates")
        return tuple(cells)

    parent_counts = _integer_tuple(section.get("parent_counts"), f"{identifier}.parent_counts")
    raw_offspring = section.get("offspring_counts")
    explicit_offspring = (
        _integer_tuple(raw_offspring, f"{identifier}.offspring_counts")
        if raw_offspring is not None
        else ()
    )
    raw_scales = section.get("mutation_scales")
    if not isinstance(raw_scales, list) or not raw_scales:
        raise ValueError(f"{identifier}.mutation_scales must be a non-empty numeric list")
    scales = tuple(float(value) for value in raw_scales)
    if any(value <= 0 for value in scales):
        raise ValueError(f"{identifier}.mutation_scales values must be positive")
    cells = []
    for parent_count in parent_counts:
        offspring_counts = (
            explicit_offspring if explicit_offspring else (MAX_POPULATION_SIZE - parent_count,)
        )
        for offspring_count in offspring_counts:
            if parent_count + offspring_count != MAX_POPULATION_SIZE:
                raise ValueError(f"{identifier} requires mu + lambda = {MAX_POPULATION_SIZE}")
            cells.extend(
                (parent_count, offspring_count, mutation_scale) for mutation_scale in scales
            )
    return tuple(cells)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/experiment-plan.json"))
    parser.add_argument(
        "--section",
        action="append",
        help="Run only RQ1 and/or RQ2 training sections; omit for the complete pipeline.",
    )
    parser.add_argument("--accelerator", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--workers",
        default="auto",
        help=(
            "Independent-run workers: auto or a positive integer. Each worker owns "
            "one independent browser, RNG, checkpoint, and result directory."
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Operational retries per failed independent run (default: 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the phase plan and persisted run states without launching Chrome.",
    )
    parser.add_argument(
        "--rq3-only",
        action="store_true",
        help=(
            "Resume only the RQ3 interventions for the frozen models in an existing "
            "campaign; requires --model-hash and never starts optimisation runs."
        ),
    )
    parser.add_argument(
        "--model-hash",
        help="Exact 12-character model hash of the existing campaign used by --rq3-only.",
    )
    return parser.parse_args()


def _planned_configs(
    plan: Mapping[str, Any], arguments: argparse.Namespace
) -> list[EvolutionConfig]:
    common = plan.get("common")
    sections = plan.get("experiments")
    if not isinstance(common, Mapping) or not isinstance(sections, list):
        raise ValueError("Plan requires object fields 'common' and 'experiments'")
    model_version = int(plan.get("model_version", 1))
    results_root = Path(str(plan.get("results_root", "results")))
    output_root = Path(str(plan.get("output_root", "artifacts/experiments")))
    training_seeds = _integer_tuple(common.get("training_seeds"), "common.training_seeds")
    validation_seeds = _integer_tuple(common.get("validation_seeds"), "common.validation_seeds")
    final_test_seeds = _integer_tuple(common.get("final_test_seeds"), "common.final_test_seeds")
    campaign_timestamp = common.get("campaign_timestamp")
    if not isinstance(campaign_timestamp, str) or not campaign_timestamp:
        raise ValueError("common.campaign_timestamp must be a non-empty string")
    partitions = (set(training_seeds), set(validation_seeds), set(final_test_seeds))
    if any(
        left & right for index, left in enumerate(partitions) for right in partitions[index + 1 :]
    ):
        raise ValueError("Training, validation, and final-test seed partitions must be disjoint")
    selected_sections = set(arguments.section or ())
    configs: list[EvolutionConfig] = []
    known_sections: set[str] = set()
    for section in sections:
        if not isinstance(section, Mapping):
            raise ValueError("Every experiment section must be an object")
        identifier = section.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Every experiment section requires a non-empty id")
        known_sections.add(identifier)
        if selected_sections and identifier not in selected_sections:
            continue
        controllers = section.get("controllers")
        if (
            not isinstance(controllers, list)
            or not controllers
            or set(controllers) - {"reactive", "proactive"}
        ):
            raise ValueError(f"{identifier}.controllers must contain reactive and/or proactive")
        seeds = _integer_tuple(section.get("optimizer_seeds"), f"{identifier}.optimizer_seeds")
        parameter_cells = _parameter_cells(section, identifier)
        adaptive_sigma = section.get("adaptive_sigma")
        if not isinstance(adaptive_sigma, bool):
            raise ValueError(f"{identifier}.adaptive_sigma must be boolean")
        generations = section.get("generations")
        max_steps = section.get("max_steps")
        if not isinstance(generations, int) or not isinstance(max_steps, int):
            raise ValueError(f"{identifier} requires integer generations and max_steps")
        for controller in controllers:
            for optimizer_seed in seeds:
                for parent_count, offspring_count, mutation_scale in parameter_cells:
                    run_id = (
                        f"{identifier}-{controller}-seed-{optimizer_seed}-"
                        f"mu-{parent_count}-lambda-{offspring_count}-"
                        f"sigma-{mutation_scale:g}"
                    )
                    config = EvolutionConfig(
                        controller_type=controller,
                        training_seeds=training_seeds,
                        validation_seeds=validation_seeds,
                        final_test_seeds=final_test_seeds,
                        generations=generations,
                        parent_count=parent_count,
                        offspring_count=offspring_count,
                        mutation_scale=mutation_scale,
                        evolution_seed=optimizer_seed,
                        max_steps=max_steps,
                        run_id=run_id,
                        output_dir=output_root,
                        adaptive_sigma=adaptive_sigma,
                        results_root=results_root,
                        model_version=model_version,
                        accelerator=arguments.accelerator,
                        study_id=identifier,
                        scientific_study_dir=output_root / identifier,
                        campaign_timestamp=campaign_timestamp,
                        result_model_hash=getattr(arguments, "model_hash", None),
                    )
                    location = result_location(config)
                    configs.append(
                        replace(
                            config,
                            output_dir=(
                                output_root
                                / f"model_v{model_version:03d}_{location.model_hash}"
                                / location.experiment_hash
                                / f"{campaign_timestamp}__{run_id}"
                            ),
                            scientific_study_dir=(
                                output_root
                                / f"model_v{model_version:03d}_{location.model_hash}"
                                / campaign_timestamp
                                / identifier
                            ),
                        )
                    )
    if selected_sections - known_sections:
        raise ValueError("Requested --section does not exist in the plan")
    return configs


def _run_metadata(config: EvolutionConfig) -> dict[str, Any]:
    return {
        "experiment_group": config.study_id,
        "controller_type": config.controller_type,
        "run_id": config.run_id,
        "optimizer_seed": config.evolution_seed,
        "parent_count": config.parent_count,
        "offspring_count": config.offspring_count,
        "mutation_scale": config.mutation_scale,
        "adaptive_sigma": config.adaptive_sigma,
        "generations": config.generations,
        "max_steps": config.max_steps,
        "campaign_timestamp": config.campaign_timestamp,
    }


def _worker_run(config: EvolutionConfig, *, require_final_test: bool) -> dict[str, Any]:
    state = inspect_run(config, require_final_test=require_final_test)
    if state.status == "complete":
        return {"run_id": config.run_id, "status": "skipped", "detail": state.message}
    checkpoint = config.output_dir / f"{config.controller_type}-checkpoint.pkl"
    if state.status == "invalid":
        return {"run_id": config.run_id, "status": "failed", "detail": state.message}
    if state.completed_generations and not checkpoint.is_file():
        return {
            "run_id": config.run_id,
            "status": "failed",
            "detail": "Parquet generations exist but the matching checkpoint is missing",
        }
    try:
        EvolutionRunner(
            config,
            checkpoint_path=checkpoint,
            progress=lambda message: print(f"[{config.run_id}] {message}", flush=True),
        ).run()
        state = inspect_run(config, require_final_test=require_final_test)
        return {"run_id": config.run_id, "status": state.status, "detail": state.message}
    except Exception as error:
        ParquetResultStore(config).mark_failed(error)
        return {
            "run_id": config.run_id,
            "status": "failed",
            "detail": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=8),
        }


def _resolve_workers(requested: str) -> int:
    if requested == "auto":
        # The dominant work is browser simulation and pixel transport, not the
        # tiny controller matrix multiply. Run-level RNG and files are isolated,
        # so independent workers cannot change one another's results.
        return min(MAX_SAFE_WORKERS, max(1, (os.cpu_count() or 1) // 2))
    try:
        workers = int(requested)
    except ValueError as error:
        raise ValueError("--workers must be 'auto' or a positive integer") from error
    if workers < 1:
        raise ValueError("--workers must be positive")
    if workers > MAX_SAFE_WORKERS:
        raise ValueError(
            f"--workers cannot exceed {MAX_SAFE_WORKERS}: every worker owns a browser "
            "and a 100-candidate pixel pipeline, and higher concurrency exhausted "
            "memory during the measured V002 campaign"
        )
    return workers


def _run_phase(
    label: str,
    configs: Iterable[EvolutionConfig],
    *,
    require_final_test: bool,
    workers: int,
    retries: int,
    worker: Any = _worker_run,
) -> list[dict[str, Any]]:
    pending = list(configs)
    reports: list[dict[str, Any]] = []
    attempts: dict[str, int] = defaultdict(int)
    while pending:
        print(f"[pipeline] {label}: {len(pending)} run(s), workers={workers}", flush=True)
        next_pending: list[EvolutionConfig] = []
        if workers == 1:
            completed = [
                worker(config, require_final_test=require_final_test) for config in pending
            ]
        else:
            completed = []
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(worker, config, require_final_test=require_final_test): config
                    for config in pending
                }
                for future in as_completed(futures):
                    config = futures[future]
                    try:
                        completed.append(future.result())
                    except Exception as error:
                        completed.append(
                            {
                                "run_id": config.run_id,
                                "status": "failed",
                                "detail": f"worker process: {type(error).__name__}: {error}",
                            }
                        )
        by_id = {config.run_id: config for config in pending}
        for report in completed:
            reports.append(report)
            print(
                f"[pipeline] {label}: {report['status']} {report['run_id']}"
                f" ({report.get('detail', '')})",
                flush=True,
            )
            run_id = str(report["run_id"])
            if report.get("status") == "failed" and any(
                token in str(report.get("detail", ""))
                for token in ("VisualTransportError", "Timeout", "OSError", "MemoryError")
            ):
                attempts[run_id] += 1
                if attempts[run_id] <= retries:
                    next_pending.append(by_id[run_id])
        pending = next_pending
    return reports


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _freeze_selection(configs: list[EvolutionConfig]) -> dict[str, Any] | None:
    if any(inspect_run(config).status != "complete" for config in configs):
        return None
    rq1_configs = [config for config in configs if config.study_id == RQ1]
    selected: list[dict[str, Any]] = []
    for controller in ("reactive", "proactive"):
        valid = []
        for config in rq1_configs:
            if config.controller_type != controller:
                continue
            values = [
                float(row["survival_steps"])
                for row in read_parquet_parts(result_location(config).run_dir / "episode-results")
                if row.get("phase") == "validation"
            ]
            if values:
                valid.append((config, float(np.mean(values))))
        if not valid:
            return None
        median = float(np.median([float(value) for _config, value in valid]))
        config, validation = min(
            valid,
            key=lambda item: (abs(float(item[1]) - median), item[0].run_id),
        )
        model_path = config.output_dir / "best-candidate.npz"
        if not model_path.is_file():
            return None
        selected.append(
            {
                **_run_metadata(config),
                "validation_mean": validation,
                "selection": "median held-out validation survival within architecture",
                "model_path": str(model_path.resolve()),
            }
        )
    location = result_location(rq1_configs[0])
    payload = {
        "model_version": location.model_version,
        "model_hash": location.model_hash,
        "campaign_timestamp": configs[0].campaign_timestamp,
        "updated_at_utc": _utc_now(),
        "selection_uses": "training and validation Parquet only",
        "locked_final_test_used": False,
        "selected_models": selected,
    }
    _atomic_json(
        model_result_directory(
            location.root,
            location.model_version,
            location.model_hash,
        )
        / FINAL_SELECTION_FILE,
        payload,
    )
    return payload


def _selected_final_configs(
    configs: Iterable[EvolutionConfig],
    frozen: Mapping[str, Any] | None,
) -> list[EvolutionConfig]:
    """Return only validation-frozen RQ1 archives for the locked test.

    The final seeds must not be spent on the other independent optimisation
    replicates.  Their role is to support the validation-only model-selection
    distribution, not to create a second test sample.
    """

    if not isinstance(frozen, Mapping):
        return []
    raw_selected = frozen.get("selected_models")
    if not isinstance(raw_selected, list):
        return []
    selected_ids = {
        str(item["run_id"])
        for item in raw_selected
        if isinstance(item, Mapping) and isinstance(item.get("run_id"), str)
    }
    if not selected_ids:
        return []
    selected = [
        replace(config, evaluate_final_test=True)
        for config in configs
        if config.study_id == RQ1 and config.run_id in selected_ids
    ]
    if len(selected) != len(selected_ids):
        raise ValueError("Frozen validation selection references an unknown RQ1 run")
    return selected


def _rq3_interventions(controller_type: str) -> list[tuple[str, dict[str, Any]]]:
    interventions: list[tuple[str, dict[str, Any]]] = [
        ("normal", {}),
        ("constant_first_frame", {"sensory_intervention": "constant_first_frame"}),
        *[
            (
                f"ablate_sensor_{index + 1}",
                {"sensory_intervention": "ablate_sensor", "ablation_sensor_index": index},
            )
            for index in range(5)
        ],
    ]
    if controller_type == "proactive":
        interventions.append(("reset_each_step", {"state_intervention": "reset_each_step"}))
    return interventions


def _assert_saved_model_compatibility(
    config: EvolutionConfig,
    evaluation: Any,
) -> None:
    """Refuse to mix changed runtime behaviour into an older campaign hash."""

    target_hash = config.result_model_hash
    current_hash = model_hash()
    if target_hash is None or target_hash == current_hash:
        return
    location = result_location(config)
    validation_path = location.run_dir / "episode-results" / "validation.parquet"
    trace_paths = sorted(
        (location.run_dir / "behavioural-traces").glob("rq3-validation-best-generation-*.parquet")
    )
    if not validation_path.is_file() or not trace_paths:
        raise RuntimeError(
            "Cannot verify the saved model against its original validation evidence "
            f"before resuming model hash {target_hash}."
        )

    expected_rows = {
        int(row["world_seed"]): row
        for row in pq.read_table(validation_path).to_pylist()
        if row.get("genome_id") == evaluation.genome_hash
    }
    observed_episodes = {int(episode.seed): episode for episode in evaluation.episodes}
    expected_seeds = set(config.validation_seeds)
    if set(expected_rows) != expected_seeds or set(observed_episodes) != expected_seeds:
        raise RuntimeError("Saved and resumed validation worlds do not match")

    for seed in sorted(expected_seeds):
        expected = expected_rows[seed]
        observed = observed_episodes[seed]
        observed_summary = {
            "survival_steps": int(observed.steps),
            "terminated": bool(observed.terminated),
            "truncated": bool(observed.truncated),
            "no_action_steps": int(observed.action_counts[0]),
            "jump_steps": int(observed.action_counts[1]),
            "duck_steps": int(observed.action_counts[2]),
            "action_switches": int(observed.action_switches),
            "pterodactyl_visible_steps": int(observed.pterodactyl_visible_steps),
            "duck_on_pterodactyl_steps": int(observed.duck_on_pterodactyl_steps),
            "cactus_visible_steps": int(observed.cactus_visible_steps),
            "duck_on_cactus_steps": int(observed.duck_on_cactus_steps),
        }
        if any(expected.get(field) != value for field, value in observed_summary.items()):
            raise RuntimeError(
                "Current runtime no longer reproduces the saved validation summary "
                f"for world {seed}; refusing to label new data as model hash {target_hash}."
            )

    saved_actions: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in pq.read_table(
        trace_paths[-1], columns=["world_seed", "timestep", "action"]
    ).to_pylist():
        saved_actions[int(row["world_seed"])].append((int(row["timestep"]), int(row["action"])))
    for seed, episode in observed_episodes.items():
        expected_actions = tuple(action for _step, action in sorted(saved_actions[seed]))
        if expected_actions != tuple(int(action) for action in episode.actions):
            raise RuntimeError(
                "Current runtime no longer reproduces the saved action trace "
                f"for world {seed}; refusing to label new data as model hash {target_hash}."
            )


def _load_frozen_selection(configs: list[EvolutionConfig]) -> dict[str, Any]:
    location = result_location(configs[0])
    model_dir = model_result_directory(
        location.root,
        location.model_version,
        location.model_hash,
    )
    path = model_dir / FINAL_SELECTION_FILE
    try:
        frozen = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read existing frozen selection at {path}: {error}") from error
    if not isinstance(frozen, dict):
        raise ValueError("Frozen selection must contain one JSON object")
    if (
        frozen.get("model_version") != location.model_version
        or frozen.get("model_hash") != location.model_hash
        or frozen.get("campaign_timestamp") != configs[0].campaign_timestamp
    ):
        raise ValueError("Frozen selection identity does not match the requested campaign")

    selected = _selected_final_configs(configs, frozen)
    if len(selected) != 2 or {config.controller_type for config in selected} != {
        "reactive",
        "proactive",
    }:
        raise ValueError("RQ3 requires exactly one frozen reactive and proactive RQ1 model")
    selected_by_id = {config.run_id: config for config in selected}
    for item in frozen.get("selected_models", []):
        if not isinstance(item, Mapping) or str(item.get("run_id")) not in selected_by_id:
            raise ValueError("Frozen selection contains an invalid selected model")
        config = selected_by_id[str(item["run_id"])]
        expected_model = (config.output_dir / "best-candidate.npz").resolve()
        stored_model = Path(str(item.get("model_path", ""))).resolve()
        if stored_model != expected_model or not expected_model.is_file():
            raise ValueError(f"Frozen model archive does not match the plan: {expected_model}")
        state = inspect_run(config, require_final_test=True)
        if state.status != "complete":
            raise ValueError(
                f"Selected run is not final-test complete: {config.run_id}: {state.message}"
            )
    return frozen


def _worker_rq3(config: EvolutionConfig, *, require_final_test: bool = False) -> dict[str, Any]:
    del require_final_test
    model_path = config.output_dir / "best-candidate.npz"
    store: ParquetResultStore | None = None
    try:
        genome, spec, _metadata = load_candidate_model(model_path)
        location = result_location(config)
        base = EpisodeConfig(
            seeds=config.validation_seeds,
            max_steps=config.max_steps,
            record_trace=True,
            accelerator=config.accelerator,
            study_id=config.study_id,
        )
        session = PopulationEvaluationSession(spec, base, population_size=1)
        try:
            for label, settings in _rq3_interventions(config.controller_type):
                trace_path = location.run_dir / "behavioural-traces" / f"rq3-{label}.parquet"
                episode_path = (
                    location.run_dir / "episode-results" / f"intervention-{label}.parquet"
                )
                if trace_path.is_file() and episode_path.is_file():
                    continue
                result = session.evaluate(
                    [genome],
                    generation=config.generations,
                    mutation_scale=config.mutation_scale,
                    episode_config=replace(base, **settings),
                )[0]
                if label == "normal":
                    _assert_saved_model_compatibility(config, result)
                if store is None:
                    store = ParquetResultStore(config)
                store.record_intervention(
                    intervention=label,
                    generation=config.generations,
                    evaluation=result,
                )
                store.record_trace(
                    trace_id=f"rq3-{label}",
                    evaluation=result,
                    controller_type=config.controller_type,
                    intervention=label,
                )
        finally:
            session.close()
        if store is None:
            store = ParquetResultStore(config)
        store.mark_complete("complete")
        return {"run_id": config.run_id, "status": "complete", "detail": "RQ3 stored"}
    except Exception as error:
        if store is not None:
            store.mark_complete("complete")
        return {
            "run_id": config.run_id,
            "status": "failed",
            "detail": f"{type(error).__name__}: {error}",
        }


def _pipeline_manifest(
    configs: list[EvolutionConfig],
    *,
    frozen: Mapping[str, Any] | None,
    rq3_reports: Iterable[Mapping[str, Any]],
) -> Path:
    location = result_location(configs[0])
    statuses: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for config in configs:
        statuses[str(config.study_id)][inspect_run(config).status] += 1
    final_configs = _selected_final_configs(configs, frozen)
    final_complete = bool(final_configs) and all(
        inspect_run(config, require_final_test=True).status == "complete"
        for config in final_configs
    )
    payload = {
        "model_version": location.model_version,
        "model_hash": location.model_hash,
        "campaign_timestamp": configs[0].campaign_timestamp,
        "updated_at_utc": _utc_now(),
        "planned_runs": [_run_metadata(config) for config in configs],
        "run_statuses": {key: dict(value) for key, value in statuses.items()},
        "frozen_selection": frozen,
        "rq3": list(rq3_reports),
        "final_test": "complete" if final_complete else "LOCKED FINAL TEST",
        "source": "Parquet parts and validated checkpoint state",
    }
    path = (
        model_result_directory(location.root, location.model_version, location.model_hash)
        / PIPELINE_MANIFEST_FILE
    )
    _atomic_json(path, payload)
    return path


def _rq3_file_state(config: EvolutionConfig) -> dict[str, list[str]]:
    run_dir = result_location(config).run_dir
    complete: list[str] = []
    missing: list[str] = []
    partial: list[str] = []
    for label, _settings in _rq3_interventions(config.controller_type):
        episode_exists = (run_dir / "episode-results" / f"intervention-{label}.parquet").is_file()
        trace_exists = (run_dir / "behavioural-traces" / f"rq3-{label}.parquet").is_file()
        if episode_exists and trace_exists:
            complete.append(label)
        elif episode_exists or trace_exists:
            partial.append(label)
        else:
            missing.append(label)
    return {"complete": complete, "partial": partial, "missing": missing}


def _resume_rq3_only(
    configs: list[EvolutionConfig],
    *,
    workers: int,
    retries: int,
    dry_run: bool,
) -> int:
    frozen = _load_frozen_selection(configs)
    selected = _selected_final_configs(configs, frozen)
    if dry_run:
        for config in selected:
            print(
                json.dumps(
                    {
                        **_run_metadata(config),
                        "results": str(result_location(config).run_dir),
                        "rq3": _rq3_file_state(config),
                    },
                    sort_keys=True,
                )
            )
        return 0

    location = result_location(selected[0])
    print(
        "[pipeline] RQ3-only resume: frozen reactive and proactive archives; "
        "no optimisation runs will be started.\n"
        f"[pipeline] Target model: {location.model_hash} (runtime hash: {model_hash()})",
        flush=True,
    )
    reports = _run_phase(
        "RQ3-only / causal interventions",
        selected,
        require_final_test=False,
        workers=workers,
        retries=retries,
        worker=_worker_rq3,
    )
    manifest_path = _pipeline_manifest(configs, frozen=frozen, rq3_reports=reports)
    index = aggregate_study_results(
        selected[0].results_root,
        model_version=selected[0].model_version,
        planned_runs=[_run_metadata(config) for config in configs],
        selected_model_hash=location.model_hash,
    )
    print(
        f"[pipeline] RQ3 aggregation complete: {manifest_path.resolve()}",
        flush=True,
    )
    reports_complete = len(reports) == len(selected) and all(
        report.get("status") == "complete" for report in reports
    )
    return 0 if reports_complete and bool(index.get("evidence_ready", {}).get("rq3")) else 1


def main() -> int:
    arguments = parse_args()
    if arguments.retries < 0:
        raise SystemExit("--retries cannot be negative")
    if arguments.rq3_only:
        if not arguments.model_hash:
            raise SystemExit("--rq3-only requires --model-hash")
        if arguments.section:
            raise SystemExit("--rq3-only cannot be combined with --section")
    elif arguments.model_hash:
        raise SystemExit("--model-hash is accepted only with --rq3-only")
    try:
        workers = _resolve_workers(arguments.workers)
        plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
        if not isinstance(plan, Mapping):
            raise ValueError("Experiment plan must contain one JSON object")
        configs = _planned_configs(plan, arguments)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Invalid experiment plan: {error}") from error
    if not configs:
        raise SystemExit("No planned experiments selected")
    if arguments.rq3_only:
        try:
            return _resume_rq3_only(
                configs,
                workers=workers,
                retries=arguments.retries,
                dry_run=arguments.dry_run,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"Cannot resume RQ3: {error}") from error
    if arguments.dry_run:
        for config in configs:
            state = inspect_run(config)
            print(
                json.dumps(
                    {
                        **_run_metadata(config),
                        "state": state.status,
                        "detail": state.message,
                        "results": str(result_location(config).run_dir),
                    },
                    sort_keys=True,
                )
            )
        return 0

    # Persist the plan before workers start so the independent results viewer
    # can report completed/planned counts during an overnight run.
    live_manifest = _pipeline_manifest(configs, frozen=None, rq3_reports=[])
    checkpoint_model_root = configs[0].output_dir.parents[1].resolve()
    results_model_root = model_result_directory(
        configs[0].results_root,
        configs[0].model_version,
        result_location(configs[0]).model_hash,
    ).resolve()
    print(
        f"[pipeline] Campaign: {configs[0].campaign_timestamp}\n"
        f"[pipeline] Restartable checkpoints/models: {checkpoint_model_root}\n"
        f"[pipeline] Scientific Parquet/figures: {results_model_root}\n"
        f"[pipeline] Live pipeline manifest: {live_manifest.resolve()}",
        flush=True,
    )

    _run_phase(
        "Phases A-B / independent RQ1 and RQ2 evolution + validation",
        configs,
        require_final_test=False,
        workers=workers,
        retries=arguments.retries,
    )
    aggregate_study_results(
        configs[0].results_root,
        model_version=configs[0].model_version,
        planned_runs=[_run_metadata(config) for config in configs],
        selected_model_hash=result_location(configs[0]).model_hash,
    )
    incomplete_core = [config for config in configs if inspect_run(config).status != "complete"]
    if incomplete_core:
        _pipeline_manifest(configs, frozen=None, rq3_reports=[])
        print(
            "[pipeline] Dependent phases blocked: "
            f"{len(incomplete_core)} core run(s) are not complete. "
            "No model selection, locked test, or RQ3 evaluation was started.",
            flush=True,
        )
        return 1

    full_pipeline = not arguments.section
    frozen = _freeze_selection(configs) if full_pipeline else None
    rq3_reports: list[dict[str, Any]] = []
    if frozen is None:
        print(
            "[pipeline] Phase C is waiting: every planned RQ1/RQ2 run must complete "
            "before final-test/RQ3 work can start.",
            flush=True,
        )
    elif full_pipeline:
        print("[pipeline] Phase C complete: frozen validation-only selection written.")
        final_configs = _selected_final_configs(configs, frozen)
        _run_phase(
            "Phase D / locked final test",
            final_configs,
            require_final_test=True,
            workers=workers,
            retries=arguments.retries,
        )
        if all(
            inspect_run(config, require_final_test=True).status == "complete"
            for config in final_configs
        ):
            selected_ids = {str(item["run_id"]) for item in frozen["selected_models"]}
            selected_configs = [config for config in configs if config.run_id in selected_ids]
            rq3_reports = _run_phase(
                "Phase E / RQ3 interventions",
                selected_configs,
                require_final_test=False,
                workers=workers,
                retries=arguments.retries,
                worker=_worker_rq3,
            )
        else:
            print("[pipeline] Phase E is waiting for every locked final-test run.")
    else:
        print("[pipeline] Requested sections complete; final test remains LOCKED.")

    first = configs[0]
    aggregate_study_results(
        first.results_root,
        model_version=first.model_version,
        planned_runs=[_run_metadata(config) for config in configs],
        selected_model_hash=result_location(first).model_hash,
    )
    manifest_path = _pipeline_manifest(configs, frozen=frozen, rq3_reports=rq3_reports)
    print(f"[pipeline] Phase F complete: results index and manifest at {manifest_path}")
    core_complete = all(inspect_run(config).status == "complete" for config in configs)
    dependent_complete = not full_pipeline or (
        frozen is not None
        and all(
            inspect_run(config, require_final_test=True).status == "complete"
            for config in _selected_final_configs(configs, frozen)
        )
        and all(report.get("status") == "complete" for report in rq3_reports)
    )
    return 0 if core_complete and dependent_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
