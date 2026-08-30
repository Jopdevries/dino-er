"""Run the reactive or proactive elitist (mu + lambda)-ES experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from dino_er.browser import VisualTransportError
from dino_er.controllers import Accelerator, ControllerType, default_controller_spec
from dino_er.evaluation import (
    PopulationEvaluationCancelled,
    PopulationRunRestartRequested,
)
from dino_er.evolution import (
    EvolutionConfig,
    EvolutionRunner,
    load_best_from_checkpoint,
)
from dino_er.perception import SENSORY_NAMES
from dino_er.scientific import export_parquet_figures, result_location


def _seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be a non-empty unique set")
    return seeds


def _local_campaign_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_UTC%z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("reactive", "proactive"), required=True)
    parser.add_argument("--parents", type=int, default=20, help="Number of retained parents (mu).")
    parser.add_argument(
        "--offspring",
        type=int,
        default=80,
        help="Number of Gaussian-mutated offspring (lambda).",
    )
    parser.add_argument(
        "--training-seeds",
        "--seeds",
        type=_seeds,
        default=(101, 202, 707, 808, 909, 1001, 1102, 1203, 1304, 1405),
    )
    generation_mode = parser.add_mutually_exclusive_group()
    generation_mode.add_argument("--generations", type=int, default=2)
    generation_mode.add_argument(
        "--continuous",
        action="store_true",
        help="Continue complete generations until stopped from the GUI or terminal.",
    )
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--mutation-scale", type=float, default=0.3)
    sigma_mode = parser.add_mutually_exclusive_group()
    sigma_mode.add_argument(
        "--adaptive-sigma",
        dest="adaptive_sigma",
        action="store_true",
        help="Adapt sigma between generations using the one-fifth success rule (default).",
    )
    sigma_mode.add_argument(
        "--fixed-sigma",
        dest="adaptive_sigma",
        action="store_false",
        help="Keep --mutation-scale fixed for every generation.",
    )
    parser.set_defaults(adaptive_sigma=True)
    parser.add_argument("--evolution-seed", type=int, default=7)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--render", choices=("none", "human"), default="none")
    parser.add_argument(
        "--simulation-speed",
        type=float,
        default=1.0,
        help=(
            "Visible execution multiplier (0.25-100); changes pacing/redraw only, "
            "never fixed-step dynamics. Headless runs are already unthrottled."
        ),
    )
    parser.add_argument(
        "--exit-after-run",
        action="store_true",
        help="Close a visible run when its finite budget is complete.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--model-version",
        type=int,
        default=2,
        help="Manually increment after a scientifically meaningful model change.",
    )
    parser.add_argument("--resume", type=Path, help="Trusted checkpoint from the same run.")
    parser.add_argument(
        "--campaign-timestamp",
        help="Filesystem-safe campaign time; generated automatically for a fresh run.",
    )
    parser.add_argument("--scientific-study")
    parser.add_argument("--study-root", type=Path, default=Path("artifacts/studies"))
    parser.add_argument("--validation-seeds", type=_seeds, default=(303, 404, 1303, 1404, 2303))
    parser.add_argument(
        "--final-test-seeds",
        type=_seeds,
        default=(
            505,
            606,
            1505,
            1606,
            2505,
            2606,
            3505,
            3606,
            4505,
            4606,
            5505,
            5606,
            6505,
            6606,
            7505,
            7606,
            8505,
            8606,
            9505,
            9606,
        ),
    )
    parser.add_argument(
        "--evaluate-final-test",
        action="store_true",
        help="Explicitly unlock one final-test evaluation after the scientific run.",
    )
    return parser.parse_args()


def _available_output_dir(requested: Path) -> Path:
    if not requested.exists():
        return requested
    for number in range(2, 10_000):
        candidate = requested.with_name(f"{requested.name}-{number:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"No available output directory beside {requested}")


def _output_for(arguments: argparse.Namespace, controller: ControllerType) -> Path:
    if arguments.output is not None:
        return arguments.output
    if arguments.scientific_study:
        return (
            arguments.study_root
            / f"model_v{arguments.model_version:03d}"
            / str(arguments.campaign_timestamp)
            / arguments.scientific_study
            / controller
            / f"evolution-seed-{arguments.evolution_seed:06d}"
        )
    return (
        Path("artifacts/runs")
        / f"model_v{arguments.model_version:03d}"
        / str(arguments.campaign_timestamp)
        / controller
    )


def _config(
    arguments: argparse.Namespace,
    *,
    controller: ControllerType,
    accelerator: Accelerator,
    output: Path,
) -> EvolutionConfig:
    study = arguments.scientific_study
    return EvolutionConfig(
        controller_type=controller,
        training_seeds=arguments.training_seeds,
        generations=arguments.generations,
        parent_count=arguments.parents,
        offspring_count=arguments.offspring,
        mutation_scale=arguments.mutation_scale,
        evolution_seed=arguments.evolution_seed,
        max_steps=arguments.max_steps,
        run_id=(
            f"{study}-{controller}-seed-{arguments.evolution_seed}"
            if study
            else f"{controller}-engineering-seed-{arguments.evolution_seed}"
        ),
        output_dir=output,
        adaptive_sigma=arguments.adaptive_sigma,
        results_root=arguments.results_root,
        model_version=arguments.model_version,
        simulation_speed=arguments.simulation_speed,
        render_mode="human" if arguments.render == "human" else None,
        continuous=arguments.continuous,
        accelerator=accelerator,
        study_id=study,
        scientific_study_dir=arguments.study_root / study if study else None,
        validation_seeds=arguments.validation_seeds if study else (),
        final_test_seeds=arguments.final_test_seeds if study else (),
        evaluate_final_test=arguments.evaluate_final_test,
        campaign_timestamp=arguments.campaign_timestamp,
    )


def main() -> int:
    arguments = parse_args()
    if arguments.continuous and arguments.render != "human":
        raise SystemExit("--continuous requires --render human")
    if arguments.continuous and arguments.scientific_study:
        raise SystemExit("Scientific runs require a finite --generations budget")
    if arguments.resume is not None and arguments.output is None:
        raise SystemExit("--resume also requires the original --output directory")
    if arguments.resume is not None and arguments.campaign_timestamp is None:
        _genome, _spec, metadata = load_best_from_checkpoint(arguments.resume)
        saved_timestamp = metadata.get("config", {}).get("campaign_timestamp")
        if not isinstance(saved_timestamp, str) or not saved_timestamp:
            raise SystemExit("The checkpoint does not contain a campaign timestamp")
        arguments.campaign_timestamp = saved_timestamp
    elif arguments.campaign_timestamp is None:
        arguments.campaign_timestamp = _local_campaign_timestamp()
    if arguments.max_steps < 2_600:
        print(
            "[evolution] Note: this short horizon is useful for smoke tests but "
            "normally ends before pterodactyls become eligible; use at least "
            "3600 steps for training duck behaviour.",
            flush=True,
        )

    controller: ControllerType = arguments.controller
    accelerator: Accelerator = arguments.accelerator
    requested_output = _output_for(arguments, controller)
    output = requested_output if arguments.resume else _available_output_dir(requested_output)
    resume = arguments.resume
    transport_restarts = 0

    while True:
        checkpoint = resume or output / f"{controller}-checkpoint.pkl"
        config = _config(
            arguments,
            controller=controller,
            accelerator=accelerator,
            output=output,
        )
        try:
            result = EvolutionRunner(
                config,
                checkpoint_path=checkpoint,
                progress=lambda message: print(f"[evolution] {message}", flush=True),
                keep_open_after_run=(arguments.render == "human" and not arguments.exit_after_run),
            ).run()
            break
        except PopulationRunRestartRequested as request:
            controller = request.controller_type
            accelerator = request.accelerator
            arguments.parents = request.parent_count
            arguments.offspring = request.offspring_count
            arguments.continuous = request.continuous
            arguments.generations = request.generations
            arguments.max_steps = request.max_steps
            arguments.mutation_scale = request.mutation_scale
            arguments.evolution_seed = request.evolution_seed
            arguments.simulation_speed = request.simulation_speed
            arguments.training_seeds = request.training_seeds
            arguments.validation_seeds = request.validation_seeds
            arguments.final_test_seeds = request.final_test_seeds
            arguments.scientific_study = request.study_id
            arguments.campaign_timestamp = _local_campaign_timestamp()
            requested_output = _output_for(arguments, controller)
            output = _available_output_dir(requested_output)
            resume = None
            print(f"[evolution] Starting GUI-configured run in {output}.", flush=True)
        except PopulationEvaluationCancelled:
            print("[evolution] Cancelled before partial generation selection.", flush=True)
            return 130
        except KeyboardInterrupt:
            print("\n[evolution] Stopped before partial generation selection.", flush=True)
            return 130
        except VisualTransportError as error:
            detail = str(error)
            was_user_close = "browser window was closed" in detail
            if arguments.render != "human" or was_user_close or transport_restarts >= 1:
                raise
            transport_restarts += 1
            resume = checkpoint
            print(
                "[evolution] The visible browser channel disconnected after a completed "
                "checkpoint. Reopening it once and resuming the next generation "
                f"from {checkpoint}.",
                flush=True,
            )

    model_path = output / "best-candidate.npz"
    parameter_counts = {
        "reactive": default_controller_spec("reactive", len(SENSORY_NAMES)).parameter_count,
        "proactive": default_controller_spec("proactive", len(SENSORY_NAMES)).parameter_count,
    }
    result_location_value = result_location(result.config)
    scientific_figures = export_parquet_figures(result_location_value)
    print(
        json.dumps(
            {
                "controller": controller,
                "parameter_count": result.controller_spec.parameter_count,
                "reactive_parameter_count": parameter_counts["reactive"],
                "proactive_parameter_count": parameter_counts["proactive"],
                "parents": arguments.parents,
                "offspring": arguments.offspring,
                "population": arguments.parents + arguments.offspring,
                "mutation_scale": arguments.mutation_scale,
                "adaptive_sigma": result.config.adaptive_sigma,
                "evolution_seed": arguments.evolution_seed,
                "accelerator": accelerator,
                "simulation_speed_initial": arguments.simulation_speed,
                "simulation_speed_affects_dynamics": False,
                "completed_generations": len(result.records),
                "best_fitness": result.best_fitness,
                "checkpoint": str(result.checkpoint_path.resolve()),
                "model": str(model_path.resolve()),
                "parquet_result_dir": str(result_location_value.run_dir.resolve()),
                "model_hash": result_location_value.model_hash,
                "experiment_hash": result_location_value.experiment_hash,
                "scientific_figures": {
                    label: str(path.resolve()) for label, path in scientific_figures.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
