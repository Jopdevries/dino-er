"""Replay a candidate from a trusted checkpoint or GUI-saved model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dino_er.evaluation import EpisodeConfig, evaluate_candidate
from dino_er.evolution import load_best_from_checkpoint, load_candidate_model
from dino_er.scientific import (
    MODEL_VERSION,
    export_behaviour_trace,
    model_hash,
    plot_behaviour_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--model", type=Path)
    source.add_argument(
        "--latest",
        dest="latest_controller",
        choices=("reactive", "proactive"),
        help=(
            "Replay the current campaign's best training archive for this controller. "
            "This is a visual diagnostic, not validation-based model selection."
        ),
    )
    parser.add_argument(
        "--evolution-seed",
        type=int,
        help="With --latest, restrict the replay to one independent optimizer seed.",
    )
    parser.add_argument("--study-id", default="rq1-main")
    parser.add_argument("--model-version", type=int, default=MODEL_VERSION)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "Game seed. Defaults to training seed 101 for --latest and to 303 "
            "for an explicit checkpoint/model replay."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument(
        "--simulation-speed",
        type=float,
        default=1.0,
        help=(
            "Wall-clock pacing target from 0.25x to 100x. Fixed 60-Hz physics, "
            "controller decisions, actions, and fitness are unchanged."
        ),
    )
    parser.add_argument("--render", choices=("human", "none", "rgb_array"), default="human")
    parser.add_argument(
        "--state-intervention",
        choices=("normal", "reset_each_step"),
        default="normal",
        help="CTRNN-state intervention for the proactive controller.",
    )
    parser.add_argument(
        "--sensory-intervention",
        choices=("normal", "constant_first_frame", "ablate_sensor"),
        default="normal",
        help="RQ3 visual-input intervention applied after perception and before control.",
    )
    parser.add_argument(
        "--ablate-sensor",
        type=int,
        help="Zero one 0-based visual sensor when --sensory-intervention ablate_sensor.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        help=(
            "Close a visible replay automatically after this many seconds. "
            "Omit this option to keep the final frame open until you close Chrome."
        ),
    )
    parser.add_argument(
        "--trace-parquet",
        type=Path,
        help="Write one selected RQ3 behavioural trace and its PNG plot.",
    )
    return parser.parse_args()


def _resolve_project_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path.resolve()
    project_path = Path(__file__).resolve().parents[1] / path
    return project_path.resolve() if project_path.exists() else path.resolve()


def _latest_checkpoint(
    *,
    controller_type: str,
    evolution_seed: int | None,
    study_id: str,
    model_version: int,
    artifacts_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Find one complete, atomically saved archive from the newest campaign.

    Checkpoint metadata, rather than directory-name parsing, defines membership.
    Within the newest campaign, omitting ``evolution_seed`` deliberately chooses
    the highest training archive only for visual debugging. Scientific model
    selection remains validation-based in the experiment pipeline.
    """

    resolved_root = _resolve_project_path(artifacts_root)
    model_dir = resolved_root / f"model_v{model_version:03d}_{model_hash()}"
    matches: list[tuple[Path, dict[str, Any], str, float]] = []
    for checkpoint in model_dir.glob(f"*/*/{controller_type}-checkpoint.pkl"):
        try:
            _genome, spec, metadata = load_best_from_checkpoint(checkpoint)
        except (OSError, ValueError):
            continue
        config = metadata.get("config")
        if not isinstance(config, dict):
            continue
        if spec.controller_type != controller_type:
            continue
        if config.get("study_id") != study_id:
            continue
        if evolution_seed is not None and config.get("evolution_seed") != evolution_seed:
            continue
        campaign = config.get("campaign_timestamp")
        best_fitness = metadata.get("best_fitness")
        if not isinstance(campaign, str) or not isinstance(best_fitness, (int, float)):
            continue
        matches.append((checkpoint.resolve(), metadata, campaign, float(best_fitness)))
    if not matches:
        seed_note = "" if evolution_seed is None else f" and optimizer seed {evolution_seed}"
        raise SystemExit(
            "No compatible completed-generation checkpoint found for "
            f"{study_id}/{controller_type}{seed_note} below {model_dir}."
        )
    newest_campaign = max(item[2] for item in matches)
    current = [item for item in matches if item[2] == newest_campaign]
    checkpoint, metadata, _campaign, _fitness = max(
        current,
        key=lambda item: (item[3], item[0].stat().st_mtime_ns, str(item[0])),
    )
    return checkpoint, metadata


def main() -> int:
    arguments = parse_args()
    if arguments.hold_seconds is not None and arguments.hold_seconds < 0:
        raise SystemExit("--hold-seconds cannot be negative")
    if arguments.sensory_intervention == "ablate_sensor" and arguments.ablate_sensor is None:
        raise SystemExit("--sensory-intervention ablate_sensor requires --ablate-sensor")
    if arguments.sensory_intervention != "ablate_sensor" and arguments.ablate_sensor is not None:
        raise SystemExit("--ablate-sensor requires --sensory-intervention ablate_sensor")
    latest_metadata: dict[str, Any] | None = None
    if arguments.latest_controller is not None:
        source_path, latest_metadata = _latest_checkpoint(
            controller_type=arguments.latest_controller,
            evolution_seed=arguments.evolution_seed,
            study_id=arguments.study_id,
            model_version=arguments.model_version,
            artifacts_root=arguments.artifacts_root,
        )
    else:
        requested_source = (
            arguments.checkpoint if arguments.checkpoint is not None else arguments.model
        )
        source_path = _resolve_project_path(requested_source)
    replay_seed = arguments.seed
    if replay_seed is None:
        replay_seed = 101 if arguments.latest_controller is not None else 303
    if not source_path.is_file():
        raise SystemExit(
            "Checkpoint/model not found: "
            f"{source_path}\n"
            "Run this command from Project-Code, or create it with evolution first."
        )
    if arguments.render == "human":
        hold_message = (
            "The final frame stays open until you close its Chrome window or press Ctrl+C."
            if arguments.hold_seconds is None
            else f"The final frame will remain open for {arguments.hold_seconds:g} seconds."
        )
        print(f"[replay] Opening one visible Chromium Dino replay. {hold_message}", flush=True)
    if arguments.checkpoint is not None or arguments.latest_controller is not None:
        genome, spec, metadata = load_best_from_checkpoint(source_path)
        saved_fitness = metadata["best_fitness"]
        source_kind = (
            "latest_training_checkpoint"
            if arguments.latest_controller is not None
            else "checkpoint"
        )
    else:
        genome, spec, metadata = load_candidate_model(source_path)
        saved_fitness = metadata.get("fitness")
        source_kind = "gui_saved_model"
    result = evaluate_candidate(
        genome,
        spec,
        EpisodeConfig(
            seeds=(replay_seed,),
            max_steps=arguments.max_steps,
            render_mode=None if arguments.render == "none" else arguments.render,
            state_intervention=arguments.state_intervention,
            sensory_intervention=arguments.sensory_intervention,
            ablation_sensor_index=arguments.ablate_sensor,
            record_trace=True,
            simulation_speed=arguments.simulation_speed,
            post_episode_hold_seconds=arguments.hold_seconds or 0.0,
            wait_for_window_close=(arguments.render == "human" and arguments.hold_seconds is None),
        ),
    )
    parquet_trace = None
    parquet_plot = None
    if arguments.trace_parquet is not None:
        parquet_trace = export_behaviour_trace(
            arguments.trace_parquet,
            evaluation=result,
            controller_type=spec.controller_type,
            source_id=result.genome_hash,
            intervention=(
                f"state={arguments.state_intervention};"
                f"sensory={arguments.sensory_intervention};"
                f"sensor={arguments.ablate_sensor}"
            ),
        )
        parquet_plot = plot_behaviour_trace(parquet_trace)
    payload = {
        "controller": spec.controller_type,
        "source": source_kind,
        "source_path": str(source_path),
        "saved_fitness": saved_fitness,
        "replay_fitness": result.fitness,
        "simulation_speed": arguments.simulation_speed,
        "evaluation_seconds": result.duration_seconds,
        "effective_control_steps_per_second": (
            sum(episode.steps for episode in result.episodes) / result.duration_seconds
            if result.duration_seconds > 0
            else None
        ),
        "campaign_timestamp": (
            latest_metadata.get("config", {}).get("campaign_timestamp")
            if latest_metadata is not None
            else None
        ),
        "evolution_seed": metadata.get("config", {}).get("evolution_seed"),
        "run_id": metadata.get("config", {}).get("run_id"),
        "genome_hash": result.genome_hash,
        "state_intervention": arguments.state_intervention,
        "sensory_intervention": arguments.sensory_intervention,
        "visible_hold": (
            ("until_window_closed" if arguments.hold_seconds is None else arguments.hold_seconds)
            if arguments.render == "human"
            else 0.0
        ),
        "episodes": [
            {
                "seed": episode.seed,
                "visual_passes": episode.visual_passes,
                "steps": episode.steps,
                "terminated": episode.terminated,
            }
            for episode in result.episodes
        ],
        "trace_parquet": str(parquet_trace.resolve()) if parquet_trace else None,
        "trace_plot": str(parquet_plot.resolve()) if parquet_plot else None,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
