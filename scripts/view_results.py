"""Generate a browser-ready HTML report from the saved Parquet results."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path
from typing import Any

from dino_er.scientific import aggregate_study_results, model_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--model-version", type=int, default=2)
    parser.add_argument(
        "--model-hash",
        help=(
            "Generate the report for one exact model hash. By default the current "
            "model is used when its manifest exists; otherwise the most complete "
            "manifested campaign is selected."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Deprecated compatibility option; no server is started.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Write the report without opening it in the default browser.",
    )
    return parser.parse_args()


def _gui_needs_build(static_dir: Path, source_paths: tuple[Path, ...]) -> bool:
    output = static_dir / "results.html"
    if not output.is_file():
        return True
    output_mtime = output.stat().st_mtime_ns
    return any(path.is_file() and path.stat().st_mtime_ns > output_mtime for path in source_paths)


def _ensure_built_gui(project_root: Path) -> Path:
    game_dir = project_root / "game"
    static_dir = game_dir / "dist"
    sources = tuple(
        game_dir / relative
        for relative in (
            "results.html",
            "src/results.ts",
            "src/results.css",
            "package.json",
            "vite.config.ts",
        )
    )
    if _gui_needs_build(static_dir, sources):
        npm = "npm.cmd" if sys.platform == "win32" else "npm"
        try:
            subprocess.run([npm, "run", "build"], cwd=game_dir, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise SystemExit(
                f"Results GUI build failed; run `{npm} run build` in game ({error})."
            ) from error
    if not (static_dir / "results.html").is_file():
        raise SystemExit("Built results GUI not found and could not be generated.")
    return static_dir


def _resolve_view_model_directory(
    results_root: Path,
    model_version: int,
    requested_model_hash: str | None,
    *,
    current_model_hash: str | None = None,
) -> Path:
    """Select a campaign without mistaking an empty engineering edit for a run."""

    prefix = f"model_v{model_version:03d}_"
    if requested_model_hash:
        if not requested_model_hash.isalnum():
            raise ValueError("--model-hash must be alphanumeric")
        requested = results_root / f"{prefix}{requested_model_hash}"
        if not requested.is_dir():
            raise FileNotFoundError(f"Results campaign does not exist: {requested}")
        return requested

    current = results_root / f"{prefix}{current_model_hash or model_hash()}"
    if (current / "results-manifest.json").is_file():
        return current
    manifested = [
        path
        for path in results_root.glob(f"{prefix}*")
        if path.is_dir() and (path / "results-manifest.json").is_file()
    ]
    if not manifested:
        raise FileNotFoundError(f"No manifested model-v{model_version} campaign in {results_root}")

    def ranking(path: Path) -> tuple[int, int]:
        try:
            manifest = json.loads((path / "results-manifest.json").read_text(encoding="utf-8"))
            statuses = manifest.get("run_statuses", {})
            completed = sum(
                int(status.get("complete", 0))
                for status in statuses.values()
                if isinstance(status, dict)
            )
        except (OSError, json.JSONDecodeError):
            completed = 0
        return completed, (path / "results-manifest.json").stat().st_mtime_ns

    return max(manifested, key=ranking)


def _planned_runs(model_dir: Path) -> list[dict[str, Any]]:
    try:
        manifest = json.loads((model_dir / "results-manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    planned = manifest.get("planned_runs", []) if isinstance(manifest, dict) else []
    return [dict(item) for item in planned if isinstance(item, dict)]


def _result_source(model_dir: Path, relative: str) -> Path:
    source = (model_dir / relative).resolve()
    if not source.is_relative_to(model_dir.resolve()) or not source.is_file():
        raise FileNotFoundError(f"Generated result is missing: {relative}")
    return source


def _copy_result(
    model_dir: Path,
    asset_root: Path,
    relative: str,
    output_relative: str | None = None,
) -> str:
    source = _result_source(model_dir, relative)
    target = asset_root / (output_relative or relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target.relative_to(asset_root).as_posix()


def _copy_table_as_csv(
    model_dir: Path,
    asset_root: Path,
    relative: str,
    output_relative: str | None = None,
) -> str:
    source = _result_source(model_dir, relative)
    csv_source = source if source.suffix.lower() == ".csv" else source.with_suffix(".csv")
    if not csv_source.is_file():
        import pyarrow.parquet as pq

        rows = pq.read_table(source).to_pylist()
        columns = list(rows[0]) if rows else list(pq.read_schema(source).names)
        with csv_source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    csv_relative = Path(output_relative or relative).with_suffix(".csv")
    target = asset_root / csv_relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_source, target)
    return csv_relative.as_posix()


def _copy_trace_csv(
    model_dir: Path,
    asset_root: Path,
    relative: str,
    world_seed: int,
    output_relative: str | None = None,
) -> str:
    import pyarrow.parquet as pq

    source = _result_source(model_dir, relative)
    rows = [
        row
        for row in pq.read_table(source).to_pylist()
        if int(row.get("world_seed", -1)) == world_seed
    ]
    if not rows:
        raise ValueError(f"Trace {relative} has no rows for world seed {world_seed}")
    csv_relative = (
        Path(output_relative).with_suffix(".csv")
        if output_relative
        else Path(relative).with_suffix("").with_name(
            f"{Path(relative).stem}_world-{world_seed}.csv"
        )
    )
    target = asset_root / csv_relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return csv_relative.as_posix()


def _inline_gui(
    static_dir: Path,
    payload: dict[str, Any],
    asset_data: dict[str, str],
) -> str:
    html = (static_dir / "results.html").read_text(encoding="utf-8")
    script_match = re.search(
        r'<script type="module"[^>]+src="/assets/([^\"]+)"[^>]*></script>', html
    )
    style_match = re.search(
        r'<link rel="stylesheet"[^>]+href="/assets/([^\"]+)"[^>]*>', html
    )
    if script_match is None or style_match is None:
        raise RuntimeError("Built results GUI does not contain the expected script and stylesheet")

    script = (static_dir / "assets" / script_match.group(1)).read_text(encoding="utf-8")
    script = re.sub(r'^import["\']\./modulepreload-polyfill-[^"\']+["\'];', "", script)
    style = (static_dir / "assets" / style_match.group(1)).read_text(encoding="utf-8")
    embedded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")
    embedded_assets = json.dumps(
        asset_data, separators=(",", ":"), ensure_ascii=True
    ).replace("</", "<\\/")

    html = html.replace(style_match.group(0), f"<style>{style}</style>")
    html = re.sub(r'\s*<link rel="modulepreload"[^>]+>\s*', "\n", html)
    html = html.replace(
        script_match.group(0),
        f"<script>window.__RESULTS_INDEX__={embedded};"
        f"window.__RESULT_ASSETS__={embedded_assets};</script>"
        f'<script type="module">{script}</script>',
    )
    return html


def _data_url(path: Path) -> str:
    media_type = "image/png" if path.suffix.lower() == ".png" else "text/csv;charset=utf-8"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _write_static_results(
    project_root: Path,
    results_root: Path,
    model_version: int,
    requested_hash: str | None,
) -> Path:
    static_dir = _ensure_built_gui(project_root)
    model_dir = _resolve_view_model_directory(results_root, model_version, requested_hash)
    payload = aggregate_study_results(
        results_root,
        model_version=model_version,
        planned_runs=_planned_runs(model_dir),
        selected_model_hash=model_dir.name.removeprefix(f"model_v{model_version:03d}_"),
    )

    staging_root = Path(tempfile.mkdtemp(prefix=".scientific-results-", dir=project_root))
    try:
        payload["figures"] = {
            label: _copy_result(
                model_dir,
                staging_root,
                str(path),
                f"figures/{Path(str(path)).name}",
            )
            for label, path in payload.get("figures", {}).items()
        }
        payload["tables"] = {
            label: _copy_table_as_csv(
                model_dir,
                staging_root,
                str(path),
                f"data/{Path(str(path)).with_suffix('.csv').name}",
            )
            for label, path in payload.get("tables", {}).items()
        }
        for trace in payload.get("rq3_traces", []):
            trace_key = str(trace["id"])
            trace["figure"] = _copy_result(
                model_dir,
                staging_root,
                str(trace["figure"]),
                f"traces/{trace_key}.png",
            )
            trace["data"] = _copy_trace_csv(
                model_dir,
                staging_root,
                str(trace["data"]),
                int(trace["world_seed"]),
                f"traces/{trace_key}.csv",
            )

        asset_data = {
            path.relative_to(staging_root).as_posix(): _data_url(path)
            for path in staging_root.rglob("*")
            if path.is_file()
        }
        html = _inline_gui(static_dir, payload, asset_data)
    finally:
        if staging_root.is_dir():
            shutil.rmtree(staging_root)

    destination = project_root / "scientific-results.html"
    destination.write_text(html, encoding="utf-8")
    return destination


def main() -> int:
    arguments = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    try:
        destination = _write_static_results(
            project_root,
            arguments.results_root,
            arguments.model_version,
            arguments.model_hash,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"[results] Wrote {destination.resolve()}", flush=True)
    if not arguments.no_open:
        webbrowser.open(destination.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
