from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.view_results as viewer
from scripts.view_results import _gui_needs_build, _write_static_results


def test_gui_build_decision_detects_stale_and_fresh_sources(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    output = dist / "results.html"
    source = tmp_path / "results.ts"
    output.write_text("built", encoding="utf-8")
    source.write_text("source", encoding="utf-8")
    now = time.time_ns()

    os.utime(output, ns=(now, now))
    os.utime(source, ns=(now - 1, now - 1))
    assert not _gui_needs_build(dist, (source,))
    os.utime(source, ns=(now + 1_000_000_000, now + 1_000_000_000))
    assert _gui_needs_build(dist, (source,))


def test_static_report_embeds_gui_and_exports_real_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    model = project / "results" / "model_v002_testhash"
    dist = project / "game" / "dist"
    (dist / "assets").mkdir(parents=True)
    (model / "final_figures").mkdir(parents=True)
    (model / "summary_tables").mkdir()
    (model / "behavioural-traces").mkdir()
    (model / "results-manifest.json").write_text(
        json.dumps({"planned_runs": []}), encoding="utf-8"
    )
    (model / "final_figures" / "figure.png").write_bytes(b"\x89PNG\r\n\x1a\nfigure")
    pq.write_table(
        pa.table({"generation": [1], "survival": [3600.0]}),
        model / "summary_tables" / "summary.parquet",
    )
    pq.write_table(
        pa.table({"world_seed": [404, 505], "action": [2, 1]}),
        model / "behavioural-traces" / "trace.parquet",
    )
    (dist / "results.html").write_text(
        '<html><head><script type="module" crossorigin src="/assets/results-x.js"></script>'
        '<link rel="modulepreload" href="/assets/modulepreload-polyfill-x.js">'
        '<link rel="stylesheet" href="/assets/results-x.css"></head><body></body></html>',
        encoding="utf-8",
    )
    (dist / "assets" / "results-x.js").write_text(
        'import"./modulepreload-polyfill-x.js";document.body.dataset.ready="yes";',
        encoding="utf-8",
    )
    (dist / "assets" / "results-x.css").write_text("body{color:#123}", encoding="utf-8")
    (dist / "assets" / "modulepreload-polyfill-x.js").write_text("", encoding="utf-8")
    monkeypatch.setattr(viewer, "_ensure_built_gui", lambda _root: dist)
    monkeypatch.setattr(
        viewer,
        "aggregate_study_results",
        lambda *_args, **_kwargs: {
            "figures": {"RQ1 learning": "final_figures/figure.png"},
            "tables": {"RQ1 summary": "summary_tables/summary.parquet"},
            "rq3_traces": [
                {
                    "id": "trace-a",
                    "figure": "final_figures/figure.png",
                    "data": "behavioural-traces/trace.parquet",
                    "world_seed": 404,
                }
            ],
        },
    )

    report = _write_static_results(project, project / "results", 2, "testhash")
    html = report.read_text(encoding="utf-8")
    assert "window.__RESULTS_INDEX__=" in html
    assert "window.__RESULT_ASSETS__=" in html
    assert "data:image/png;base64," in html
    assert "data:text/csv;charset=utf-8;base64," in html
    assert 'type="module">document.body.dataset.ready="yes";' in html
    assert "<style>body{color:#123}</style>" in html
    assert 'src="/assets/' not in html
    assert 'href="/assets/' not in html
    assert not (project / "scientific-results-assets").exists()
