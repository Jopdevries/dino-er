# Evolved Visual Control for Chromium Dino

> A parameter-matched comparison of reactive and recurrent neural controllers,
> evolved from rendered pixels in a deterministic Chromium Dino environment.

**AE4350 Bio-inspired Intelligence and Learning for Aerospace Applications**

TU Delft · Research software · Python 3.11 · TypeScript · Chromium/Playwright

[Install](#installation) · [Commands](#command-reference) ·
[Research](#research-questions) · [Results](#results-at-a-glance) ·
[Reproduction](#reproducing-the-study) · [Citation](#citation)

## Overview

This repository contains the complete software pipeline for evolving, evaluating,
and inspecting visual controllers in a local Chromium Dino task. It compares a
reactive feed-forward neural network (FFNN) with a proactive continuous-time
recurrent neural network (CTRNN).

Both controllers:

- observe the same five measurements extracted from the current rendered frame;
- choose from the same three held-key actions;
- contain exactly 93 evolved parameters;
- use the same seeded worlds, fitness definition, and candidate budget.

The TypeScript application implements the browser game, renderer, and batch
bridge. The Python package implements perception, neural controllers, evolution,
evaluation, causal interventions, persisted Parquet evidence, and scientific
visualisation.

## Installation

### Requirements

- Python `>=3.11,<3.12`;
- [`uv`](https://docs.astral.sh/uv/);
- a current Node.js/npm installation;
- Chromium installed through Playwright.

### 1. Install Python dependencies and Chromium

Run these commands from the repository root:

```console
uv sync --extra dev
uv run --no-sync playwright install chromium
uv run --no-sync python scripts/check_system.py --json
```

The final command verifies the Python environment, Node/npm, browser runtime,
game build prerequisites, and important project paths.

### 2. Install and build the game

```powershell
# Windows PowerShell
npm.cmd --prefix game ci
npm.cmd --prefix game run build
```

```console
# macOS or Linux
npm --prefix game ci
npm --prefix game run build
```

### 3. Run a small end-to-end experiment

This reduced population verifies the browser–perception–controller loop without
starting the full scientific campaign:

```console
uv run --no-sync python scripts/evolve.py --controller proactive --parents 2 --offspring 8 --continuous --render human
```

The visible interface can pause execution, stop after a complete generation,
save a candidate, and inspect progress. Run
`uv run --no-sync python scripts/evolve.py --help` for finite budgets, seed sets,
checkpoint options, output paths, and accelerator selection.

Optional CUDA inference is available with:

```console
uv sync --extra dev --extra cuda
```

CPU inference remains the portable default and does not alter the controller
definition.

## Command reference

| Task | Command |
|---|---|
| Verify the local setup | `uv run --no-sync python scripts/check_system.py --json` |
| Run a custom evolution | `uv run --no-sync python scripts/evolve.py --help` |
| Inspect the campaign plan | `uv run --no-sync python scripts/run_experiments.py --dry-run` |
| Run the complete campaign | `uv run --no-sync python scripts/run_experiments.py --accelerator auto --workers auto` |
| Resume missing RQ3 evidence | `uv run --no-sync python scripts/run_experiments.py --rq3-only --model-hash 46ccc63ff585 --accelerator auto --workers 2` |
| Generate the results viewer | `uv run --no-sync python scripts/view_results.py` |
| Generate without opening a browser | `uv run --no-sync python scripts/view_results.py --no-open` |
| Replay the latest proactive archive | `uv run --no-sync python scripts/replay.py --latest proactive` |
| Run Python tests | `uv run --no-sync python -m pytest -q` |
| Run linting | `uv run --no-sync python -m ruff check .` |
| Run type checking | `uv run --no-sync python -m mypy src scripts tests` |
| Build the game on Windows | `npm.cmd --prefix game run build` |
| Run TypeScript tests on Windows | `npm.cmd --prefix game run test:ts` |
| Run visual-regression tests on Windows | `npm.cmd --prefix game run test:visual` |

On macOS or Linux, replace `npm.cmd` with `npm`.

## Research questions

The pre-declared study addresses three questions:

1. **RQ1 — Learning and generalisation:** How do reactive and proactive
   controllers differ in finite-budget learning and held-out survival?
2. **RQ2 — Evolution-parameter sensitivity:** How sensitive is the observed
   architecture contrast to the configured Gaussian scale and the
   parent–offspring split?
3. **RQ3 — Strategy and internal state:** How do the evolved strategies differ,
   and does the proactive controller functionally depend on recurrent state?

## Results at a glance

The figure below shows selected-parent training survival over the equal
2,500-candidate evaluation budget. Thin lines are the three independent optimizer
runs; thick lines are architecture means. Survival is capped at 3,600 control
steps.

![RQ1 selected-parent learning curves](media/rq1_learning.png)

*RQ1 learning curves from the saved model-v2 campaign. The dotted line is the
270-step no-action reference used by the analysis.*

| Evidence stage | Reactive FFNN | Proactive CTRNN | Uncertainty unit |
|---|---:|---:|---|
| RQ1 final archive training survival | 3,600.0 ± 0.0 | 2,589.7 ± 289.7 | 95% Student-t half-width across 3 optimizer runs |
| RQ1 validation survival | 2,873.1 ± 495.6 | 2,615.9 ± 748.4 | 95% Student-t half-width across 3 optimizer runs |
| Locked final-test survival | 3,379.4 ± 285.7 | 1,625.4 ± 411.9 | 95% Student-t half-width across 20 worlds for one frozen representative |
| Locked final-test ceiling episodes | 17 / 20 | 0 / 20 | Descriptive count |

The paired RQ1 validation contrast was **proactive − reactive = −257.1 ±
1,094.2 control steps** across the three matched optimizer seeds. The interval
includes zero, so the study does not establish a consistent architecture-level
validation advantage. The locked final test compares one validation-frozen
representative per architecture; its 20 worlds quantify environment-seed
variation, not independent evolutionary replication.

RQ3 provides evidence that the selected CTRNN *uses* recurrent state: resetting
its state before every controller update reduced mean survival from 2,683.0 to
270.6 steps over five matched worlds. This establishes functional dependence for
that selected policy, but does not by itself establish a general performance
benefit of recurrent capacity.

## Experimental design

### Controllers and observations

| Property | Reactive | Proactive |
|---|---|---|
| Architecture | FFNN, `5 → 10 → 3` | CTRNN, `5 → 6 recurrent → 3` |
| Persistent controller state | None | Six recurrent states |
| Evolved parameters | 93 | 93 |
| Output decision | Raw-score `argmax` | Raw-score `argmax` |

Both controllers receive the same normalized five-vector:

1. horizontal gap to the nearest obstacle;
2. obstacle top relative to the Dino top;
3. obstacle bounding-box width;
4. obstacle bounding-box height;
5. Dino bounding-box top coordinate.

The policy receives these measurements from the rendered pixels rather than
game-engine state. Its action space is `0` no key, `1` jump held, and `2`
duck/fast-drop held. Rendering, perception, controller evaluation, action
application, and physics use one logical 60 Hz control step. Wall-clock
acceleration changes pacing, not the simulated time step.

### Evolution and evidence separation

| Stage | Design | Purpose |
|---|---|---|
| RQ1 | `(20 + 80)` elitist ES, 25 generations, adaptive scale, optimizer seeds 7/17/27 | Primary learning comparison |
| RQ2 | Five fixed-scale / parent–offspring cells, both architectures, seeds 7/17/27 | Sensitivity analysis |
| Selection | Median validation-performing RQ1 archive within each architecture | Freeze one representative per architecture |
| RQ3 | Paired normal, visual-input, sensor-zero, and CTRNN-state-reset replays | Strategy and causal interventions |
| Final test | 20 untouched world seeds after selection is frozen | Locked representative comparison |

Each candidate is evaluated on the same ten training worlds within a run, and
fitness is arithmetic mean survival with a 3,600-step ceiling. Validation uses
five disjoint worlds; the final test uses another 20 disjoint worlds. The exact
seed lists and experiment matrix are versioned in
[`configs/experiment-plan.json`](configs/experiment-plan.json).

> [!IMPORTANT]
> In the implemented RQ2 pipeline, the configured Gaussian scale controls both
> initial genome spread and mutation magnitude. RQ2 therefore measures
> sensitivity to that combined scale setting; it is not a mutation-only
> intervention.

## Repository structure

```text
Project-Code/
├── configs/                       pre-declared experiment plan and seed partitions
├── game/                          Chromium-derived TypeScript/Canvas environment
│   ├── src/                       engine, renderer, input, and batch bridge
│   └── tests/                     engine and visual-regression tests
├── media/                         curated image used by this README
├── scripts/                       experiment, replay, system-check, and viewer CLIs
├── src/dino_er/                   Python research implementation
│   ├── controllers.py             FFNN and CTRNN definitions
│   ├── perception.py              rendered-frame measurement pipeline
│   ├── environment.py             environment and controller execution
│   ├── evaluation.py              seeded evaluation and interventions
│   ├── evolution.py               evolution strategy and checkpoints
│   └── scientific.py              statistics, figures, and exports
├── tests/                         Python unit and integration tests
└── third_party/                   pinned Chromium reference source and licences
```

## Reproducing the study

First inspect the persisted state and execution plan without launching workers:

```console
uv run --no-sync python scripts/run_experiments.py --dry-run
```

Launch the complete dependency-ordered pipeline:

```console
uv run --no-sync python scripts/run_experiments.py --accelerator auto --workers auto
```

The runner executes independent optimization runs, freezes validation-based
representatives, and only then performs RQ3 interventions and the locked final
test. Completed runs are skipped; compatible partial runs can resume from their
last complete checkpoint. Each worker owns its browser, RNG, checkpoint, and
result path.

To resume only missing RQ3 evidence for the existing model-v2 campaign:

```console
uv run --no-sync python scripts/run_experiments.py --rq3-only --model-hash 46ccc63ff585 --accelerator auto --workers 2
```

The full campaign comprises six RQ1 optimization runs and 30 RQ2 runs. At 2,500
candidate evaluations and ten training episodes per run, it is computationally
expensive; use the dry run and small visible run before starting it.

### Inspecting and replaying results

Generate the self-contained scientific results viewer:

```console
uv run --no-sync python scripts/view_results.py
```

This reads the saved Parquet evidence, writes `scientific-results.html`, embeds
figures and CSV exports, and opens the file in the default browser. Add
`--no-open` for non-interactive generation or select the campaign explicitly:

```console
uv run --no-sync python scripts/view_results.py --model-version 2 --model-hash 46ccc63ff585 --no-open
```

Replay an archived policy:

```console
uv run --no-sync python scripts/replay.py --latest proactive
```

Use `scripts/replay.py --help` for explicit models, world seeds, visual or
recurrent-state interventions, and Parquet trace export. A replay is a diagnostic
inspection tool and is not independent statistical evidence.

## Verification

Run the Python quality and regression suite:

```console
uv run --no-sync python -m pytest -q
uv run --no-sync python -m ruff check .
uv run --no-sync python -m mypy src scripts tests
```

Run the TypeScript and visual-regression suites:

```powershell
# Windows PowerShell
npm.cmd --prefix game run test:ts
npm.cmd --prefix game run test:visual
```

```console
# macOS or Linux
npm --prefix game run test:ts
npm --prefix game run test:visual
```

The tests cover controller dimensions and parameter use, perception,
determinism, seeded environments, evolution behavior, checkpointing,
interventions, Chromium-reference fidelity, and results generation.

## Interpretation and scope

- The task establishes partial observability because the five visual measurements
  omit dynamic state such as vertical velocity and game speed; it does not prove
  that every observation is perceptually aliased.
- A state-reset performance loss demonstrates functional state use by the selected
  CTRNN. It does not show that recurrent state is always beneficial or necessary.
- Sensor-zero interventions are strong counterfactual perturbations and may move
  observations outside the policy's training distribution.
- The locked final-test confidence intervals describe variation across worlds for
  two fixed representatives, not uncertainty across independently evolved models.
- Survival is right-censored at 3,600 steps, so ceiling-heavy results contain less
  information about performance beyond the evaluation horizon.

## Provenance and licences

The game adaptation is derived from Chromium's
`components/neterror/resources/dino_game` at revision
`1ccb91e11f09fbbdec4f8f754d0e2f7d28246660`. Source attribution, local
modifications, sprite provenance, and the Chromium BSD-style licence are recorded
in [`THIRD_PARTY_NOTICES.txt`](THIRD_PARTY_NOTICES.txt) and
[`third_party/`](third_party/).

Original project code is released under the [MIT License](LICENSE).

## Citation

If you use this software or experimental design, cite the accompanying report and
the repository version used for the analysis. Until an archival DOI is available,
the following software citation can be used:

```bibtex
@software{de_vries_dino_er_2026,
  author  = {de Vries, J. H.},
  title   = {Evolved Visual Control for Chromium Dino},
  year    = {2026},
  version = {0.3.0},
  note    = {AE4350 Bio-inspired Intelligence and Learning for Aerospace Applications,
             Delft University of Technology}
}
```
