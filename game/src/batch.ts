import './style.css';

import {
  controlIntervalMilliseconds,
  installPopulationArenaBridge,
  publicRenderingEnabled,
  validSimulationSpeed,
} from './batch_bridge';
import type {
  ArenaBridgeHandle,
  ArenaBridgeView,
  ArenaDiagnostics,
  ArenaRunMetadata,
  ArenaStatusMessage,
  CandidateDiagnostics,
  ConfigureArenaMessage,
  GenerationHistoryPoint,
} from './batch_bridge';
import type {Action} from './engine';
import {DinoPopulationEngine} from './engine';
import {VirtualDinoInput} from './input';
import type {CandidateDisplayStyle} from './renderer';
import {DinoRenderer} from './renderer';

const parameters = new URLSearchParams(window.location.search);
const populationSize = Math.max(
  1,
  Math.min(100, Number.parseInt(parameters.get('instances') ?? '100', 10)),
);
const bridgeUrl = parameters.get('bridge');
const realTimeDisplay = parameters.get('realtime') === '1';
const initialSeed = Number.parseInt(parameters.get('seed') ?? '7', 10);
// A visible session starts at Chromium's normal pace.  Higher values are an
// explicit throughput choice, never an implicit GUI default.
const requestedSpeed = Number.parseFloat(parameters.get('speed') ?? '1');
const initialSimulationSpeed = validSimulationSpeed(requestedSpeed)
  ? requestedSpeed
  : 1;

function required<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`Required UI element is missing: ${selector}`);
  }
  return element;
}

function setText(selector: string, value: string | number): void {
  required<HTMLElement>(selector).textContent = String(value);
}

function mean(values: readonly number[]): number {
  return values.length === 0
    ? 0
    : values.reduce((total, value) => total + value, 0) / values.length;
}

function parseSeeds(selector: string): number[] | null {
  const values = required<HTMLInputElement>(selector).value
    .split(',')
    .map((part) => Number(part.trim()));
  return values.length > 0
    && values.every(Number.isSafeInteger)
    && new Set(values).size === values.length
    ? values
    : null;
}

class PopulationView implements ArenaBridgeView {
  readonly populationSize: number;
  readonly privateObservationAtlas: HTMLCanvasElement;
  privateObservationCandidateIds: readonly number[];

  private readonly engine: DinoPopulationEngine;
  private readonly inputs: VirtualDinoInput[];
  private readonly publicRenderer: DinoRenderer;
  private readonly sharedWorldCanvas = document.createElement('canvas');
  private readonly sharedWorldRenderer: DinoRenderer;
  private readonly atlasRenderer: DinoRenderer;
  private readonly readyPromise: Promise<void>;
  private metadata: ArenaRunMetadata;
  private diagnostics: ArenaDiagnostics | null = null;
  private selectedIndex = 0;
  private nextControlDeadline: number | null = null;
  private simulationSpeed: number;

  constructor(size: number, seed: number, simulationSpeed: number) {
    this.populationSize = size;
    this.privateObservationCandidateIds = Array.from(
      {length: size},
      (_value, index) => index,
    );
    this.simulationSpeed = simulationSpeed;
    this.engine = new DinoPopulationEngine(size, seed);
    this.inputs = Array.from({length: size}, () => new VirtualDinoInput());
    this.metadata = {
      controllerType: 'replay',
      generation: 0,
      mutationScale: null,
      accelerator: bridgeUrl ? 'connecting' : 'not connected',
    };

    this.publicRenderer = new DinoRenderer(
      required<HTMLCanvasElement>('#population-game'),
      1,
    );
    this.sharedWorldRenderer = new DinoRenderer(this.sharedWorldCanvas, 1);
    this.privateObservationAtlas = document.createElement('canvas');
    this.privateObservationAtlas.dataset.candidateCount = String(size);
    required<HTMLElement>('#private-observations').append(this.privateObservationAtlas);
    this.atlasRenderer = new DinoRenderer(this.privateObservationAtlas, 1, true);
    this.readyPromise = Promise.all([
      this.publicRenderer.ready(),
      this.sharedWorldRenderer.ready(),
      this.atlasRenderer.ready(),
    ]).then(() => undefined);
  }

  async ready(): Promise<void> {
    await this.readyPromise;
    this.renderPrivateFrames(undefined, true);
    this.renderGame();
    this.renderSummary();
    this.renderSimulationSpeed();
  }

  async configure(message: ConfigureArenaMessage): Promise<void> {
    if (!Number.isInteger(message.seed)) {
      throw new Error('The world seed must be an integer');
    }
    this.metadata = {...message.metadata};
    this.diagnostics = null;
    this.selectedIndex = 0;
    this.nextControlDeadline = null;
    this.engine.reset(message.seed);
    this.inputs.forEach((input) => input.reset());
    await this.readyPromise;
    this.renderPrivateFrames(undefined, true);
    this.renderGame();
    this.renderSummary();
    this.renderSimulationSpeed();
    setText('#seed', message.seed);
    setText('#run-status', 'Generation ready');
  }

  async step(actions: readonly Action[], active: readonly boolean[]): Promise<void> {
    const started = performance.now();
    const models = this.engine.candidateRenderModels();
    const canonical = actions.map((action, index) => {
      const input = this.inputs[index];
      const model = models[index];
      if (!input || !model || model.crashed || !active[index]) {
        return 0;
      }
      input.setAction(action);
      return input.action();
    });
    this.engine.step(canonical, active);
    this.renderPrivateFrames(active);
    if (publicRenderingEnabled(this.simulationSpeed)) {
      this.renderGame();
    }

    if (bridgeUrl && realTimeDisplay) {
      const deadline = Math.max(
        started,
        (this.nextControlDeadline ?? started)
          + controlIntervalMilliseconds(this.simulationSpeed),
      );
      this.nextControlDeadline = deadline;
      const delay = deadline - performance.now();
      if (delay > 0) {
        await new Promise((resolve) => window.setTimeout(resolve, delay));
      }
    }
  }

  updateDiagnostics(message: ArenaDiagnostics): void {
    this.diagnostics = message;
    this.simulationSpeed = message.simulationSpeed;
    if (publicRenderingEnabled(this.simulationSpeed)) {
      this.renderGame();
    }
    this.renderSimulationSpeed();
    this.renderSummary();
    this.renderSelected();
    this.renderHistory(message.generationHistory);
    setText(
      '#accelerator',
      `${message.accelerator} | ${message.controlTicksPerSecond.toFixed(1)} steps/s`,
    );
  }

  setSimulationSpeed(speed: number): void {
    if (!validSimulationSpeed(speed)) {
      throw new Error('Simulation speed must be between 0.25x and 100x');
    }
    this.simulationSpeed = speed;
    this.nextControlDeadline = null;
    if (publicRenderingEnabled(speed)) {
      this.renderGame();
    }
    this.renderSimulationSpeed();
  }

  currentSimulationSpeed(): number {
    return this.simulationSpeed;
  }

  updateStatus(message: ArenaStatusMessage): void {
    setText('#run-status', message.message);
    if (message.generationHistory) {
      this.renderHistory(message.generationHistory);
    }
    if (message.mutationScale !== undefined) {
      setText('#mutation-scale', message.mutationScale.toFixed(4));
    }
    if (message.savedModelPath) {
      setText('#saved-model', `Saved: ${message.savedModelPath}`);
    }
    if (
      message.scientificOutputPath
      || message.aggregateOutputPath
      || message.resultDirectory
      || message.scientificFigures
    ) {
      required<HTMLElement>('#scientific-files').hidden = false;
      if (message.resultDirectory) {
        setText('#result-directory', message.resultDirectory);
      }
      if (message.scientificOutputPath) {
        setText('#run-output', message.scientificOutputPath);
      }
      if (message.aggregateOutputPath) {
        setText('#study-output', message.aggregateOutputPath);
      }
      if (message.scientificFigures) {
        this.renderScientificFigures(message.scientificFigures);
      }
    }
    if (['run_complete', 'stopped', 'replay_complete'].includes(message.phase)) {
      for (const id of ['resume', 'pause', 'stop']) {
        required<HTMLButtonElement>(`#${id}`).disabled = true;
      }
    }
  }

  selectRelative(offset: -1 | 1): void {
    this.selectedIndex = (
      this.selectedIndex + offset + this.populationSize
    ) % this.populationSize;
    if (publicRenderingEnabled(this.simulationSpeed)) {
      this.renderGame();
    }
    this.renderSelected();
  }

  selectBest(): void {
    const best = this.bestCandidate();
    const bestIndex = this.diagnostics?.candidates.findIndex(
      (candidate) => candidate.id === best?.id,
    ) ?? -1;
    if (bestIndex >= 0) {
      this.selectedIndex = bestIndex;
      if (publicRenderingEnabled(this.simulationSpeed)) {
        this.renderGame();
      }
      this.renderSelected();
    }
  }

  selectedCandidateId(): number | null {
    return this.diagnostics?.candidates[this.selectedIndex]?.id ?? null;
  }

  bestCandidateId(): number | null {
    return this.bestCandidate()?.id ?? null;
  }

  private bestCandidate(): CandidateDiagnostics | undefined {
    return [...(this.diagnostics?.candidates ?? [])].sort(
      (left, right) => right.currentFitness - left.currentFitness || left.id - right.id,
    )[0];
  }

  private renderPrivateFrames(
    active?: readonly boolean[],
    reset = false,
  ): void {
    // These are the controller's only legal observations. Active candidates
    // are redrawn exactly every step. A previously detected inactive Dino has
    // fixed fitness and is intentionally left frozen in the transport atlas.
    this.sharedWorldRenderer.drawWorld(this.engine.worldRenderModel());
    const allCandidates = this.engine.candidateRenderModels();
    this.privateObservationCandidateIds = allCandidates
      .map((_candidate, index) => index)
      .filter((index) => active?.[index] ?? true);
    const compactCandidates = this.privateObservationCandidateIds.map(
      (candidateId) => {
        const candidate = allCandidates[candidateId];
        if (!candidate) {
          throw new Error('Active candidate ordering is incomplete');
        }
        return candidate;
      },
    );
    this.atlasRenderer.drawPrivatePixelAtlas(
      this.sharedWorldCanvas,
      compactCandidates,
      compactCandidates.map(() => true),
      reset,
    );
  }

  private renderGame(): void {
    const world = this.engine.worldRenderModel();
    const candidates = this.engine.candidateRenderModels();
    const alive = Math.max(1, candidates.filter((candidate) => !candidate.crashed).length);
    const styles: CandidateDisplayStyle[] = candidates.map((candidate, index) => ({
      opacity: candidate.crashed
        ? 0.03
        : index === this.selectedIndex
          ? 1
          : Math.min(0.28, 0.10 * Math.sqrt(this.populationSize / alive)),
      verticalOffset: 0,
      accent: index === this.selectedIndex ? '#202124' : '#6f777d',
      showMarker: false,
    }));
    this.publicRenderer.drawPopulation(
      this.sharedWorldCanvas,
      world,
      candidates,
      styles,
    );
  }

  private renderSummary(): void {
    setText('#controller', this.metadata.controllerType);
    setText('#generation', this.metadata.generation + 1);
    setText('#mutation-scale', this.metadata.mutationScale?.toFixed(4) ?? 'n/a');
    setText('#accelerator', this.metadata.accelerator);
    setText('#evidence-label', this.metadata.runKind === 'scientific'
      ? 'SCIENTIFIC RUN'
      : 'ENGINEERING RUN');
    const candidates = this.diagnostics?.candidates ?? [];
    const fitness = candidates.map((candidate) => candidate.currentFitness);
    setText('#alive', `${this.diagnostics?.alive ?? this.populationSize} / ${this.populationSize}`);
    setText('#fitness-mean', mean(fitness).toFixed(1));
    setText('#fitness-best', Math.max(0, ...fitness).toFixed(1));
    this.renderSelected();
  }

  private renderSelected(): void {
    const candidate = this.diagnostics?.candidates[this.selectedIndex];
    setText('#selected-id', candidate?.id ?? this.selectedIndex);
    setText(
      '#selected-status',
      candidate?.alive ? 'alive' : candidate ? 'finished' : 'waiting',
    );
    setText(
      '#selected-action',
      candidate ? ['none', 'jump', 'duck'][candidate.action] ?? 'unknown' : 'none',
    );
    setText('#selected-fitness', candidate?.currentFitness.toFixed(1) ?? '0');
    setText(
      '#selected-scores',
      candidate ? candidate.actionScores.map((score) => score.toFixed(3)).join(' / ') : '0 / 0 / 0',
    );
    setText('#selected-genome', candidate?.genomeHash.slice(0, 12) ?? 'waiting');
    setText('#selected-obstacle', candidate?.obstacleClass ?? 'none detected');
    const sensors = required<HTMLDListElement>('#sensors');
    sensors.replaceChildren();
    if (!candidate) {
      return;
    }
    candidate.sensoryNames.forEach((name, index) => {
      const row = document.createElement('div');
      const term = document.createElement('dt');
      const value = document.createElement('dd');
      term.textContent = name;
      value.textContent = (candidate.sensory[index] ?? 0).toFixed(3);
      row.append(term, value);
      sensors.append(row);
    });
  }

  private renderHistory(history: readonly GenerationHistoryPoint[]): void {
    const canvas = required<HTMLCanvasElement>('#generation-history');
    const context = canvas.getContext('2d');
    if (!context) {
      return;
    }
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, canvas.width, canvas.height);
    if (history.length === 0) {
      return;
    }
    const maximum = Math.max(1, ...history.flatMap((point) => [point.best, point.mean]));
    const draw = (field: 'best' | 'mean', colour: string): void => {
      context.beginPath();
      history.forEach((point, index) => {
        const x = history.length === 1
          ? canvas.width / 2
          : 12 + index / (history.length - 1) * (canvas.width - 24);
        const y = canvas.height - 12 - point[field] / maximum * (canvas.height - 24);
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
      context.strokeStyle = colour;
      context.lineWidth = 2;
      context.stroke();
    };
    draw('best', '#1f5f50');
    draw('mean', '#b46a3c');
    context.font = '12px system-ui, sans-serif';
    context.fillStyle = '#1f5f50';
    context.fillText('Generation best', 14, 16);
    context.fillStyle = '#b46a3c';
    context.fillText('Selected-parent mean', 132, 16);
  }

  private renderSimulationSpeed(): void {
    const input = required<HTMLInputElement>('#simulation-speed');
    input.value = String(this.simulationSpeed);
    required<HTMLOutputElement>('#simulation-speed-value').value =
      `${this.simulationSpeed.toFixed(2)}x`;
  }

  private renderScientificFigures(figures: Readonly<Record<string, string>>): void {
    const container = required<HTMLElement>('#scientific-figures');
    container.replaceChildren();
    Object.entries(figures).forEach(([label, dataUrl]) => {
      const figure = document.createElement('figure');
      const image = document.createElement('img');
      const caption = document.createElement('figcaption');
      const exportLink = document.createElement('a');
      image.src = dataUrl;
      image.alt = label;
      caption.textContent = label;
      exportLink.href = dataUrl;
      exportLink.download = `${label.toLowerCase().replaceAll(' ', '-')}.png`;
      exportLink.textContent = 'Export PNG';
      figure.append(image, caption, exportLink);
      container.append(figure);
    });
  }
}

const view = new PopulationView(populationSize, initialSeed, initialSimulationSpeed);
await view.ready();
let bridge: ArenaBridgeHandle | null = bridgeUrl
  ? installPopulationArenaBridge(view, bridgeUrl)
  : null;

const controlled = [
  'resume',
  'pause',
  'stop',
  'save-best',
  'save-selected',
  'run-settings',
];
if (!bridge) {
  setText('#run-status', 'Preview only. Start Python to run evolution.');
  controlled.forEach((id) => {
    const element = required<HTMLElement>(`#${id}`);
    if (element instanceof HTMLButtonElement || element instanceof HTMLFieldSetElement) {
      element.setAttribute('disabled', '');
    }
  });
}

required<HTMLButtonElement>('#resume').addEventListener('click', () => {
  bridge?.sendControl('next_generation');
  setText('#run-status', 'Resume requested');
});
required<HTMLButtonElement>('#pause').addEventListener('click', () => {
  bridge?.sendControl('pause_now');
  setText('#run-status', 'Pause requested');
});
required<HTMLButtonElement>('#stop').addEventListener('click', () => {
  bridge?.sendControl('stop_after_generation');
  setText('#run-status', 'Will stop after this complete generation');
});
required<HTMLButtonElement>('#previous-dino').addEventListener(
  'click',
  () => view.selectRelative(-1),
);
required<HTMLButtonElement>('#next-dino').addEventListener(
  'click',
  () => view.selectRelative(1),
);
required<HTMLButtonElement>('#best-dino').addEventListener('click', () => view.selectBest());
required<HTMLButtonElement>('#save-selected').addEventListener('click', () => {
  const id = view.selectedCandidateId();
  if (id !== null) bridge?.saveCandidate(id);
});
required<HTMLButtonElement>('#save-best').addEventListener('click', () => {
  const id = view.bestCandidateId();
  if (id !== null) bridge?.saveCandidate(id);
});
const applySimulationSpeed = (speed: number): void => {
  if (!validSimulationSpeed(speed)) {
    return;
  }
  view.setSimulationSpeed(speed);
  bridge?.setSimulationSpeed(speed);
};
required<HTMLInputElement>('#simulation-speed').addEventListener('input', (event) => {
  applySimulationSpeed((event.currentTarget as HTMLInputElement).valueAsNumber);
});
required<HTMLButtonElement>('#reset-simulation-speed').addEventListener('click', () => {
  applySimulationSpeed(1);
});

const mode = required<HTMLSelectElement>('#new-mode');
const updateScientificSettings = (): void => {
  const scientific = mode.value === 'scientific';
  document.querySelectorAll<HTMLElement>('.scientific-setting').forEach((element) => {
    element.hidden = !scientific;
  });
};
mode.addEventListener('change', updateScientificSettings);
updateScientificSettings();

required<HTMLFormElement>('#run-settings').addEventListener('submit', (event) => {
  event.preventDefault();
  const parentCount = required<HTMLInputElement>('#new-parents').valueAsNumber;
  const offspringCount = required<HTMLInputElement>('#new-offspring').valueAsNumber;
  const controller = required<HTMLSelectElement>('#new-controller').value;
  const accelerator = required<HTMLSelectElement>('#new-accelerator').value;
  const runMode = mode.value;
  const generations = required<HTMLInputElement>('#new-generations').valueAsNumber;
  const maxSteps = required<HTMLInputElement>('#new-max-steps').valueAsNumber;
  const mutationScale = required<HTMLInputElement>('#new-mutation-scale').valueAsNumber;
  const evolutionSeed = required<HTMLInputElement>('#new-evolution-seed').valueAsNumber;
  const trainingSeeds = parseSeeds('#new-training-seeds');
  const validationSeeds = parseSeeds('#new-validation-seeds');
  const finalTestSeeds = parseSeeds('#new-test-seeds');
  const studyId = required<HTMLInputElement>('#new-study-id').value.trim();
  const valid = Number.isInteger(parentCount)
    && parentCount >= 1
    && Number.isInteger(offspringCount)
    && offspringCount >= 1
    && parentCount <= 99
    && offspringCount <= 99
    && parentCount + offspringCount <= 100
    && ['reactive', 'proactive'].includes(controller)
    && ['auto', 'cpu', 'cuda'].includes(accelerator)
    && ['continuous', 'scientific'].includes(runMode)
    && Number.isInteger(generations)
    && generations > 0
    && Number.isInteger(maxSteps)
    && maxSteps >= 30
    && mutationScale > 0
    && Number.isSafeInteger(evolutionSeed)
    && trainingSeeds !== null
    && validationSeeds !== null
    && finalTestSeeds !== null
    && (runMode !== 'scientific' || /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(studyId));
  const partitions = [trainingSeeds ?? [], validationSeeds ?? [], finalTestSeeds ?? []];
  const overlap = partitions.some((left, index) => partitions
    .slice(index + 1)
    .some((right) => left.some((seed) => right.includes(seed))));
  if (!valid || (runMode === 'scientific' && overlap)) {
    setText('#run-status', 'Invalid run settings or overlapping scientific seeds');
    return;
  }
  bridge?.startNewRun({
    parentCount,
    offspringCount,
    controllerType: controller as 'reactive' | 'proactive',
    accelerator: accelerator as 'auto' | 'cpu' | 'cuda',
    mode: runMode as 'continuous' | 'scientific',
    generations,
    maxSteps,
    mutationScale,
    evolutionSeed,
    simulationSpeed: view.currentSimulationSpeed(),
    trainingSeeds,
    validationSeeds,
    finalTestSeeds,
    studyId,
  });
  setText('#run-status', 'Starting a new run...');
});

window.addEventListener('beforeunload', () => bridge?.close());
