import './results.css';

type Scalar = boolean | number | string | null;
type SummaryRow = Record<string, Scalar>;

interface Progress {
  completed: number;
  failed: number;
  interrupted?: number;
  not_started?: number;
  planned: number;
  running?: number;
}

interface Rq3Trace {
  id: string;
  controller: string;
  run_id: string;
  world_seed: number;
  intervention: string;
  trace_id: string;
  label: string;
  figure: string;
  data: string;
}

interface ResultsIndex {
  campaign_timestamp: string | null;
  model_hash: string;
  model_version: number;
  progress: Record<string, Progress>;
  figures: Record<string, string>;
  tables: Record<string, string>;
  rq1_summary: SummaryRow[];
  rq2_summary: SummaryRow[];
  rq3_summary: SummaryRow[];
  rq3_causal_summary?: SummaryRow[];
  rq3_traces?: Rq3Trace[];
  final_test_summary: SummaryRow[];
  final_test_locked: boolean;
  evidence_ready?: Record<string, boolean>;
  source: string;
}

declare global {
  interface Window {
    __RESULTS_INDEX__?: ResultsIndex;
    __RESULT_ASSETS__?: Record<string, string>;
  }
}

interface SectionDefinition {
  title: string;
  question: string;
  evidence: string;
}

const sections: Record<string, SectionDefinition> = {
  rq1: {
    title: 'RQ1 · Learning and generalisation',
    question: 'How do reactive and proactive controllers differ in learning and held-out survival?',
    evidence: 'Primary evidence: selected-parent survival in control steps across three independent optimiser runs per controller, followed by paired held-out validation.',
  },
  rq2: {
    title: 'RQ2 · Evolution-parameter sensitivity',
    question: 'How sensitive is the architecture difference to the main evolutionary parameters?',
    evidence: 'Primary evidence: the five pre-declared fixed-sigma one-factor-at-a-time conditions under an equal candidate-evaluation budget.',
  },
  rq3: {
    title: 'RQ3 · Strategy and internal state',
    question: 'How do the evolved strategies differ, and does the proactive controller use its internal state?',
    evidence: 'Primary evidence: normal-play action occupancy and paired visual-input and CTRNN-state interventions. Individual traces are supplementary.',
  },
  final: {
    title: 'Locked final test',
    question: 'How do the validation-frozen representatives perform on the untouched final-test seeds?',
    evidence: 'The final test remains locked until model selection is frozen from training and validation evidence.',
  },
};

const groups = Object.fromEntries(
  Object.entries(sections).map(([key, section]) => [key, section.title]),
) as Record<string, string>;

const figureDescriptions: Record<string, {title: string; caption: string}> = {
  'RQ1 learning': {
    title: 'Selected-parent learning across generations',
    caption: 'Thin lines are independent optimiser runs; thick lines are architecture means over the shared completed-generation horizon. Selected-parent mean survival is reported in control steps over the ten training seeds. The top axis reports the equivalent candidate-evaluation budget; the dotted line is the no-action reference. Survival is capped at 3,600 steps.',
  },
  'RQ1 held-out performance': {
    title: 'Matched validation survival and paired difference',
    caption: 'Paired optimizer-seed observations are connected. Panel B reports proactive minus reactive survival and its two-sided 95% Student-t interval across n=3 paired runs.',
  },
  'RQ2 OFAT sensitivity': {
    title: 'Paired architecture contrast: mutation scale and parent--offspring split',
    caption: 'Both panels show raw paired proactive-minus-reactive differences, means, and two-sided 95% Student-t intervals across three optimizer runs. Conditions are one-factor-at-a-time.',
  },
  'RQ2 controller heatmaps': {
    title: 'Controller-specific sensitivity heatmaps',
    caption: 'The x-axis is fixed mutation scale, the y-axis is the parent--offspring split, and colour is mean held-out survival. Reactive and proactive use the same fixed 0–3,600 control-step scale; grey cells were not tested in the pre-declared OFAT design.',
  },
  'RQ3 strategy and causal interventions': {
    title: 'Evolved strategy and causal intervention effects',
    caption: 'Panel A compares normal-play action occupancy. Panel B measures paired survival loss for constant-first-frame and each of the five individual sensor-zero visual ablations. Panel C measures proactive loss when the CTRNN state is reset every control step. Hidden-state change alone does not establish functional memory; state-reset intervention is required. Positive differences indicate removed information was useful; they are causal intervention effects, not correlations.',
  },
  'Final test comparison': {
    title: 'Locked final-test comparison',
    caption: 'Only validation-frozen representatives and untouched final-test seeds enter this figure.',
  },
  'Final test paired': {
    title: 'Paired locked final-test performance',
    caption: 'Each point compares the two validation-frozen representatives on the same locked environment seed. The difference panel reports environment-seed variation, not controller replication.',
  },
};

function assetHref(path: string): string {
  return window.__RESULT_ASSETS__?.[path]
    ?? (window.__RESULTS_INDEX__ ? `scientific-results-assets/${path}` : `/${path}`);
}

function exportHref(path: string, worldSeed?: number): string {
  if (window.__RESULTS_INDEX__) return assetHref(path);
  const world = worldSeed === undefined ? '' : `&world_seed=${encodeURIComponent(String(worldSeed))}`;
  return `/download?path=${encodeURIComponent(path)}${world}`;
}

const visibleColumns: Record<string, string[]> = {
  rq1: [
    'controller', 'completed_runs', 'archive_training_mean',
    'validation_mean', 'training_minus_validation_gap',
  ],
  rq2: [
    'parent_count', 'offspring_count', 'mutation_scale',
    'reactive_validation_mean', 'proactive_validation_mean',
    'proactive_minus_reactive', 'paired_optimizer_runs',
  ],
  rq3: [
    'controller', 'intervention', 'mean_survival', 'survival_95ci', 'episodes',
    'no_action_fraction', 'jump_held_fraction', 'duck_fast_drop_fraction',
  ],
  final: [
    'summary_type', 'controller', 'selected_run_id', 'selected_optimizer_seed',
    'locked_environment_seeds', 'mean_final_test_survival',
    'median_final_test_survival', 'q1_final_test_survival',
    'q3_final_test_survival', 'minimum_final_test_survival',
    'maximum_final_test_survival', 'ceiling_count', 'ceiling_fraction',
    'environment_seed_95ci', 'paired_reactive_minus_proactive_mean',
    'paired_reactive_minus_proactive_median', 'paired_reactive_minus_proactive_95ci',
  ],
};

const compactColumnLabels: Record<string, string> = {
  archive_training_mean: 'Training survival (control steps) ± CI',
  completed_runs: 'Runs complete / planned',
  proactive_minus_reactive: 'Proactive − reactive ± CI',
  proactive_validation_mean: 'Proactive validation ± CI',
  reactive_validation_mean: 'Reactive validation ± CI',
  training_minus_validation_gap: 'Train − validation ± CI',
  validation_mean: 'Validation survival (control steps) ± CI',
  paired_reactive_minus_proactive_mean: 'Paired reactive − proactive mean',
  paired_reactive_minus_proactive_median: 'Paired reactive − proactive median',
  paired_reactive_minus_proactive_95ci: 'Paired reactive − proactive 95% CI',
  summary_type: 'Summary type',
};

const intervalCompanions: Record<string, string> = {
  archive_training_mean: 'archive_training_95ci',
  proactive_minus_reactive: 'proactive_minus_reactive_95ci',
  proactive_validation_mean: 'proactive_validation_95ci',
  reactive_validation_mean: 'reactive_validation_95ci',
  training_minus_validation_gap: 'training_minus_validation_gap_95ci',
  validation_mean: 'validation_95ci',
};

const causalColumns = [
  'controller', 'run_id', 'intervention', 'shared_prefix_steps', 'n_matched_worlds',
  'mean_action_disagreement_fraction', 'mean_output_score_l1_difference_per_step',
  'mean_normal_minus_constant_survival', 'normal_minus_intervention_survival',
  'delta_t_visual', 'delta_t_state', 'paired_trace_seeds',
  'normal_pterodactyl_exposure_episodes',
];

const progressGroups: Record<string, string | undefined> = {
  final: 'final_test',
  rq1: 'rq1-main',
  rq2: 'rq2-sensitivity',
  rq3: 'rq3',
};

const evidenceGroups: Record<string, string> = {
  final: 'final',
  rq1: 'rq1',
  rq2: 'rq2',
  rq3: 'rq3',
};

function required<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Required result-view element is missing: ${selector}`);
  return element;
}

function titleCase(value: string): string {
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function display(value: Scalar): string {
  if (value === null) return 'Not available yet';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(3);
  return String(value);
}

function displayColumn(row: SummaryRow, column: string): string {
  if (column === 'completed_runs') {
    return `${display(row.completed_runs ?? null)} / ${display(row.planned_runs ?? null)}`;
  }
  const intervalColumn = intervalCompanions[column];
  if (intervalColumn) {
    const mean = row[column] ?? null;
    const interval = row[intervalColumn] ?? null;
    if (mean === null || interval === null) return 'Not available yet';
    return `${display(mean)} ± ${display(interval)}`;
  }
  return display(row[column] ?? null);
}

function matchesGroup(label: string, group: string): boolean {
  const normal = label.toLowerCase();
  if (group === 'rq1') return normal.startsWith('rq1');
  if (group === 'rq2') return normal.startsWith('rq2');
  if (group === 'rq3') return normal.startsWith('rq3');
  return normal.startsWith('final');
}

function summaryFor(results: ResultsIndex, group: string): SummaryRow[] {
  if (group === 'rq1') return results.rq1_summary;
  if (group === 'rq2') return results.rq2_summary;
  if (group === 'rq3') return results.rq3_summary;
  return results.final_test_summary;
}

function appendMessage(panel: HTMLElement, text: string): void {
  const message = document.createElement('div');
  message.className = 'alert';
  message.textContent = text;
  panel.append(message);
}

function appendSectionIntroduction(panel: HTMLElement, group: string): void {
  const section = sections[group];
  if (!section) return;
  const header = document.createElement('header');
  header.className = 'mb-4 space-y-2';
  const heading = document.createElement('h2');
  heading.className = 'text-xl font-semibold';
  heading.textContent = section.title;
  const question = document.createElement('p');
  question.className = 'max-w-4xl text-base';
  question.textContent = section.question;
  const evidence = document.createElement('p');
  evidence.className = 'max-w-4xl text-sm opacity-70';
  evidence.textContent = section.evidence;
  header.append(heading, question, evidence);
  panel.append(header);
}

function appendProgress(panel: HTMLElement, progress: Progress | undefined): void {
  if (!progress) return;
  const card = document.createElement('div');
  card.className = 'card card-border bg-base-100';
  const body = document.createElement('div');
  body.className = 'card-body p-4';
  const heading = document.createElement('h2');
  heading.className = 'card-title text-base';
  heading.textContent = 'Evidence progress';
  const count = document.createElement('p');
  const states = [
    `${progress.completed ?? 0}/${progress.planned ?? 0} complete`,
    `${progress.running ?? 0} running`,
    `${progress.failed ?? 0} failed`,
    `${progress.interrupted ?? 0} interrupted`,
    `${progress.not_started ?? 0} not started`,
  ];
  count.textContent = `${states.join('; ')}.`;
  body.append(heading, count);
  card.append(body);
  panel.append(card);
}

function appendEvidenceStatus(panel: HTMLElement, group: string, results: ResultsIndex): void {
  const key = evidenceGroups[group];
  const ready = key ? results.evidence_ready?.[key] : false;
  const message = document.createElement('div');
  message.className = ready ? 'alert alert-success' : 'alert alert-warning';
  message.textContent = ready
    ? 'Evidence complete for the pre-declared protocol.'
    : 'Preliminary evidence only. Partial runs remain visible but must not be interpreted as final results.';
  panel.append(message);
}

function appendFigures(panel: HTMLElement, results: ResultsIndex, group: string): void {
  const entries = Object.entries(results.figures).filter(([label]) => matchesGroup(label, group));
  if (entries.length === 0) {
    appendMessage(panel, 'Not available yet. This view never fabricates missing figures.');
    return;
  }
  const grid = document.createElement('div');
  grid.className =
    group === 'rq1'
      ? 'grid grid-cols-1 gap-4'
      : 'grid grid-cols-[repeat(auto-fit,minmax(300px,1fr))] gap-4';
  for (const [label, path] of entries) {
    const description = figureDescriptions[label] ?? {title: label, caption: ''};
    const card = document.createElement('article');
    card.className = 'card card-border bg-base-100';
    const body = document.createElement('div');
    body.className = 'card-body gap-3 p-4';
    const heading = document.createElement('h2');
    heading.className = 'card-title text-base';
    heading.textContent = description.title;
    const caption = document.createElement('p');
    caption.className = 'max-w-4xl text-sm leading-relaxed opacity-75';
    caption.textContent = description.caption;
    const figure = document.createElement('figure');
    const image = document.createElement('img');
    image.src = assetHref(path);
    image.alt = `${description.title}. ${description.caption}`;
    image.className = 'block h-auto w-full';
    figure.append(image);
    const actions = document.createElement('div');
    actions.className = 'card-actions justify-end';
    const exportLink = document.createElement('a');
    exportLink.className = 'btn btn-sm';
    exportLink.href = exportHref(path);
    exportLink.download = path.split('/').at(-1) ?? 'figure.png';
    exportLink.setAttribute('data-export', 'figure');
    exportLink.textContent = 'Export PNG';
    actions.append(exportLink);
    const figureData = results.tables[`${label} data`];
    if (figureData) {
      const dataLink = document.createElement('a');
      dataLink.className = 'btn btn-sm';
      dataLink.href = exportHref(figureData);
      dataLink.download = figureData.split('/').at(-1)?.replace(/\.parquet$/, '.csv') ?? 'figure-data.csv';
      dataLink.textContent = 'Export Data';
      dataLink.setAttribute('data-export', 'figure-data');
      actions.append(dataLink);
    }
    body.append(heading, caption, figure, actions);
    card.append(body);
    grid.append(card);
  }
  panel.append(grid);
}

function appendRowsTable(
  panel: HTMLElement,
  headingText: string,
  rows: SummaryRow[],
  exportPath?: string,
  requestedColumns?: string[],
): void {
  if (rows.length === 0) return;
  const card = document.createElement('article');
  card.className = 'card card-border bg-base-100';
  const body = document.createElement('div');
  body.className = 'card-body gap-3 p-4';
  const heading = document.createElement('h2');
  heading.className = 'card-title text-base';
  heading.textContent = headingText;
  const wrapper = document.createElement('div');
  wrapper.className = 'overflow-x-auto';
  const table = document.createElement('table');
  table.className = 'table table-zebra table-sm text-xs';
  const availableColumns = new Set(rows.flatMap((row) => Object.keys(row)));
  const columns = requestedColumns?.filter((column) => availableColumns.has(column))
    ?? [...availableColumns];
  const head = document.createElement('thead');
  const headRow = document.createElement('tr');
  columns.forEach((column) => {
    const cell = document.createElement('th');
    cell.className = 'whitespace-normal align-bottom leading-tight';
    cell.textContent = compactColumnLabels[column] ?? titleCase(column);
    headRow.append(cell);
  });
  head.append(headRow);
  const bodyRows = document.createElement('tbody');
  rows.forEach((row) => {
    const tableRow = document.createElement('tr');
    columns.forEach((column) => {
      const cell = document.createElement('td');
      cell.textContent = displayColumn(row, column);
      tableRow.append(cell);
    });
    bodyRows.append(tableRow);
  });
  table.append(head, bodyRows);
  wrapper.append(table);
  const actions = document.createElement('div');
  actions.className = 'card-actions justify-end';
  if (exportPath) {
    const link = document.createElement('a');
    link.className = 'btn btn-sm';
    link.href = exportHref(exportPath);
    link.download = exportPath.split('/').at(-1)?.replace(/\.parquet$/, '.csv') ?? 'summary.csv';
    link.setAttribute('data-export', 'data');
    link.textContent = 'Export Data';
    actions.append(link);
  }
  body.append(heading, wrapper, actions);
  card.append(body);
  panel.append(card);
}

function appendTable(panel: HTMLElement, results: ResultsIndex, group: string): void {
  const summaryLabels: Record<string, string> = {
    rq1: 'RQ1 summary',
    rq2: 'RQ2 summary',
    rq3: 'RQ3 summary',
    final: 'Final test summary',
  };
  const summaryLabel = summaryLabels[group];
  if (group === 'rq3' && (results.rq3_causal_summary?.length ?? 0) > 0) {
    appendRowsTable(
      panel,
      'Paired causal sensor-dependence evidence',
      results.rq3_causal_summary ?? [],
      results.tables['RQ3 causal sensor dependence'],
      causalColumns,
    );
  }
  appendRowsTable(
    panel,
    'Numerical summary',
    summaryFor(results, group),
    summaryLabel ? results.tables[summaryLabel] : undefined,
    visibleColumns[group],
  );
}

function appendRq3TraceSelector(panel: HTMLElement, results: ResultsIndex): void {
  const traces = results.rq3_traces ?? [];
  if (traces.length === 0) return;
  const card = document.createElement('article');
  card.className = 'card card-border bg-base-100';
  const body = document.createElement('div');
  body.className = 'card-body gap-3 p-4';
  const heading = document.createElement('h2');
  heading.className = 'card-title text-base';
  heading.textContent = 'Supplementary episode inspection';
  const description = document.createElement('p');
  description.className = 'max-w-4xl text-sm leading-relaxed opacity-75';
  description.textContent = 'Use this trace to inspect one archived policy and world seed. It illustrates sensor, output, action, and recurrent-state timing; it is not independent evidence and is not the primary RQ3 result.';
  const label = document.createElement('label');
  label.className = 'grid max-w-2xl gap-1 text-sm font-medium';
  label.htmlFor = 'rq3-trace-select';
  label.textContent = 'Archived episode';
  const select = document.createElement('select');
  select.id = 'rq3-trace-select';
  select.className = 'select w-full font-normal';
  traces.forEach((trace) => {
    const option = document.createElement('option');
    option.value = trace.id;
    option.textContent = trace.label;
    select.append(option);
  });
  label.append(select);
  const figure = document.createElement('figure');
  const image = document.createElement('img');
  image.className = 'block h-auto w-full';
  figure.append(image);
  const metadata = document.createElement('p');
  metadata.className = 'break-all text-xs opacity-65';
  const actions = document.createElement('div');
  actions.className = 'card-actions justify-end';
  const png = document.createElement('a');
  png.className = 'btn btn-sm';
  png.textContent = 'Export Trace PNG';
  const data = document.createElement('a');
  data.className = 'btn btn-sm';
  data.textContent = 'Export Trace Data';
  actions.append(png, data);

  const renderTrace = (): void => {
    const trace = traces.find((entry) => entry.id === select.value);
    if (!trace) return;
    image.src = assetHref(trace.figure);
    image.alt = trace.label;
    metadata.textContent = `Run: ${trace.run_id} · intervention: ${trace.intervention} · world seed: ${trace.world_seed}`;
    png.href = exportHref(trace.figure);
    png.download = trace.figure.split('/').at(-1) ?? 'rq3-trace.png';
    data.href = exportHref(trace.data, trace.world_seed);
    data.download = window.__RESULTS_INDEX__
      ? trace.data.split('/').at(-1) ?? 'rq3-trace.csv'
      : `${trace.data.split('/').at(-1)?.replace(/\.parquet$/, '') ?? 'rq3-trace'}_world-${trace.world_seed}.csv`;
    png.setAttribute('data-export', 'trace-figure');
    data.setAttribute('data-export', 'trace-data');
  };
  select.addEventListener('change', renderTrace);
  renderTrace();
  body.append(heading, description, label, metadata, figure, actions);
  card.append(body);
  panel.append(card);
}

function render(results: ResultsIndex): void {
  const status = required<HTMLElement>('#result-status');
  const campaign = results.campaign_timestamp ? `; campaign ${results.campaign_timestamp}` : '';
  status.textContent = `Model v${results.model_version} (${results.model_hash})${campaign}; ${results.source}.`;
  for (const group of Object.keys(groups)) {
    const panel = required<HTMLElement>(`#panel-${group}`);
    panel.replaceChildren();
    appendSectionIntroduction(panel, group);
    if (group === 'final' && results.final_test_locked) {
      appendMessage(panel, 'LOCKED FINAL TEST — no final-test Parquet evidence is available yet.');
    } else if (group === 'final' && !results.evidence_ready?.final) {
      appendMessage(panel, 'Final-test selection is frozen; the authorised final evidence is incomplete.');
    }
    appendProgress(panel, results.progress[progressGroups[group] ?? '']);
    appendEvidenceStatus(panel, group, results);
    appendFigures(panel, results, group);
    appendTable(panel, results, group);
    if (group === 'rq3') appendRq3TraceSelector(panel, results);
  }
}

function selectTab(group: string): void {
  for (const name of Object.keys(groups)) {
    const tab = required<HTMLButtonElement>(`#tab-${name}`);
    const panel = required<HTMLElement>(`#panel-${name}`);
    const selected = name === group;
    tab.classList.toggle('tab-active', selected);
    tab.setAttribute('aria-selected', String(selected));
    panel.hidden = !selected;
  }
}

async function refresh(): Promise<void> {
  const status = required<HTMLElement>('#result-status');
  status.textContent = 'Reading immutable Parquet results…';
  try {
    if (window.__RESULTS_INDEX__) {
      render(window.__RESULTS_INDEX__);
      return;
    }
    const response = await fetch('/results-index', {cache: 'no-store'});
    if (!response.ok) throw new Error(`Results request failed (${response.status})`);
    render(await response.json() as ResultsIndex);
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : String(error);
  }
}

for (const group of Object.keys(groups)) {
  required<HTMLButtonElement>(`#tab-${group}`).addEventListener('click', () => selectTab(group));
}
required<HTMLButtonElement>('#refresh-results').addEventListener('click', () => void refresh());
await refresh();
