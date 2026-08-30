import {expect, test} from '@playwright/test';
import type {Page} from '@playwright/test';
import {existsSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

type Action = 0 | 1 | 2;

test('generated scientific report works directly from file', async ({page}) => {
  const report = resolve('..', 'scientific-results.html');
  test.skip(!existsSync(report), 'Generate scientific-results.html before static viewer QA');
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  await page.setViewportSize({width: 900, height: 1000});
  await page.goto(pathToFileURL(report).href, {waitUntil: 'load'});
  await expect(page.getByRole('heading', {name: 'Dino ER scientific evidence'})).toBeVisible();
  await expect(page.locator('#result-status')).toContainText('Model v2 (46ccc63ff585)');
  await expect(page.locator('#panel-rq1 img')).toHaveCount(2);
  await expect.poll(async () => page.locator('#panel-rq1 img').evaluateAll(
    (images) => images.every((image) => (image as HTMLImageElement).naturalWidth > 0),
  )).toBe(true);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);

  const figureLink = page.locator('#panel-rq1 [data-export="figure"]').first();
  const dataLink = page.locator('#panel-rq1 [data-export="figure-data"]').first();
  expect(await figureLink.evaluate((link) => (link as HTMLAnchorElement).href)).toMatch(
    /^data:image\/png;base64,/,
  );
  expect(await dataLink.evaluate((link) => (link as HTMLAnchorElement).href)).toMatch(
    /^data:text\/csv;charset=utf-8;base64,/,
  );
  const [figureDownload] = await Promise.all([page.waitForEvent('download'), figureLink.click()]);
  expect(await figureDownload.failure()).toBeNull();
  expect(figureDownload.suggestedFilename()).toMatch(/\.png$/);
  const [dataDownload] = await Promise.all([page.waitForEvent('download'), dataLink.click()]);
  expect(await dataDownload.failure()).toBeNull();
  expect(dataDownload.suggestedFilename()).toMatch(/\.csv$/);

  for (const tab of ['RQ2: Sensitivity', 'RQ3: Behaviour', 'Final Test']) {
    await page.getByRole('tab', {name: tab}).click();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    if (tab === 'RQ2: Sensitivity') {
      await expect(page.getByRole('heading', {name: 'Controller-specific sensitivity heatmaps'}))
        .toBeVisible();
      await expect(page.locator('#panel-rq2 img')).toHaveCount(2);
      await expect.poll(async () => page.locator('#panel-rq2 img').evaluateAll(
        (images) => images.every((image) => (image as HTMLImageElement).naturalWidth > 0),
      )).toBe(true);
      const heatmapExport = page.locator('#panel-rq2 [data-export="figure"]').nth(1);
      const [heatmapDownload] = await Promise.all([
        page.waitForEvent('download'),
        heatmapExport.click(),
      ]);
      expect(await heatmapDownload.failure()).toBeNull();
      expect(heatmapDownload.suggestedFilename()).toBe('rq2_controller_heatmaps.png');
    }
  }
  await expect(page.locator('#panel-rq3')).toContainText(
    'Evidence complete for the pre-declared protocol.',
  );
  await expect(page.locator('#panel-final')).toContainText(
    'Evidence complete for the pre-declared protocol.',
  );
  expect(errors).toEqual([]);
});

interface MockArenaWindow {
  __arenaMessages: Array<Record<string, unknown>>;
  __arenaSocket: EventTarget;
}

async function installMockSocket(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const messages: Array<Record<string, unknown>> = [];
    class MockWebSocket extends EventTarget {
      static readonly OPEN = 1;
      readonly readyState = MockWebSocket.OPEN;

      constructor(_url: string) {
        super();
        Object.assign(window, {
          __arenaMessages: messages,
          __arenaSocket: this,
        });
        queueMicrotask(() => this.dispatchEvent(new Event('open')));
      }

      send(data: string | ArrayBuffer): void {
        if (typeof data === 'string') {
          messages.push(JSON.parse(data) as Record<string, unknown>);
        }
      }

      close(): void {
        return;
      }
    }
    Object.defineProperty(window, 'WebSocket', {
      configurable: true,
      value: MockWebSocket,
    });
  });
}

async function browserMessages(page: Page): Promise<Array<Record<string, unknown>>> {
  return page.evaluate(() => (
    window as unknown as MockArenaWindow
  ).__arenaMessages);
}

async function send(page: Page, message: Record<string, unknown>): Promise<void> {
  const frameCount = (await browserMessages(page)).filter(
    (item) => item.type === 'frame_batch',
  ).length;
  await page.evaluate((payload) => {
    const socket = (window as unknown as MockArenaWindow).__arenaSocket;
    socket.dispatchEvent(new MessageEvent('message', {
      data: JSON.stringify(payload),
    }));
  }, message);
  if (message.type === 'configure_arena' || message.type === 'action_batch') {
    await expect.poll(async () => (await browserMessages(page)).filter(
      (item) => item.type === 'frame_batch',
    ).length).toBe(frameCount + 1);
  }
}

function metadata(): Record<string, unknown> {
  return {
    controllerType: 'reactive',
    generation: 0,
    mutationScale: 0.3,
    accelerator: 'cpu',
  };
}

async function configure(page: Page, seed = 7): Promise<void> {
  await send(page, {
    type: 'configure_arena',
    requestId: 1,
    seed,
    metadata: metadata(),
  });
}

async function openArena(page: Page, instances = 1, seed = 7): Promise<void> {
  await installMockSocket(page);
  await page.goto(`/batch.html?instances=${instances}&bridge=ws://test&speed=1`);
  await expect.poll(async () => (await browserMessages(page)).some(
    (message) => message.type === 'ready',
  )).toBe(true);
  await configure(page, seed);
}

async function actions(page: Page, sequence: readonly Action[]): Promise<void> {
  for (const [index, action] of sequence.entries()) {
    await send(page, {
      type: 'action_batch',
      requestId: index + 2,
      actions: [action],
      active: [true],
    });
  }
}

test('population size one preserves canonical Dino frames', async ({page}) => {
  await openArena(page);
  const canvas = page.locator('#population-game');

  await actions(page, Array<Action>(8).fill(0));
  await expect(canvas).toHaveScreenshot('dino-running.png');

  await configure(page);
  await actions(page, [1, 1]);
  await expect(canvas).toHaveScreenshot('dino-jumping.png');

  await configure(page);
  await actions(page, [2]);
  await expect(canvas).toHaveScreenshot('dino-ducking.png');
});

test('one page contains the shared game and hidden private pixels for 100 Dinos', async ({page}) => {
  await page.goto('/batch.html?instances=100');
  await expect(page.locator('#population-game')).toBeVisible();
  await expect(page.locator('#private-observations')).toBeHidden();
  await expect(page.locator('#private-observations canvas')).toHaveAttribute(
    'data-candidate-count',
    '100',
  );
  await expect(page.getByRole('heading', {name: 'Evolutionary Chrome Dino'})).toBeVisible();
  await expect(page.getByRole('button', {name: 'Pause'})).toBeDisabled();
  await expect(page.locator('#simulation-speed-value')).toHaveText('1.00x');
});

test('high display speed freezes the public canvas but keeps private pixels current', async ({page}) => {
  await openArena(page);
  const canvas = page.locator('#population-game');
  const before = await canvas.screenshot();
  const privateCanvas = page.locator('#private-observations canvas');
  const privateBefore = await privateCanvas.evaluate((element) => (
    (element as HTMLCanvasElement).toDataURL()
  ));

  await page.getByLabel('Simulation speed').fill('100');
  await actions(page, [1, 1]);

  // Above 1x, public rendering is intentionally skipped. The private atlas is
  // still redrawn because it is the controller's black-box observation.
  expect(await canvas.screenshot()).toEqual(before);
  expect(await privateCanvas.evaluate((element) => (
    (element as HTMLCanvasElement).toDataURL()
  ))).not.toEqual(privateBefore);
});

test('scientific results are rebuilt through the standalone Parquet viewer', async ({page}) => {
  await page.route('**/results-index', async (route) => route.fulfill({
    json: {
      campaign_timestamp: '2026-08-29_10-47-10_CEST',
      model_version: 2,
      model_hash: 'test-model',
      source: 'immutable Parquet generation, episode, and behavioural-trace parts',
      progress: {
        'rq1-main': {completed: 3, planned: 6, failed: 0},
        'rq2-sensitivity': {completed: 2, planned: 30, failed: 1},
        rq3: {completed: 2, planned: 2, failed: 0, not_started: 0},
        final_test: {completed: 2, planned: 2, failed: 0, not_started: 0},
      },
      figures: {
        'RQ1 learning': 'final_figures/rq1_learning.png',
        'RQ1 held-out performance': 'final_figures/rq1_heldout.png',
        'RQ2 OFAT sensitivity': 'final_figures/rq2_ofat_sensitivity.png',
        'RQ3 strategy and causal interventions': 'final_figures/rq3_strategy.png',
        'Final test comparison': 'final_figures/final_test_paired.png',
      },
      tables: {
        'RQ1 summary': 'summary_tables/rq1_summary.parquet',
        'RQ1 learning data': 'summary_tables/rq1_learning.parquet',
        'RQ1 held-out performance data': 'summary_tables/rq1_heldout.parquet',
        'RQ2 OFAT sensitivity data': 'summary_tables/rq2_ofat.parquet',
        'RQ3 strategy and causal interventions data': 'summary_tables/rq3_strategy.parquet',
        'Final test comparison data': 'summary_tables/final_test.parquet',
        'RQ3 causal sensor dependence': 'summary_tables/rq3_causal_sensor_dependence.parquet',
        'RQ2 summary': 'summary_tables/rq2_summary.parquet',
        'Final test summary': 'summary_tables/final_test_summary.parquet',
      },
      rq1_summary: [{
        controller: 'reactive',
        completed_runs: 3,
        planned_runs: 6,
        validation_mean: 123.0,
      }],
      rq2_summary: [{
        parent_count: 20,
        offspring_count: 80,
        mutation_scale: 0.3,
        reactive_runs: 2,
        proactive_runs: 2,
      }],
      rq3_summary: [],
      rq3_causal_summary: [{
        controller: 'reactive',
        intervention: 'constant_first_frame',
        shared_prefix_steps: 20,
        n_matched_worlds: 5,
        paired_trace_seeds: 5,
        mean_action_disagreement_fraction: 0.4,
        discrete_actions_depend_on_visual_input: true,
      }],
      final_test_summary: [{
        summary_type: 'frozen representative', controller: 'reactive',
        selected_run_id: 'run-seed-7', selected_optimizer_seed: 7,
        locked_environment_seeds: 20, mean_final_test_survival: 1200,
        median_final_test_survival: 1100, q1_final_test_survival: 800,
        q3_final_test_survival: 1600, minimum_final_test_survival: 400,
        maximum_final_test_survival: 3600, ceiling_count: 2, ceiling_fraction: 0.1,
        environment_seed_95ci: 200,
      }],
      final_test_locked: false,
      evidence_ready: {rq1: false, rq2: false, rq3: true, final: true},
      rq3_traces: [{
        id: 'trace-a', controller: 'proactive', run_id: 'run-seed-27', world_seed: 404,
        intervention: 'normal', trace_id: 'a', label: 'world seed 404',
        figure: 'final_figures/trace-a.png', data: 'behavioural-traces/trace-a.parquet',
      }, {
        id: 'trace-b', controller: 'proactive', run_id: 'run-seed-27', world_seed: 505,
        intervention: 'normal', trace_id: 'b', label: 'world seed 505',
        figure: 'final_figures/trace-b.png', data: 'behavioural-traces/trace-b.parquet',
      }],
    },
  }));
  await page.route('**/download**', async (route) => {
    const path = new URL(route.request().url()).searchParams.get('path') ?? '';
    const world = new URL(route.request().url()).searchParams.get('world_seed');
    const filename = path.endsWith('.parquet')
      ? path.split('/').at(-1)!.replace('.parquet', `${world ? `_world-${world}` : ''}.csv`) : path.split('/').at(-1)!;
    const body = filename.endsWith('.png')
      ? Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), Buffer.from('fixture')])
      : Buffer.from(world ? `world_seed,action\n${world},2\n` : 'generation,survival\n1,3600\n');
    await route.fulfill({status: 200, body, headers: {
      'Content-Type': filename.endsWith('.png') ? 'image/png' : 'text/csv',
      'Content-Disposition': `attachment; filename="${filename}"`,
      'Content-Length': String(body.length),
    }});
  });

  await page.goto('/results.html');
  await expect(page.getByRole('heading', {name: 'Dino ER scientific evidence'})).toBeVisible();
  await expect(page.locator('#panel-rq1')).toContainText('3/6 complete');
  await expect(page.locator('#panel-rq1')).toContainText(
    'How do reactive and proactive controllers differ in learning and held-out survival?',
  );
  const downloadedHrefs = new Set<string>();
  const downloadVisibleExports = async (): Promise<string[]> => {
    const filenames: string[] = [];
    const links = page.locator('[data-export]:visible');
    for (let index = 0; index < await links.count(); index += 1) {
      const link = links.nth(index);
      const href = await link.getAttribute('href');
      if (!href || downloadedHrefs.has(href)) continue;
      const response = await link.evaluate(async (element) => {
        const anchor = element as HTMLAnchorElement;
        const downloadResponse = await fetch(anchor.href);
        const payload = await downloadResponse.arrayBuffer();
        const objectUrl = URL.createObjectURL(new Blob([payload], {
          type: downloadResponse.headers.get('Content-Type') ?? 'application/octet-stream',
        }));
        anchor.href = objectUrl;
        return {
          contentDisposition: downloadResponse.headers.get('Content-Disposition'),
          length: payload.byteLength,
          status: downloadResponse.status,
        };
      });
      expect(response.status).toBe(200);
      expect(response.length).toBeGreaterThan(8);
      expect(response.contentDisposition).toContain('attachment; filename=');
      const [download] = await Promise.all([page.waitForEvent('download'), link.click()]);
      const filename = download.suggestedFilename();
      expect(await download.failure(), `${filename} download failure`).toBeNull();
      const stream = await download.createReadStream();
      const chunks: Buffer[] = [];
      for await (const chunk of stream) chunks.push(Buffer.from(chunk));
      const payload = Buffer.concat(chunks);
      expect(payload.length).toBeGreaterThan(8);
      if (filename.endsWith('.png')) {
        expect(payload.subarray(0, 8)).toEqual(
          Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
        );
      } else {
        expect(filename).toMatch(/\.csv$/);
        expect(payload.toString('utf8').trim().split('\n').length).toBeGreaterThan(1);
      }
      downloadedHrefs.add(href);
      filenames.push(filename);
    }
    return filenames;
  };
  expect(await downloadVisibleExports()).toEqual(expect.arrayContaining([
    'rq1_learning.png', 'rq1_learning.csv', 'rq1_heldout.png', 'rq1_heldout.csv',
    'rq1_summary.csv',
  ]));

  await page.getByRole('tab', {name: 'RQ2: Sensitivity'}).click();
  await expect(page.locator('#panel-rq2')).toContainText('2/30 complete');
  await expect(page.locator('#panel-rq2')).toContainText('1 failed');
  expect(await downloadVisibleExports()).toEqual(expect.arrayContaining([
    'rq2_ofat_sensitivity.png', 'rq2_ofat.csv', 'rq2_summary.csv',
  ]));
  await page.getByRole('tab', {name: 'RQ3: Behaviour'}).click();
  await expect(page.locator('#panel-rq3')).toContainText(
    'Paired causal sensor-dependence evidence',
  );
  await expect(page.locator('#panel-rq3')).toContainText('Shared Prefix Steps');
  await expect(page.locator('#panel-rq3')).toContainText('Intervention');
  const traceSelect = page.locator('#rq3-trace-select');
  await traceSelect.selectOption('trace-b');
  await expect(page.getByRole('link', {name: 'Export Trace PNG'})).toHaveAttribute(
    'href', /path=final_figures%2Ftrace-b\.png/,
  );
  await expect(page.getByRole('link', {name: 'Export Trace Data'})).toHaveAttribute(
    'href', /world_seed=505/,
  );
  expect(await downloadVisibleExports()).toEqual(expect.arrayContaining([
    'rq3_strategy.png', 'rq3_strategy.csv', 'rq3_causal_sensor_dependence.csv',
    'trace-b.png', 'trace-b_world-505.csv',
  ]));
  await page.getByRole('tab', {name: 'Final Test'}).click();
  await expect(page.locator('#panel-final')).toContainText('2/2 complete');
  await expect(page.locator('#panel-final')).toContainText('0 not started');
  await expect(page.locator('#panel-final')).toContainText(
    'Evidence complete for the pre-declared protocol.',
  );
  await expect(page.locator('#panel-final')).toContainText('Selected Optimizer Seed');
  await expect(page.locator('#panel-final')).toContainText('Ceiling Fraction');
  expect(await downloadVisibleExports()).toEqual(expect.arrayContaining([
    'final_test_paired.png', 'final_test.csv', 'final_test_summary.csv',
  ]));
});

test('minimal controls and diagnostics use the Python bridge', async ({page}) => {
  await openArena(page, 4, 101);
  await send(page, {
    type: 'arena_diagnostics',
    alive: 3,
    candidates: Array.from({length: 4}, (_, index) => ({
      id: 100 + index,
      alive: index !== 1,
      action: index % 3,
      actionScores: [0.1, 0.2, 0.3],
      genomeHash: `genome-${index}`,
      sensory: Array.from({length: 5}, (__, sensorIndex) => (
        index / 10 + sensorIndex / 100
      )),
      sensoryNames: [
        'obstacle_x',
        'obstacle_y',
        'obstacle_width',
        'obstacle_height',
        'dino_y',
      ],
      currentFitness: index,
      obstacleClass: index === 3 ? 'pterodactyl' : 'small_cactus',
    })),
    generationHistory: [
      {generation: 0, best: 1, mean: 0.5},
      {generation: 1, best: 3, mean: 1.5},
    ],
    accelerator: 'cuda:0 | test GPU',
    controlTicksPerSecond: 29.8,
    simulationSpeed: 3,
  });

  await expect(page.locator('#alive')).toHaveText('3 / 4');
  await expect(page.locator('#fitness-best')).toHaveText('3.0');
  await expect(page.locator('#accelerator')).toContainText('cuda:0');
  await page.getByRole('button', {name: 'Best', exact: true}).click();
  await expect(page.locator('#selected-id')).toHaveText('103');
  await expect(page.locator('#sensors dd')).toHaveCount(5);
  await expect(page.locator('#selected-obstacle')).toHaveText('pterodactyl');

  await page.getByLabel('Simulation speed').fill('100');
  await expect(page.locator('#simulation-speed-value')).toHaveText('100.00x');

  await page.getByRole('button', {name: 'Save selected'}).click();
  await page.getByRole('button', {name: 'Save best'}).click();
  await page.getByRole('button', {name: 'Pause'}).click();
  await page.getByRole('button', {name: 'Resume'}).click();
  await page.getByRole('button', {name: 'Stop after generation'}).click();
  await expect.poll(async () => (await browserMessages(page))
    .filter((message) => message.type === 'arena_control')
    .map((message) => message.action)).toEqual([
    'set_simulation_speed',
    'save_candidate',
    'save_candidate',
    'pause_now',
    'next_generation',
    'stop_after_generation',
  ]);
  await expect.poll(async () => (await browserMessages(page))
    .filter((message) => message.action === 'save_candidate')
    .map((message) => message.candidateId)).toEqual([103, 103]);
  await expect.poll(async () => (await browserMessages(page))
    .find((message) => message.action === 'set_simulation_speed')).toMatchObject({
    simulationSpeed: 100,
  });

  await page.getByText('Start a new run').click();
  await page.getByLabel('Parents (mu)').fill('2');
  await page.getByLabel('Offspring (lambda)').fill('4');
  await page.locator('#new-controller').selectOption('proactive');
  await page.locator('#new-accelerator').selectOption('cpu');
  await page.getByRole('button', {name: 'Start new run'}).click();
  await expect.poll(async () => (await browserMessages(page)).at(-1)).toMatchObject({
    type: 'arena_control',
    action: 'start_new_run',
    parentCount: 2,
    offspringCount: 4,
    controllerType: 'proactive',
    accelerator: 'cpu',
    simulationSpeed: 100,
  });
});
