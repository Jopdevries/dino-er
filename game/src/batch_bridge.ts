type Action = 0 | 1 | 2;

export const MIN_SIMULATION_SPEED = 0.25;
export const MAX_SIMULATION_SPEED = 100;
/** One controller decision per original Chromium 60-Hz physics frame. */
export const BASE_CONTROL_HZ = 60;

export function validSimulationSpeed(value: number): boolean {
  return Number.isFinite(value)
    && value >= MIN_SIMULATION_SPEED
    && value <= MAX_SIMULATION_SPEED;
}

export function controlIntervalMilliseconds(speed: number): number {
  if (!validSimulationSpeed(speed)) {
    throw new Error('Simulation speed must be between 0.25x and 100x');
  }
  return 1000 / (BASE_CONTROL_HZ * speed);
}

export function publicRenderingEnabled(speed: number): boolean {
  if (!validSimulationSpeed(speed)) {
    throw new Error('Simulation speed must be between 0.25x and 100x');
  }
  return speed <= 1;
}

export interface ArenaRunMetadata {
  controllerType: 'proactive' | 'reactive' | 'replay';
  generation: number;
  mutationScale: number | null;
  accelerator: string;
  runKind?: 'engineering' | 'scientific';
}

export interface CandidateDiagnostics {
  id: number;
  alive: boolean;
  action: Action;
  actionScores: [number, number, number];
  genomeHash: string;
  sensory: number[];
  sensoryNames: string[];
  currentFitness: number;
  obstacleClass: 'large_cactus' | 'pterodactyl' | 'small_cactus' | null;
}

export interface GenerationHistoryPoint {
  generation: number;
  best: number;
  mean: number;
  bestEver?: number;
  selectionMean?: number;
}

export interface ArenaDiagnostics {
  type: 'arena_diagnostics';
  alive: number;
  candidates: CandidateDiagnostics[];
  generationHistory: GenerationHistoryPoint[];
  accelerator: string;
  controlTicksPerSecond: number;
  simulationSpeed: number;
}

export interface ArenaStatusMessage {
  type: 'arena_status';
  phase:
    | 'candidate_saved'
    | 'evaluating'
    | 'generation_complete'
    | 'paused'
    | 'preparing'
    | 'replay_complete'
    | 'restarting'
    | 'run_complete'
    | 'stopped';
  message: string;
  generationHistory?: GenerationHistoryPoint[];
  totalEvaluations?: number;
  mutationScale?: number;
  savedModelPath?: string;
  scientificOutputPath?: string;
  aggregateOutputPath?: string;
  resultDirectory?: string;
  scientificFigures?: Record<string, string>;
}

export interface ConfigureArenaMessage {
  type: 'configure_arena';
  requestId: number;
  seed: number;
  metadata: ArenaRunMetadata;
}

interface ActionArenaMessage {
  type: 'action_batch';
  requestId: number;
  actions: Action[];
  active: boolean[];
}

type ArenaMessage =
  | ActionArenaMessage
  | ArenaDiagnostics
  | ArenaStatusMessage
  | ConfigureArenaMessage;

export interface ArenaBridgeView {
  readonly populationSize: number;
  readonly privateObservationAtlas: HTMLCanvasElement;
  readonly privateObservationCandidateIds: readonly number[];
  configure(message: ConfigureArenaMessage): Promise<void>;
  step(actions: readonly Action[], active: readonly boolean[]): Promise<void>;
  updateDiagnostics(message: ArenaDiagnostics): void;
  updateStatus(message: ArenaStatusMessage): void;
}

export interface NewRunSettings {
  parentCount: number;
  offspringCount: number;
  controllerType: 'proactive' | 'reactive';
  accelerator: 'auto' | 'cpu' | 'cuda';
  mode: 'continuous' | 'scientific';
  generations: number;
  maxSteps: number;
  mutationScale: number;
  evolutionSeed: number;
  simulationSpeed: number;
  trainingSeeds: number[];
  validationSeeds: number[];
  finalTestSeeds: number[];
  studyId: string;
}

export type ArenaControlAction =
  | 'cancel'
  | 'next_generation'
  | 'pause_now'
  | 'save_candidate'
  | 'set_simulation_speed'
  | 'start_new_run'
  | 'stop_after_generation';

export interface ArenaBridgeHandle {
  close(): void;
  sendControl(action: ArenaControlAction): void;
  saveCandidate(candidateId: number): void;
  setSimulationSpeed(speed: number): void;
  startNewRun(settings: NewRunSettings): void;
}

interface EncodedGrayscale {
  data: ArrayBuffer;
  current: Uint8Array;
  encoding:
    | 'zlib_grayscale_u8'
    | 'zlib_xor_grayscale_u8'
    | 'sparse_xor_runs_u8';
}

async function canvasEncodedGrayscale(
  canvas: HTMLCanvasElement,
  previous: Uint8Array | null,
  forceKeyFrame: boolean,
): Promise<EncodedGrayscale> {
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Private pixel atlas Canvas is unavailable');
  }
  const rgba = context.getImageData(
    0,
    0,
    canvas.width,
    canvas.height,
  ).data;
  const pixels = new Uint32Array(
    rgba.buffer,
    rgba.byteOffset,
    rgba.byteLength / 4,
  );
  const grayscale = new Uint8Array(pixels.length);
  for (let index = 0; index < pixels.length; index += 1) {
    grayscale[index] = (pixels[index] ?? 0) & 0xff;
  }
  if (!forceKeyFrame && previous && previous.length === grayscale.length) {
    // The decoder reconstructs the same current pixels before perception.
    // Previous pixels exist only as a transport codec state, never as a
    // controller input, feature, reward, or game-state read.
    {
      const starts: number[] = [];
      const lengths: number[] = [];
      let byteLength = 0;
      let index = 0;
      while (index < grayscale.length) {
        if ((grayscale[index] ?? 0) === (previous[index] ?? 0)) {
          index += 1;
          continue;
        }
        const start = index;
        let length = 0;
        while (
          index < grayscale.length
          && length < 0xffff
          && (grayscale[index] ?? 0) !== (previous[index] ?? 0)
        ) {
          index += 1;
          length += 1;
        }
        starts.push(start);
        lengths.push(length);
        byteLength += 6 + length;
      }
      if (byteLength <= grayscale.length / 4) {
        const payload = new Uint8Array(byteLength);
        const view = new DataView(payload.buffer);
        let offset = 0;
        starts.forEach((start, runIndex) => {
          const length = lengths[runIndex] ?? 0;
          view.setUint32(offset, start, true);
          view.setUint16(offset + 4, length, true);
          offset += 6;
          for (let valueIndex = 0; valueIndex < length; valueIndex += 1) {
            const sourceIndex = start + valueIndex;
            payload[offset + valueIndex] =
              (grayscale[sourceIndex] ?? 0) ^ (previous[sourceIndex] ?? 0);
          }
          offset += length;
        });
        return {
          data: payload.buffer,
          current: grayscale,
          encoding: 'sparse_xor_runs_u8',
        };
      }
    }
    const delta = new Uint8Array(grayscale.length);
    for (let deltaIndex = 0; deltaIndex < grayscale.length; deltaIndex += 1) {
      delta[deltaIndex] =
        (grayscale[deltaIndex] ?? 0) ^ (previous[deltaIndex] ?? 0);
    }
    const compressed = new Blob([delta])
      .stream()
      .pipeThrough(new CompressionStream('deflate'));
    return {
      data: await new Response(compressed).arrayBuffer(),
      current: grayscale,
      encoding: 'zlib_xor_grayscale_u8',
    };
  }
  const compressed = new Blob([grayscale])
    .stream()
    .pipeThrough(new CompressionStream('deflate'));
  return {
    data: await new Response(compressed).arrayBuffer(),
    current: grayscale,
    encoding: 'zlib_grayscale_u8',
  };
}

export function installPopulationArenaBridge(
  view: ArenaBridgeView,
  url: string,
): ArenaBridgeHandle {
  const socket = new WebSocket(url);
  let previousGrayscale: Uint8Array | null = null;
  const sendPrivateFrames = async (
    requestId: number,
    forceKeyFrame = false,
  ): Promise<void> => {
    const encoded = await canvasEncodedGrayscale(
      view.privateObservationAtlas,
      previousGrayscale,
      forceKeyFrame,
    );
    previousGrayscale = encoded.current;
    const candidateIds = [...view.privateObservationCandidateIds];
    const compactCount = candidateIds.length;
    if (
      compactCount < 1
      || new Set(candidateIds).size !== compactCount
      || candidateIds.some(
        (candidateId) => !Number.isInteger(candidateId)
          || candidateId < 0
          || candidateId >= view.populationSize,
      )
    ) {
      throw new Error('Private pixel atlas candidate ordering is invalid');
    }
    const dinoColumns = Math.min(20, compactCount);
    const dinoRows = Math.ceil(compactCount / dinoColumns);
    const gameOverTextColumns = Math.min(6, compactCount);
    const gameOverTextPatchY = 150 + dinoRows * 150;
    const gameOverTextRows = Math.ceil(
      compactCount / gameOverTextColumns,
    );
    const restartColumns = Math.min(33, compactCount);
    const restartPatchY = gameOverTextPatchY + gameOverTextRows * 11;
    const restartRows = Math.ceil(compactCount / restartColumns);
    socket.send(JSON.stringify({
      type: 'frame_batch',
      requestId,
      width: 600,
      height: 150,
      count: compactCount,
      populationSize: view.populationSize,
      candidateIds,
      atlasLayout: 'shared_world_active_private_pixel_patches_v4',
      atlasWidth: Math.max(
        600,
        dinoColumns * 60,
        gameOverTextColumns * 191,
        restartColumns * 36,
      ),
      atlasHeight: restartPatchY + restartRows * 32,
      dinoColumns,
      gameOverTextColumns,
      gameOverTextPatchY,
      restartColumns,
      restartPatchY,
      observation: 'private_clean_candidate_frames',
      encoding: encoded.encoding,
      rawByteLength:
        view.privateObservationAtlas.width *
        view.privateObservationAtlas.height,
      byteLength: encoded.data.byteLength,
    }));
    socket.send(encoded.data);
  };

  socket.addEventListener('open', () => {
    socket.send(JSON.stringify({
      type: 'ready',
      protocol: 6,
      instances: view.populationSize,
      presentation: 'one_shared_game_multiple_dinos',
      observation: 'private_clean_candidate_frames',
    }));
  });

  let messageQueue = Promise.resolve();
  socket.addEventListener('message', (event) => {
    messageQueue = messageQueue.then(async () => {
      let message: ArenaMessage;
      try {
        message = JSON.parse(String(event.data)) as ArenaMessage;
      } catch {
        socket.send(JSON.stringify({
          type: 'transport_error',
          error: 'Invalid JSON arena command',
        }));
        return;
      }
      if (
        message.type === 'configure_arena'
      ) {
        await view.configure(message);
        await sendPrivateFrames(message.requestId, true);
        return;
      }
      if (
        message.type === 'action_batch' &&
        message.actions.length === view.populationSize &&
        message.active.length === view.populationSize &&
        message.actions.every((action) => [0, 1, 2].includes(action))
      ) {
        await view.step(message.actions, message.active);
        await sendPrivateFrames(message.requestId);
        return;
      }
      if (message.type === 'arena_diagnostics') {
        view.updateDiagnostics(message);
        return;
      }
      if (message.type === 'arena_status') {
        view.updateStatus(message);
        return;
      }
      socket.send(JSON.stringify({
        type: 'transport_error',
        error: 'Unsupported arena command',
      }));
    }).catch((error: unknown) => {
      socket.send(JSON.stringify({
        type: 'transport_error',
        error: error instanceof Error ? error.message : String(error),
      }));
    });
  });

  const send = (message: object): void => {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    }
  };
  return {
    close: () => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.close(1000, 'shared game closed');
      }
    },
    sendControl: (action) => send({
      type: 'arena_control',
      action,
    }),
    saveCandidate: (candidateId) => send({
      type: 'arena_control',
      action: 'save_candidate',
      candidateId,
    }),
    setSimulationSpeed: (speed) => {
      if (!validSimulationSpeed(speed)) {
        throw new Error('Simulation speed must be between 0.25x and 100x');
      }
      send({
        type: 'arena_control',
        action: 'set_simulation_speed',
        simulationSpeed: speed,
      });
    },
    startNewRun: (settings) => send({
      type: 'arena_control',
      action: 'start_new_run',
      ...settings,
    }),
  };
}
