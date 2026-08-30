// Copyright 2025 The Chromium Authors
// Use of this source code is governed by a BSD-style license recorded in
// THIRD_PARTY_NOTICES.txt.
//
// Faithful local adaptation of Chromium's dino_game at commit
// 1ccb91e11f09fbbdec4f8f754d0e2f7d28246660. Browser integration, fixed-step
// scheduling and seeded randomness are project adaptations.

export type Action = 0 | 1 | 2;
export type ObstacleKind = 'cactusSmall' | 'cactusLarge' | 'pterodactyl';

export interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface ObstacleDefinition {
  kind: ObstacleKind;
  width: number;
  height: number;
  yPositions: readonly number[];
  multipleSpeed: number;
  minGap: number;
  minSpeed: number;
  speedOffset: number;
  collisionBoxes: readonly Box[];
}

export interface RenderObstacle {
  kind: ObstacleKind;
  x: number;
  y: number;
  width: number;
  height: number;
  size: number;
  animationFrame: number;
}

export interface RenderCloud {
  x: number;
  y: number;
}

export interface RenderStar {
  x: number;
  y: number;
  sprite: number;
}

export interface WorldRenderModel {
  tick: number;
  speed: number;
  distance: number;
  score: number;
  highScore: number;
  scoreVisible: boolean;
  inverted: boolean;
  nightPhase: number;
  nightOpacity: number;
  nightMoonX: number;
  nightStarsVisible: boolean;
  nightStars: readonly RenderStar[];
  obstacles: readonly RenderObstacle[];
  clouds: readonly RenderCloud[];
  horizonX: readonly [number, number];
  horizonSource: readonly [number, number];
}

export interface PlayerRenderModel {
  dinoX: number;
  dinoY: number;
  grounded: boolean;
  ducking: boolean;
  crashed: boolean;
  crashMs: number;
}

export interface RenderModel extends WorldRenderModel, PlayerRenderModel {}

export interface PlayerPhysicsSnapshot extends PlayerRenderModel {
  jumpVelocity: number;
  speedDrop: boolean;
  reachedMinHeight: boolean;
}

export interface PopulationStepResult {
  newlyCrashed: readonly number[];
  alive: number;
  terminated: number;
}

interface Obstacle extends RenderObstacle {
  definition: ObstacleDefinition;
  gap: number;
  speedOffset: number;
  collisionBoxes: Box[];
  followingCreated: boolean;
  animationTimer: number;
}

interface Cloud extends RenderCloud {
  gap: number;
}

const FPS = 60;
const FRAME_MS = 1000 / FPS;
// The controller observes and acts at the same 60-Hz cadence as Chromium's
// fixed physics.  A control step therefore advances exactly one physics frame.
const SUBSTEPS_PER_CONTROL = 1;
const CANVAS_WIDTH = 600;
const CANVAS_HEIGHT = 150;
const DINO_X = 50;
const DINO_WIDTH = 44;
const DINO_HEIGHT = 47;
const GROUND_Y = CANVAS_HEIGHT - DINO_HEIGHT - 10;
const DISTANCE_COEFFICIENT = 0.025;
const INVERT_DISTANCE = 700;
const INVERT_FADE_DURATION_MS = 12_000;
const SCORE_FLASH_DURATION_MS = 250;
const SCORE_FLASH_ITERATIONS = 3;
// The pinned Horizon source passes dimensions.height (150) to NightMode as
// its container width. Preserve that observable implementation detail.
const NIGHT_CONTAINER_WIDTH = CANVAS_HEIGHT;

const DINO_COLLISION_BOXES: Record<'running' | 'ducking', readonly Box[]> = {
  ducking: [{x: 1, y: 18, width: 55, height: 25}],
  running: [
    {x: 22, y: 0, width: 17, height: 16},
    {x: 1, y: 18, width: 30, height: 9},
    {x: 10, y: 35, width: 14, height: 8},
    {x: 1, y: 24, width: 29, height: 5},
    {x: 5, y: 30, width: 21, height: 4},
    {x: 9, y: 34, width: 15, height: 4},
  ],
};

const OBSTACLE_DEFINITIONS: readonly ObstacleDefinition[] = [
  {
    kind: 'cactusSmall',
    width: 17,
    height: 35,
    yPositions: [105],
    multipleSpeed: 4,
    minGap: 120,
    minSpeed: 0,
    speedOffset: 0,
    collisionBoxes: [
      {x: 0, y: 7, width: 5, height: 27},
      {x: 4, y: 0, width: 6, height: 34},
      {x: 10, y: 4, width: 7, height: 14},
    ],
  },
  {
    kind: 'cactusLarge',
    width: 25,
    height: 50,
    yPositions: [90],
    multipleSpeed: 7,
    minGap: 120,
    minSpeed: 0,
    speedOffset: 0,
    collisionBoxes: [
      {x: 0, y: 12, width: 7, height: 38},
      {x: 8, y: 0, width: 7, height: 49},
      {x: 13, y: 10, width: 10, height: 38},
    ],
  },
  {
    kind: 'pterodactyl',
    width: 46,
    height: 40,
    yPositions: [100, 75, 50],
    multipleSpeed: 999,
    minGap: 150,
    minSpeed: 8.5,
    speedOffset: 0.8,
    collisionBoxes: [
      {x: 15, y: 15, width: 16, height: 5},
      {x: 18, y: 21, width: 24, height: 6},
      {x: 2, y: 14, width: 4, height: 3},
      {x: 6, y: 10, width: 4, height: 7},
      {x: 10, y: 8, width: 6, height: 9},
    ],
  },
];

class SeededRandom {
  private state = 0;

  reset(seed: number): void {
    this.state = seed >>> 0;
  }

  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let value = this.state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  }

  integer(minimum: number, maximum: number): number {
    return Math.floor(this.next() * (maximum - minimum + 1)) + minimum;
  }
}

function intersects(left: Box, right: Box): boolean {
  return left.x < right.x + right.width &&
      left.x + left.width > right.x &&
      left.y < right.y + right.height &&
      left.y + left.height > right.y;
}

class DinoWorld {
  private readonly random = new SeededRandom();
  private tick = 0;
  private runningMs = 0;
  private speed = 6;
  private distance = 0;
  private highScore = 0;
  private scoreVisible = true;
  private achievement = false;
  private flashTimer = 0;
  private flashIterations = 0;
  private inverted = false;
  private invertTimer = 0;
  private nightPhase = 0;
  private nightOpacity = 0;
  private nightMoonX = 0;
  private nightStarsVisible = false;
  private nightStars: RenderStar[] = [];
  private obstacles: Obstacle[] = [];
  private obstacleHistory: ObstacleKind[] = [];
  private clouds: Cloud[] = [];
  private horizonX: [number, number] = [0, 600];
  private horizonSource: [number, number] = [0, 600];

  constructor(seed = 0) {
    this.reset(seed);
  }

  reset(seed: number, preserveHighScore = false): void {
    this.random.reset(seed);
    this.tick = 0;
    this.runningMs = 0;
    this.speed = 6;
    this.distance = 0;
    if (!preserveHighScore) {
      this.highScore = 0;
    }
    this.scoreVisible = true;
    this.achievement = false;
    this.flashTimer = 0;
    this.flashIterations = 0;
    this.inverted = false;
    this.invertTimer = 0;
    this.nightPhase = 0;
    this.nightOpacity = 0;
    this.nightMoonX = 0;
    this.nightStarsVisible = false;
    this.nightStars = [];
    this.obstacles = [];
    this.obstacleHistory = [];
    this.clouds = [];
    this.horizonX = [0, 600];
    // Chromium starts with the two fixed, adjacent horizon crops. A random
    // crop is selected only when a segment wraps around.
    this.horizonSource = [0, 600];
    this.addCloud();
    // NightMode is constructed after the first cloud and immediately places
    // two stars. Each star consumes an x and y draw.
    this.placeStars();
    // The original Trex constructor schedules its first blink after the
    // Horizon creates the initial cloud and NightMode. It consumes one
    // Math.random() draw before the first obstacle is selected.
    this.random.next();
  }

  get currentSpeed(): number {
    return this.speed;
  }

  beginFrame(): boolean {
    this.runningMs += FRAME_MS;
    const obstaclesActive = this.runningMs > 3000;
    this.advanceWorld(obstaclesActive);
    this.tick += 1;
    return obstaclesActive;
  }

  finishFrame(): void {
    this.distance += this.speed;
    this.updateScore();
    this.updateInversion();
    if (this.speed < 13) {
      this.speed += 0.001;
    }
  }

  recordCrash(): void {
    this.highScore = Math.max(this.highScore, this.actualScore());
  }

  collides(player: PlayerRenderModel): boolean {
    return this.hasCollision(player);
  }

  renderModel(): WorldRenderModel {
    return {
      tick: this.tick,
      speed: this.speed,
      distance: this.distance,
      score: this.actualScore(),
      highScore: this.highScore,
      scoreVisible: this.scoreVisible,
      inverted: this.inverted,
      nightPhase: this.nightPhase,
      nightOpacity: this.nightOpacity,
      nightMoonX: this.nightMoonX,
      nightStarsVisible: this.nightStarsVisible,
      nightStars: this.nightStars,
      obstacles: this.obstacles,
      clouds: this.clouds,
      horizonX: this.horizonX,
      horizonSource: this.horizonSource,
    };
  }

  private advanceWorld(obstaclesActive: boolean): void {
    const increment = Math.floor(this.speed);
    const active = this.horizonX[0] <= 0 ? 0 : 1;
    const other = 1 - active;
    this.horizonX[active] -= increment;
    this.horizonX[other] = this.horizonX[active] + 600;
    if (this.horizonX[active] <= -600) {
      this.horizonX[active] += 1200;
      this.horizonX[other] = this.horizonX[active] - 600;
      this.horizonSource[active] = this.random.integer(0, 1) * 600;
    }

    // Chromium updates NightMode before clouds and obstacles. In daylight the
    // zero-opacity branch still re-places both stars every frame, consuming
    // four Math.random() values that affect the exact obstacle sequence.
    this.advanceNight();

    const cloudIncrement = Math.ceil(0.2 / 1000 * FRAME_MS * this.speed);
    const cloudCount = this.clouds.length;
    for (const cloud of this.clouds) {
      cloud.x -= cloudIncrement;
    }
    const lastCloud = this.clouds.at(-1);
    if (cloudCount === 0) {
      this.addCloud();
    } else if (lastCloud &&
      cloudCount < 6 &&
      CANVAS_WIDTH - lastCloud.x > lastCloud.gap &&
      this.random.next() < 0.5
    ) {
      this.addCloud();
    }
    this.clouds = this.clouds.filter((cloud) => cloud.x + 46 > 0);

    if (!obstaclesActive) {
      return;
    }
    for (const obstacle of this.obstacles) {
      obstacle.x -= Math.floor(this.speed + obstacle.speedOffset);
      if (obstacle.kind === 'pterodactyl') {
        obstacle.animationTimer += FRAME_MS;
        if (obstacle.animationTimer >= 1000 / 6) {
          obstacle.animationFrame = 1 - obstacle.animationFrame;
          obstacle.animationTimer = 0;
        }
      }
    }
    this.obstacles = this.obstacles.filter(
      (obstacle) => obstacle.x + obstacle.width > 0,
    );
    const lastObstacle = this.obstacles.at(-1);
    if (!lastObstacle) {
      this.addObstacle();
    } else if (
      !lastObstacle.followingCreated &&
      lastObstacle.x + lastObstacle.width + lastObstacle.gap < CANVAS_WIDTH
    ) {
      this.addObstacle();
      lastObstacle.followingCreated = true;
    }
  }

  private addCloud(): void {
    // Chromium's Cloud constructor draws gap first, then vertical position.
    const gap = this.random.integer(100, 400);
    const y = this.random.integer(30, 71);
    this.clouds.push({
      x: CANVAS_WIDTH,
      y,
      gap,
    });
  }

  private placeStars(): void {
    const segmentSize = Math.round(NIGHT_CONTAINER_WIDTH / 2);
    this.nightStars = [0, 1].map((segment) => ({
      x: this.random.integer(
        segmentSize * segment,
        segmentSize * (segment + 1),
      ),
      y: this.random.integer(0, 70),
      sprite: segment,
    }));
  }

  private advanceNight(): void {
    if (this.inverted && this.nightOpacity === 0) {
      this.nightPhase = (this.nightPhase + 1) % 7;
    }

    if (this.inverted && (this.nightOpacity < 1 || this.nightOpacity === 0)) {
      this.nightOpacity += 0.035;
    } else {
      if (this.nightOpacity > 0) {
        this.nightOpacity -= 0.035;
      }
    }

    if (this.nightOpacity > 0) {
      this.nightMoonX = this.updateNightX(this.nightMoonX, 0.25);
      if (this.nightStarsVisible) {
        this.nightStars = this.nightStars.map((star) => ({
          ...star,
          x: this.updateNightX(star.x, 0.3),
        }));
      }
    } else {
      this.nightOpacity = 0;
      this.placeStars();
    }
    this.nightStarsVisible = true;
  }

  private updateNightX(position: number, speed: number): number {
    if (position < -20) {
      return NIGHT_CONTAINER_WIDTH;
    }
    return position - speed;
  }

  private actualScore(): number {
    return Math.round(Math.ceil(this.distance) * DISTANCE_COEFFICIENT);
  }

  private updateScore(): void {
    if (this.achievement) {
      this.flashTimer += FRAME_MS;
      if (this.flashTimer < SCORE_FLASH_DURATION_MS) {
        this.scoreVisible = false;
      } else if (this.flashTimer > SCORE_FLASH_DURATION_MS * 2) {
        this.flashTimer = 0;
        this.flashIterations += 1;
        this.scoreVisible = true;
        if (this.flashIterations > SCORE_FLASH_ITERATIONS) {
          this.achievement = false;
          this.flashIterations = 0;
        }
      } else {
        this.scoreVisible = true;
      }
      return;
    }

    const score = this.actualScore();
    if (score > 0 && score % 100 === 0) {
      this.achievement = true;
      this.flashTimer = 0;
      this.flashIterations = 0;
    }
    this.scoreVisible = true;
  }

  private updateInversion(): void {
    if (this.invertTimer > INVERT_FADE_DURATION_MS) {
      this.invertTimer = 0;
      this.inverted = false;
      return;
    }
    if (this.invertTimer > 0) {
      this.invertTimer += FRAME_MS;
      return;
    }
    const score = this.actualScore();
    if (score > 0 && score % INVERT_DISTANCE === 0) {
      this.invertTimer += FRAME_MS;
      this.inverted = true;
    }
  }

  private addObstacle(): void {
    // Chromium draws from every original obstacle type first and retries when
    // the selected type is unavailable at the current speed or would exceed
    // MAX_OBSTACLE_DUPLICATION. Filtering before drawing changes both the
    // distribution and every subsequent random value.
    let definition: ObstacleDefinition | undefined;
    while (!definition) {
      const candidate = OBSTACLE_DEFINITIONS[
        this.random.integer(0, OBSTACLE_DEFINITIONS.length - 1)
      ];
      if (
        candidate &&
        this.speed >= candidate.minSpeed &&
        !this.isDuplicate(candidate.kind)
      ) {
        definition = candidate;
      }
    }
    let size = this.random.integer(1, 3);
    if (size > 1 && definition.multipleSpeed > this.speed) {
      size = 1;
    }
    const width = definition.width * size;
    const collisionBoxes = definition.collisionBoxes.map((box) => ({...box}));
    if (size > 1) {
      const left = collisionBoxes[0];
      const middle = collisionBoxes[1];
      const right = collisionBoxes[2];
      if (left && middle && right) {
        middle.width = width - left.width - right.width;
        right.x = width - right.width;
      }
    }
    const y = definition.yPositions[
      this.random.integer(0, definition.yPositions.length - 1)
    ];
    if (y === undefined) {
      throw new Error('Obstacle y-position is missing');
    }
    const speedOffset = definition.speedOffset === 0
      ? 0
      : (this.random.next() > 0.5 ? definition.speedOffset : -definition.speedOffset);
    const minGap = Math.round(width * this.speed + definition.minGap * 0.6);
    this.obstacles.push({
      definition,
      kind: definition.kind,
      x: CANVAS_WIDTH + definition.width,
      y,
      width,
      height: definition.height,
      size,
      gap: this.random.integer(minGap, Math.round(minGap * 1.5)),
      speedOffset,
      collisionBoxes,
      followingCreated: false,
      animationFrame: 0,
      animationTimer: 0,
    });
    this.obstacleHistory.unshift(definition.kind);
    this.obstacleHistory.length = Math.min(this.obstacleHistory.length, 2);
  }

  private isDuplicate(kind: ObstacleKind): boolean {
    let duplicateCount = 0;
    for (const previous of this.obstacleHistory) {
      duplicateCount = previous === kind ? duplicateCount + 1 : 0;
    }
    return duplicateCount >= 2;
  }

  private hasCollision(player: PlayerRenderModel): boolean {
    const obstacle = this.obstacles[0];
    if (!obstacle) {
      return false;
    }
    const dinoOuter: Box = {
      x: DINO_X + 1,
      y: player.dinoY + 1,
      width: DINO_WIDTH - 2,
      height: DINO_HEIGHT - 2,
    };
    const obstacleOuter: Box = {
      x: obstacle.x + 1,
      y: obstacle.y + 1,
      width: obstacle.width - 2,
      height: obstacle.height - 2,
    };
    if (!intersects(dinoOuter, obstacleOuter)) {
      return false;
    }
    const dinoBoxes = DINO_COLLISION_BOXES[
      player.ducking ? 'ducking' : 'running'
    ];
    return dinoBoxes.some((dinoBox) => {
      const adjustedDino = {
        ...dinoBox,
        x: dinoOuter.x + dinoBox.x,
        y: dinoOuter.y + dinoBox.y,
      };
      return obstacle.collisionBoxes.some((obstacleBox) => intersects(
        adjustedDino,
        {
          ...obstacleBox,
          x: obstacleOuter.x + obstacleBox.x,
          y: obstacleOuter.y + obstacleBox.y,
        },
      ));
    });
  }
}

class DinoPlayer {
  private dinoY = GROUND_Y;
  private jumpVelocity = 0;
  private grounded = true;
  private ducking = false;
  private jumpHeld = false;
  private speedDrop = false;
  private reachedMinHeight = false;
  private crashed = false;
  private crashMs = 0;

  reset(): void {
    this.dinoY = GROUND_Y;
    this.jumpVelocity = 0;
    this.grounded = true;
    this.ducking = false;
    this.jumpHeld = false;
    this.speedDrop = false;
    this.reachedMinHeight = false;
    this.crashed = false;
    this.crashMs = 0;
  }

  get isCrashed(): boolean {
    return this.crashed;
  }

  advance(action: Action, speed: number): void {
    if (this.crashed) {
      throw new Error('A crashed player cannot receive another action');
    }
    this.applyAction(action, speed);
    this.advanceDino();
  }

  advanceCrashClock(): void {
    if (this.crashed) {
      this.crashMs += FRAME_MS;
    }
  }

  markCrashed(): void {
    this.crashed = true;
    this.crashMs = 0;
  }

  renderModel(): PlayerRenderModel {
    return {
      dinoX: DINO_X,
      dinoY: this.dinoY,
      grounded: this.grounded,
      ducking: this.ducking,
      crashed: this.crashed,
      crashMs: this.crashMs,
    };
  }

  physicsSnapshot(): PlayerPhysicsSnapshot {
    return {
      ...this.renderModel(),
      jumpVelocity: this.jumpVelocity,
      speedDrop: this.speedDrop,
      reachedMinHeight: this.reachedMinHeight,
    };
  }

  private applyAction(action: Action, speed: number): void {
    if (action === 1) {
      // Chromium starts a jump on a keydown event, not on every frame for
      // which the key remains held. Holding still controls jump height, but a
      // new jump requires an intervening key release.
      if (!this.jumpHeld && this.grounded && !this.ducking) {
        this.jumpVelocity = -10 - speed / 10;
        this.grounded = false;
        this.reachedMinHeight = false;
        this.speedDrop = false;
      }
      this.jumpHeld = true;
      return;
    }
    this.jumpHeld = false;
    if (action === 2) {
      if (this.grounded) {
        this.ducking = true;
      } else if (!this.speedDrop) {
        this.speedDrop = true;
        this.jumpVelocity = 1;
      }
      return;
    }
    this.ducking = false;
    this.endJump();
  }

  private endJump(): void {
    if (this.reachedMinHeight && this.jumpVelocity < -5) {
      this.jumpVelocity = -5;
    }
  }

  private advanceDino(): void {
    if (this.grounded) {
      return;
    }
    const multiplier = this.speedDrop ? 3 : 1;
    this.dinoY += Math.round(this.jumpVelocity * multiplier);
    this.jumpVelocity += 0.6;
    if (this.dinoY < GROUND_Y - 30 || this.speedDrop) {
      this.reachedMinHeight = true;
    }
    if (this.dinoY < 30 || this.speedDrop) {
      this.endJump();
    }
    if (this.dinoY > GROUND_Y) {
      this.dinoY = GROUND_Y;
      this.jumpVelocity = 0;
      this.grounded = true;
      this.ducking = this.speedDrop;
      this.speedDrop = false;
      this.reachedMinHeight = false;
    }
  }
}

/**
 * One shared Chromium world plus isolated player physics for an evolved population.
 *
 * Candidate actions can change only their own DinoPlayer. The world clock,
 * speed, RNG, obstacle schedule, clouds, horizon and score advance exactly
 * once per simulation frame, independent of population size and actions.
 */
export class DinoPopulationEngine {
  private readonly world: DinoWorld;
  private players: DinoPlayer[];

  constructor(
    populationSize = 100,
    seed = 0,
  ) {
    if (!Number.isInteger(populationSize) || populationSize < 1 || populationSize > 100) {
      throw new Error('populationSize must be an integer between 1 and 100');
    }
    this.world = new DinoWorld(seed);
    this.players = Array.from({length: populationSize}, () => new DinoPlayer());
  }

  get populationSize(): number {
    return this.players.length;
  }

  reset(seed: number): void {
    this.world.reset(seed);
    for (const player of this.players) {
      player.reset();
    }
  }

  step(
    actions: readonly Action[],
    active: readonly boolean[] = actions.map(() => true),
  ): PopulationStepResult {
    if (actions.length !== this.players.length) {
      throw new Error('Exactly one action is required for every candidate');
    }
    if (active.length !== this.players.length) {
      throw new Error('Exactly one active flag is required for every candidate');
    }
    const newlyCrashed = new Set<number>();
    for (let substep = 0; substep < SUBSTEPS_PER_CONTROL; substep += 1) {
      for (let index = 0; index < this.players.length; index += 1) {
        const player = this.players[index];
        const action = actions[index];
        if (!player || action === undefined) {
          throw new Error('Candidate action ordering is incomplete');
        }
        if (!active[index]) {
          continue;
        }
        if (player.isCrashed) {
          player.advanceCrashClock();
        } else {
          player.advance(action, this.world.currentSpeed);
        }
      }
      const obstaclesActive = this.world.beginFrame();
      if (obstaclesActive) {
        for (let index = 0; index < this.players.length; index += 1) {
          const player = this.players[index];
          if (
            player &&
            active[index] &&
            !player.isCrashed &&
            this.world.collides(player.renderModel())
          ) {
            player.markCrashed();
            newlyCrashed.add(index);
            this.world.recordCrash();
          }
        }
      }
      this.world.finishFrame();
    }
    const terminated = this.players.filter((player) => player.isCrashed).length;
    return {
      newlyCrashed: [...newlyCrashed],
      alive: this.players.length - terminated,
      terminated,
    };
  }

  worldRenderModel(): WorldRenderModel {
    return this.world.renderModel();
  }

  candidateRenderModel(index: number): RenderModel {
    const player = this.players[index];
    if (!player) {
      throw new Error(`Candidate ${index} does not exist`);
    }
    return {
      ...this.world.renderModel(),
      ...player.renderModel(),
    };
  }

  candidatePhysics(index: number): PlayerPhysicsSnapshot {
    const player = this.players[index];
    if (!player) {
      throw new Error(`Candidate ${index} does not exist`);
    }
    return player.physicsSnapshot();
  }

  candidateRenderModels(): readonly RenderModel[] {
    const world = this.world.renderModel();
    return this.players.map((player) => ({
      ...world,
      ...player.renderModel(),
    }));
  }
}
