// Copyright 2025 The Chromium Authors
// Use of this source code is governed by a BSD-style license recorded in
// THIRD_PARTY_NOTICES.txt.

import type {
  RenderModel,
  RenderObstacle,
  WorldRenderModel,
} from './engine';

const LOGICAL_WIDTH = 600;
const LOGICAL_HEIGHT = 150;
const MOON_PHASES = [140, 120, 100, 60, 40, 20, 0] as const;
// Exact draw bounds: do not transport blank private-canvas margins.
const PRIVATE_DINO_PATCH = {x: 50, y: 0, width: 60, height: 150} as const;
const PRIVATE_GAME_OVER_TEXT_PATCH = {
  x: 205,
  y: 42,
  width: 191,
  height: 11,
} as const;
const PRIVATE_RESTART_PATCH = {
  x: 284,
  y: 75,
  width: 36,
  height: 32,
} as const;

export interface PrivatePixelAtlasLayout {
  width: number;
  height: number;
  sharedWorldY: number;
  dinoPatchY: number;
  dinoColumns: number;
  gameOverTextPatchY: number;
  gameOverTextColumns: number;
  restartPatchY: number;
  restartColumns: number;
}

export function privatePixelAtlasLayout(
  candidateCount: number,
): PrivatePixelAtlasLayout {
  const dinoColumns = Math.min(20, Math.max(1, candidateCount));
  const gameOverTextColumns = Math.min(6, Math.max(1, candidateCount));
  const restartColumns = Math.min(33, Math.max(1, candidateCount));
  const dinoRows = Math.ceil(candidateCount / dinoColumns);
  const gameOverTextRows = Math.ceil(
    candidateCount / gameOverTextColumns,
  );
  const restartRows = Math.ceil(candidateCount / restartColumns);
  const sharedWorldY = 0;
  const dinoPatchY = LOGICAL_HEIGHT;
  const gameOverTextPatchY =
    dinoPatchY + dinoRows * PRIVATE_DINO_PATCH.height;
  const restartPatchY =
    gameOverTextPatchY +
    gameOverTextRows * PRIVATE_GAME_OVER_TEXT_PATCH.height;
  return {
    width: Math.max(
      LOGICAL_WIDTH,
      dinoColumns * PRIVATE_DINO_PATCH.width,
      gameOverTextColumns * PRIVATE_GAME_OVER_TEXT_PATCH.width,
      restartColumns * PRIVATE_RESTART_PATCH.width,
    ),
    height:
      restartPatchY + restartRows * PRIVATE_RESTART_PATCH.height,
    sharedWorldY,
    dinoPatchY,
    dinoColumns,
    gameOverTextPatchY,
    gameOverTextColumns,
    restartPatchY,
    restartColumns,
  };
}

interface Point {
  x: number;
  y: number;
}

export interface CandidateDisplayStyle {
  opacity: number;
  verticalOffset: number;
  accent: string;
  showMarker: boolean;
}

const LDPI: Record<string, Point> = {
  cactusLarge: {x: 332, y: 2},
  cactusSmall: {x: 228, y: 2},
  cloud: {x: 86, y: 2},
  moon: {x: 484, y: 2},
  pterodactyl: {x: 134, y: 2},
  restart: {x: 2, y: 68},
  star: {x: 645, y: 2},
  text: {x: 655, y: 2},
  tRex: {x: 848, y: 2},
};

const HDPI: Record<string, Point> = {
  cactusLarge: {x: 652, y: 2},
  cactusSmall: {x: 446, y: 2},
  cloud: {x: 166, y: 2},
  moon: {x: 954, y: 2},
  pterodactyl: {x: 260, y: 2},
  restart: {x: 2, y: 130},
  star: {x: 1276, y: 2},
  text: {x: 1294, y: 2},
  tRex: {x: 1678, y: 2},
};

export class DinoRenderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly context: CanvasRenderingContext2D;
  private readonly sprite = new Image();
  private readonly canvasScale: number;
  private readonly spriteScale: 1 | 2;
  private readonly positions: Record<string, Point>;
  private readyPromise: Promise<void>;

  constructor(
    canvas: HTMLCanvasElement,
    previewScale?: number,
    readFrequently = false,
  ) {
    const context = canvas.getContext('2d', {
      alpha: false,
      willReadFrequently: readFrequently,
    });
    if (!context) {
      throw new Error('2D Canvas is unavailable');
    }
    this.canvas = canvas;
    this.context = context;
    this.canvasScale = previewScale ?? (window.devicePixelRatio >= 2 ? 2 : 1);
    this.spriteScale = this.canvasScale >= 2 ? 2 : 1;
    this.positions = this.spriteScale === 2 ? HDPI : LDPI;
    canvas.width = Math.round(LOGICAL_WIDTH * this.canvasScale);
    canvas.height = Math.round(LOGICAL_HEIGHT * this.canvasScale);
    canvas.style.aspectRatio = `${LOGICAL_WIDTH} / ${LOGICAL_HEIGHT}`;
    context.setTransform(
      this.canvasScale,
      0,
      0,
      this.canvasScale,
      0,
      0,
    );
    context.imageSmoothingEnabled = false;
    this.readyPromise = new Promise((resolve, reject) => {
      this.sprite.addEventListener('load', () => resolve(), {once: true});
      this.sprite.addEventListener(
        'error',
        () => reject(new Error('Chromium sprite sheet could not be loaded')),
        {once: true},
      );
    });
    this.sprite.src = this.spriteScale === 2
      ? '/assets/chromium/200-offline-sprite.png'
      : '/assets/chromium/100-offline-sprite.png';
  }

  ready(): Promise<void> {
    return this.readyPromise;
  }

  draw(model: RenderModel, waiting = false): void {
    this.drawWorld(model);
    this.beginLogicalDrawing(model.inverted);
    this.drawDino(model, waiting);
    if (model.crashed) {
      this.drawGameOver(model.crashMs);
    }
    this.context.restore();
  }

  drawWorld(model: WorldRenderModel): void {
    const context = this.context;
    this.beginLogicalDrawing(model.inverted);
    context.fillStyle = '#f7f7f7';
    context.fillRect(0, 0, LOGICAL_WIDTH, LOGICAL_HEIGHT);

    for (let index = 0; index < 2; index += 1) {
      const sourceOffset = model.horizonSource[index] ?? 0;
      const destinationX = model.horizonX[index] ?? 0;
      this.blit(
        2 + sourceOffset * this.spriteScale,
        52 * this.spriteScale,
        600,
        12,
        destinationX,
        127,
        600,
        12,
      );
    }
    this.drawNight(model);
    for (const cloud of model.clouds) {
      const position = this.positions.cloud;
      if (position) {
        this.blit(position.x, position.y, 46, 14, cloud.x, cloud.y, 46, 14);
      }
    }
    for (const obstacle of model.obstacles) {
      this.drawObstacle(obstacle);
    }
    this.drawScore(model);
    context.restore();
  }

  drawPrivatePixelAtlas(
    sharedWorld: HTMLCanvasElement,
    candidates: readonly RenderModel[],
    active: readonly boolean[] = candidates.map(() => true),
    reset = false,
  ): PrivatePixelAtlasLayout {
    if (active.length !== candidates.length) {
      throw new Error('Private pixel atlas requires one active flag per candidate');
    }
    const layout = privatePixelAtlasLayout(candidates.length);
    const requiredWidth = Math.round(layout.width * this.canvasScale);
    const requiredHeight = Math.round(layout.height * this.canvasScale);
    if (
      this.canvas.width !== requiredWidth ||
      this.canvas.height !== requiredHeight
    ) {
      this.canvas.width = requiredWidth;
      this.canvas.height = requiredHeight;
      this.context.imageSmoothingEnabled = false;
      reset = true;
    }
    this.context.save();
    this.context.setTransform(1, 0, 0, 1, 0, 0);
    this.context.filter = 'none';
    this.context.globalAlpha = 1;
    if (reset) {
      this.context.fillStyle = '#000';
      this.context.fillRect(0, 0, requiredWidth, requiredHeight);
    }
    this.context.drawImage(
      sharedWorld,
      0,
      layout.sharedWorldY * this.canvasScale,
      LOGICAL_WIDTH * this.canvasScale,
      LOGICAL_HEIGHT * this.canvasScale,
    );
    for (let index = 0; index < candidates.length; index += 1) {
      if (!active[index]) {
        continue;
      }
      const column = index % layout.dinoColumns;
      const row = Math.floor(index / layout.dinoColumns);
      this.context.drawImage(
        sharedWorld,
        PRIVATE_DINO_PATCH.x * this.canvasScale,
        PRIVATE_DINO_PATCH.y * this.canvasScale,
        PRIVATE_DINO_PATCH.width * this.canvasScale,
        PRIVATE_DINO_PATCH.height * this.canvasScale,
        column * PRIVATE_DINO_PATCH.width * this.canvasScale,
        (layout.dinoPatchY + row * PRIVATE_DINO_PATCH.height) *
          this.canvasScale,
        PRIVATE_DINO_PATCH.width * this.canvasScale,
        PRIVATE_DINO_PATCH.height * this.canvasScale,
      );
    }
    for (let index = 0; index < candidates.length; index += 1) {
      if (!active[index]) {
        continue;
      }
      const column = index % layout.gameOverTextColumns;
      const row = Math.floor(index / layout.gameOverTextColumns);
      this.context.drawImage(
        sharedWorld,
        PRIVATE_GAME_OVER_TEXT_PATCH.x * this.canvasScale,
        PRIVATE_GAME_OVER_TEXT_PATCH.y * this.canvasScale,
        PRIVATE_GAME_OVER_TEXT_PATCH.width * this.canvasScale,
        PRIVATE_GAME_OVER_TEXT_PATCH.height * this.canvasScale,
        column * PRIVATE_GAME_OVER_TEXT_PATCH.width * this.canvasScale,
        (layout.gameOverTextPatchY +
          row * PRIVATE_GAME_OVER_TEXT_PATCH.height) *
          this.canvasScale,
        PRIVATE_GAME_OVER_TEXT_PATCH.width * this.canvasScale,
        PRIVATE_GAME_OVER_TEXT_PATCH.height * this.canvasScale,
      );
    }
    for (let index = 0; index < candidates.length; index += 1) {
      if (!active[index]) {
        continue;
      }
      const column = index % layout.restartColumns;
      const row = Math.floor(index / layout.restartColumns);
      this.context.drawImage(
        sharedWorld,
        PRIVATE_RESTART_PATCH.x * this.canvasScale,
        PRIVATE_RESTART_PATCH.y * this.canvasScale,
        PRIVATE_RESTART_PATCH.width * this.canvasScale,
        PRIVATE_RESTART_PATCH.height * this.canvasScale,
        column * PRIVATE_RESTART_PATCH.width * this.canvasScale,
        (layout.restartPatchY + row * PRIVATE_RESTART_PATCH.height) *
          this.canvasScale,
        PRIVATE_RESTART_PATCH.width * this.canvasScale,
        PRIVATE_RESTART_PATCH.height * this.canvasScale,
      );
    }
    this.context.restore();

    for (let index = 0; index < candidates.length; index += 1) {
      const candidate = candidates[index];
      if (!candidate || !active[index]) {
        continue;
      }
      const column = index % layout.dinoColumns;
      const row = Math.floor(index / layout.dinoColumns);
      const destinationX = column * PRIVATE_DINO_PATCH.width;
      const destinationY =
        layout.dinoPatchY + row * PRIVATE_DINO_PATCH.height;
      this.context.save();
      this.context.beginPath();
      this.context.rect(
        destinationX,
        destinationY,
        PRIVATE_DINO_PATCH.width,
        PRIVATE_DINO_PATCH.height,
      );
      this.context.clip();
      this.context.setTransform(
        this.canvasScale,
        0,
        0,
        this.canvasScale,
        (destinationX - PRIVATE_DINO_PATCH.x) * this.canvasScale,
        (destinationY - PRIVATE_DINO_PATCH.y) * this.canvasScale,
      );
      this.context.filter = candidate.inverted ? 'invert(1)' : 'none';
      this.drawDino(candidate, false);
      this.context.restore();

      if (candidate.crashed) {
        const gameOverColumn = index % layout.gameOverTextColumns;
        const gameOverRow = Math.floor(index / layout.gameOverTextColumns);
        const gameOverX =
          gameOverColumn * PRIVATE_GAME_OVER_TEXT_PATCH.width;
        const gameOverY =
          layout.gameOverTextPatchY +
          gameOverRow * PRIVATE_GAME_OVER_TEXT_PATCH.height;
        this.context.save();
        this.context.beginPath();
        this.context.rect(
          gameOverX,
          gameOverY,
          PRIVATE_GAME_OVER_TEXT_PATCH.width,
          PRIVATE_GAME_OVER_TEXT_PATCH.height,
        );
        this.context.clip();
        this.context.setTransform(
          this.canvasScale,
          0,
          0,
          this.canvasScale,
          (gameOverX - PRIVATE_GAME_OVER_TEXT_PATCH.x) *
            this.canvasScale,
          (gameOverY - PRIVATE_GAME_OVER_TEXT_PATCH.y) *
            this.canvasScale,
        );
        this.context.filter = candidate.inverted ? 'invert(1)' : 'none';
        this.drawGameOver(candidate.crashMs);
        this.context.restore();

        const restartColumn = index % layout.restartColumns;
        const restartRow = Math.floor(index / layout.restartColumns);
        const restartX = restartColumn * PRIVATE_RESTART_PATCH.width;
        const restartY =
          layout.restartPatchY + restartRow * PRIVATE_RESTART_PATCH.height;
        this.context.save();
        this.context.beginPath();
        this.context.rect(
          restartX,
          restartY,
          PRIVATE_RESTART_PATCH.width,
          PRIVATE_RESTART_PATCH.height,
        );
        this.context.clip();
        this.context.setTransform(
          this.canvasScale,
          0,
          0,
          this.canvasScale,
          (restartX - PRIVATE_RESTART_PATCH.x) * this.canvasScale,
          (restartY - PRIVATE_RESTART_PATCH.y) * this.canvasScale,
        );
        this.context.filter = candidate.inverted ? 'invert(1)' : 'none';
        this.drawGameOver(candidate.crashMs);
        this.context.restore();
      }
    }
    return layout;
  }

  drawPopulation(
    sharedWorld: HTMLCanvasElement,
    world: WorldRenderModel,
    candidates: readonly RenderModel[],
    styles: readonly CandidateDisplayStyle[],
  ): void {
    if (styles.length !== candidates.length) {
      throw new Error('Every population candidate requires one display style');
    }
    this.copySharedWorld(sharedWorld);
    this.beginLogicalDrawing(world.inverted);
    for (let index = 0; index < candidates.length; index += 1) {
      const candidate = candidates[index];
      const style = styles[index];
      if (!candidate || !style || style.opacity <= 0) {
        continue;
      }
      this.context.save();
      this.context.globalAlpha = Math.min(1, Math.max(0, style.opacity));
      this.context.translate(0, style.verticalOffset);
      this.drawDino(candidate, false);
      if (style.showMarker) {
        this.context.fillStyle = style.accent;
        this.context.fillRect(
          Math.round(candidate.dinoX + 19),
          Math.round(candidate.dinoY - 3),
          6,
          2,
        );
      }
      this.context.restore();
    }
    this.context.restore();
  }

  private beginLogicalDrawing(inverted: boolean): void {
    this.context.save();
    this.context.setTransform(
      this.canvasScale,
      0,
      0,
      this.canvasScale,
      0,
      0,
    );
    this.context.filter = inverted ? 'invert(1)' : 'none';
  }

  private copySharedWorld(sharedWorld: HTMLCanvasElement): void {
    this.context.save();
    this.context.setTransform(1, 0, 0, 1, 0, 0);
    this.context.filter = 'none';
    this.context.globalAlpha = 1;
    this.context.drawImage(
      sharedWorld,
      0,
      0,
      this.canvas.width,
      this.canvas.height,
    );
    this.context.restore();
  }

  private drawNight(model: WorldRenderModel): void {
    const moon = this.positions.moon;
    const star = this.positions.star;
    const phaseOffset = MOON_PHASES[model.nightPhase];
    if (!moon || !star || phaseOffset === undefined || model.nightOpacity <= 0) {
      return;
    }
    this.context.save();
    this.context.globalAlpha = model.nightOpacity;
    if (model.nightStarsVisible) {
      for (const item of model.nightStars) {
        this.blit(
          star.x,
          star.y + 9 * item.sprite * this.spriteScale,
          9,
          9,
          item.x,
          item.y,
          9,
          9,
        );
      }
    }
    const moonWidth = model.nightPhase === 3 ? 40 : 20;
    this.blit(
      moon.x + phaseOffset * this.spriteScale,
      moon.y,
      moonWidth,
      40,
      model.nightMoonX,
      30,
      moonWidth,
      40,
    );
    this.context.restore();
  }

  private drawObstacle(obstacle: RenderObstacle): void {
    const position = this.positions[obstacle.kind];
    if (!position) {
      return;
    }
    let sourceX = position.x;
    if (obstacle.kind === 'pterodactyl') {
      sourceX += obstacle.height === 40
        ? obstacle.width / obstacle.size *
          obstacle.animationFrame *
          this.spriteScale
        : 0;
    } else {
      const singleWidth = obstacle.width / obstacle.size;
      sourceX += singleWidth *
        obstacle.size *
        (obstacle.size - 1) /
        2 *
        this.spriteScale;
    }
    this.blit(
      sourceX,
      position.y,
      obstacle.width,
      obstacle.height,
      obstacle.x,
      obstacle.y,
      obstacle.width,
      obstacle.height,
    );
  }

  private drawDino(model: RenderModel, waiting: boolean): void {
    const position = this.positions.tRex;
    if (!position) {
      return;
    }
    let frame = 0;
    let width = 44;
    if (waiting) {
      frame = 0;
    } else if (model.crashed) {
      frame = 220;
    } else if (!model.grounded) {
      frame = 0;
    } else if (model.ducking) {
      frame = [264, 323][Math.floor(model.tick / 8) % 2] ?? 264;
      width = 59;
    } else {
      frame = [88, 132][Math.floor(model.tick / 5) % 2] ?? 88;
    }
    this.blit(
      position.x + frame * this.spriteScale,
      position.y,
      width,
      47,
      model.crashed && model.ducking ? model.dinoX + 1 : model.dinoX,
      model.dinoY,
      width,
      47,
    );
  }

  private drawScore(model: WorldRenderModel): void {
    const position = this.positions.text;
    if (!position) {
      return;
    }
    const units = model.score > 99999 ? 6 : 5;
    const x = LOGICAL_WIDTH - 11 * (units + 1);
    if (model.scoreVisible) {
      const digits = model.score.toString().padStart(units, '0').slice(-units);
      for (let index = 0; index < digits.length; index += 1) {
        this.drawScoreCharacter(position, x, index, Number(digits[index]));
      }
    }
    if (model.highScore > 0) {
      const highScore = `HI ${model.highScore
        .toString()
        .padStart(units, '0')
        .slice(-units)}`;
      const highScoreX = x - units * 2 * 10;
      this.context.save();
      this.context.globalAlpha = 0.8;
      for (let index = 0; index < highScore.length; index += 1) {
        const character = highScore[index];
        if (character === ' ') {
          continue;
        }
        const value = character === 'H' ? 10 :
          character === 'I' ? 11 : Number(character);
        this.drawScoreCharacter(position, highScoreX, index, value);
      }
      this.context.restore();
    }
  }

  private drawScoreCharacter(
    position: Point,
    x: number,
    index: number,
    value: number,
  ): void {
    this.blit(
      position.x + value * 10 * this.spriteScale,
      position.y,
      10,
      13,
      x + index * 11,
      5,
      10,
      13,
    );
  }

  private drawGameOver(crashMs: number): void {
    const text = this.positions.text;
    const restart = this.positions.restart;
    if (!text || !restart) {
      return;
    }
    this.blit(
      text.x,
      text.y + 13 * this.spriteScale,
      191,
      11,
      205,
      42,
      191,
      11,
    );
    const animationTime = Math.max(0, crashMs - 875);
    const animationFrame = crashMs <= 875
      ? 0
      : Math.min(7, 1 + Math.floor(animationTime / (875 / 8)));
    this.blit(
      restart.x + animationFrame * 36 * this.spriteScale,
      restart.y,
      36,
      32,
      284,
      75,
      36,
      32,
    );
  }

  private blit(
    sourceX: number,
    sourceY: number,
    sourceWidth: number,
    sourceHeight: number,
    targetX: number,
    targetY: number,
    targetWidth: number,
    targetHeight: number,
  ): void {
    this.context.drawImage(
      this.sprite,
      sourceX,
      sourceY,
      sourceWidth * this.spriteScale,
      sourceHeight * this.spriteScale,
      Math.round(targetX),
      Math.round(targetY),
      targetWidth,
      targetHeight,
    );
  }
}
