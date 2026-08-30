import {describe, expect, test} from 'vitest';

import type {Action, RenderModel} from '../src/engine';
import {DinoPopulationEngine} from '../src/engine';
import {CanonicalDinoInput, VirtualDinoInput} from '../src/input';
import {
  controlIntervalMilliseconds,
  publicRenderingEnabled,
} from '../src/batch_bridge';

function rollout(seed: number, actions: readonly Action[]): string[] {
  const engine = new DinoPopulationEngine(1, seed);
  const models: string[] = [JSON.stringify(engine.candidateRenderModel(0))];
  for (const action of actions) {
    engine.step([action]);
    models.push(JSON.stringify(engine.candidateRenderModel(0)));
  }
  return models;
}

interface AvoidanceState {
  phase: 'ready' | 'hold' | 'airborne';
  holdTicks: number;
}

function chromiumAvoidanceAction(
  model: RenderModel,
  state: AvoidanceState,
): Action {
  if (state.phase === 'hold') {
    state.holdTicks -= 1;
    if (state.holdTicks > 0) {
      return 1;
    }
    state.phase = 'airborne';
    return 0;
  }
  if (state.phase === 'airborne') {
    if (model.grounded) {
      state.phase = 'ready';
    }
    return 0;
  }
  const obstacle = model.obstacles[0];
  if (!obstacle) {
    return 0;
  }
  const distance = obstacle.x - model.dinoX;
  if (obstacle.kind === 'pterodactyl') {
    if (obstacle.y === 75 && distance < 160 && distance > -60) {
      return 2;
    }
    if (obstacle.y !== 100 || distance >= 140 || distance <= 0) {
      return 0;
    }
  } else if (distance >= 145 || distance <= 0) {
    return 0;
  }
  state.phase = 'hold';
  state.holdTicks = 10;
  return 1;
}

describe('Chromium-derived fixed-step engine', () => {
  test('execution speed changes pacing and redraw only', () => {
    expect(controlIntervalMilliseconds(1)).toBeCloseTo(1000 / 60, 12);
    expect(controlIntervalMilliseconds(3)).toBeCloseTo(1000 / 180, 12);
    expect(publicRenderingEnabled(1)).toBe(true);
    expect(publicRenderingEnabled(1.01)).toBe(false);
    expect(publicRenderingEnabled(100)).toBe(false);
    const actions: Action[] = [...Array<Action>(20).fill(0), 1, 1, 0, 0];
    const oneTimes = rollout(71, actions);
    const threeTimes = rollout(71, actions);
    expect(threeTimes).toEqual(oneTimes);
  });

  test('is deterministic for equal seeds and action sequences', () => {
    const actions: Action[] = [
      ...Array<Action>(10).fill(0),
      ...Array<Action>(8).fill(1),
      ...Array<Action>(12).fill(0),
      ...Array<Action>(3).fill(2),
    ];
    expect(rollout(17, actions)).toEqual(rollout(17, actions));
    expect(rollout(17, actions)).not.toEqual(rollout(18, actions));
  });

  test('matches the audited 60 Hz jump and speed update', () => {
    const engine = new DinoPopulationEngine(1, 7);
    expect(engine.candidateRenderModel(0).dinoY).toBe(93);
    engine.step([1]);
    const model = engine.candidateRenderModel(0);
    expect(model.dinoY).toBe(82);
    expect(model.speed).toBeCloseTo(6.001, 12);
    expect(model.distance).toBeCloseTo(6, 12);
    expect(model.tick).toBe(1);
  });

  test.each([
    {steps: 1_300, expectedScore: 216},
    {steps: 1_490, expectedScore: 251},
  ])(
    'maps $steps fixed steps to Chromium distance score $expectedScore',
    ({steps, expectedScore}) => {
      const engine = new DinoPopulationEngine(1, 7);
      for (let step = 0; step < steps; step += 1) {
        engine.step([0]);
      }
      const model = engine.candidateRenderModel(0);
      expect(model.tick).toBe(steps);
      expect(model.score).toBe(expectedScore);
    },
  );

  test('delays obstacles for the Chromium clear time', () => {
    const engine = new DinoPopulationEngine(1, 11);
    for (let index = 0; index < 180; index += 1) {
      engine.step([0]);
    }
    expect(engine.candidateRenderModel(0).obstacles).toHaveLength(0);
    engine.step([0]);
    expect(engine.candidateRenderModel(0).obstacles).toHaveLength(1);
  });

  test.each([4, 10, 25, 50, 100])(
    'initialises a %i-candidate shared-world population',
    (populationSize) => {
      const arena = new DinoPopulationEngine(populationSize, 41);
      expect(arena.populationSize).toBe(populationSize);
      expect(arena.candidateRenderModels()).toHaveLength(populationSize);
      expect(new Set(
        arena.candidateRenderModels().map((candidate) =>
          JSON.stringify(candidate.obstacles)
        ),
      )).toEqual(new Set(['[]']));
    },
  );

  test('candidate actions cannot change the shared world or another player', () => {
    const jumping = new DinoPopulationEngine(4, 77);
    const idle = new DinoPopulationEngine(4, 77);
    const untouchedBefore = jumping.candidatePhysics(1);
    jumping.step([1, 0, 2, 0]);
    idle.step([0, 0, 0, 0]);

    expect(jumping.worldRenderModel()).toEqual(idle.worldRenderModel());
    expect(jumping.candidatePhysics(1)).toEqual(untouchedBefore);
    expect(jumping.candidatePhysics(0).dinoY).toBeLessThan(
      untouchedBefore.dinoY,
    );
    expect(jumping.candidatePhysics(2).ducking).toBe(true);
  });

  test('one collision eliminates only that candidate', () => {
    const arena = new DinoPopulationEngine(2, 7);
    const avoidance = {phase: 'ready', holdTicks: 0} satisfies AvoidanceState;
    let separated = false;
    for (let control = 0; control < 400; control += 1) {
      const model = arena.candidateRenderModel(0);
      const action = chromiumAvoidanceAction(model, avoidance);
      arena.step([action, 0]);
      const safe = arena.candidateRenderModel(0);
      const weak = arena.candidateRenderModel(1);
      if (weak.crashed) {
        expect(safe.crashed).toBe(false);
        separated = true;
        break;
      }
    }
    expect(separated).toBe(true);
  });

  test('virtual and canonical inputs have identical action semantics', () => {
    const canonical = new CanonicalDinoInput();
    const virtual = new VirtualDinoInput();

    canonical.handle('jump_press');
    virtual.setAction(1);
    expect(virtual.action()).toBe(canonical.action());
    virtual.setAction(1);
    expect(virtual.action()).toBe(canonical.action());
    canonical.handle('jump_release');
    virtual.setAction(0);
    expect(virtual.action()).toBe(canonical.action());

    canonical.handle('duck_press');
    virtual.setAction(2);
    expect(virtual.action()).toBe(canonical.action());
    canonical.handle('duck_release');
    virtual.setAction(0);
    expect(virtual.action()).toBe(canonical.action());
  });

  test('holding jump across control ticks produces a higher Chromium jump', () => {
    const shortPress = new DinoPopulationEngine(1, 23);
    const longPress = new DinoPopulationEngine(1, 23);
    let shortApex = shortPress.candidateRenderModel(0).dinoY;
    let longApex = longPress.candidateRenderModel(0).dinoY;
    for (let control = 0; control < 20; control += 1) {
      shortPress.step([control === 0 ? 1 : 0]);
      longPress.step([control < 5 ? 1 : 0]);
      shortApex = Math.min(shortApex, shortPress.candidateRenderModel(0).dinoY);
      longApex = Math.min(longApex, longPress.candidateRenderModel(0).dinoY);
    }
    expect(longApex).toBeLessThan(shortApex);
  });

  test('a held jump cannot retrigger until the virtual key is released', () => {
    const engine = new DinoPopulationEngine(1, 23);
    engine.step([1]);
    expect(engine.candidateRenderModel(0).grounded).toBe(false);

    for (
      let control = 0;
      control < 60 && !engine.candidateRenderModel(0).grounded;
      control += 1
    ) {
      engine.step([1]);
    }
    expect(engine.candidateRenderModel(0).grounded).toBe(true);

    engine.step([1]);
    expect(engine.candidateRenderModel(0).grounded).toBe(true);
    engine.step([0]);
    engine.step([1]);
    expect(engine.candidateRenderModel(0).grounded).toBe(false);
  });

  test('matches the audited Chromium obstacle retry and spawn sequence', () => {
    const engine = new DinoPopulationEngine(1, 7);
    const avoidance = {phase: 'ready', holdTicks: 0} satisfies AvoidanceState;
    const seen = new Set<object>();
    const events: Array<{
      control: number;
      kind: string;
      y: number;
      size: number;
      width: number;
    }> = [];
    for (let control = 1; control <= 2_800; control += 1) {
      const model = engine.candidateRenderModel(0);
      engine.step([chromiumAvoidanceAction(model, avoidance)]);
      for (const obstacle of engine.candidateRenderModel(0).obstacles) {
        if (!seen.has(obstacle)) {
          seen.add(obstacle);
          events.push({
            control,
            kind: obstacle.kind,
            y: obstacle.y,
            size: obstacle.size,
            width: obstacle.width,
          });
        }
      }
    }
    expect(events.slice(0, 6)).toEqual([
      {control: 181, kind: 'cactusSmall', y: 105, size: 3, width: 51},
      {control: 275, kind: 'cactusLarge', y: 90, size: 1, width: 25},
      {control: 327, kind: 'cactusLarge', y: 90, size: 1, width: 25},
      {control: 388, kind: 'cactusSmall', y: 105, size: 3, width: 51},
      {control: 472, kind: 'cactusSmall', y: 105, size: 3, width: 51},
      {control: 565, kind: 'cactusLarge', y: 90, size: 1, width: 25},
    ]);
    expect(events.find((event) => event.kind === 'pterodactyl')).toEqual({
      control: 2643,
      kind: 'pterodactyl',
      y: 50,
      size: 1,
      width: 46,
    });
  });

  test('reaches the Chromium score-triggered night presentation', () => {
    const engine = new DinoPopulationEngine(1, 7);
    const avoidance = {phase: 'ready', holdTicks: 0} satisfies AvoidanceState;
    let sawInversion = false;
    let sawNightArtwork = false;
    for (let control = 0; control < 4_400; control += 1) {
      const action = chromiumAvoidanceAction(engine.candidateRenderModel(0), avoidance);
      engine.step([action]);
      const model = engine.candidateRenderModel(0);
      sawInversion ||= model.inverted;
      sawNightArtwork ||= model.nightOpacity > 0;
      if (model.nightOpacity >= 0.95) {
        break;
      }
    }
    expect(engine.candidateRenderModel(0).crashed).toBe(false);
    expect(sawInversion).toBe(true);
    expect(sawNightArtwork).toBe(true);
  });

});
