import type {Action} from './engine';

export type DinoInputEvent =
  | 'jump_press'
  | 'jump_release'
  | 'duck_press'
  | 'duck_release';

/** Canonical input state consumed by Dino physics for both human and virtual input. */
export class CanonicalDinoInput {
  private jumpPressed = false;
  private jumpReleased = false;
  private duckPressed = false;

  handle(event: DinoInputEvent): void {
    if (event === 'jump_press') {
      this.jumpPressed = true;
    } else if (event === 'jump_release') {
      this.jumpPressed = false;
      this.jumpReleased = true;
    } else if (event === 'duck_press') {
      this.duckPressed = true;
    } else {
      this.duckPressed = false;
    }
  }

  action(): Action {
    if (this.duckPressed) {
      return 2;
    }
    if (this.jumpPressed) {
      return 1;
    }
    return 0;
  }

  consumeJumpRelease(): boolean {
    const released = this.jumpReleased;
    this.jumpReleased = false;
    return released;
  }
}

/**
 * Per-candidate virtual key channel. It never calls player physics directly.
 */
export class VirtualDinoInput {
  private readonly canonical = new CanonicalDinoInput();
  private jumpHeld = false;
  private duckHeld = false;

  setAction(action: Action): void {
    if (action === 1 && !this.jumpHeld) {
      this.canonical.handle('jump_press');
      this.jumpHeld = true;
    } else if (action !== 1 && this.jumpHeld) {
      this.canonical.handle('jump_release');
      this.jumpHeld = false;
    }
    if (action === 2 && !this.duckHeld) {
      this.canonical.handle('duck_press');
      this.duckHeld = true;
    } else if (action !== 2 && this.duckHeld) {
      this.canonical.handle('duck_release');
      this.duckHeld = false;
    }
  }

  action(): Action {
    return this.canonical.action();
  }

  reset(): void {
    this.setAction(0);
    this.canonical.consumeJumpRelease();
  }
}
