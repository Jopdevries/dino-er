import {defineConfig} from '@playwright/test';
import {tmpdir} from 'node:os';
import {resolve} from 'node:path';

export default defineConfig({
  outputDir: resolve(tmpdir(), 'dino-er-playwright-results'),
  testDir: 'tests',
  testMatch: '**/*.visual.spec.ts',
  timeout: 30_000,
  expect: {
    toHaveScreenshot: {
      animations: 'disabled',
      maxDiffPixelRatio: 0.001,
      threshold: 0.1,
    },
  },
  use: {
    baseURL: 'http://127.0.0.1:4173',
    channel: 'chrome',
    deviceScaleFactor: 1,
    headless: true,
    viewport: {width: 760, height: 300},
  },
  webServer: {
    command: 'npm.cmd run dev -- --port 4173',
    url: 'http://127.0.0.1:4173/batch.html',
    reuseExistingServer: true,
  },
});
