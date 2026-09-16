import { defineConfig } from '@playwright/test'

// Tests explicitly cover 320/768/1440; no listener or production service.
export default defineConfig({
  testDir: '.', testMatch: 'regime-prospective.spec.ts', workers: 1,
  reporter: [['list']], outputDir: '/tmp/bettersaataa-regime-forward-browser',
  use: { channel: 'chrome', headless: true, trace: 'retain-on-failure' },
})
