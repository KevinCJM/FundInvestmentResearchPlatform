import { defineConfig } from '@playwright/test'
// Same three viewport assertions, no listener and no production service.
export default defineConfig({
  testDir: '.',
  testMatch: 'regime-reference-confidence.spec.ts',
  fullyParallel: true,
  reporter: [['list']],
  outputDir: '/tmp/bettersaataa-regime-browser-results',
  use: { baseURL: 'http://regime-fixture.test', channel: 'chrome', headless: true, trace: 'retain-on-failure' },
  projects: [
    { name: 'mobile-320', use: { viewport: { width: 320, height: 800 } } },
    { name: 'tablet-768', use: { viewport: { width: 768, height: 1024 } } },
    { name: 'desktop-1440', use: { viewport: { width: 1440, height: 1000 } } },
  ],
})
