import { defineConfig } from '@playwright/test'

delete process.env.ALL_PROXY

export default defineConfig({
  testDir: './e2e', testMatch: 'tactical-allocation.spec.ts', fullyParallel: true,
  reporter: [['list']], outputDir: '/tmp/taa-ui-evidence',
  use: { baseURL: 'http://127.0.0.1:4199', channel: 'chrome', headless: true, trace: 'retain-on-failure' },
  projects: [
    { name: 'desktop-1440', use: { viewport: { width: 1440, height: 1000 } } },
    { name: 'mobile-390', use: { viewport: { width: 390, height: 844 } } },
  ],
  webServer: { command: 'npm run dev -- --host 127.0.0.1 --port 4199 --strictPort', url: 'http://127.0.0.1:4199/pre-investment/taa', reuseExistingServer: false, timeout: 120000 },
})
