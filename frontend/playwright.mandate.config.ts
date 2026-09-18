import { defineConfig } from '@playwright/test'

delete process.env.ALL_PROXY
export default defineConfig({
  testDir: './e2e', testMatch: 'mandate-boundaries.spec.ts', fullyParallel: false, workers: 1,
  timeout: 120000, reporter: [['list']], outputDir: '../.pytest_cache/mandate-boundaries/browser',
  use: { baseURL: 'http://127.0.0.1:4326', channel: 'chrome', headless: true, actionTimeout: 15000, trace: 'retain-on-failure' },
  projects: [
    { name: 'desktop-1440', use: { viewport: { width: 1440, height: 1000 } } },
    { name: 'tablet-768', use: { viewport: { width: 768, height: 1024 } } },
    { name: 'mobile-320', use: { viewport: { width: 320, height: 844 } } },
  ],
  webServer: [
    { command: 'cd .. && PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m uvicorn backend.tests.mandate_boundary_app:app --host 127.0.0.1 --port 8126', url: 'http://127.0.0.1:8126/ready', reuseExistingServer: false, timeout: 180000 },
    { command: 'npm run dev -- --host 127.0.0.1 --port 4326 --strictPort', env: { VITE_API_TARGET: 'http://127.0.0.1:8126' }, url: 'http://127.0.0.1:4326', reuseExistingServer: false, timeout: 120000 },
  ],
})
