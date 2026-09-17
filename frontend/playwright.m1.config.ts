import { defineConfig } from '@playwright/test'
// Local loopback fixture, matching the existing strategic acceptance harness.
delete process.env.ALL_PROXY
export default defineConfig({
  testDir: './e2e', testMatch: 'bettersaataa-m1.spec.ts', fullyParallel: false, workers: 1,
  timeout: 120000, reporter: [['list']], outputDir: '../.pytest_cache/bettersaataa-m1/browser',
  use: { baseURL: 'http://127.0.0.1:4419', channel: 'chrome', headless: true, viewport: { width: 1440, height: 1000 }, trace: 'retain-on-failure' },
  webServer: [
    { command: 'cd .. && PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m uvicorn backend.tests.bettersaataa_m1_app:app --host 127.0.0.1 --port 8129', url: 'http://127.0.0.1:8129/ready', reuseExistingServer: false, timeout: 180000 },
    { command: 'npm run dev -- --host 127.0.0.1 --port 4419 --strictPort', url: 'http://127.0.0.1:4419', reuseExistingServer: false, timeout: 120000 },
  ],
})
