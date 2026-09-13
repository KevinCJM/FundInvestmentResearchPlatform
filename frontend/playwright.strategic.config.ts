import { defineConfig } from '@playwright/test'

delete process.env.ALL_PROXY
export default defineConfig({
  testDir: './e2e', testMatch: 'strategic-allocation.spec.ts', fullyParallel: false, workers: 1,
  timeout: 120000, reporter: [['list']], outputDir: '../.pytest_cache/top-down-allocation/browser',
  use: { baseURL: 'http://127.0.0.1:4317', channel: 'chrome', headless: true, actionTimeout: 12000, trace: 'retain-on-failure' },
  projects: [
    { name: 'desktop-1440', use: { viewport: { width: 1440, height: 1000 } } },
    { name: 'mobile-390', use: { viewport: { width: 390, height: 844 } } },
  ],
  webServer: [
    { command: 'cd .. && PYTHONPATH=.:backend /Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m uvicorn backend.tests.strategic_allocation_app:app --host 127.0.0.1 --port 8118', url: 'http://127.0.0.1:8118/ready', reuseExistingServer: false, timeout: 180000 },
    { command: 'npm run dev -- --host 127.0.0.1 --port 4317 --strictPort', url: 'http://127.0.0.1:4317', reuseExistingServer: false, timeout: 120000 },
  ],
})
