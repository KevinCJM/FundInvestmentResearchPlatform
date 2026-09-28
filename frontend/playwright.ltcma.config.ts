import { defineConfig } from '@playwright/test'

delete process.env.ALL_PROXY
const python = process.env.INDICATOR_TEST_PYTHON || '/Users/chenjunming/Desktop/myenv_312/bin/python3.12'
const api = process.env.LTCMA_TEST_API || 'http://127.0.0.1:8128'
const web = process.env.LTCMA_TEST_WEB || 'http://127.0.0.1:4328'
export default defineConfig({
  testDir: './e2e', testMatch: ['ltcma.spec.ts', 'scope-feasibility.spec.ts'], fullyParallel: false, workers: 1,
  timeout: 90000, reporter: [['list']], outputDir: '../.tmp_ltcma_20260918/browser',
  use: { baseURL: web, channel: 'chrome', headless: true, actionTimeout: 12000, trace: 'retain-on-failure' },
  projects: [
    { name: 'desktop-1440', use: { viewport: { width: 1440, height: 1000 } } },
    { name: 'tablet-768', use: { viewport: { width: 768, height: 1000 } } },
    { name: 'mobile-320', use: { viewport: { width: 320, height: 900 } } },
  ],
  webServer: [
    { command: `cd .. && PYTHONPATH=.:backend "${python}" -m uvicorn backend.tests.ltcma_app:app --host 127.0.0.1 --port ${new URL(api).port}`,
      url: `${api}/ready`, reuseExistingServer: false, timeout: 180000 },
    { command: `npm run dev -- --host 127.0.0.1 --port ${new URL(web).port} --strictPort`, url: web, reuseExistingServer: false, timeout: 120000 },
  ],
})
