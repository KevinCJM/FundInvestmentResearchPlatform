import { defineConfig } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../', import.meta.url))
const python = process.env.INDICATOR_TEST_PYTHON || process.env.PYTHON || 'python3'
const api = 'http://127.0.0.1:8769'
process.env.PYTHONPATH = [root, path.join(root, 'backend'), path.join(root, 'backend/tests')].join(path.delimiter)
delete process.env.ALL_PROXY

export default defineConfig({
  testDir: './e2e',
  testMatch: 'regime-full-stack.spec.ts',
  workers: 1,
  timeout: 90_000,
  expect: { timeout: 15_000 },
  reporter: [['list']],
  outputDir: '/tmp/regime-completion-integration-results',
  use: {
    baseURL: 'http://127.0.0.1:4184', channel: 'chrome', headless: true,
    trace: 'retain-on-failure', viewport: { width: 1440, height: 1000 },
  },
  webServer: [
    {
      command: `"${python}" -m uvicorn regime_completion_app:app --host 127.0.0.1 --port 8769`,
      cwd: root, url: `${api}/ready`, reuseExistingServer: false, timeout: 180_000,
      env: { OPENBLAS_NUM_THREADS: '1', OMP_NUM_THREADS: '1', NUMBA_NUM_THREADS: '1' },
    },
    {
      command: `node node_modules/vite/bin/vite.js preview --host 127.0.0.1 --port 4184 --outDir "${process.env.REGIME_INTEGRATION_BUILD || 'dist'}"`,
      url: 'http://127.0.0.1:4184', reuseExistingServer: false, timeout: 120_000,
    },
  ],
})
