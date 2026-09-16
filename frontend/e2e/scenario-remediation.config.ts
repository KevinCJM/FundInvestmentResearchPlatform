import { defineConfig } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
const root = fileURLToPath(new URL('../../', import.meta.url))
const python = process.env.INDICATOR_TEST_PYTHON || 'python3'
export default defineConfig({
  testDir: '.', testMatch: ['scenario-remediation.spec.ts', 'regime-full-stack.spec.ts'], workers: 1, timeout: 90_000,
  expect: { timeout: 15_000 }, reporter: [['list']],
  outputDir: process.env.SCENARIO_AUDIT_OUTPUT_DIR || path.join(root, 'frontend/test-results/scenario-remediation'),
  use: { baseURL: 'http://127.0.0.1:4184', channel: 'chrome', headless: true, trace: 'retain-on-failure' },
  projects: [
    { name: 'mobile-320', use: { viewport: { width: 320, height: 800 } } },
    { name: 'tablet-768', use: { viewport: { width: 768, height: 1024 } } },
    { name: 'desktop-1440', use: { viewport: { width: 1440, height: 1000 } } },
  ],
  webServer: [
    { command: `"${python}" -m uvicorn scenario_remediation_app:app --host 127.0.0.1 --port 8769`, cwd: root,
      url: 'http://127.0.0.1:8769/ready', reuseExistingServer: false, timeout: 180_000,
      env: { PYTHONPATH: [root, path.join(root, 'backend'), path.join(root, 'backend/tests')].join(path.delimiter),
        OPENBLAS_NUM_THREADS: '1', OMP_NUM_THREADS: '1', NUMBA_NUM_THREADS: '1' } },
    { command: `node node_modules/vite/bin/vite.js preview --host 127.0.0.1 --port 4184 --outDir "${process.env.REGIME_INTEGRATION_BUILD || 'dist'}"`,
      cwd: path.join(root, 'frontend'), url: 'http://127.0.0.1:4184', reuseExistingServer: false, timeout: 120_000 },
  ],
})
