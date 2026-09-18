import { defineConfig } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../', import.meta.url))
const python = process.env.INDICATOR_TEST_PYTHON || '/Users/chenjunming/Desktop/myenv_312/bin/python3.12'
const apiPort = Number(process.env.RISK_SCALE_API_PORT || 8877)
const uiPort = Number(process.env.RISK_SCALE_UI_PORT || 4199)
const api = `http://127.0.0.1:${apiPort}`
const build = process.env.RISK_SCALE_BUILD || '/tmp/risk-scales-m3-build'
// Local fixture traffic must never use the workstation's external proxy.
for (const key of ['ALL_PROXY', 'all_proxy', 'HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy']) delete process.env[key]
export default defineConfig({
  testDir: './e2e', testMatch: 'risk-scales.spec.ts', workers: 1, timeout: 180_000,
  expect: { timeout: 20_000 }, reporter: [['list']], outputDir: '/tmp/risk-scales-m3-playwright',
  use: { baseURL: `http://127.0.0.1:${uiPort}`, channel: 'chrome', headless: true, viewport: { width: 1440, height: 1000 }, trace: 'retain-on-failure' },
  webServer: [
    { command: `"${python}" -m uvicorn backend.tests.risk_scale_app:app --host 127.0.0.1 --port ${apiPort}`, cwd: root, url: `${api}/ready`, reuseExistingServer: false, timeout: 180_000,
      env: { PYTHONPATH: [root, path.join(root, 'backend')].join(path.delimiter), PYTHONDONTWRITEBYTECODE: '1', NUMBA_CACHE_DIR: '/tmp/risk-scales-m3-numba', OPENBLAS_NUM_THREADS: '1', OMP_NUM_THREADS: '1', NUMBA_NUM_THREADS: '1' } },
    { command: `node node_modules/vite/bin/vite.js build --outDir ${JSON.stringify(build)} && node --input-type=module -e 'import { preview } from "vite"; await preview({build:{outDir:${JSON.stringify(build)}},preview:{host:"127.0.0.1",port:${uiPort},strictPort:true,proxy:{"/api":{target:${JSON.stringify(api)},changeOrigin:true}}}})'`, cwd: path.join(root, 'frontend'), url: `http://127.0.0.1:${uiPort}`, reuseExistingServer: false, timeout: 120_000 },
  ],
})
