import { defineConfig, devices } from '@playwright/test'
import { fileURLToPath } from 'node:url'
import { homedir } from 'node:os'
import { join } from 'node:path'

delete process.env.ALL_PROXY
const root = fileURLToPath(new URL('../', import.meta.url))
const python = process.env.TEST_PYTHON || join(homedir(), 'Desktop', 'myenv_312', 'bin', 'python3.12')
const quote = (value: string) => "'" + value.replace(/'/g, "'\\''") + "'"
const port = Number(process.env.FACTOR_TEST_PORT || '8011')
if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error('Invalid local acceptance port')

export default defineConfig({
  testDir: './e2e', testMatch: 'factor-research-live.spec.ts', workers: 1,
  timeout: 180_000, expect: { timeout: 15_000 }, reporter: [['list']],
  outputDir: './factor-live-results',
  use: { ...devices['Desktop Chrome'], channel: 'chrome', viewport: { width: 1440, height: 1000 },
    baseURL: 'http://127.0.0.1:' + port, trace: 'retain-on-failure' },
  webServer: {
    command: quote(python) + ' -m uvicorn backend.app:app --host 127.0.0.1 --port ' + port,
    cwd: root, url: 'http://127.0.0.1:' + port + '/api/factor-research/catalog',
    timeout: 180_000, reuseExistingServer: true,
  },
})
