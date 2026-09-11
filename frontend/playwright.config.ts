import { defineConfig, devices } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// Fixture servers import both backend.* and backend-root packages. Resolve
// from this config, not the shell's cwd or an unrelated developer PYTHONPATH.
const projectRoot = fileURLToPath(new URL('../', import.meta.url))
process.env.PYTHONPATH = [projectRoot, path.join(projectRoot, 'backend'), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter)
if (process.env.INDICATOR_TEST_PYTHON) process.env.PYTHON = process.env.INDICATOR_TEST_PYTHON

// The desktop host may expose a SOCKS-only ALL_PROXY for other tools. Node's
// web-server readiness probe only accepts HTTP proxies, and local E2E traffic
// must never leave 127.0.0.1.
delete process.env.ALL_PROXY

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  reporter: [['list']],
  use: {
    baseURL: 'http://127.0.0.1:4173',
    channel: 'chrome',
    headless: true,
    trace: 'retain-on-failure',
  },
  projects: [
    { name: 'mobile-320', use: { ...devices['Desktop Chrome'], viewport: { width: 320, height: 800 } } },
    { name: 'tablet-768', use: { ...devices['Desktop Chrome'], viewport: { width: 768, height: 1024 } } },
    { name: 'desktop-1440', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 1000 } } },
  ],
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 4173',
    url: 'http://127.0.0.1:4173/indicator-studio',
    reuseExistingServer: true,
    timeout: 120_000,
  },
})
