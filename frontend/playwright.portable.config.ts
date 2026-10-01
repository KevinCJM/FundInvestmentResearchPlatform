import { defineConfig } from '@playwright/test'

delete process.env.ALL_PROXY
const origin = 'http://127.0.0.1:14173'
export default defineConfig({
  testDir: './e2e', testMatch: 'portable-agent.spec.ts', workers: 1,
  timeout: 120_000, expect: { timeout: 20_000 }, reporter: 'list',
  use: { baseURL: origin, channel: 'chrome', headless: true, trace: 'retain-on-failure' },
  projects: [
    { name: 'mobile-320', use: { viewport: { width: 320, height: 800 } } },
    { name: 'tablet-768', use: { viewport: { width: 768, height: 1024 } } },
    { name: 'desktop-1440', use: { viewport: { width: 1440, height: 1000 } } },
  ],
  webServer: [
    { command: `${process.env.INDICATOR_TEST_PYTHON || 'python3'} ../scripts/run_portable_integration.py`,
      url: 'http://127.0.0.1:18080/fixture/ready', reuseExistingServer: false, timeout: 180_000 },
    { command: 'npm run dev -- --host 127.0.0.1 --port 14173 --strictPort', url: origin,
      env: { VITE_API_TARGET: 'http://127.0.0.1:18080', VITE_PORTABLE_AGENT_TARGET: 'http://127.0.0.1:18787' },
      reuseExistingServer: false, timeout: 120_000 },
  ],
})
