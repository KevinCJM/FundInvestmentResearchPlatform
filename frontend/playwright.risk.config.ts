import { defineConfig } from '@playwright/test'
import base from './playwright.config'

// Keep this acceptance run independent of other workspaces' browser servers
// and test-results cleanup. The existing /output/ directory is locally ignored.
export default defineConfig({
  ...base,
  testMatch: 'published-risk.spec.ts',
  outputDir: '../output/published-risk-browser',
  use: { ...base.use, baseURL: 'http://127.0.0.1:4197' },
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 4197 --strictPort',
    url: 'http://127.0.0.1:4197/settings/risk-models',
    reuseExistingServer: false,
    timeout: 120_000,
  },
})
