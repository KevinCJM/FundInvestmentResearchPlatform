import { defineConfig } from '@playwright/test'
import base from './playwright.config'

// An old local preview may occupy 4173; these offline checks own one strict port.
export default defineConfig({
  ...base,
  testMatch: 'agent-research-pages.spec.ts',
  use: { ...base.use, baseURL: 'http://127.0.0.1:4276' },
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 4276 --strictPort',
    url: 'http://127.0.0.1:4276', reuseExistingServer: false, timeout: 120_000,
  },
})
