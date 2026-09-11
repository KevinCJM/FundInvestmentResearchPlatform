import { defineConfig } from '@playwright/test'
import base from './playwright.config'

// Acceptance uses the shipped bundle, not a hot-reloading development page.
// All indicator APIs in this spec are routed to a disposable backend workspace.
export default defineConfig({
  ...base,
  testMatch: 'indicator-primitives.spec.ts',
  use: { ...base.use, baseURL: 'http://127.0.0.1:4178' },
  webServer: {
    command: 'npm run build && node node_modules/vite/bin/vite.js preview --host 127.0.0.1 --port 4178 --strictPort',
    url: 'http://127.0.0.1:4178/settings/indicators-models',
    reuseExistingServer: false,
    timeout: 120_000,
  },
})
