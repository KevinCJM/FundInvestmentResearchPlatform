import { defineConfig } from '@playwright/test'
import strategic from './playwright.strategic.config'

// Same real, temporary-data API as strategic acceptance; no production service.
export default defineConfig({
  ...strategic,
  testMatch: 'better-saataa-01-04.spec.ts',
  outputDir: '../.pytest_cache/better-saataa/browser',
  projects: [
    { name: 'mobile-320', use: { viewport: { width: 320, height: 900 } } },
    { name: 'tablet-768', use: { viewport: { width: 768, height: 1000 } } },
    { name: 'desktop-1440', use: { viewport: { width: 1440, height: 1000 } } },
  ],
})
