import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { localSourceWrites } from './dev/sourceWriteGuard'

export default defineConfig({
  plugins: [react(), localSourceWrites()],
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    globals: true,
    css: true,
    exclude: ['**/e2e/**', '**/node_modules/**', '**/dist/**']
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      }
    }
  }
})
