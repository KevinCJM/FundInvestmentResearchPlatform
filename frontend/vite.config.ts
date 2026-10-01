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
    exclude: ['**/e2e/**', '**/node_modules/**', '**/dist/**', '**/.run/**']
  },
  server: {
    port: 5173,
    proxy: {
      '/assistant': {
        target: process.env.VITE_PORTABLE_AGENT_TARGET || 'http://127.0.0.1:8787',
        changeOrigin: true,
        rewrite: path => path.replace(/^\/assistant(?=\/)/, ''),
      },
      '/api': {
        target: process.env.VITE_API_TARGET || 'http://127.0.0.1:8000',
        changeOrigin: true
      }
    }
  }
})
