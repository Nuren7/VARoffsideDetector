import react from '@vitejs/plugin-react'
import { cwd } from 'node:process'
import { defineConfig, loadEnv } from 'vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, cwd(), '')

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api': {
          target: env.VITE_BACKEND_URL || 'http://127.0.0.1:8000',
          changeOrigin: true,
          timeout: 20 * 60 * 1000,
          proxyTimeout: 20 * 60 * 1000,
        },
      },
    },
  }
})
