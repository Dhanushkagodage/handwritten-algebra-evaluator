import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // With VITE_GATEWAY_URL blank, the app calls /api/v1/... on its own origin
    // and this forwards it to the gateway — so CORS never enters the picture
    // in development. Set VITE_GATEWAY_URL to bypass the proxy.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8090',
        changeOrigin: true,
      },
    },
  },
})



