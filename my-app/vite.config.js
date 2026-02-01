import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/run': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, _res) => {
            console.log('proxy error', err);
          });
          proxy.on('proxyReq', (proxyReq, req, _res) => {
            console.log('Sending Request to ADK:', req.method, req.url);
          });
          proxy.on('proxyRes', (proxyRes, req, _res) => {
            console.log('Received Response from ADK:', proxyRes.statusCode, req.url);
          });
        },
      },
      '/run_sse': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
      '/apps': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
      '/list-apps': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      }
    }
  }
})
