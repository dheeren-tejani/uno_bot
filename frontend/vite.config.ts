import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Dev proxy so the real backend can be used same-origin (set VITE_USE_MOCK=false)
    proxy: { '/api': { target: 'http://localhost:8000', changeOrigin: true } },
  },
});