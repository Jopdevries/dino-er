import tailwindcss from '@tailwindcss/vite';
import {defineConfig} from 'vite';
import {resolve} from 'node:path';

export default defineConfig({
  root: '.',
  plugins: [tailwindcss()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        app: resolve(import.meta.dirname, 'batch.html'),
        results: resolve(import.meta.dirname, 'results.html'),
      },
    },
  },
});
