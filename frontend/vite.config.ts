import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Development only: the API runs on 5177; production serves the build from FastAPI.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5178,
    strictPort: true,
    proxy: { "/api": "http://127.0.0.1:5177" },
  },
  // Never inline assets as data: URIs; the server CSP allows fonts only from the same origin.
  build: { outDir: "dist", sourcemap: false, assetsInlineLimit: 0 },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["src/test-setup.ts"],
  },
});
