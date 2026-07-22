import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, devices } from "@playwright/test";

const webRoot = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(webRoot, "../..");

export default defineConfig({
  testDir: "./e2e",
  timeout: 180_000,
  fullyParallel: true,
  workers: 1,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"], ["html", { outputFolder: "artifacts/playwright-report", open: "never" }]],
  use: {
    baseURL: "http://127.0.0.1:4173",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile", use: { ...devices["Pixel 5"] } },
  ],
  webServer: [
    {
      command: "python scripts/e2e_server.py --root . --host 127.0.0.1 --port 8000",
      cwd: repoRoot,
      url: "http://127.0.0.1:8000/healthz",
      timeout: 180_000,
      reuseExistingServer: false,
      env: { PYTHONPATH: repoRoot },
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 4173",
      cwd: webRoot,
      url: "http://127.0.0.1:4173",
      timeout: 60_000,
      reuseExistingServer: false,
      env: { VITE_API_BASE_URL: "http://127.0.0.1:8000" },
    },
  ],
});
