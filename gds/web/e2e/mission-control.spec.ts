import { expect, test } from "@playwright/test";

async function admitRoi(page: import("@playwright/test").Page) {
  page.on("pageerror", (error) => console.log(`[pageerror] ${error.stack ?? error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") console.log(`[console.error] ${message.text()}`);
  });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Mission control" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Preview & transmit" })).toBeEnabled({ timeout: 30_000 });
  await page.getByLabel("width").fill("256");
  await page.getByLabel("height").fill("256");
  await page.getByRole("button", { name: "Select ROI" }).click();
  await page.getByRole("button", { name: "Preview & transmit" }).click();
  const dialog = page.getByRole("dialog", { name: "Confirm ROI analysis" });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("HTTP IDEMPOTENCY-KEY");
  await dialog.getByRole("button", { name: "Transmit analysis" }).click();
  await expect(page.getByRole("status")).toContainText("ROI analysis admitted", { timeout: 30_000 });
  await expect.poll(
    async () => {
      const response = await page.request.get("http://127.0.0.1:8000/api/state");
      const payload = await response.json() as { state: { commands: unknown; jobs: unknown; products: Record<string, { state?: string }> } };
      return JSON.stringify({
        published: Object.values(payload.state.products).some((item) => item.state === "PUBLISHED"),
        commands: payload.state.commands,
        jobs: payload.state.jobs,
        products: payload.state.products,
      });
    },
    { timeout: 120_000, intervals: [1_000, 2_000, 5_000] },
  ).toContain('"published":true');
  await expect(page.locator(".lifecycle-row").filter({ hasText: "PUBLISHED" }).first()).toBeVisible({ timeout: 120_000 });
  await expect(page.locator(".product-row").filter({ hasText: "SHA256_MATCH" }).first()).toBeVisible();
}

test("desktop completes scene to verified product workflow", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "desktop workflow is scoped to the desktop project");
  await admitRoi(page);
  await expect(page.getByText("local_sil / host loopback")).toBeVisible();
});

test("mobile can edit a valid ROI and reach the confirmation gate", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile", "mobile ROI workflow is scoped to the mobile project");
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Mission control" })).toBeVisible();
  await page.getByLabel("width").fill("256");
  await page.getByLabel("height").fill("256");
  await page.getByRole("button", { name: "Select ROI" }).click();
  await page.getByRole("button", { name: "Preview & transmit" }).click();
  const dialog = page.getByRole("dialog", { name: "Confirm ROI analysis" });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("ROI PIXELS");
  await dialog.getByRole("button", { name: "Cancel" }).click();
  await expect(dialog).toBeHidden();
});
