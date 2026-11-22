const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const {pathToFileURL} = require("url");

const playwrightCandidates = [
  path.resolve(path.dirname(process.execPath), "..", "node_modules", "playwright"),
  process.env.USERPROFILE && path.join(process.env.USERPROFILE, ".cache", "codex-runtimes",
    "codex-primary-runtime", "dependencies", "node", "node_modules", "playwright"),
].filter(Boolean);
const playwrightRoot = playwrightCandidates.find(candidate => fs.existsSync(candidate));
if (!playwrightRoot) throw Error(`Playwright runtime not found; checked: ${playwrightCandidates.join(", ")}`);
const {chromium} = require(playwrightRoot);
const browserExecutableCandidates = [
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
];
const browserExecutable = browserExecutableCandidates.find(candidate => fs.existsSync(candidate));
if (!browserExecutable) throw Error("Neither Microsoft Edge nor Google Chrome is available");

function arg(name) {
  const index = process.argv.indexOf(name);
  if (index < 0 || index + 1 >= process.argv.length) throw Error(`missing ${name}`);
  return process.argv[index + 1];
}
function optionalArg(name) {
  const index = process.argv.indexOf(name);
  return index < 0 || index + 1 >= process.argv.length ? null : process.argv[index + 1];
}
function sha256(file) { return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex"); }

(async () => {
  const urlIndex = process.argv.indexOf("--url");
  const pagePath = urlIndex < 0 ? path.resolve(arg("--page")) : null;
  const sourceUrl = pagePath ? pathToFileURL(pagePath).href : arg("--url");
  if (!sourceUrl.startsWith("file:") && !/^http:\/\/127\.0\.0\.1:\d+\//.test(sourceUrl)) {
    throw Error("acceptance URL must use a local file or 127.0.0.1 isolated server");
  }
  const output = path.resolve(arg("--output"));
  const workbenchUrl = optionalArg("--workbench-url");
  const hiddenWorkbenchUrl = optionalArg("--hidden-workbench-url");
  if (workbenchUrl && !/^http:\/\/127\.0\.0\.1:\d+\/$/.test(workbenchUrl)) {
    throw Error("workbench URL must be a 127.0.0.1 isolated server root");
  }
  if (hiddenWorkbenchUrl && !/^http:\/\/127\.0\.0\.1:\d+\/$/.test(hiddenWorkbenchUrl)) {
    throw Error("hidden workbench URL must be a 127.0.0.1 isolated server root");
  }
  if (fs.existsSync(output)) throw Error("output directory already exists");
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({headless: true, executablePath: browserExecutable});
  try {
    const context = await browser.newContext({viewport: {width: 1440, height: 1000}, deviceScaleFactor: 1});
    const page = await context.newPage();
    await page.goto(sourceUrl, {waitUntil: "load"});
    const statValues = await page.locator("[data-stat] strong").allTextContents();
    const tableRows = await page.locator("tbody tr").count();
    const images = await page.locator(".scene img").evaluateAll(nodes => nodes.map(node => ({complete: node.complete, width: node.naturalWidth, height: node.naturalHeight})));
    const deployment = await page.locator("[data-deployment]").getAttribute("data-deployment");
    const buttons = await page.locator("button").count();
    const screenshot = path.join(output, "review-page.png");
    await page.screenshot({path: screenshot, fullPage: true});
    let workbenchEntry = null;
    let workbenchScreenshotSha256 = null;
    let hiddenWorkbenchEntry = null;
    let hiddenWorkbenchScreenshotSha256 = null;
    if (workbenchUrl) {
      const shell = await context.newPage();
      await shell.goto(workbenchUrl, {waitUntil: "load"});
      const entry = shell.locator("#palletCandidateReviewEntry");
      await entry.waitFor({state: "visible", timeout: 5000});
      const href = await entry.getAttribute("href");
      const badge = await shell.locator("#palletCandidateReviewBadge").textContent();
      const target = await entry.getAttribute("target");
      const workbenchScreenshot = path.join(output, "workbench-entry.png");
      await shell.screenshot({path: workbenchScreenshot, fullPage: false});
      workbenchScreenshotSha256 = sha256(workbenchScreenshot);
      const popupPromise = context.waitForEvent("page");
      await entry.click();
      const popup = await popupPromise;
      await popup.waitForLoadState("load");
      workbenchEntry = {
        visible: await entry.isVisible(), href, badge, target,
        opened_read_only_page: (await popup.locator("[data-deployment]").getAttribute(
          "data-deployment")) === "not-deployed" && await popup.locator("button").count() === 0,
      };
      await popup.close();
      await shell.close();
    }
    if (hiddenWorkbenchUrl) {
      const shell = await context.newPage();
      await shell.goto(hiddenWorkbenchUrl, {waitUntil: "load"});
      await shell.waitForTimeout(500);
      const entry = shell.locator("#palletCandidateReviewEntry");
      const apiStatus = await shell.evaluate(async () => {
        const response = await fetch("/api/pallet-candidate-review", {cache: "no-store"});
        return response.status;
      });
      hiddenWorkbenchEntry = {
        hidden: await entry.isHidden(),
        data_ready: await entry.getAttribute("data-ready"),
        badge: await shell.locator("#palletCandidateReviewBadge").textContent(),
        api_status: apiStatus,
      };
      const hiddenScreenshot = path.join(output, "workbench-entry-hidden.png");
      await shell.screenshot({path: hiddenScreenshot, fullPage: false});
      hiddenWorkbenchScreenshotSha256 = sha256(hiddenScreenshot);
      await shell.close();
    }
    const checks = {
      stat_values: statValues,
      table_rows: tableRows,
      images,
      deployment,
      buttons,
      workbench_entry: workbenchEntry,
      hidden_workbench_entry: hiddenWorkbenchEntry,
      passed: JSON.stringify(statValues) === JSON.stringify(["5", "4", "1", "5"])
        && tableRows === 5 && images.length === 2
        && images.every(item => item.complete && item.width === 640 && item.height === 512)
        && deployment === "not-deployed" && buttons === 0
        && (!workbenchUrl || (workbenchEntry.visible
          && workbenchEntry.href === "/pallet-candidate-review/"
          && workbenchEntry.badge === "5" && workbenchEntry.target === "_blank"
          && workbenchEntry.opened_read_only_page))
        && (!hiddenWorkbenchUrl || (hiddenWorkbenchEntry.hidden
          && hiddenWorkbenchEntry.data_ready === null
          && hiddenWorkbenchEntry.badge === "0"
          && hiddenWorkbenchEntry.api_status === 404)),
    };
    const receipt = {schema_version: "labelagent.pallet_candidate_review_browser_acceptance.v1",
      source: pagePath ? "local_file" : "isolated_http",
      page_sha256: pagePath ? sha256(pagePath) : null,
      screenshot_sha256: sha256(screenshot),
      workbench_screenshot_sha256: workbenchScreenshotSha256,
      hidden_workbench_screenshot_sha256: hiddenWorkbenchScreenshotSha256,
      ...checks};
    fs.writeFileSync(path.join(output, "acceptance.json"), JSON.stringify(receipt, null, 2) + "\n");
    console.log(JSON.stringify(receipt, null, 2));
    if (!checks.passed) process.exitCode = 2;
  } finally { await browser.close(); }
})().catch(error => { console.error(error.stack || error); process.exit(1); });
