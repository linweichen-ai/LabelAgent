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
function sha256(file) { return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex"); }

(async () => {
  const urlIndex = process.argv.indexOf("--url");
  const pagePath = urlIndex < 0 ? path.resolve(arg("--page")) : null;
  const sourceUrl = pagePath ? pathToFileURL(pagePath).href : arg("--url");
  if (!sourceUrl.startsWith("file:") && !/^http:\/\/127\.0\.0\.1:\d+\//.test(sourceUrl)) {
    throw Error("acceptance URL must use a local file or 127.0.0.1 isolated server");
  }
  const output = path.resolve(arg("--output"));
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
    const checks = {
      stat_values: statValues,
      table_rows: tableRows,
      images,
      deployment,
      buttons,
      passed: JSON.stringify(statValues) === JSON.stringify(["5", "4", "1", "5"])
        && tableRows === 5 && images.length === 2
        && images.every(item => item.complete && item.width === 640 && item.height === 512)
        && deployment === "not-deployed" && buttons === 0,
    };
    const receipt = {schema_version: "labelagent.pallet_candidate_review_browser_acceptance.v1",
      source: pagePath ? "local_file" : "isolated_http",
      page_sha256: pagePath ? sha256(pagePath) : null,
      screenshot_sha256: sha256(screenshot), ...checks};
    fs.writeFileSync(path.join(output, "acceptance.json"), JSON.stringify(receipt, null, 2) + "\n");
    console.log(JSON.stringify(receipt, null, 2));
    if (!checks.passed) process.exitCode = 2;
  } finally { await browser.close(); }
})().catch(error => { console.error(error.stack || error); process.exit(1); });
