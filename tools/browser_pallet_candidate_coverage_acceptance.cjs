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
if (!playwrightRoot) throw Error("Playwright runtime not found");
const {chromium} = require(playwrightRoot);
const executable = [
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
].find(candidate => fs.existsSync(candidate));
if (!executable) throw Error("Edge or Chrome is required");

function arg(name) {
  const index = process.argv.indexOf(name);
  if (index < 0 || index + 1 >= process.argv.length) throw Error(`missing ${name}`);
  return process.argv[index + 1];
}
function sha256(file) {
  return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

(async () => {
  const pagePath = path.resolve(arg("--page"));
  const output = path.resolve(arg("--output"));
  if (fs.existsSync(output)) throw Error("output directory already exists");
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({headless: true, executablePath: executable});
  try {
    const context = await browser.newContext({viewport: {width: 1440, height: 1000}});
    const page = await context.newPage();
    const externalRequests = [];
    page.on("request", request => {
      if (!request.url().startsWith("file:")) externalRequests.push(request.url());
    });
    await page.goto(pathToFileURL(pagePath).href, {waitUntil: "load"});
    const stats = await page.locator(".stat strong").allTextContents();
    const rows = await page.locator("tbody tr").count();
    const safeSkips = await page.locator("tbody .skip").count();
    const triggers = await page.locator("tbody .trigger").count();
    const buttons = await page.locator("button").count();
    const contact = await page.locator(".contact").evaluate(image => ({
      complete: image.complete, width: image.naturalWidth, height: image.naturalHeight,
    }));
    const screenshot = path.join(output, "coverage-page.png");
    await page.screenshot({path: screenshot, fullPage: true});
    const passed = JSON.stringify(stats) === JSON.stringify(["12", "10", "2", "0 / 0"])
      && rows === 12 && safeSkips === 10 && triggers === 2 && buttons === 0
      && contact.complete && contact.width > 0 && contact.height > 0
      && externalRequests.length === 0;
    const receipt = {
      schema_version: "labelagent.pallet_candidate_coverage_browser_acceptance.v1",
      page_sha256: sha256(pagePath), screenshot_sha256: sha256(screenshot),
      stats, rows, safe_skips: safeSkips, triggers, buttons,
      contact_sheet: contact, external_requests: externalRequests, passed,
    };
    fs.writeFileSync(path.join(output, "acceptance.json"),
      JSON.stringify(receipt, null, 2) + "\n");
    console.log(JSON.stringify(receipt, null, 2));
    if (!passed) process.exitCode = 2;
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error.stack || error);
  process.exit(1);
});
