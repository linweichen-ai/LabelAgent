#!/usr/bin/env node
"use strict";

const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const http = require("node:http");
const os = require("node:os");
const path = require("node:path");

const playwrightCandidates = [
  path.resolve(path.dirname(process.execPath), "..", "node_modules", "playwright"),
  process.env.USERPROFILE && path.join(process.env.USERPROFILE, ".cache", "codex-runtimes",
    "codex-primary-runtime", "dependencies", "node", "node_modules", "playwright"),
].filter(Boolean);
const playwrightRoot = playwrightCandidates.find(candidate => fs.existsSync(candidate));
if (!playwrightRoot) throw new Error(`Playwright runtime not found; checked: ${playwrightCandidates.join(", ")}`);
const {chromium} = require(playwrightRoot);

function parseArgs(argv) {
  const result = {};
  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    if (key === "--screenshot") {
      result.screenshot = true;
      continue;
    }
    if (!["--page", "--output"].includes(key) || i + 1 >= argv.length) {
      throw new Error("usage: node tools/browser_cvat_human_signoff_draft_acceptance.cjs --page PAGE --output NEW_DIR [--screenshot]");
    }
    result[key.slice(2)] = argv[++i];
  }
  if (!result.page || !result.output) throw new Error("--page and --output are required");
  return result;
}

function sha256File(file) {
  return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

function writeExclusive(file, value) {
  fs.writeFileSync(file, value, {encoding: "utf8", flag: "wx"});
}

async function startLoopbackServer(pageFile) {
  const body = fs.readFileSync(pageFile);
  const server = http.createServer((request, response) => {
    const url = new URL(request.url, "http://127.0.0.1");
    if (request.method !== "GET" || !["/", "/index.html"].includes(url.pathname)) {
      response.writeHead(404, {"Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store"});
      response.end("not found");
      return;
    }
    response.writeHead(200, {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store"});
    response.end(body);
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  assert.equal(address.address, "127.0.0.1");
  return {server, base: `http://127.0.0.1:${address.port}`};
}

async function closeServer(server) {
  await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
}

async function captureDownload(page, selector, destination) {
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.locator(selector).click(),
  ]);
  await download.saveAs(destination);
  assert.equal(await download.failure(), null);
  return JSON.parse(fs.readFileSync(destination, "utf8"));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const pageFile = path.resolve(args.page);
  const output = path.resolve(args.output);
  if (!fs.statSync(pageFile).isFile()) throw new Error("reviewer page is not a file");
  if (fs.existsSync(output)) throw new Error("output directory already exists");
  const outputParent = path.dirname(output);
  if (!fs.existsSync(outputParent)) fs.mkdirSync(outputParent, {recursive: true});
  fs.mkdirSync(output, {recursive: false});

  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "labelagent-signoff-browser-"));
  const draftPath = path.join(temp, "human-signoff-draft.json");
  const fullDraftPath = path.join(temp, "human-signoff-full-draft.json");
  const badDraftPath = path.join(temp, "human-signoff-bad-draft.json");
  const officialPath = path.join(temp, "human-signoff.json");
  let browser;
  let server;
  const blockedRequests = [];
  const pageErrors = [];
  const started = Date.now();
  try {
    const local = await startLoopbackServer(pageFile);
    server = local.server;
    browser = await chromium.launch({channel: "msedge", headless: true});
    const page = await browser.newPage({viewport: {width: 1440, height: 960}, acceptDownloads: true});
    page.on("pageerror", error => pageErrors.push(error.message));
    page.on("console", message => { if (message.type() === "error") pageErrors.push(message.text()); });
    await page.route("**/*", async route => {
      const url = new URL(route.request().url());
      if (url.protocol === "data:" || url.protocol === "blob:"
          || (url.protocol === "http:" && url.hostname === "127.0.0.1" && url.port === new URL(local.base).port)) {
        await route.continue();
      } else {
        blockedRequests.push(`${route.request().method()} ${url.origin}${url.pathname}`);
        await route.abort("blockedbyclient");
      }
    });
    await page.goto(local.base, {waitUntil: "domcontentloaded", timeout: 30000});
    await page.locator("#position").waitFor({state: "visible"});
    assert.equal(await page.locator("#position").textContent(), "1 / 81");
    assert.equal(await page.locator("#export").isDisabled(), true);

    await page.locator('input[name="decision"][value="uncertain"]').check();
    await page.locator("#reviewer").fill("synthetic-browser-reviewer");
    await page.locator("#reviewedAt").fill("2026-11-22T12:00:00+08:00");
    await page.locator("#note").fill("synthetic browser draft roundtrip");
    const draft = await captureDownload(page, "#exportDraft", draftPath);
    assert.equal(draft.schema_version, "labelagent.cvat-human-signoff-draft.v1");
    assert.equal(draft.rows.length, 81);
    assert.equal(draft.rows[0].decision, "uncertain");
    assert.equal(draft.rows[0].reviewer, "synthetic-browser-reviewer");
    assert.equal(Object.hasOwn(draft, "training_authorized"), false);

    await page.reload({waitUntil: "domcontentloaded"});
    assert.equal(await page.locator("#reviewer").inputValue(), "");
    await page.locator("#importDraft").setInputFiles(draftPath);
    await page.waitForFunction(() => document.querySelector("#status")?.textContent === "草稿已导入");
    assert.equal(await page.locator("#reviewer").inputValue(), "synthetic-browser-reviewer");
    assert.equal(await page.locator("#note").inputValue(), "synthetic browser draft roundtrip");
    assert.equal(await page.locator('input[name="decision"][value="uncertain"]').isChecked(), true);
    assert.equal(await page.locator("#export").isDisabled(), true);

    const badDraft = structuredClone(draft);
    badDraft.rows[0].focused_image_sha256 = "0".repeat(64);
    writeExclusive(badDraftPath, `${JSON.stringify(badDraft, null, 2)}\n`);
    await page.locator("#importDraft").setInputFiles(badDraftPath);
    await page.waitForFunction(() => document.querySelector("#status")?.textContent === "草稿导入失败");
    assert.equal(await page.locator("#reviewer").inputValue(), "synthetic-browser-reviewer");
    assert.equal(await page.locator("#note").inputValue(), "synthetic browser draft roundtrip");
    assert.equal(await page.locator('input[name="decision"][value="uncertain"]').isChecked(), true);

    const fullDraft = structuredClone(draft);
    fullDraft.rows.forEach((row, index) => {
      row.decision = index % 3 === 0 ? "pass" : index % 3 === 1 ? "reject" : "uncertain";
      row.reviewer = "synthetic-browser-reviewer";
      row.reviewed_at = "2026-11-22T12:00:00+08:00";
      row.note = `synthetic-${index}`;
    });
    writeExclusive(fullDraftPath, `${JSON.stringify(fullDraft, null, 2)}\n`);
    await page.locator("#importDraft").setInputFiles(fullDraftPath);
    await page.waitForFunction(() => document.querySelector("#status")?.textContent === "草稿已导入");
    assert.equal(await page.locator("#counter").textContent(), "未完成 0 / 81");
    assert.equal(await page.locator("#export").isEnabled(), true);
    const official = await captureDownload(page, "#export", officialPath);
    assert.equal(official.schema_version, "labelagent.cvat-human-instance-signoff.v1");
    assert.equal(official.human_signoff_count, 81);
    assert.equal(official.rows.length, 81);
    assert.equal(official.training_authorized, false);
    assert.equal(official.automatic_approval, false);
    assert.equal(official.approved_for_training, false);
    assert.equal(official.human_pass_is_training_authorization, false);
    assert.equal(official.rows.every(row => row.approved_for_training === false), true);
    assert.deepEqual(blockedRequests, []);
    assert.deepEqual(pageErrors, []);

    if (args.screenshot) {
      await page.screenshot({path: path.join(output, "reviewer-browser-acceptance.png"), fullPage: true});
    }
    const receipt = {
      schema_version: "labelagent.cvat-human-signoff-browser-acceptance.v1",
      passed: true,
      reviewer_page_sha256: sha256File(pageFile),
      candidate_count: 81,
      partial_draft_roundtrip: true,
      refresh_cleared_memory_before_import: true,
      invalid_binding_rejected_without_state_change: true,
      official_export_blocked_before_completion: true,
      official_export_enabled_after_complete_synthetic_draft: true,
      downloaded_draft_sha256: sha256File(draftPath),
      downloaded_official_sha256: sha256File(officialPath),
      training_authorized: official.training_authorized,
      automatic_approval: official.automatic_approval,
      approved_for_training: official.approved_for_training,
      external_requests: blockedRequests.length,
      page_errors: pageErrors,
      screenshot_captured: Boolean(args.screenshot),
      elapsed_ms: Date.now() - started,
    };
    writeExclusive(path.join(output, "receipt.json"), `${JSON.stringify(receipt, null, 2)}\n`);
    process.stdout.write(`${JSON.stringify(receipt)}\n`);
  } finally {
    if (browser) await browser.close();
    if (server) await closeServer(server);
    fs.rmSync(temp, {recursive: true, force: true});
  }
}

main().catch(error => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
