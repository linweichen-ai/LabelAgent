#!/usr/bin/env node
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {chromium} = require(
  "C:/Users/seer/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright");

const base = (process.env.LABELAGENT_ACCEPTANCE_URL || "http://127.0.0.1:8765").replace(/\/$/, "");
const output = path.resolve(process.env.LABELAGENT_ACCEPTANCE_OUTPUT || "artifacts/live-progress-browser-acceptance");
const expectedTask = process.env.LABELAGENT_EXPECTED_TASK_ID || "";

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({channel: "msedge", headless: true});
  const page = await browser.newPage({viewport: {width: 1760, height: 1120}});
  const pageErrors = [], failedRequests = [], httpErrors = [], truthWrites = [];
  page.on("pageerror", error => pageErrors.push(error.message));
  page.on("console", message => { if (message.type() === "error") pageErrors.push(message.text()); });
  page.on("requestfailed", request => failedRequests.push(`${request.method()} ${request.url()} ${request.failure()?.errorText || ""}`));
  page.on("response", response => { if (response.status() >= 400) httpErrors.push(`${response.status()} ${response.url()}`); });
  page.on("request", request => {
    if (request.method() !== "GET" && /\/api\/(truth|human_review)(?:$|\?)/.test(request.url())) {
      truthWrites.push(`${request.method()} ${request.url()}`);
    }
  });
  try {
    const response = await page.goto(base, {waitUntil: "domcontentloaded", timeout: 60000});
    assert.equal(response.status(), 200);
    await page.evaluate(() => window.labelAgentBootstrapReady);
    await page.locator("#autoModeButton").click();
    await page.waitForFunction(() => document.body.dataset.workspace === "auto");
    await page.waitForFunction(() => document.querySelector("#runProgressState")?.classList.contains("running"));
    await page.waitForFunction(() => {
      const text = document.querySelector("#runProgressPercent")?.textContent || "";
      return text.startsWith("约 ") && text !== "约 0%";
    });
    const progress = await page.evaluate(async () => {
      const job = await fetch("/api/run/status", {cache: "no-store"}).then(response => response.json());
      return {
        state: document.querySelector("#runProgressState")?.textContent,
        title: document.querySelector("#runProgressTitle")?.textContent,
        detail: document.querySelector("#runProgressDetail")?.textContent,
        percent: document.querySelector("#runProgressPercent")?.textContent,
        frames: document.querySelector("#runProgressFrames")?.textContent,
        frame: document.querySelector("#runProgressFrame")?.textContent,
        round: document.querySelector("#runProgressRound")?.textContent,
        elapsed: document.querySelector("#runProgressElapsed")?.textContent,
        task_id: job.task_id || null,
      };
    });
    if (expectedTask) assert.equal(progress.task_id, expectedTask);
    assert.notEqual(progress.percent, "约 0%");
    assert.notEqual(progress.frames, "0 / 0");
    assert.deepEqual(truthWrites, []);
    assert.deepEqual(pageErrors, []);
    assert.deepEqual(failedRequests, []);
    assert.deepEqual(httpErrors, []);
    const screenshot = path.join(output, "live-progress.png");
    await page.screenshot({path: screenshot, fullPage: true});
    const result = {
      schema_version: "labelagent.live-progress-browser-acceptance.v1",
      passed: true,
      progress,
      truth_write_requests: truthWrites,
      page_errors: pageErrors,
      failed_requests: failedRequests,
      http_errors: httpErrors,
      screenshot,
    };
    fs.writeFileSync(path.join(output, "result.json"), JSON.stringify(result, null, 2) + "\n", "utf8");
    process.stdout.write(JSON.stringify(result) + "\n");
  } finally {
    await browser.close();
  }
})().catch(error => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
