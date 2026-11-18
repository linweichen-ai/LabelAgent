#!/usr/bin/env node
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {chromium} = require(
  "C:/Users/seer/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright");

const base = (process.env.LABELAGENT_ACCEPTANCE_URL || "http://127.0.0.1:8765").replace(/\/$/, "");
const output = path.resolve(process.env.LABELAGENT_ACCEPTANCE_OUTPUT || "artifacts/dataset-coverage-browser-acceptance");

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({channel: "msedge", headless: true});
  const page = await browser.newPage({viewport: {width: 1760, height: 1120}, deviceScaleFactor: 1});
  const errors = [], failedRequests = [], httpErrors = [], truthWrites = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
  page.on("requestfailed", request => failedRequests.push(`${request.method()} ${request.url()} ${request.failure()?.errorText || ""}`));
  page.on("response", response => { if (response.status() >= 400) httpErrors.push(`${response.request().method()} ${response.status()} ${response.url()}`); });
  page.on("request", request => {
    if (request.method() !== "GET" && /\/api\/(truth|human_review)(?:$|\?)/.test(request.url())) {
      truthWrites.push(`${request.method()} ${request.url()}`);
    }
  });
  const started = Date.now();
  try {
    const response = await page.goto(base, {waitUntil: "networkidle", timeout: 60000});
    assert.equal(response.status(), 200);
    await page.evaluate(() => window.labelAgentBootstrapReady);
    await page.waitForFunction(() => document.querySelector("#datasetIdentityName")?.textContent.includes("SFL-CPD15-Y"));
    const identity = await page.evaluate(() => ({
      kind: document.querySelector("#datasetIdentityKind")?.textContent,
      name: document.querySelector("#datasetIdentityName")?.textContent,
      detail: document.querySelector("#datasetIdentityDetail")?.textContent,
      title: document.querySelector("#datasetIdentity")?.title,
    }));
    assert.equal(identity.kind, "新数据集");
    assert(identity.detail.includes("/capture/camera_"));

    await page.locator("#datasetIdentitySwitch").click();
    await page.locator("#fileManager[open]").waitFor({state: "visible"});
    const newRow = page.locator(".fileRow", {hasText: "SFL-CPD15-Y · 批次 A（0920）· 000"});
    await newRow.waitFor({state: "visible"});
    const fileRow = await newRow.innerText();
    assert(fileRow.includes("新数据集"));
    assert(fileRow.includes("当前已加载"));
    await page.locator("#closeFilesDialog").click();

    await page.locator("#autoModeButton").click();
    await page.waitForFunction(() => document.querySelectorAll("#autoLabelOptions input:checked").length === 8);
    await page.waitForFunction(() => document.querySelectorAll("#candidateSummary > span").length >= 8);
    const coverage = await page.evaluate(() => ({
      checked: [...document.querySelectorAll("#autoLabelOptions input:checked")].map(item => item.value),
      summary: document.querySelector("#candidateSummary")?.innerText,
      hint: document.querySelector("#autoLabelHint")?.textContent,
    }));
    assert.equal(coverage.checked.length, 8);
    assert(coverage.summary.includes("已选 8/8 类"));
    assert(coverage.hint.includes("不等于每帧必有该类"));

    await page.evaluate(() => window.openReferencePreparation());
    await page.locator("#referencePrepDialog[open]").waitFor({state: "visible"});
    const allSelectedOption = page.locator("#referenceAutoLabel option", {hasText: "逐类尝试全部已选推理类别"});
    assert.equal(await allSelectedOption.count(), 1);
    await page.locator("#referencePrepClose").click();
    await page.screenshot({path: path.join(output, "new-dataset-class-coverage.png"), fullPage: true});

    assert.deepEqual(truthWrites, []);
    assert.deepEqual(errors, []);
    assert.deepEqual(failedRequests, []);
    assert.deepEqual(httpErrors, []);
    const result = {
      schema_version: "labelagent.dataset-coverage-browser-acceptance.v1",
      passed: true,
      elapsed_ms: Date.now() - started,
      identity,
      file_row: fileRow,
      coverage,
      forced_all_selected_option: true,
      truth_write_requests: truthWrites,
      page_errors: errors,
      failed_requests: failedRequests,
      http_errors: httpErrors,
      screenshot: path.join(output, "new-dataset-class-coverage.png"),
    };
    fs.writeFileSync(path.join(output, "result.json"), JSON.stringify(result, null, 2) + "\n", "utf8");
    process.stdout.write(JSON.stringify(result) + "\n");
  } finally {
    await browser.close();
  }
})().catch(error => { process.stderr.write((error.stack || error) + "\n"); process.exitCode = 1; });
