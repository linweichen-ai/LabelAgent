#!/usr/bin/env node
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {chromium} = require(
  "C:/Users/seer/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright");

const base = (process.env.LABELAGENT_ACCEPTANCE_URL || "http://127.0.0.1:8765").replace(/\/$/, "");
const frame = Number(process.env.LABELAGENT_ACCEPTANCE_FRAME || 46);
const output = path.resolve(process.env.LABELAGENT_ACCEPTANCE_OUTPUT || "artifacts/gpt-trace-browser-acceptance");

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({channel: "msedge", headless: true});
  // The inspection workbench uses a 100dvh layout with internal scrolling.
  // A taller acceptance viewport keeps the selected GPT stage visibly inside
  // the captured page instead of producing a one-pixel scroll-container crop.
  const page = await browser.newPage({viewport: {width: 1760, height: 2200}});
  const pageErrors = [], failedRequests = [], httpErrors = [], truthWrites = [];
  page.on("pageerror", error => pageErrors.push(error.message));
  page.on("requestfailed", request => failedRequests.push({
    method: request.method(), url: request.url(), error: request.failure()?.errorText || "unknown",
  }));
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
    await page.evaluate(target => {
      const input = document.querySelector("#frameNumber");
      input.value = String(target);
      input.dispatchEvent(new Event("change", {bubbles: true}));
    }, frame);
    await page.waitForFunction(target => Number(document.querySelector("#frameNumber")?.value) === target, frame);
    await page.waitForFunction(() => document.querySelector("#frameImage")?.complete);
    await page.waitForFunction(() => document.querySelector("#inspectionStage3")?.dataset.available === "true");

    await page.locator("#inspectionStage3").click();
    await page.waitForFunction(() => document.querySelector("#inspectionBody")?.innerText.includes("再次发送给 GPT 的完整 Prompt"));
    const reviewInput = await page.locator("#inspectionBody").innerText();
    assert(reviewInput.includes("GPT 输入图"));
    assert(!reviewInput.includes("暂无此阶段记录"));
    const reviewScreenshot = path.join(output, `frame${frame}-gpt-review-input.png`);
    await page.locator("#inspectionBody").evaluate(element => { element.scrollTop = 0; });
    await page.screenshot({path: reviewScreenshot, fullPage: true});

    await page.locator("#inspectionStage4").click();
    await page.waitForFunction(() => document.querySelector("#inspectionBody")?.innerText.includes("GPT 第 1 轮决策"));
    const decisionOutput = await page.locator("#inspectionBody").innerText();
    assert(decisionOutput.includes("判断与依据"));
    assert(decisionOutput.includes("要求 Agent 执行的操作"));
    assert(!decisionOutput.includes("[]"));
    const decisionScreenshot = path.join(output, `frame${frame}-gpt-decision-output.png`);
    await page.locator("#inspectionBody").evaluate(element => { element.scrollTop = 0; });
    await page.screenshot({path: decisionScreenshot, fullPage: true});

    const result = {
      passed: true,
      frame,
      review_input_visible: true,
      decision_output_visible: true,
      review_input_excerpt: reviewInput.slice(0, 600),
      decision_output_excerpt: decisionOutput.slice(0, 600),
      review_screenshot: reviewScreenshot,
      decision_screenshot: decisionScreenshot,
      truth_writes: truthWrites,
      page_errors: pageErrors,
      failed_requests: failedRequests,
      http_errors: httpErrors,
    };
    assert.deepEqual(truthWrites, []);
    assert.deepEqual(pageErrors, []);
    const blockingFailures = failedRequests.filter(item => !(
      item.method === "GET" && /\/api\/frame\/\d+/.test(item.url) && item.error === "net::ERR_ABORTED"));
    assert.deepEqual(blockingFailures, []);
    assert.deepEqual(httpErrors, []);
    fs.writeFileSync(path.join(output, "result.json"), JSON.stringify(result, null, 2));
    process.stdout.write(JSON.stringify(result, null, 2) + "\n");
  } finally {
    await browser.close();
  }
})().catch(error => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
