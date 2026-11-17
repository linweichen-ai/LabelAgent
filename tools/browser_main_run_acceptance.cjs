#!/usr/bin/env node
"use strict";

// Real browser acceptance for the primary "开始自动标注" workflow.
// Run only against an isolated workbench snapshot with a verified reference.

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(
  "C:/Users/seer/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright");

function required(name) {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function integer(name, fallback) {
  const value = Number(process.env[name] ?? fallback);
  if (!Number.isInteger(value) || value < 0) throw new Error(`${name} must be a non-negative integer`);
  return value;
}

(async () => {
  const base = required("BASE_URL").replace(/\/$/, "");
  const outputJson = path.resolve(required("OUTPUT_JSON"));
  const screenshot = path.resolve(required("SCREENSHOT"));
  const frame = integer("FRAME", 0);
  const count = integer("COUNT", 1);
  const topic = process.env.TOPIC || "/capture/camera_1/image";
  assert(count >= 1 && count <= 100);
  fs.mkdirSync(path.dirname(outputJson), { recursive: true });
  fs.mkdirSync(path.dirname(screenshot), { recursive: true });

  const browser = await chromium.launch({ channel: "msedge", headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1050 } });
    const pageErrors = [];
    const requestCounts = new Map();
    page.on("pageerror", error => pageErrors.push(error.message));
    page.on("request", request => {
      const endpoint = new URL(request.url()).pathname;
      requestCounts.set(endpoint, (requestCounts.get(endpoint) || 0) + 1);
    });

    const started = Date.now();
    const response = await page.goto(base, { waitUntil: "networkidle" });
    assert.equal(response.status(), 200);
    await page.evaluate(() => window.labelAgentBootstrapReady);
    await page.locator("#autoModeButton").click();

    const primaryReady = async () => await page.locator("#primaryWindow").isVisible()
      && await page.locator("#primaryTitle").textContent() === topic;
    if (!await primaryReady()) {
      await page.locator("#addWindowButton").click();
      await page.locator("#addWindowDialog").waitFor({ state: "visible" });
      const topicOptions = page.locator("#windowSources input");
      let topicOption = null;
      for (let index = 0; index < await topicOptions.count(); index += 1) {
        const option = topicOptions.nth(index);
        if (await option.inputValue() === topic) { topicOption = option; break; }
      }
      assert(topicOption, `topic option not found: ${topic}`);
      await topicOption.check();
      await page.locator("#addWindowForm button[type=submit]").click();
      if (!await primaryReady()) {
        const sensorWindow = page.locator(".sensorWindow:visible")
          .filter({ hasText: `图像 · ${topic}` }).last();
        await sensorWindow.waitFor({ state: "visible", timeout: 30000 });
        await sensorWindow.locator('button[title="进入标注模式"]').click();
      }
    }
    await page.locator("#primaryWindow").waitFor({ state: "visible", timeout: 30000 });
    await page.waitForFunction(expected =>
      document.getElementById("primaryTitle")?.textContent === expected
        && Number((document.getElementById("frameTotal")?.textContent || "").replace(/\D/g, "")) > 0,
    topic, { timeout: 30000 });

    await page.locator("#frameNumber").evaluate(async (input, expected) => {
      input.value = String(expected);
      await input.onchange({ target: input });
    }, frame);
    assert.equal(Number(await page.locator("#frameNumber").inputValue()), frame);
    await page.locator("#taskCount").fill(String(count));

    const initial = await page.evaluate(() => {
      const state = window.currentWorkbenchFrameState();
      return {
        file: state.file,
        topic: state.topic,
        frame: Number(document.getElementById("frameNumber")?.value),
        truth: state.truth,
        candidates: state.candidates,
        reference_frames: state.reference_frames,
      };
    });
    assert.equal(initial.frame, frame);
    assert.equal(Object.hasOwn(initial.truth, String(frame)), false);
    assert.equal(Object.hasOwn(initial.candidates, String(frame)), false);

    await page.evaluate(() => {
      window.__labelAgentMainRunProgress = [];
      window.__labelAgentMainRunStartedAt = performance.now();
      const panel = document.getElementById("runProgress");
      const capture = () => {
        const sample = {
          at_ms: Math.round(performance.now() - window.__labelAgentMainRunStartedAt),
          hidden: Boolean(panel?.hidden),
          state: document.getElementById("runProgressState")?.textContent || "",
          title: document.getElementById("runProgressTitle")?.textContent || "",
          detail: document.getElementById("runProgressDetail")?.textContent || "",
          percent: document.getElementById("runProgressPercent")?.textContent || "",
          frames: document.getElementById("runProgressFrames")?.textContent || "",
          button: document.getElementById("runButton")?.textContent || "",
        };
        const prior = window.__labelAgentMainRunProgress.at(-1);
        if (!prior || JSON.stringify({ ...prior, at_ms: 0 }) !== JSON.stringify({ ...sample, at_ms: 0 })) {
          window.__labelAgentMainRunProgress.push(sample);
        }
      };
      new MutationObserver(capture).observe(document.body, {
        subtree: true, childList: true, characterData: true, attributes: true,
      });
      capture();
    });

    await page.locator("#runButton").click();
    await page.locator("#runReadinessDialog").waitFor({ state: "visible" });
    const readiness = await page.locator("#runReadinessDialog [data-readiness]").allTextContents();
    assert.equal(await page.locator("#runReadinessConfirm").isEnabled(), true);

    const runResponsePromise = page.waitForResponse(item =>
      new URL(item.url()).pathname === "/api/run" && item.request().method() === "POST",
    { timeout: 300000 });
    const runStarted = Date.now();
    await page.locator("#runReadinessConfirm").click();
    await page.waitForFunction(() => {
      const text = document.getElementById("runProgressPercent")?.textContent || "";
      const value = Number.parseInt(text, 10);
      return Number.isFinite(value) && value > 0;
    }, null, { timeout: 30000 });
    const firstNonzeroProgressMs = Date.now() - runStarted;

    const runResponse = await runResponsePromise;
    const runPayload = await runResponse.json();
    assert.equal(runResponse.status(), 200);
    await page.waitForFunction(() => {
      const value = document.getElementById("runProgressState")?.textContent || "";
      return value.includes("完成") || value.includes("失败") || value.includes("停止");
    }, null, { timeout: 30000 });
    await page.waitForTimeout(500);

    const final = await page.evaluate(() => {
      const state = window.currentWorkbenchFrameState();
      return {
        file: state.file,
        topic: state.topic,
        frame: Number(document.getElementById("frameNumber")?.value),
        truth: state.truth,
        candidates: state.candidates,
        reference_frames: state.reference_frames,
        progress: window.__labelAgentMainRunProgress,
        progress_state: document.getElementById("runProgressState")?.textContent || "",
        progress_title: document.getElementById("runProgressTitle")?.textContent || "",
        progress_percent: document.getElementById("runProgressPercent")?.textContent || "",
        run_button: document.getElementById("runButton")?.textContent || "",
      };
    });
    assert.deepEqual(final.truth, initial.truth);
    assert.deepEqual(final.reference_frames, initial.reference_frames);
    assert.deepEqual(pageErrors, []);

    const frameResult = Array.isArray(runPayload.frames)
      ? runPayload.frames.find(item => item.frame === frame) : null;
    assert(frameResult, "run response does not contain the requested frame");
    assert.equal(Boolean(frameResult.failed), false);
    const candidates = final.candidates?.[String(frame)] || [];
    const visibleProgress = final.progress.filter(item => !item.hidden);
    assert(visibleProgress.length > 0);
    assert(visibleProgress.some(item => Number.parseInt(item.percent, 10) > 0));

    await page.screenshot({ path: screenshot, fullPage: true });
    const result = {
      schema_version: "labelagent.browser-main-run-acceptance.v1",
      base_url: base,
      page_status: response.status(),
      frame,
      count,
      readiness,
      first_nonzero_progress_ms: firstNonzeroProgressMs,
      run_response_ms: Date.now() - runStarted,
      total_ms: Date.now() - started,
      run_task_id: runPayload.task_id,
      run_stopped: runPayload.stopped === true,
      failed_frames: runPayload.failed_frames || [],
      candidate_count: candidates.length,
      candidate_counts_by_label: Object.fromEntries([...new Set(candidates.map(x => x.label))]
        .sort().map(label => [label, candidates.filter(x => x.label === label).length])),
      truth_unchanged: true,
      reference_frames_unchanged: true,
      progress_final: {
        state: final.progress_state,
        title: final.progress_title,
        percent: final.progress_percent,
        button: final.run_button,
      },
      progress_samples: final.progress,
      endpoint_counts: Object.fromEntries([...requestCounts.entries()].sort()),
      page_errors: pageErrors,
      screenshot,
    };
    fs.writeFileSync(outputJson, JSON.stringify(result, null, 2) + "\n", "utf8");
    process.stdout.write(JSON.stringify(result) + "\n");
  } finally {
    await browser.close();
  }
})().catch(error => {
  process.stderr.write(error.stack + "\n");
  process.exitCode = 1;
});
