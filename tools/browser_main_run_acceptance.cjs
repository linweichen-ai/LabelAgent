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

    let frameSelected = false;
    for (let attempt = 0; attempt < 3 && !frameSelected; attempt += 1) {
      await page.locator("#frameNumber").fill(String(frame));
      await page.locator("#frameNumber").dispatchEvent("change");
      try {
        await page.waitForFunction(expected => {
          const input = document.getElementById("frameNumber");
          const image = document.getElementById("frameImage");
          if (Number(input?.value) !== expected || !image?.complete || !image.src) return false;
          return new URL(image.src).pathname === `/api/frame/${expected}`;
        }, frame, { timeout: 15000 });
        await page.waitForTimeout(300);
        frameSelected = Number(await page.locator("#frameNumber").inputValue()) === frame;
      } catch (error) {
        if (attempt === 2) throw error;
      }
    }
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

    await page.locator("#toast").evaluate(element => {
      element.textContent = "";
      element.style.display = "none";
    });
    const runRequestPromise = page.waitForRequest(item =>
      new URL(item.url()).pathname === "/api/run" && item.method() === "POST",
    { timeout: 60000 }).then(runRequest => ({ runRequest }));
    const preflightFailurePromise = page.waitForFunction(() => {
      const button = document.getElementById("runButton");
      const toast = document.getElementById("toast");
      return button?.getAttribute("aria-busy") === "false"
        && toast?.style.display === "block" && Boolean(toast.textContent?.trim())
        ? toast.textContent.trim() : false;
    }, null, { timeout: 60000 }).then(async handle => ({ preflightFailure: await handle.jsonValue() }))
      .catch(() => null);
    const refinementResponsePromise = page.waitForResponse(item =>
      new URL(item.url()).pathname === "/api/reference_candidates/refine"
        && item.request().method() === "POST",
    { timeout: 300000 }).catch(() => null);
    const runStarted = Date.now();
    await page.locator("#runReadinessConfirm").click();
    await page.waitForFunction(() => {
      const text = document.getElementById("runProgressPercent")?.textContent || "";
      const value = Number.parseInt(text, 10);
      return Number.isFinite(value) && value > 0;
    }, null, { timeout: 30000 });
    const firstNonzeroProgressMs = Date.now() - runStarted;

    const runOutcome = await Promise.race([runRequestPromise, preflightFailurePromise]);
    assert(runOutcome, "automatic-label preflight produced no request or explicit failure");
    assert(!runOutcome.preflightFailure,
      `automatic-label preflight rejected the run: ${runOutcome.preflightFailure}`);
    const runResponse = await runOutcome.runRequest.response();
    assert(runResponse, "automatic-label request ended without an HTTP response");
    const runPayload = await runResponse.json();
    const runResponseMs = Date.now() - runStarted;
    assert.equal(runResponse.status(), 200);
    await page.waitForFunction(requestedFrame => {
      const state = window.currentWorkbenchFrameState();
      return Array.isArray(state.candidates?.[String(requestedFrame)])
        && state.candidates[String(requestedFrame)].length > 0;
    }, frame, { timeout: 30000 });
    const candidateVisibleMs = Date.now() - runStarted;
    let refinementResponse = null;
    if (runPayload.agent_review_deferred === true) {
      refinementResponse = await refinementResponsePromise;
      assert(refinementResponse, "deferred run did not submit Agent refinement");
      assert.equal(refinementResponse.status(), 200);
      await page.waitForFunction(requestedFrame => {
        const state = window.currentWorkbenchFrameState();
        return state.agent_runs?.[String(requestedFrame)]?.agent_review_status === "complete";
      }, frame, { timeout: 30000 });
    }
    await page.waitForFunction(() => {
      const value = document.getElementById("runProgressState")?.textContent || "";
      return value.includes("完成") || value.includes("失败") || value.includes("停止");
    }, null, { timeout: 30000 });
    await page.waitForTimeout(300);

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
      run_response_ms: runResponseMs,
      candidate_visible_ms: candidateVisibleMs,
      agent_review_deferred: runPayload.agent_review_deferred === true,
      refinement_response_status: refinementResponse?.status() ?? null,
      background_complete_ms: Date.now() - runStarted,
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
