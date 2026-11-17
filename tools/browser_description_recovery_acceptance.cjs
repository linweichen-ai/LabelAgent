#!/usr/bin/env node
"use strict";

// Browser acceptance for the recoverable description gate. Run only against an
// isolated work directory: the test intentionally regenerates a derived
// description and starts one candidate-only inference.

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

(async () => {
  const base = required("BASE_URL").replace(/\/$/, "");
  const outputJson = path.resolve(required("OUTPUT_JSON"));
  const screenshot = path.resolve(required("SCREENSHOT"));
  const frame = Number(process.env.FRAME || 34);
  const topic = process.env.TOPIC || "/capture/camera_1/image";
  const gateOnly = process.env.GATE_ONLY === "1";
  assert(Number.isInteger(frame) && frame >= 0);
  fs.mkdirSync(path.dirname(outputJson), { recursive: true });
  fs.mkdirSync(path.dirname(screenshot), { recursive: true });

  const browser = await chromium.launch({ channel: "msedge", headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1050 } });
    const pageErrors = [];
    const requests = [];
    page.on("pageerror", error => pageErrors.push(error.message));
    page.on("request", request => requests.push({
      path: new URL(request.url()).pathname,
      method: request.method(),
      at: Date.now(),
    }));

    const started = Date.now();
    const response = await page.goto(base, { waitUntil: "networkidle" });
    assert.equal(response.status(), 200);
    await page.evaluate(() => window.labelAgentBootstrapReady);
    await page.locator("#autoModeButton").click();
    await page.waitForFunction(expected =>
      document.getElementById("primaryTitle")?.textContent === expected
        && Number((document.getElementById("frameTotal")?.textContent || "").replace(/\D/g, "")) > 0,
    topic, { timeout: 30000 });

    await page.locator("#frameNumber").fill(String(frame));
    await page.locator("#frameNumber").dispatchEvent("change");
    await page.waitForFunction(expected => {
      const image = document.getElementById("frameImage");
      return Number(document.getElementById("frameNumber")?.value) === expected
        && image?.complete && image.src && new URL(image.src).pathname === `/api/frame/${expected}`;
    }, frame, { timeout: 30000 });
    await page.locator("#taskCount").fill("1");

    const initial = await page.evaluate(() => {
      const state = window.currentWorkbenchFrameState();
      return {
        truth: state.truth,
        reference_frames: state.reference_frames,
        candidate_present: Object.hasOwn(state.candidates || {}, document.getElementById("frameNumber").value),
      };
    });
    assert.equal(Object.hasOwn(initial.truth, String(frame)), false);
    assert.equal(initial.candidate_present, false);

    await page.evaluate(() => {
      window.__descriptionRecoveryProgress = [];
      const capture = () => window.__descriptionRecoveryProgress.push({
        at: Date.now(),
        prepare: document.getElementById("runReadinessPrepare")?.textContent || "",
        phase: document.getElementById("runProgressTitle")?.textContent || "",
        percent: document.getElementById("runProgressPercent")?.textContent || "",
      });
      new MutationObserver(capture).observe(document.body, {
        subtree: true, childList: true, characterData: true, attributes: true,
      });
      capture();
    });

    await page.locator("#runButton").click();
    await page.locator("#runReadinessDialog").waitFor({ state: "visible" });
    const descriptionRow = page.locator('[data-readiness="description"]');
    assert.equal(await descriptionRow.getAttribute("class"), "blocked");
    assert.match(await descriptionRow.textContent(), /描述已过期/);
    assert.equal(await page.locator("#runReadinessConfirm").isEnabled(), false);
    assert.equal(await page.locator("#runReadinessPrepare").isVisible(), true);
    assert.equal(await page.locator("#runReadinessPrepare").textContent(), "生成描述并继续");

    if (gateOnly) {
      await page.screenshot({ path: screenshot, fullPage: true });
      const result = {
        schema_version: "labelagent.browser-description-gate.v1",
        base_url: base,
        frame,
        topic,
        initial_gate: "description_stale",
        description_row: await descriptionRow.textContent(),
        confirm_enabled: await page.locator("#runReadinessConfirm").isEnabled(),
        recovery_button: await page.locator("#runReadinessPrepare").textContent(),
        describe_post_count: requests.filter(item => item.path === "/api/describe" && item.method === "POST").length,
        run_post_count: requests.filter(item => item.path === "/api/run" && item.method === "POST").length,
        formal_state_mutation_requested: false,
        page_errors: pageErrors,
        total_ms: Date.now() - started,
      };
      assert.equal(result.describe_post_count, 0);
      assert.equal(result.run_post_count, 0);
      assert.deepEqual(result.page_errors, []);
      fs.writeFileSync(outputJson, JSON.stringify(result, null, 2) + "\n", "utf8");
      console.log(JSON.stringify(result));
      return;
    }

    const describeResponsePromise = page.waitForResponse(item =>
      new URL(item.url()).pathname === "/api/describe" && item.request().method() === "POST",
    { timeout: 120000 });
    const runResponsePromise = page.waitForResponse(item =>
      new URL(item.url()).pathname === "/api/run" && item.request().method() === "POST",
    { timeout: 180000 });
    const refinementRequestPromise = page.waitForRequest(item =>
      new URL(item.url()).pathname === "/api/reference_candidates/refine"
        && item.method() === "POST", { timeout: 30000 }).catch(() => null);

    const clicked = Date.now();
    await page.locator("#runReadinessPrepare").click();
    await page.waitForFunction(() =>
      document.getElementById("runReadinessPrepare")?.textContent === "正在生成描述…",
    null, { timeout: 5000 });
    const describingVisibleMs = Date.now() - clicked;

    const describeResponse = await describeResponsePromise;
    assert.equal(describeResponse.status(), 200);
    const description = await describeResponse.json();
    assert.equal(typeof description.reference_signature, "string");
    const descriptionResponseMs = Date.now() - clicked;

    const runResponse = await runResponsePromise;
    assert.equal(runResponse.status(), 200);
    const runPayload = await runResponse.json();
    const runResponseMs = Date.now() - clicked;
    assert.equal(runPayload.failed_frames?.length || 0, 0);

    await page.waitForFunction(target => {
      const state = window.currentWorkbenchFrameState();
      return Array.isArray(state.candidates?.[String(target)])
        && state.candidates[String(target)].length > 0;
    }, frame, { timeout: 30000 });
    const candidateVisibleMs = Date.now() - clicked;

    const refinementRequest = await refinementRequestPromise;
    let stopStatus = null;
    if (refinementRequest) {
      const stopped = await page.evaluate(async () => {
        const response = await fetch("/api/run/stop", {
          method: "POST", headers: {"Content-Type": "application/json"}, body: "{}",
        });
        return {status: response.status, body: await response.json()};
      });
      stopStatus = stopped.status;
      assert.equal(stopStatus, 200);
    }

    const final = await page.evaluate(() => {
      const state = window.currentWorkbenchFrameState();
      return {
        truth: state.truth,
        reference_frames: state.reference_frames,
        candidates: state.candidates,
        progress: window.__descriptionRecoveryProgress,
      };
    });
    assert.deepEqual(final.truth, initial.truth);
    assert.deepEqual(final.reference_frames, initial.reference_frames);
    assert.deepEqual(pageErrors, []);
    assert(final.progress.some(item => item.prepare === "正在生成描述…"));

    const readiness = await page.evaluate(async () => {
      const response = await fetch("/api/run/readiness", {cache: "no-store"});
      return response.json();
    });
    assert.equal(readiness.ready, true);
    const describeCalls = requests.filter(item => item.path === "/api/describe" && item.method === "POST");
    const runCalls = requests.filter(item => item.path === "/api/run" && item.method === "POST");
    assert.equal(describeCalls.length, 1);
    assert.equal(runCalls.length, 1);

    await page.screenshot({ path: screenshot, fullPage: true });
    const candidates = final.candidates[String(frame)] || [];
    const result = {
      schema_version: "labelagent.browser-description-recovery.v1",
      base_url: base,
      frame,
      topic,
      initial_gate: "description_stale",
      describing_visible_ms: describingVisibleMs,
      description_response_ms: descriptionResponseMs,
      run_response_ms: runResponseMs,
      candidate_visible_ms: candidateVisibleMs,
      describe_post_count: describeCalls.length,
      run_post_count: runCalls.length,
      readiness_after_recovery: readiness,
      candidate_count: candidates.length,
      candidate_only: true,
      truth_unchanged: true,
      reference_frames_unchanged: true,
      page_errors: pageErrors,
      deferred_refinement_started: Boolean(refinementRequest),
      deferred_refinement_stop_status: stopStatus,
      total_ms: Date.now() - started,
    };
    fs.writeFileSync(outputJson, JSON.stringify(result, null, 2) + "\n", "utf8");
    console.log(JSON.stringify(result));
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
