#!/usr/bin/env node
"use strict";

// Real-browser acceptance for cancelling an obsolete deferred Agent refinement.
// Run only against an isolated workbench whose selected frame has pending candidates.

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

async function waitForVlmIdle(timeoutMs = 90000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const response = await fetch("http://127.0.0.1:28868/health");
    assert.equal(response.status, 200);
    const health = await response.json();
    if (health.exact_requests_inflight === 0) return Date.now() - started;
    await new Promise(resolve => setTimeout(resolve, 200));
  }
  throw new Error("VLM remained busy after the cancelled browser refinement");
}

(async () => {
  const base = required("BASE_URL").replace(/\/$/, "");
  const outputJson = path.resolve(required("OUTPUT_JSON"));
  const screenshot = path.resolve(required("SCREENSHOT"));
  const frame = integer("FRAME", 34);
  const topic = process.env.TOPIC || "/capture/camera_1/image";
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

    const pageResponse = await page.goto(base, { waitUntil: "networkidle" });
    assert.equal(pageResponse.status(), 200);
    const bootstrap = await page.evaluate(() => window.labelAgentBootstrapReady);
    assert.equal(bootstrap.topic, topic);
    const stateBefore = await page.evaluate(async ({ frame, topic, sourcePath }) => {
      const response = await fetch(`/api/frame_state/${frame}?topic=${encodeURIComponent(topic)}&path=${encodeURIComponent(sourcePath)}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`frame state failed: ${response.status}`);
      return response.json();
    }, { frame, topic, sourcePath: bootstrap.path });
    const key = String(frame);
    const initialCandidates = stateBefore.candidates?.[key];
    const initialRun = stateBefore.agent_runs?.[key];
    assert(Array.isArray(initialCandidates) && initialCandidates.length > 1);
    assert.equal(initialRun?.agent_review_status, "pending");
    assert.equal(initialRun?.agent_review_deferred, true);
    assert.equal(Object.hasOwn(stateBefore.truth || {}, key), false);

    const refineRequestPromise = page.waitForRequest(request =>
      new URL(request.url()).pathname === "/api/reference_candidates/refine"
        && request.method() === "POST", { timeout: 30000 }).catch(error => ({ waitError: error }));
    const refineResponsePromise = page.waitForResponse(response =>
      new URL(response.url()).pathname === "/api/reference_candidates/refine"
        && response.request().method() === "POST", { timeout: 90000 }).catch(error => ({ waitError: error }));
    const stopResponsePromise = page.waitForResponse(response =>
      new URL(response.url()).pathname === "/api/reference_candidates/refine/stop"
        && response.request().method() === "POST", { timeout: 30000 }).catch(error => ({ waitError: error }));

    await page.evaluate(({ frame, candidates, run }) =>
      window.prepareReferenceCandidateDraft(frame, candidates, run),
    { frame, candidates: initialCandidates, run: initialRun });
    await page.waitForFunction(expected =>
      document.querySelectorAll("#shapeList .shapeItem").length === expected,
    initialCandidates.length, { timeout: 10000 });
    const refineRequest = await refineRequestPromise;
    if (refineRequest.waitError) throw refineRequest.waitError;

    const beforeEditCount = await page.locator("#shapeList .shapeItem").count();
    assert.equal(beforeEditCount, initialCandidates.length);
    const editStarted = Date.now();
    await page.locator("#shapeList .shapeItem button").first().click();
    await page.waitForFunction(expected =>
      document.querySelectorAll("#shapeList .shapeItem").length === expected,
    beforeEditCount - 1, { timeout: 5000 });
    const stopResponse = await stopResponsePromise;
    if (stopResponse.waitError) throw stopResponse.waitError;
    const stopPayload = await stopResponse.json();
    const stopResponseMs = Date.now() - editStarted;
    assert.equal(stopResponse.status(), 200);
    assert.equal(stopPayload.cancellation_requested, true);

    const refineResponse = await refineResponsePromise;
    if (refineResponse.waitError) throw refineResponse.waitError;
    const refinePayload = await refineResponse.json();
    const workbenchIdleMs = Date.now() - editStarted;
    assert.equal(refineResponse.status(), 200);
    assert.equal(refinePayload.stopped, true);
    assert.equal(refinePayload.cancellation_reason, "manual_edit");
    assert.equal(refinePayload.refinement_id, stopPayload.refinement_id);
    await page.waitForFunction(async () => {
      const response = await fetch("/api/run/status", { cache: "no-store" });
      const status = await response.json();
      return status.status === "stopped";
    }, null, { timeout: 5000 });
    await page.waitForTimeout(1000);

    const stateAfter = await page.evaluate(async ({ frame, topic, sourcePath }) => {
      const response = await fetch(`/api/frame_state/${frame}?topic=${encodeURIComponent(topic)}&path=${encodeURIComponent(sourcePath)}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`frame state failed: ${response.status}`);
      return response.json();
    }, { frame, topic, sourcePath: bootstrap.path });
    assert.deepEqual(stateAfter.candidates?.[key], initialCandidates);
    assert.deepEqual(stateAfter.agent_runs?.[key], initialRun);
    assert.equal(Object.hasOwn(stateAfter.truth || {}, key), false);
    assert.equal(requestCounts.get("/api/reference_candidates/audit") || 0, 0);
    assert.deepEqual(pageErrors, []);

    await page.screenshot({ path: screenshot, fullPage: true });
    const vlmIdleAfterUiMs = await waitForVlmIdle();
    const result = {
      schema_version: "labelagent.browser-refinement-edit-cancel-acceptance.v1",
      base_url: base,
      frame,
      topic,
      initial_candidate_count: initialCandidates.length,
      edited_draft_count: beforeEditCount - 1,
      stop_response_ms: stopResponseMs,
      workbench_stopped_ms: workbenchIdleMs,
      vlm_idle_after_workbench_ms: vlmIdleAfterUiMs,
      matching_refinement_id: refinePayload.refinement_id === stopPayload.refinement_id,
      active_task_was_already_registered: stopPayload.stopped === true,
      server_candidates_unchanged: true,
      server_agent_run_unchanged: true,
      truth_unchanged: true,
      audit_requests_after_edit: requestCounts.get("/api/reference_candidates/audit") || 0,
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
