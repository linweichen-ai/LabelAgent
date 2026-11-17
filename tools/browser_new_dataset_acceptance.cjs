#!/usr/bin/env node
"use strict";

// Real browser acceptance for one isolated, candidate-only new-dataset image.
// The caller must prepare an empty isolated workbench and model configuration.

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
  const image = path.resolve(required("IMAGE"));
  const outputJson = path.resolve(required("OUTPUT_JSON"));
  const screenshot = path.resolve(required("SCREENSHOT"));
  const waitForDwellPrefetch = process.env.WAIT_FOR_DWELL_PREFETCH === "1";
  assert(fs.statSync(image).isFile());
  fs.mkdirSync(path.dirname(outputJson), { recursive: true });
  fs.mkdirSync(path.dirname(screenshot), { recursive: true });

  const browser = await chromium.launch({ channel: "msedge", headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1050 } });
    const pageErrors = [];
    const requests = [];
    page.on("pageerror", error => pageErrors.push(error.message));
    page.on("request", request => requests.push(new URL(request.url()).pathname));

    const started = Date.now();
    const response = await page.goto(base, { waitUntil: "networkidle" });
    assert.equal(response.status(), 200);
    await page.evaluate(() => window.labelAgentBootstrapReady);

    await page.locator("#openFilesDialog").click();
    await page.locator("#fileManager").waitFor({ state: "visible" });
    const uploadStarted = Date.now();
    await page.locator("#mcapFile").setInputFiles(image);
    await page.waitForFunction(() =>
      document.getElementById("fileManagerStatus")?.textContent.includes("已添加"));
    const uploadMs = Date.now() - uploadStarted;
    const row = page.locator(".fileRow").filter({ hasText: path.basename(image) });
    await row.waitFor({ state: "visible" });
    const dwellPrefetchPromise = waitForDwellPrefetch
      ? page.waitForResponse(item =>
        new URL(item.url()).pathname === "/api/reference_discovery/prefetch",
      { timeout: 120000 })
      : null;
    const loadStarted = Date.now();
    await row.getByRole("button", { name: "加载", exact: true }).click();
    await page.waitForFunction(() =>
      document.getElementById("fileManagerStatus")?.textContent.includes("兼容检查通过"));
    const loadMs = Date.now() - loadStarted;
    const loadStatus = await page.locator("#fileManagerStatus").innerText();
    await page.locator("#closeFilesDialog").click();
    let dwellPrefetchMs = null;
    let dwellPrefetchPayload = null;
    if (dwellPrefetchPromise) {
      const dwellPrefetchResponse = await dwellPrefetchPromise;
      dwellPrefetchMs = Date.now() - loadStarted;
      dwellPrefetchPayload = await dwellPrefetchResponse.json();
      assert.equal(dwellPrefetchResponse.status(), 200);
      assert.equal(dwellPrefetchPayload.prefetched, true);
      assert.equal(dwellPrefetchPayload.candidate_only, true);
      assert.equal(dwellPrefetchPayload.human_truth, false);
      assert.equal(dwellPrefetchPayload.approved_for_training, false);
    }

    await page.evaluate(() => {
      window.__labelAgentAcceptanceProgress = [];
      const progress = document.getElementById("referenceDraftProgress");
      const capture = () => {
        const sample = {
          at_ms: Math.round(performance.now()),
          hidden: Boolean(progress?.hidden),
          title: document.getElementById("referenceDraftProgressTitle")?.textContent || "",
          detail: document.getElementById("referenceDraftProgressDetail")?.textContent || "",
          elapsed: document.getElementById("referenceDraftElapsed")?.textContent || "",
        };
        const prior = window.__labelAgentAcceptanceProgress.at(-1);
        if (!prior || JSON.stringify({...prior, at_ms: 0}) !== JSON.stringify({...sample, at_ms: 0})) {
          window.__labelAgentAcceptanceProgress.push(sample);
        }
      };
      new MutationObserver(capture).observe(progress, {
        subtree: true, childList: true, characterData: true, attributes: true,
      });
      capture();
    });

    await page.evaluate(() => window.openReferencePreparation());
    await page.locator("#referencePrepDialog").waitFor({ state: "visible" });
    await page.waitForFunction(() => {
      const button = document.getElementById("runReferenceAutoDraft");
      return button && !button.disabled;
    }, null, { timeout: 30000 });
    const initialState = await page.evaluate(() => window.currentWorkbenchFrameState());
    assert.deepEqual(initialState.truth, {});
    assert.deepEqual(initialState.candidates, {});

    const draftResponsePromise = page.waitForResponse(
      item => new URL(item.url()).pathname === "/api/reference_draft",
      { timeout: 60000 });
    const draftStarted = Date.now();
    await page.locator("#runReferenceAutoDraft").click();
    await page.waitForFunction(() => {
      const state = window.currentWorkbenchFrameState();
      return Array.isArray(state.candidates?.["0"]) && state.candidates["0"].length > 0;
    }, null, { timeout: 60000 });
    const candidateVisibleMs = Date.now() - draftStarted;
    const draftResponse = await draftResponsePromise;
    const draftPayload = await draftResponse.json();
    assert.equal(draftResponse.status(), 200);
    assert.equal(draftPayload.candidate_only, true);
    assert.equal(draftPayload.human_truth, false);
    assert.equal(draftPayload.approved_for_training, false);

    const candidateState = await page.evaluate(() => window.currentWorkbenchFrameState());
    const candidates = candidateState.candidates["0"];
    assert(candidates.length > 0);
    assert.equal(Object.hasOwn(candidateState.truth, "0"), false);
    assert.equal(candidateState.reference_frames?.[candidateState.topic]?.includes(0) || false, false);

    await page.waitForFunction(() => {
      const run = window.currentWorkbenchFrameState().agent_runs?.["0"];
      return run?.agent_review_status === "complete"
        && run?.candidate_class_audit_status === "complete";
    }, null, { timeout: 60000 });
    const backgroundCompleteMs = Date.now() - draftStarted;
    const finalState = await page.evaluate(() => window.currentWorkbenchFrameState());
    const progressSamples = await page.evaluate(() => window.__labelAgentAcceptanceProgress);
    const visibleProgress = progressSamples.filter(item => !item.hidden);
    assert(visibleProgress.length > 0);
    assert(visibleProgress.some(item => item.title && !item.title.includes("0%")));
    assert.deepEqual(pageErrors, []);

    await page.screenshot({ path: screenshot, fullPage: true });
    const result = {
      schema_version: "labelagent.browser-new-dataset-acceptance.v1",
      base_url: base,
      input: image,
      input_bytes: fs.statSync(image).size,
      page_status: response.status(),
      upload_ms: uploadMs,
      load_ms: loadMs,
      load_status: loadStatus,
      dwell_prefetch_waited: waitForDwellPrefetch,
      dwell_prefetch_ms: dwellPrefetchMs,
      dwell_prefetch: dwellPrefetchPayload && {
        prefetched: dwellPrefetchPayload.prefetched,
        input_sha256: dwellPrefetchPayload.input_sha256,
        discovery_sha256: dwellPrefetchPayload.discovery_sha256,
        discovery_elapsed_ms: dwellPrefetchPayload.discovery_elapsed_ms,
        candidate_only: dwellPrefetchPayload.candidate_only,
        human_truth: dwellPrefetchPayload.human_truth,
        approved_for_training: dwellPrefetchPayload.approved_for_training,
        sam_prefetched: dwellPrefetchPayload.sam_prefetch?.prefetched === true,
      },
      candidate_visible_ms: candidateVisibleMs,
      background_complete_ms: backgroundCompleteMs,
      candidate_count: candidates.length,
      candidate_counts_by_label: Object.fromEntries([...new Set(candidates.map(x => x.label))]
        .sort().map(label => [label, candidates.filter(x => x.label === label).length])),
      draft_response: {
        candidate_only: draftPayload.candidate_only,
        human_truth: draftPayload.human_truth,
        approved_for_training: draftPayload.approved_for_training,
        agent_review_deferred: draftPayload.agent_review_deferred,
        candidate_class_audit_deferred: draftPayload.candidate_class_audit_deferred,
      },
      final_agent_status: {
        agent_review_status: finalState.agent_runs["0"].agent_review_status,
        candidate_class_audit_status: finalState.agent_runs["0"].candidate_class_audit_status,
      },
      truth_frame_created: Object.hasOwn(finalState.truth, "0"),
      reference_frame_created: Boolean(finalState.reference_frames?.[finalState.topic]?.includes(0)),
      progress_samples: progressSamples,
      endpoint_counts: Object.fromEntries([...new Set(requests)].sort()
        .map(endpoint => [endpoint, requests.filter(value => value === endpoint).length])),
      page_errors: pageErrors,
      screenshot,
      total_ms: Date.now() - started,
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
