const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const playwrightCandidates = [
  process.env.PLAYWRIGHT_NODE_MODULES,
  path.resolve(path.dirname(process.execPath), '..', 'node_modules', 'playwright'),
  'C:/Users/seer/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright',
].filter(Boolean);
const playwrightRoot = playwrightCandidates.find(candidate => fs.existsSync(candidate));
if (!playwrightRoot) {
  throw Error(`Playwright runtime not found; checked: ${playwrightCandidates.join(', ')}`);
}
const {chromium} = require(playwrightRoot);

const base = process.argv[2] || 'http://127.0.0.1:8765';
const output = process.argv[3];
const targetFrame = Number(process.argv[4] || 17);
if (!output) throw Error('Usage: node browser_result_overlay_acceptance.cjs BASE_URL OUTPUT_DIRECTORY [FRAME]');

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  const report = {target_frame: targetFrame, checks: {}, page_errors: []};
  try {
    const page = await browser.newPage({viewport: {width: 1600, height: 1000}});
    page.setDefaultTimeout(60000);
    page.on('pageerror', error => report.page_errors.push(error.message));
    await page.goto(base, {waitUntil: 'networkidle'});
    await page.evaluate(() => window.labelAgentBootstrapReady);
    await page.evaluate(() => window.setWorkspaceMode('auto', false));
    await page.waitForFunction(() => window.workspaceMode === 'auto');
    await page.evaluate(async frameNumber => {
      const data = await readCompactFrameStateSnapshot(frameNumber, topic, loadedDataPath);
      applyCompactFrameState(data, frameNumber, topic, loadedDataPath);
      frame = frameNumber;
      document.getElementById('frameNumber').value = String(frameNumber);
      const image = document.getElementById('frameImage');
      await new Promise((resolve, reject) => {
        image.addEventListener('load', resolve, {once: true});
        image.addEventListener('error', reject, {once: true});
        image.src = `/api/frame/${frameNumber}?topic=${encodeURIComponent(topic)}&v=${Date.now()}`;
      });
      frame = frameNumber;
      document.getElementById('frameNumber').value = String(frameNumber);
      draw();
      publishWorkbenchFrameState();
    }, targetFrame);
    await page.waitForTimeout(500);

    const splitter = page.locator('#workbenchSplitter');
    const splitterBox = await splitter.boundingBox();
    if (!splitterBox) throw Error('workbench splitter is not visible');
    await page.mouse.move(splitterBox.x + splitterBox.width / 2, splitterBox.y + splitterBox.height / 2);
    await page.mouse.down();
    await page.mouse.move(splitterBox.x + splitterBox.width / 2, splitterBox.y - 140, {steps: 8});
    await page.mouse.up();
    await page.waitForTimeout(300);

    report.checks = await page.evaluate(frameNumber => {
      const state = window.currentWorkbenchFrameState();
      const candidateCanvas = document.getElementById('compareCandidateCanvas');
      const pendingCanvas = document.getElementById('comparePendingCanvas');
      const truthCanvas = document.getElementById('compareTruthCanvas');
      const visiblePixels = canvas => {
        if (!canvas || !canvas.width || !canvas.height) return 0;
        const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
        let count = 0;
        for (let index = 3; index < data.length; index += 4) if (data[index]) count += 1;
        return count;
      };
      const stored = state.truth?.[String(frameNumber)] || [];
      const verified = state.truth_review_status?.[String(frameNumber)]?.reusable_as_reference === true;
      const baseRect = document.getElementById('canvas')?.getBoundingClientRect();
      const candidateRect = candidateCanvas?.getBoundingClientRect();
      const layoutDelta = baseRect && candidateRect ? Math.max(
        Math.abs(baseRect.left - candidateRect.left), Math.abs(baseRect.top - candidateRect.top),
        Math.abs(baseRect.width - candidateRect.width), Math.abs(baseRect.height - candidateRect.height),
      ) : Number.POSITIVE_INFINITY;
      return {
        workspace: document.body.dataset.workspace,
        current_frame: Number(document.getElementById('frameNumber').value),
        candidate_count: (state.candidates?.[String(frameNumber)] || []).length,
        pending_count: verified ? 0 : stored.length,
        truth_count: verified ? stored.length : 0,
        candidate_canvas_hidden: candidateCanvas?.hidden,
        candidate_canvas_width: candidateCanvas?.width || 0,
        candidate_visible_pixels: visiblePixels(candidateCanvas),
        pending_canvas_hidden: pendingCanvas?.hidden,
        pending_visible_pixels: visiblePixels(pendingCanvas),
        truth_visible_pixels: visiblePixels(truthCanvas),
        overlay_layout_max_delta_px: layoutDelta,
        compare_status: document.getElementById('compareStatus')?.textContent || '',
        vlm_readiness: null,
      };
    }, targetFrame);

    assert.equal(report.checks.workspace, 'auto');
    assert.equal(report.checks.current_frame, targetFrame);
    assert.ok(report.checks.candidate_count + report.checks.pending_count > 0,
      'target frame must have machine candidates or pending-review annotations');
    if (report.checks.candidate_count) {
      assert.equal(report.checks.candidate_canvas_hidden, false, 'candidate overlay must be visible');
      assert.ok(report.checks.candidate_visible_pixels > 0, 'candidate overlay must paint visible pixels');
    }
    if (report.checks.pending_count) {
      assert.equal(report.checks.pending_canvas_hidden, false, 'pending-review overlay must be visible');
      assert.ok(report.checks.pending_visible_pixels > 0, 'pending-review overlay must paint visible pixels');
    }
    assert.ok(report.checks.overlay_layout_max_delta_px <= 1,
      `overlay must remain aligned after dragging splitter; delta=${report.checks.overlay_layout_max_delta_px}`);
    const readiness = await page.evaluate(async () => {
      const response = await fetch('/api/model_readiness', {cache: 'no-store'});
      return response.json();
    });
    report.checks.vlm_readiness = readiness.vlm;
    assert.equal(readiness.vlm?.ready, true, readiness.vlm?.error || 'Local VLM must be online');
    assert.deepEqual(report.page_errors, []);
    await page.screenshot({path: path.join(output, `frame-${targetFrame}-result-overlay.png`), fullPage: true});
    report.passed = true;
  } catch (error) {
    report.passed = false;
    report.error = error.stack;
    throw error;
  } finally {
    fs.writeFileSync(path.join(output, 'result-overlay-receipt.json'), JSON.stringify(report, null, 2));
    await browser.close();
  }
})().catch(error => {
  console.error(error.stack);
  process.exitCode = 1;
});
