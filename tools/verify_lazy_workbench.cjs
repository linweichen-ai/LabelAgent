// Requires playwright-core (or playwright) and an installed Edge browser.
// Read-only: it opens the current dataset, switches frames, and takes a screenshot.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
let chromium;
try {
  ({chromium} = require('playwright-core'));
} catch {
  ({chromium} = require('playwright'));
}

const base = process.argv[2] || 'http://127.0.0.1:8765';
const output = process.argv[3];
const selectedTopic = process.argv[4] || '/capture/camera_1/image';
if (!output) throw Error('Usage: node verify_lazy_workbench.cjs BASE_URL OUTPUT_DIRECTORY [TOPIC]');

(async () => {
  fs.mkdirSync(output, {recursive:true});
  const browser = await chromium.launch({
    executablePath:'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    headless:true,
  });
  const report = {
    page_errors:[], console_errors:[], failed_requests:[], http_errors:[],
    responses:[], frames:[],
  };
  try {
    const page = await browser.newPage({viewport:{width:1600,height:1000}});
    page.setDefaultTimeout(90000);
    page.on('pageerror', error => report.page_errors.push(error.message));
    page.on('console', message => {
      if (message.type() === 'error') report.console_errors.push(message.text());
    });
    page.on('requestfailed', request => report.failed_requests.push({
      url:request.url(), error:request.failure()?.errorText,
    }));
    page.on('response', async response => {
      const url = response.url();
      if (response.status() >= 400) report.http_errors.push({url,status:response.status()});
      if (!url.includes('/api/topic') && !url.includes('/api/frame_state/')) return;
      let bytes = Number(response.headers()['content-length'] || 0);
      if (!bytes) {
        try { bytes = (await response.body()).length; } catch {}
      }
      report.responses.push({url:url.replace(/^.*\/api/,'/api/'),status:response.status(),bytes});
    });

    let started = Date.now();
    await page.goto(base, {waitUntil:'networkidle'});
    report.cold_page_ms = Date.now() - started;
    await page.waitForFunction(() => Object.keys(topics).length > 0);
    await page.locator('#addWindowButton').click();
    await page.locator('#windowType').selectOption('image');
    await page.locator('#windowSources input').evaluateAll((inputs, topic) => {
      for (const input of inputs) input.checked = input.value === topic;
    }, selectedTopic);
    await page.locator('#addWindowForm button[type="submit"]').click();
    await page.waitForFunction(topic => [...extraWindows.values()].some(view =>
      view.sources[0] === topic && view.layers[0]?.video?.readyState >= 1), selectedTopic);
    const view = page.locator('#windowGrid > .sensorWindow').filter({
      has:page.locator('.windowHeader > span', {hasText:`图像 · ${selectedTopic}`}),
    });
    started = Date.now();
    await view.locator('button[title="进入标注模式"]').click();
    await page.waitForFunction(() => annotationMode && !loading && imageWidth > 0);
    report.enter_annotation_ms = Date.now() - started;
    report.initial = await page.evaluate(() => ({
      compactGeometryMode, frame, truth_keys:Object.keys(shapes),
      candidate_keys:Object.keys(candidates), truth_index:truthFrameIndex.size,
      candidate_index:Object.keys(candidateCountIndex).length, topic, path:loadedDataPath,
    }));
    assert.equal(report.initial.compactGeometryMode, true);
    assert.equal(report.initial.topic, selectedTopic);

    const maxFrame = await page.evaluate(() => topics[topic].count - 1);
    const sequence = [Math.min(50,maxFrame), Math.min(1,maxFrame),
      Math.min(101,maxFrame), maxFrame, Math.min(50,maxFrame)];
    for (const index of sequence) {
      started = Date.now();
      await page.locator('#frameNumber').fill(String(index));
      await page.locator('#frameNumber').press('Tab');
      await page.waitForFunction(expected => frame === expected && !loading, index);
      await page.waitForTimeout(50);
      const state = await page.evaluate(() => ({
        frame, truth_keys:Object.keys(shapes), candidate_keys:Object.keys(candidates),
        selected_count:selectedShapes().length,
        candidate_count:(candidates[String(frame)] || []).length,
        compactGeometryMode,
      }));
      state.elapsed_ms = Date.now() - started;
      report.frames.push(state);
      assert.equal(state.frame, index);
      assert.equal(state.compactGeometryMode, true);
      assert.ok(state.truth_keys.every(key => key === String(index)));
      assert.ok(state.candidate_keys.every(key => key === String(index)));
    }
    await page.screenshot({path:path.join(output,'lazy-frame-workbench-round19.png'),fullPage:true});
    assert.deepEqual(report.page_errors, []);
    assert.deepEqual(report.failed_requests, []);
    assert.deepEqual(report.http_errors.filter(item => !item.url.endsWith('/favicon.ico')), []);
    report.passed = true;
  } catch (error) {
    report.error = error.stack;
    throw error;
  } finally {
    fs.writeFileSync(path.join(output,'lazy-frame-browser-round19.json'),JSON.stringify(report,null,2));
    await browser.close();
  }
  console.log(JSON.stringify(report,null,2));
})().catch(error => {
  console.error(error.stack);
  process.exitCode = 1;
});
