// Requires playwright-core (or playwright) and an installed Edge browser.
// Read-only except for writing the receipt and screenshot into OUTPUT_DIRECTORY.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
let chromium;
try { ({chromium} = require('playwright-core')); }
catch { ({chromium} = require('playwright')); }

const base = process.argv[2] || 'http://127.0.0.1:8765';
const output = process.argv[3];
const selectedTopic = process.argv[4] || '/capture/camera_1/image';
if (!output) throw Error('Usage: node verify_cold_start.cjs BASE_URL OUTPUT_DIRECTORY [TOPIC]');

(async () => {
  fs.mkdirSync(output, {recursive:true});
  const browser = await chromium.launch({
    executablePath:'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    headless:true,
  });
  const report = {responses:[],page_errors:[],console_errors:[],failed_requests:[]};
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
      const pathname = new URL(response.url()).pathname;
      if (!['/api/bootstrap','/api/state','/api/state/summary','/api/topic'].includes(pathname)
          && !pathname.startsWith('/api/frame_state/')) return;
      let bytes = Number(response.headers()['content-length'] || 0);
      if (!bytes) try { bytes = (await response.body()).length; } catch {}
      report.responses.push({pathname,status:response.status(),bytes});
    });

    let started = Date.now();
    await page.goto(base, {waitUntil:'networkidle'});
    await page.waitForFunction(() => window.labelAgentBootstrapReady && Object.keys(topics).length > 0);
    report.cold_start_ms = Date.now() - started;
    report.cold_responses = report.responses.slice();
    assert.equal(report.cold_responses.filter(item => item.pathname === '/api/state').length, 0);
    assert.equal(await page.locator('#batchReviewPanel').evaluate(node => node.classList.contains('compact')), true);

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
    await view.locator('button[title="进入标注模式"]').click();
    await page.waitForFunction(() => annotationMode && !loading && compactGeometryMode);

    const maxFrame = await page.evaluate(() => topics[topic].count - 1);
    const sequence = [Math.min(50,maxFrame), Math.min(1,maxFrame),
      Math.min(101,maxFrame), maxFrame];
    started = Date.now();
    await page.evaluate(sequence => Promise.allSettled(sequence.map(index => showFrame(index))), sequence);
    await page.waitForFunction(expected => frame === expected && !loading, sequence.at(-1));
    report.rapid_navigation_ms = Date.now() - started;
    report.rapid_navigation = await page.evaluate(() => ({
      frame, truth_keys:Object.keys(shapes), candidate_keys:Object.keys(candidates),
      compactGeometryMode, review_context:document.querySelector('#candidateReviewContext')?.textContent,
    }));
    assert.equal(report.rapid_navigation.frame, sequence.at(-1));
    assert.equal(report.rapid_navigation.compactGeometryMode, true);
    assert.ok(report.rapid_navigation.truth_keys.every(key => key === String(sequence.at(-1))));
    assert.ok(report.rapid_navigation.candidate_keys.every(key => key === String(sequence.at(-1))));

    const fullBefore = report.responses.filter(item => item.pathname === '/api/state').length;
    started = Date.now();
    await page.locator('#batchReviewToggle').click();
    await page.waitForFunction(() => !document.querySelector('#batchReviewPanel').classList.contains('compact'));
    await page.waitForFunction(() => document.querySelector('#batchReviewContext').textContent.includes('帧候选'));
    report.batch_expand_ms = Date.now() - started;
    const fullAfter = report.responses.filter(item => item.pathname === '/api/state').length;
    report.batch_full_state_requests = fullAfter - fullBefore;
    report.batch_context = await page.locator('#batchReviewContext').textContent();
    assert.equal(report.batch_full_state_requests, 1);

    await page.screenshot({path:path.join(output,'cold-start-workbench-round20.png'),fullPage:true});
    assert.deepEqual(report.page_errors, []);
    assert.deepEqual(report.console_errors, []);
    report.expected_aborted_frame_requests = report.failed_requests.filter(item =>
      item.error === 'net::ERR_ABORTED' && new URL(item.url).pathname.startsWith('/api/frame/'));
    report.unexpected_failed_requests = report.failed_requests.filter(item =>
      !report.expected_aborted_frame_requests.includes(item));
    assert.deepEqual(report.unexpected_failed_requests, []);
    report.passed = true;
  } catch (error) {
    report.error = error.stack;
    throw error;
  } finally {
    fs.writeFileSync(path.join(output,'cold-start-browser-round20.json'),JSON.stringify(report,null,2));
    await browser.close();
  }
  console.log(JSON.stringify(report,null,2));
})().catch(error => {
  console.error(error.stack);
  process.exitCode = 1;
});
