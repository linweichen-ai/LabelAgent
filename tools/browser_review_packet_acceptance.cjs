const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(
  'C:/Users/seer/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const base = process.env.LABELAGENT_ACCEPTANCE_URL || 'http://127.0.0.1:8875';
const output = path.resolve(process.env.LABELAGENT_ACCEPTANCE_OUTPUT || 'artifacts/review-packet-browser-acceptance');

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  const page = await browser.newPage({viewport: {width: 1600, height: 1000}, deviceScaleFactor: 1});
  const errors = [], failedRequests = [], httpErrors = [], writes = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('requestfailed', request => failedRequests.push(`${request.method()} ${request.url()} ${request.failure()?.errorText || ''}`));
  page.on('response', response => {
    if (response.status() >= 400) {
      const request = response.request();
      httpErrors.push(`${request.method()} ${response.status()} ${response.url()}`);
    }
  });
  page.on('request', request => {
    if (request.method() !== 'GET' && (/\/api\/(truth|human_review)(?:$|\?)/.test(request.url())
        || /\/api\/review_packets\/[^/]+\/decisions(?:$|\?)/.test(request.url()))) {
      writes.push(`${request.method()} ${request.url()}`);
    }
  });
  const started = Date.now();
  try {
    await page.goto(base, {waitUntil: 'networkidle', timeout: 60000});
    await page.locator('#openReviewPacketsButton').click();
    await page.locator('#reviewPacketDialog[open]').waitFor({state: 'visible'});
    await page.waitForFunction(() => document.querySelectorAll('.reviewPacketItem').length === 32);
    const before = await page.evaluate(() => ({
      packetOptions: document.querySelectorAll('#reviewPacketSelect option').length,
      rows: document.querySelectorAll('.reviewPacketItem').length,
      status: document.querySelector('#reviewPacketStatus')?.textContent,
      bodyHidden: document.querySelector('#reviewPacketBody')?.hidden,
    }));
    await page.locator('#reviewPacketActivate').click();
    await page.waitForFunction(() => document.querySelector('#reviewPacketStatus')?.textContent.includes('均已通过'), null, {timeout: 60000});
    const row74 = page.locator('.reviewPacketItem', {hasText: '第 74 帧'});
    await row74.click();
    await page.waitForFunction(() => document.querySelector('#frameNumber')?.value === '74'
      && !document.querySelector('#reviewPacketDock')?.hidden
      && document.querySelector('#frameImage')?.naturalWidth > 0, null, {timeout: 30000});
    const readback = await page.evaluate(async () => {
      const state = window.reviewPacketWorkbenchSnapshot();
      const response = await fetch('/api/human_review_binding', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({frame: 74, topic: state.topic})});
      return {state, binding: await response.json(), status: response.status,
        dock: document.querySelector('#reviewPacketDockDetail')?.textContent,
        assistant: document.querySelector('#reviewPacketAssistant')?.textContent,
        decision: document.querySelector('#reviewPacketDecision')?.value,
        decisionSubmitDisabled: document.querySelector('#reviewPacketDecisionSubmit')?.disabled,
        dialogOpen: document.querySelector('#reviewPacketDialog')?.open};
    });
    await page.screenshot({path: path.join(output, 'review-packet-frame74.png'), fullPage: true});
    if (readback.status !== 200 || readback.binding.frame_sha256 !== '10d8ed84741c0031c935349dd200804b465c149956728e50a37a2096a151c34c') {
      throw new Error('Frame 74 binding does not match the packet');
    }
    if (writes.length) throw new Error(`Unexpected review write requests: ${writes.join(', ')}`);
    if (!readback.assistant?.includes('助手预审') || readback.decision !== ''
        || readback.decisionSubmitDisabled !== true) {
      throw new Error('Assistant suggestion was not shown with a blank, disabled human decision gate');
    }
    if (errors.length || failedRequests.length || httpErrors.length) {
      throw new Error(`Browser errors: ${[...errors, ...failedRequests, ...httpErrors].join(' | ')}`);
    }
    const result = {schema_version: 'labelagent.review-packet-browser-acceptance.v1', passed: true,
      elapsed_ms: Date.now() - started, before, frame74: {
        path: readback.state.path, topic: readback.state.topic, frame: readback.state.frame,
        annotation_state: readback.state.annotation_state, truth_verified: readback.state.truth_verified,
        frame_sha256: readback.binding.frame_sha256, dock: readback.dock,
        assistant: readback.assistant, decision_preselected: readback.decision,
        decision_submit_disabled: readback.decisionSubmitDisabled,
      }, dialog_closed_after_navigation: readback.dialogOpen === false,
      truth_write_requests: writes, page_errors: errors, failed_requests: failedRequests,
      http_errors: httpErrors,
      screenshot: path.join(output, 'review-packet-frame74.png')};
    fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify(result, null, 2));
    process.stdout.write(`${JSON.stringify(result)}\n`);
  } finally { await browser.close(); }
})().catch(error => { console.error(error.stack || error); process.exitCode = 1; });
