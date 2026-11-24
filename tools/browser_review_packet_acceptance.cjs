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

const base = process.env.LABELAGENT_ACCEPTANCE_URL || 'http://127.0.0.1:8875';
const output = path.resolve(process.env.LABELAGENT_ACCEPTANCE_OUTPUT || 'artifacts/review-packet-browser-acceptance');
const expectedPriorityFrames = String(process.env.LABELAGENT_ACCEPTANCE_PRIORITY_FRAMES || '')
  .split(',').map(value => Number(value.trim())).filter(Number.isInteger);
const priorityOnly = process.env.LABELAGENT_ACCEPTANCE_PRIORITY_ONLY === '1';
const viewportWidth = Number(process.env.LABELAGENT_ACCEPTANCE_VIEWPORT_WIDTH || 1600);
const viewportHeight = Number(process.env.LABELAGENT_ACCEPTANCE_VIEWPORT_HEIGHT || 1000);
const darkMode = process.env.LABELAGENT_ACCEPTANCE_DARK_MODE === '1';
const isolatedInspectionStubs = process.env.LABELAGENT_ACCEPTANCE_ISOLATED_INSPECTION_STUBS === '1';

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  const page = await browser.newPage({viewport: {width: viewportWidth, height: viewportHeight}, deviceScaleFactor: 1});
  if (isolatedInspectionStubs) {
    const transparentPng = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M/wHwAF/gL+XxKqWQAAAABJRU5ErkJggg==', 'base64');
    await page.route('**/api/inspection/image/**', route => route.fulfill({status: 200, contentType: 'image/png', body: transparentPng}));
  }
  const errors = [], failedRequests = [], httpErrors = [], ignoredHttpErrors = [], writes = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('requestfailed', request => failedRequests.push(`${request.method()} ${request.url()} ${request.failure()?.errorText || ''}`));
  page.on('response', response => {
    if (response.status() >= 400) {
      const request = response.request();
      const detail = `${request.method()} ${response.status()} ${response.url()}`;
      if (response.status() === 404 && new URL(response.url()).pathname === '/api/pallet-candidate-review') {
        ignoredHttpErrors.push(detail);
      } else httpErrors.push(detail);
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
    await page.goto(base, {waitUntil: 'domcontentloaded', timeout: 60000});
    await page.evaluate(() => window.labelAgentWorkspaceReady);
    await page.locator('#openReviewPacketsButton').waitFor({state: 'visible', timeout: 30000});
    if (darkMode) {
      await page.locator('#themeButton').click();
      await page.waitForFunction(() => document.body.classList.contains('darkTheme'));
    }
    await page.locator('#openReviewPacketsButton').click();
    await page.locator('#reviewPacketDialog[open]').waitFor({state: 'visible'});
    await page.waitForFunction(() => document.querySelectorAll('.reviewPacketItem').length === 32);
    const before = await page.evaluate(() => ({
      packetOptions: document.querySelectorAll('#reviewPacketSelect option').length,
      rows: document.querySelectorAll('.reviewPacketItem').length,
      firstFrames: [...document.querySelectorAll('.reviewPacketItem')].slice(0, 4)
        .map(item => Number(item.dataset.packetFrame)),
      priorityLabels: [...document.querySelectorAll('.reviewPacketItem.priority strong')]
        .map(item => item.textContent),
      status: document.querySelector('#reviewPacketStatus')?.textContent,
      bodyHidden: document.querySelector('#reviewPacketBody')?.hidden,
    }));
    await page.locator('[data-packet-filter="priority"]').evaluate(button => button.click());
    const priorityFrames = await page.locator('.reviewPacketItem').evaluateAll(items =>
      items.map(item => Number(item.dataset.packetFrame)));
    await page.screenshot({path: path.join(output, 'review-packet-priority-queue.png'), fullPage: true});
    await page.locator('[data-packet-filter="all"]').evaluate(button => button.click());
    if (expectedPriorityFrames.length
        && JSON.stringify(priorityFrames) !== JSON.stringify(expectedPriorityFrames)) {
      throw new Error(`Priority queue mismatch: ${JSON.stringify(priorityFrames)}`);
    }
    if (priorityOnly) {
      const relevantErrors = errors.filter(message => !(ignoredHttpErrors.length
        && message === 'Failed to load resource: the server responded with a status of 404 (NOT FOUND)'));
      if (writes.length) throw new Error(`Unexpected review write requests: ${writes.join(', ')}`);
      if (relevantErrors.length || failedRequests.length || httpErrors.length) {
        throw new Error(`Browser errors: ${[...relevantErrors, ...failedRequests, ...httpErrors].join(' | ')}`);
      }
      const result = {schema_version: 'labelagent.review-packet-priority-browser-acceptance.v1',
        passed: true, elapsed_ms: Date.now() - started, viewport: {
          width: viewportWidth, height: viewportHeight, dark_mode: darkMode,
        }, before, priorityFrames,
        truth_write_requests: writes, page_errors: relevantErrors, failed_requests: failedRequests,
        http_errors: httpErrors, ignored_optional_http_errors: ignoredHttpErrors,
        screenshot: path.join(output, 'review-packet-priority-queue.png')};
      fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify(result, null, 2));
      process.stdout.write(`${JSON.stringify(result)}\n`);
      return;
    }
    const readyBeforeActivation = await page.evaluate(() => {
      const status = document.querySelector('#reviewPacketStatus');
      return status?.classList.contains('ready') && status.textContent.includes('topic 已就绪');
    });
    await page.locator('#reviewPacketActivate').evaluate(button => button.click());
    await page.waitForFunction(() => {
      const scope = window.currentWorkbenchFrameState?.();
      const status = document.querySelector('#reviewPacketStatus');
      return Number.isInteger(scope?.file_activation_generation)
        && scope.file_activation_generation > 0
        && status?.classList.contains('ready');
    }, null, {timeout: 60000});
    const row74 = page.locator('.reviewPacketItem', {hasText: '第 74 帧'});
    await row74.click();
    try {
      await page.waitForFunction(() => document.querySelector('#frameNumber')?.value === '74'
        && !document.querySelector('#reviewPacketDock')?.hidden
        && document.querySelector('#frameImage')?.naturalWidth > 0, null, {timeout: 30000});
    } catch (error) {
      const diagnostic = await page.evaluate(() => ({
        frame: document.querySelector('#frameNumber')?.value,
        dockHidden: document.querySelector('#reviewPacketDock')?.hidden,
        imageWidth: document.querySelector('#frameImage')?.naturalWidth,
        status: document.querySelector('#reviewPacketStatus')?.textContent,
        toast: document.querySelector('#toast')?.textContent,
        dialogOpen: document.querySelector('#reviewPacketDialog')?.open,
      }));
      throw new Error(`Frame 74 did not open: ${JSON.stringify({diagnostic, errors, failedRequests, httpErrors})}; ${error.message}`);
    }
    const readback = await page.evaluate(async () => {
      const state = window.reviewPacketWorkbenchSnapshot();
      const response = await fetch('/api/human_review_binding', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({frame: 74, topic: state.topic,
          file_activation_generation: state.file_activation_generation})});
      return {state, binding: await response.json(), status: response.status,
        dock: document.querySelector('#reviewPacketDockDetail')?.textContent,
        assistant: document.querySelector('#reviewPacketAssistant')?.textContent,
        decision: document.querySelector('#reviewPacketDecision')?.value,
        decisionSaveText: document.querySelector('#reviewPacketDecisionSave')?.textContent,
        decisionSubmitText: document.querySelector('#reviewPacketDecisionSubmit')?.textContent,
        shortcutHint: document.querySelector('#reviewPacketShortcutHint')?.textContent,
        decisionSubmitDisabled: document.querySelector('#reviewPacketDecisionSubmit')?.disabled,
        dialogOpen: document.querySelector('#reviewPacketDialog')?.open};
    });
    await page.keyboard.press('Alt+Digit3');
    const keyboardSelection = await page.locator('#reviewPacketDecision').inputValue();
    await page.locator('#reviewPacketDecision').selectOption('');
    await page.screenshot({path: path.join(output, 'review-packet-frame74.png'), fullPage: true});
    if (readback.status !== 200 || readback.binding.frame_sha256 !== '10d8ed84741c0031c935349dd200804b465c149956728e50a37a2096a151c34c'
        || readback.binding.file_activation_generation !== readback.state.file_activation_generation) {
      throw new Error(`Frame 74 binding does not match the packet: ${JSON.stringify({
        status: readback.status, frame_sha256: readback.binding.frame_sha256,
        binding_generation: readback.binding.file_activation_generation,
        state_generation: readback.state.file_activation_generation,
      })}`);
    }
    if (writes.length) throw new Error(`Unexpected review write requests: ${writes.join(', ')}`);
    if (!readback.assistant?.includes('助手预审') || readback.decision !== ''
        || readback.decisionSaveText?.trim() !== '仅签署'
        || readback.decisionSubmitText?.trim() !== '签署并下一帧'
        || !readback.shortcutHint?.includes('Ctrl+Enter')
        || keyboardSelection !== 'no_slope_present'
        || readback.decisionSubmitDisabled !== true) {
      throw new Error('Assistant suggestion was not shown with a blank, disabled human decision gate');
    }
    if (errors.length || failedRequests.length || httpErrors.length) {
      throw new Error(`Browser errors: ${[...errors, ...failedRequests, ...httpErrors].join(' | ')}`);
    }
    const result = {schema_version: 'labelagent.review-packet-browser-acceptance.v1', passed: true,
      elapsed_ms: Date.now() - started, before, priorityFrames, frame74: {
        path: readback.state.path, topic: readback.state.topic, frame: readback.state.frame,
        annotation_state: readback.state.annotation_state, truth_verified: readback.state.truth_verified,
        frame_sha256: readback.binding.frame_sha256, dock: readback.dock,
        assistant: readback.assistant, decision_preselected: readback.decision,
        keyboard_selection: keyboardSelection,
        decision_save_text: readback.decisionSaveText,
        decision_submit_text: readback.decisionSubmitText,
        shortcut_hint: readback.shortcutHint,
        decision_submit_disabled: readback.decisionSubmitDisabled,
      }, dialog_closed_after_navigation: readback.dialogOpen === false,
      packet_was_ready_before_activation: readyBeforeActivation,
      isolated_inspection_stubs: isolatedInspectionStubs,
      truth_write_requests: writes, page_errors: errors, failed_requests: failedRequests,
      http_errors: httpErrors,
      screenshots: [path.join(output, 'review-packet-priority-queue.png'),
        path.join(output, 'review-packet-frame74.png')]};
    fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify(result, null, 2));
    process.stdout.write(`${JSON.stringify(result)}\n`);
  } finally { await browser.close(); }
})().catch(error => { console.error(error.stack || error); process.exitCode = 1; });
