const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const baseUrl = 'http://127.0.0.1:18793';
const sample = 'D:\\Apps\\LabelAgent\\work\\datasets\\SFL_CPD15Y_20261104_smoke\\003-1793759959809514987.png';
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

async function jsonRequest(endpoint, options = {}) {
  const response = await fetch(baseUrl + endpoint, options);
  const data = await response.json();
  if (!response.ok) throw new Error(`${endpoint}: ${data.error || response.status}`);
  return data;
}

async function waitJson(url, timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try { const response = await fetch(url); if (response.ok) return response.json(); } catch {}
    await delay(200);
  }
  throw new Error(`wait timeout: ${url}`);
}

async function connect(url) {
  const socket = new WebSocket(url);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });
  let nextId = 1;
  const pending = new Map();
  const listeners = new Map();
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const task = pending.get(message.id); pending.delete(message.id);
      return message.error ? task.reject(new Error(message.error.message)) : task.resolve(message.result || {});
    }
    for (const listener of listeners.get(message.method) || []) listener(message.params || {});
  });
  return {
    socket,
    on(method, listener) { if (!listeners.has(method)) listeners.set(method, []); listeners.get(method).push(listener); },
    send(method, params = {}) {
      const id = nextId++;
      return new Promise((resolve, reject) => { pending.set(id, { resolve, reject }); socket.send(JSON.stringify({ id, method, params })); });
    },
  };
}

async function evaluate(cdp, expression) {
  const response = await cdp.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (response.exceptionDetails) throw new Error(response.exceptionDetails.text || 'page evaluation failed');
  return response.result?.value;
}

async function waitFor(cdp, expression, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await evaluate(cdp, expression)) return;
    await delay(150);
  }
  throw new Error(`page timeout: ${expression}`);
}

async function frameState(bootstrap) {
  const query = new URLSearchParams({
    topic: bootstrap.topic,
    path: bootstrap.path,
    file_activation_generation: String(bootstrap.file_activation_generation),
  });
  return jsonRequest(`/api/frame_state/0?${query}`);
}

(async () => {
  const outDir = __dirname;
  const edgePath = ['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', 'C:/Program Files/Microsoft/Edge/Application/msedge.exe'].find(fs.existsSync);
  if (!edgePath) throw new Error('Edge not found');
  const files = await jsonRequest('/api/files');
  const item = files.files.find(entry => entry.path === sample);
  if (!item) throw new Error('003 sample missing from isolated file library');
  await jsonRequest(`/api/files/${item.id}/load`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });

  const edge = spawn(edgePath, [
    '--headless=new', '--remote-debugging-port=9331', `--user-data-dir=${path.join(outDir, 'edge-profile')}`,
    '--no-first-run', '--disable-default-apps', '--disable-background-networking', 'about:blank',
  ], { windowsHide: true, stdio: 'ignore' });
  const result = { schema_version: 'labelagent.candidate-coverage-risk-browser-acceptance.v1', sample, page_errors: [], http_errors: [] };
  let cdp;
  try {
    const targets = await waitJson('http://127.0.0.1:9331/json/list');
    cdp = await connect(targets.find(target => target.type === 'page').webSocketDebuggerUrl);
    cdp.on('Runtime.exceptionThrown', event => result.page_errors.push(event.exceptionDetails?.text || 'exception'));
    cdp.on('Network.responseReceived', event => { if (event.response?.status >= 400) result.http_errors.push({ status: event.response.status, url: event.response.url }); });
    await cdp.send('Page.enable'); await cdp.send('Runtime.enable'); await cdp.send('Network.enable');
    await cdp.send('Page.navigate', { url: baseUrl + '/' });
    await waitFor(cdp, `document.readyState === 'complete' && document.querySelectorAll('#candidateSummary .candidateCoverageChip').length === 8`);
    await waitFor(cdp, `document.querySelector('#datasetIdentityName').textContent.includes('问题包 003')`);
    await waitFor(cdp, `[...document.querySelectorAll('#candidateSummary .candidateCoverageChip')].some(item => item.dataset.coverageLabel === 'person' && item.textContent.includes('已生成'))`);
    result.ui = await evaluate(cdp, `(() => {
      const chip = [...document.querySelectorAll('#candidateSummary .candidateCoverageChip')].find(item => item.dataset.coverageLabel === 'person');
      chip.click();
      return {
        identity: document.querySelector('#datasetIdentityName').textContent.trim(),
        person_chip: { text: chip.textContent.trim(), class_name: chip.className, title: chip.title },
        detail: document.querySelector('#candidateCoverageDetail').innerText.trim(),
        chips: [...document.querySelectorAll('#candidateSummary .candidateCoverageChip')].map(item => item.textContent.trim()),
      };
    })()`);
    const bootstrap = await jsonRequest('/api/bootstrap');
    const state = await frameState(bootstrap);
    result.boundary = {
      truth_frames: bootstrap.truth_frames || [],
      reference_frames: bootstrap.reference_frames || {},
      candidate_count: state.candidates?.['0']?.length || 0,
      agent_review_status: state.agent_runs?.['0']?.agent_review_status,
      class_audit_status: state.agent_runs?.['0']?.candidate_class_audit_status,
    };
    const metrics = await cdp.send('Page.getLayoutMetrics');
    const width = Math.max(1600, Math.ceil(metrics.cssContentSize?.width || 1600));
    const height = Math.min(4000, Math.max(1000, Math.ceil(metrics.cssContentSize?.height || 1000)));
    await cdp.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });
    const shot = await cdp.send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
    fs.writeFileSync(path.join(outDir, 'batchb-003-risk-chip.png'), Buffer.from(shot.data, 'base64'));
    result.passed = result.ui.person_chip.text.includes('2 个需复核')
      && result.ui.person_chip.class_name.includes('candidateNeedsReview')
      && result.ui.detail.includes('不会据此自动删除或改写真值')
      && result.boundary.candidate_count === 17
      && result.boundary.truth_frames.length === 0
      && Object.keys(result.boundary.reference_frames).length === 0
      && result.page_errors.length === 0 && result.http_errors.length === 0;
  } finally {
    fs.writeFileSync(path.join(outDir, 'result.json'), JSON.stringify(result, null, 2));
    if (cdp) { try { await cdp.send('Browser.close'); } catch {} cdp.socket.close(); }
    if (!edge.killed) edge.kill();
  }
  console.log(JSON.stringify(result, null, 2));
  if (!result.passed) process.exitCode = 1;
})().catch(error => { console.error(error); process.exitCode = 1; });
