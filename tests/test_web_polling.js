const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function source(name) {
  return fs.readFileSync(path.join(__dirname, '../source/web', name), 'utf8');
}
function section(text, start, end) {
  const begin = text.indexOf(start);
  const finish = text.indexOf(end, begin);
  assert.ok(begin >= 0 && finish > begin, `找不到测试代码段 ${start}`);
  return text.slice(begin, finish);
}
function harness(extra = {}) {
  let now = 0, next = 1;
  const timers = new Map(), listeners = new Map();
  const context = {
    document: {
      hidden: false,
      addEventListener(name, callback) {
        if (!listeners.has(name)) listeners.set(name, []);
        listeners.get(name).push(callback);
      },
    },
    setTimeout(callback, delay) {
      const id = next++;
      timers.set(id, { callback, at: now + delay });
      return id;
    },
    clearTimeout(id) { timers.delete(id); },
    ...extra,
  };
  context.window = context;
  context.workspaceMode = 'auto';
  vm.createContext(context);
  async function flush() { for (let i = 0; i < 12; i++) await Promise.resolve(); }
  return {
    context, timers,
    run(code) { return vm.runInContext(code, context); },
    async emit(name, detail) {
      for (const callback of listeners.get(name) || []) callback({ detail });
      await flush();
    },
    async advance(ms) {
      const until = now + ms;
      for (;;) {
        const due = [...timers].filter(([, item]) => item.at <= until).sort((a, b) => a[1].at - b[1].at)[0];
        if (!due) break;
        now = due[1].at;
        timers.delete(due[0]);
        due[1].callback();
        await flush();
      }
      now = until;
      await flush();
    },
    flush,
  };
}

test('状态请求隐藏期间完成不重建定时器，恢复后立即轮询且保持单链', async () => {
  let calls = 0, release;
  const h = harness({
    latestAgentJob: { status: 'running' },
    refreshAgentProgress() { calls++; return new Promise(resolve => { release = resolve; }); },
  });
  h.run(section(source('app.js'), 'let runStatusPollTimer=', "fetch('/api/state').then"));
  assert.equal(calls, 1);
  h.context.document.hidden = true;
  await h.emit('visibilitychange');
  release();
  await h.flush();
  assert.equal(h.timers.size, 0);
  h.context.document.hidden = false;
  await h.emit('visibilitychange');
  assert.equal(calls, 2);
  release();
  await h.flush();
  assert.equal(h.timers.size, 1);
  await h.advance(1000);
  assert.equal(calls, 3);
  assert.equal(h.timers.size, 0);
  release();
  await h.flush();
  assert.equal(h.timers.size, 1);
});

test('复核台按帧完成即时读取，长阶段只保留低频恢复轮询', async () => {
  let calls = 0;
  const h = harness({ async refresh() { calls++; } });
  h.run(section(source('review.js'), '  let reviewPollTimer =', '\n})();'));
  await h.emit('labelagent:job-status', { status: 'running', phase: 'starting' });
  await h.advance(0);
  assert.equal(calls, 1);
  for (let i = 0; i < 6; i++) {
    await h.advance(1000);
    await h.emit('labelagent:job-status', { status: 'running', phase: `phase-${i}`, frame: i, task_id: 'a', completed: 0 });
  }
  assert.equal(calls, 1);
  await h.emit('labelagent:job-status', { status: 'running', phase: 'saved', task_id: 'a', completed: 1 });
  await h.advance(0);
  assert.equal(calls, 2);
  await h.emit('labelagent:job-status', { status: 'completed', task_id: 'a', completed: 1 });
  await h.advance(0);
  assert.equal(calls, 3);
  assert.equal(h.timers.size, 1);
});

test('状态请求尚未完成时隐藏后恢复不会丢失后续轮询', async () => {
  let calls = 0, release;
  const h = harness({
    latestAgentJob: { status: 'idle' },
    refreshAgentProgress() { calls++; return new Promise(resolve => { release = resolve; }); },
  });
  h.run(section(source('app.js'), 'let runStatusPollTimer=', "fetch('/api/state').then"));
  h.context.document.hidden = true;
  await h.emit('visibilitychange');
  h.context.document.hidden = false;
  await h.emit('visibilitychange');
  assert.equal(calls, 1);
  release();
  await h.flush();
  assert.equal(h.timers.size, 1);
  await h.advance(15000);
  assert.equal(calls, 2);
});

test('引导台任务开始事件立即刷新并切换到活跃间隔', async () => {
  let calls = 0;
  const h = harness({ async refreshGuide() { calls++; }, latestRunStatus: 'idle', guidePollTimer: null, guideRefreshBusy: false });
  const text = source('guide.js');
  h.run(section(text, '  function scheduleGuidePoll(', '\n  function showWorkbench'));
  h.run(section(text, "  document.addEventListener('labelagent:job-status'", "  byId('manualModeButton')"));
  h.run('scheduleGuidePoll()');
  await h.emit('labelagent:job-status', { status: 'running' });
  assert.equal(calls, 1);
  await h.advance(3000);
  assert.equal(calls, 2);
});

test('VLM 类别识别阶段显示非零进度并保持运行动画', () => {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      textContent: '', className: '', style: {}, attributes: {},
      classList: { values: new Set(), toggle(name, enabled) {
        if (enabled) this.values.add(name); else this.values.delete(name);
      } },
      setAttribute(name, value) { this.attributes[name] = value; },
      children: id === 'runProgressStages' ? Array.from({length: 5}, () => ({
        classList: { toggle() {} },
      })) : [],
    });
    return elements.get(id);
  }
  const h = harness({ $: element });
  const text = source('app.js');
  h.run(section(text, ' const agentPhase=', "$('downloadDebug').onclick"));
  h.run(section(text, 'function formatAgentDuration', 'function renderRunButton'));
  h.run("renderAgentProgressPanel({status:'running',phase:'discovering',total:1,completed:0,frame:31,phase_elapsed_seconds:70,events:[]},70)");
  assert.equal(element('runProgressPercent').textContent, '6%');
  assert.equal(element('runProgressTitle').textContent, 'VLM 识别当前帧可见类别');
  assert.ok(element('runProgressDetail').textContent.includes('后台仍在线'));
  assert.ok(element('runProgressBar').classList.values.has('running'));
  assert.ok(element('runProgressBar').classList.values.has('longPhase'));
});

test('瞬时故障重试时同一任务进度不倒退，新任务重新计数', () => {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      textContent: '', className: '', style: {}, attributes: {},
      classList: { values: new Set(), toggle(name, enabled) {
        if (enabled) this.values.add(name); else this.values.delete(name);
      } },
      setAttribute(name, value) { this.attributes[name] = value; },
      children: id === 'runProgressStages' ? Array.from({length: 5}, () => ({
        classList: { toggle() {} },
      })) : [],
    });
    return elements.get(id);
  }
  const h = harness({ $: element });
  const text = source('app.js');
  h.run(section(text, ' const agentPhase=', "$('downloadDebug').onclick"));
  h.run(section(text, 'function formatAgentDuration', 'function renderRunButton'));
  h.run("renderAgentProgressPanel({task_id:'a',status:'running',phase:'observing',total:1,completed:0,events:[]},10)");
  assert.equal(element('runProgressPercent').textContent, '58%');
  h.run("renderAgentProgressPanel({task_id:'a',status:'running',phase:'retrying',total:1,completed:0,events:[]},20)");
  assert.equal(element('runProgressPercent').textContent, '58%');
  assert.equal(element('runProgressTitle').textContent, '服务短暂异常，正在自动重试本帧');
  h.run("renderAgentProgressPanel({task_id:'b',status:'running',phase:'retrying',total:1,completed:0,events:[]},2)");
  assert.equal(element('runProgressPercent').textContent, '18%');
  h.run("renderAgentProgressPanel({task_id:'b',status:'running',phase:'vlm_waiting',total:1,completed:0,round:2,max_rounds:6,retry_in_seconds:10,events:[]},12)");
  assert.equal(element('runProgressTitle').textContent, 'VLM 正忙，任务正在排队等待');
  assert.ok(element('runProgressDetail').textContent.includes('约 10 秒后'));
  assert.equal(element('runProgressRound').textContent, '2 / 6');
});

test('状态读取超过八秒会解除轮询占用并在主面板提示恢复', async () => {
  let renders = 0;
  class FakeAbortController {
    constructor() { this.signal = {}; }
    abort() { if (this.signal.onabort) this.signal.onabort(); }
  }
  const button = { title: '', removeAttribute() {} };
  const h = harness({
    AbortController: FakeAbortController,
    progressFetching: false,
    runStatusFailureCount: 0,
    runStatusConnectionMessage: '',
    latestAgentJob: {status: 'running'},
    $() { return button; },
    renderRunButton() { renders++; },
    fetch(url, options) {
      return new Promise((resolve, reject) => {
        options.signal.onabort = () => {
          const error = new Error('aborted'); error.name = 'AbortError'; reject(error);
        };
      });
    },
  });
  h.run(section(source('app.js'), 'async function fetchAgentStatus', 'async function ensureAutoLabelReadiness'));
  const pending = h.run('refreshAgentProgress()');
  await h.advance(8000);
  await pending;
  assert.equal(h.context.progressFetching, false);
  assert.ok(h.context.runStatusConnectionMessage.includes('超过 8 秒'));
  assert.equal(renders, 1);
});

test('模型未就绪时预检明确失败且不伪装为推理已启动', async () => {
  let calls = 0;
  const h = harness({
    async fetch() {
      calls++;
      return calls === 1
        ? {ok: true, async json() { return {ready: false, sam: {ready: true}, vlm: {ready: false, error: '服务未连接'}}; }}
        : {ok: true, async json() { return {ready: true}; }};
    },
  });
  h.run(section(source('app.js'), 'async function ensureAutoLabelReadiness', "$('runButton').onclick"));
  await assert.rejects(() => h.context.ensureAutoLabelReadiness(), /尚未启动推理.*VLM：服务未连接/);
  assert.equal(calls, 2);
});

test('参考帧描述失效时预检阻止提交长推理', async () => {
  let calls = 0;
  const h = harness({
    async fetch() {
      calls++;
      return calls === 1
        ? {ok: true, async json() { return {ready: true, sam: {ready: true}, vlm: {ready: true}}; }}
        : {ok: true, async json() { return {ready: false, message: '参考帧 0 的描述已失效，请重新生成'}; }};
    },
  });
  h.run(section(source('app.js'), 'async function ensureAutoLabelReadiness', "$('runButton').onclick"));
  await assert.rejects(() => h.context.ensureAutoLabelReadiness(), /流程：参考帧 0 的描述已失效/);
});
