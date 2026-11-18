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

test('参考起稿公开高风险复核与安全保存阶段', () => {
  const text = source('reference-prep.js');
  assert.match(text, /auditing: 'VLM 正在复核高风险候选'/);
  assert.match(text, /committing: '正在安全保存机器候选'/);
  assert.match(text, /高风险复核 \$\{status\.audit_completed \|\| 0\}\/\$\{status\.audit_total\}/);
  assert.match(text, /新图首批机器候选通常约 15～25 秒出现/);
});

test('参考起稿先展示候选并在后台完成高风险类别复核', () => {
  const prep = source('reference-prep.js');
  const app = source('app.js');
  assert.match(prep, /defer_class_audit: true/);
  assert.match(prep, /defer_agent_review: true/);
  assert.match(prep, /Agent 正在后台精修，可先检查轮廓/);
  assert.match(app, /agent_review_status==='pending'/);
  assert.match(app, /refineReferenceCandidateSnapshot\(target,candidate\)\.then/);
  assert.match(app, /系统没有覆盖当前人工草稿/);
  assert.match(app, /finishAudit\(result\.candidates\)/);
  assert.match(app, /结果不会自动变成人工真值/);
});

test('候选生成、Agent 精修和类别复核只读回当前帧绑定', () => {
  const prep = source('reference-prep.js');
  const app = source('app.js');
  assert.match(app, /function readCompactFrameStateSnapshot/);
  assert.match(app, /\/api\/frame_state\/\$\{target\}/);
  assert.doesNotMatch(app.slice(app.indexOf('async function auditReferenceCandidateSnapshot'), app.indexOf('function invalidateChangedCandidateAudits')), /fetch\('\/api\/state'/);
  assert.doesNotMatch(app.slice(app.indexOf('async function refineReferenceCandidateSnapshot'), app.indexOf('function invalidateChangedCandidateAudits')), /fetch\('\/api\/state'/);
  assert.match(prep, /window\.fetchCompactFrameState\?\.\(frameId\)/);
});

test('参考准备弹窗只读取轻量目录并在选帧后按需读取几何', () => {
  const prep = source('reference-prep.js');
  assert.match(prep, /fetch\('\/api\/reference-preparation'/);
  assert.match(prep, /window\.readCompactFrameStateSnapshot\(frameId, expected\.topic, expected\.path\)/);
  assert.doesNotMatch(prep, /fetch\('\/api\/state'/);
  assert.match(prep, /state\?\.truth_index \|\| state\?\.truth/);
  assert.match(prep, /state\?\.candidate_index \|\| state\?\.candidates/);
});

test('参考准备弹窗为当前空白帧预取可见类别并显示SAM预热状态', () => {
  const prep = source('reference-prep.js');
  const prefetch = prep.slice(prep.indexOf('async function prefetchVisibleDiscovery'),
    prep.indexOf('\n  function render(state)'));
  assert.match(prefetch, /hasTruthFrame\(state, frameId\) \|\| currentFrameCandidate\(state, frameId\)/);
  assert.match(prefetch, /fetch\('\/api\/reference_discovery\/prefetch'/);
  assert.match(prefetch, /expected_path: state\.path/);
  assert.match(prefetch, /expected_topic: state\.topic/);
  assert.match(prefetch, /data\.schema_version === 'labelagent\.reference-discovery-prefetch\.v1'/);
  assert.match(prefetch, /samReady: data\.sam_prefetch\?\.prefetched === true/);
  assert.match(prep, /void maybePrefetchVisibleDiscovery\(generation\)/);
  assert.match(prep, /正在后台预备当前图片的 VLM 类别识别/);
  assert.match(prep, /当前图片的 VLM 类别识别与整图 SAM 已预备/);
});

test('图片稳定停留后提前预取但跳过真值候选和快速导航', () => {
  const prep = source('reference-prep.js');
  const dwell = prep.slice(prep.indexOf('function scheduleFrameDwellPrefetch'),
    prep.indexOf('\n  function render(state)'));
  assert.match(prep, /const FRAME_DWELL_PREFETCH_MS = 1200/);
  assert.match(prep, /state\?\.loaded === true \|\| Boolean\(state\?\.path && state\?\.topic\)/);
  assert.match(dwell, /clearTimeout\(frameDwellPrefetchTimer\)/);
  assert.match(dwell, /hasTruthFrame\(state, frameId\) \|\| currentFrameCandidate\(state, frameId\)/);
  assert.match(dwell, /fetch\('\/api\/model_readiness'/);
  assert.match(dwell, /generation !== frameDwellPrefetchGeneration \|\| currentFrame\(\) !== frameId/);
  assert.match(dwell, /await prefetchVisibleDiscovery\(state, frameId, key/);
  assert.match(prep, /discoveryPrefetch = \{key, status: 'idle', completedAt: 0, samReady: false\}/);
  assert.match(prep, /document\.addEventListener\('labelagent:frame-state'/);
});

test('当前候选返回后只在空闲时预备紧邻下一帧且切帧立即取消', () => {
  const prep = source('reference-prep.js');
  const adjacent = prep.slice(prep.indexOf('function cancelAdjacentFramePrefetch'),
    prep.indexOf('\n  function render(state)'));
  assert.match(prep, /const ADJACENT_PREFETCH_DELAY_MS = 3000/);
  assert.match(prep, /const ADJACENT_PREFETCH_MAX_ATTEMPTS = 12/);
  assert.match(adjacent, /const nextFrame = Number\(sourceFrame\) \+ 1/);
  assert.match(adjacent, /nextFrame >= frameCount/);
  assert.match(adjacent, /hasTruthFrame\(state, nextFrame\)/);
  assert.match(adjacent, /currentFrameCandidate\(state, nextFrame\)/);
  assert.match(adjacent, /fetch\('\/api\/run\/status'/);
  assert.match(adjacent, /status\.status !== 'idle'/);
  assert.match(adjacent, /frame: nextFrame, expected_path: state\.path/);
  assert.match(adjacent, /\['active_workflow', 'vlm_busy', 'sam_busy'\]/);
  assert.match(prep, /scheduleAdjacentFramePrefetch\(fresh, frameId\)/);
  assert.match(prep, /currentFrame\(\) !== adjacentPrefetchSourceFrame[\s\S]*cancelAdjacentFramePrefetch\(\)/);
});

test('相邻帧预取在繁忙时低频重试、空闲后只请求下一帧且取消后不再发送', async () => {
  const prep = source('reference-prep.js');
  const adjacent = prep.slice(prep.indexOf('function cancelAdjacentFramePrefetch'),
    prep.indexOf('\n  function render(state)'));
  const calls = [];
  let statusReads = 0;
  const h = harness({
    frameValue: 30,
    async fetch(url, options = {}) {
      calls.push({url, options});
      if (url === '/api/run/status') {
        statusReads += 1;
        return {ok: true, async json() {
          return {status: statusReads === 1 ? 'running' : 'idle'};
        }};
      }
      return {ok: true, async json() { return {prefetched: true}; }};
    },
  });
  h.context.currentWorkbenchFrameState = () => ({
    path: 'dataset.mcap', topic: '/camera', labels: ['floor'],
  });
  h.run(`
    let adjacentPrefetchTimer = null;
    let adjacentPrefetchGeneration = 0;
    let adjacentPrefetchSourceFrame = null;
    const ADJACENT_PREFETCH_DELAY_MS = 3000;
    const ADJACENT_PREFETCH_RETRY_MS = 5000;
    const ADJACENT_PREFETCH_MAX_ATTEMPTS = 12;
    const currentFrame = () => frameValue;
    const discoveryPrefetchKey = (state, frame) => state
      ? [state.path, state.topic, (state.labels || []).join(','), frame].join('|') : '';
    const hasTruthFrame = (state, frame) => Boolean(state.truth_index?.[String(frame)]);
    const currentFrameCandidate = (state, frame) => state.candidate_index?.[String(frame)] || null;
    ${adjacent}
  `);
  const state = {path: 'dataset.mcap', topic: '/camera', labels: ['floor'],
    frame_count: 40, truth_index: {}, candidate_index: {}};
  h.context.state = state;
  h.run('scheduleAdjacentFramePrefetch(state, 30)');
  await h.advance(2999);
  assert.equal(calls.length, 0);
  await h.advance(1);
  assert.deepEqual(calls.map(item => item.url), ['/api/run/status']);
  await h.advance(4999);
  assert.equal(calls.length, 1);
  await h.advance(1);
  assert.deepEqual(calls.map(item => item.url), [
    '/api/run/status', '/api/run/status', '/api/reference_discovery/prefetch']);
  assert.deepEqual(JSON.parse(calls[2].options.body), {
    frame: 31, expected_path: 'dataset.mcap', expected_topic: '/camera'});

  calls.length = 0;
  h.run('scheduleAdjacentFramePrefetch(state, 30); frameValue = 31; cancelAdjacentFramePrefetch()');
  await h.advance(3000);
  assert.equal(calls.length, 0);
});

test('保存真值和锁定参考帧只读回目标帧且保留强绑定校验', () => {
  const app = source('app.js');
  const lock = app.slice(app.indexOf('async function lockReferenceFrame'), app.indexOf("$('referenceButton').onclick"));
  const save = app.slice(app.indexOf('async function saveCurrentTruth'), app.indexOf("$('truthSave').onclick"));
  assert.match(lock, /readCompactFrameStateSnapshot\(target,targetTopic,targetPath\)/);
  assert.match(save, /readCompactFrameStateSnapshot\(targetFrame,targetTopic,targetPath\)/);
  assert.doesNotMatch(lock, /fetch\('\/api\/state'/);
  assert.doesNotMatch(save, /fetch\('\/api\/state'/);
  assert.match(lock, /review\?\.truth_sha256!==expectedHash/);
  assert.match(save, /review\?\.truth_sha256!==expectedTruthHash/);
});

test('单帧候选复核恢复轮询只读当前帧而不是整套状态', () => {
  const review = source('review.js');
  const refresh = review.slice(review.indexOf('async function refresh()'), review.indexOf('function adoptFrameState'));
  assert.match(refresh, /window\.currentWorkbenchFrameState\?\.\(\)/);
  assert.match(refresh, /window\.readCompactFrameStateSnapshot\(frame, current\.topic, current\.path\)/);
  assert.doesNotMatch(refresh, /fetch\('\/api\/state'/);
});

test('批量复核队列使用无几何风险索引并复用候选内容种子', () => {
  const batch = source('batch-review.js');
  assert.match(batch, /fetch\('\/api\/review\/index'/);
  assert.match(batch, /labelagent\.batch-review-index\.v1/);
  assert.match(batch, /shape\?\.review_seed/);
  assert.doesNotMatch(batch, /fetch\('\/api\/state'/);
  assert.match(batch, /只读取帧级风险索引/);
  assert.match(batch, /next\.compact_geometry === true[\s\S]*!panel\.classList\.contains\('compact'\)[\s\S]*refresh\(\)/);
});

test('状态请求隐藏期间完成不重建定时器，恢复后立即轮询且保持单链', async () => {
  let calls = 0, release;
  const h = harness({
    latestAgentJob: { status: 'running' },
    refreshAgentProgress() { calls++; return new Promise(resolve => { release = resolve; }); },
  });
  h.run(section(source('app.js'), 'let runStatusPollTimer=', 'window.labelAgentBootstrapReady='));
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
  h.run(section(source('app.js'), 'let runStatusPollTimer=', 'window.labelAgentBootstrapReady='));
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

test('VLM 类别识别阶段按实测基线平滑显示预计进度并保持运行动画', () => {
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
  assert.equal(element('runProgressPercent').textContent, '约 19%');
  assert.equal(element('runProgressTitle').textContent, 'VLM 识别当前帧可见类别');
  assert.ok(element('runProgressDetail').textContent.includes('当前 4090 基线通常约 16 秒'));
  assert.ok(element('runProgressDetail').textContent.includes('后台仍在线'));
  assert.ok(element('runProgressBar').classList.values.has('running'));
  assert.ok(element('runProgressBar').classList.values.has('longPhase'));
});

test('VLM 预计进度随等待时间单调前进但不会越过下一阶段', () => {
  const h = harness();
  const text = source('app.js');
  h.run(section(text, ' const agentPhase=', "$('downloadDebug').onclick"));
  h.run(section(text, 'function formatAgentDuration', 'function renderAgentProgressPanel'));
  const weights = [0, 8, 16, 60].map(seconds =>
    h.run(`estimatedAgentPhaseWeight('discovering',${seconds})`));
  assert.deepEqual(weights.map(value => Math.round(value * 100)), [6, 13, 19, 19]);
  assert.ok(weights.every(value => value < 0.20));
});

test('后台 Agent 精修显示明确阶段而不是看似卡在百分之一', () => {
  const text = source('app.js');
  const phases = section(text, 'const agentPhase=', 'let progressTaskId');
  assert.match(phases, /agent_refining:'VLM 后台精修候选'/);
  assert.match(phases, /agent_refining:2/);
  assert.match(phases, /agent_refining:0\.58/);
});

test('空闲状态明确显示待开始而不是误导性的零进度', () => {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      textContent: '', className: '', style: {}, attributes: {},
      classList: { toggle() {} },
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
  h.run("renderAgentProgressPanel({status:'idle',events:[]},0)");
  assert.equal(element('runProgressPercent').textContent, '待开始');
  assert.equal(element('runProgressBar').attributes['aria-valuenow'], '0');
});

test('刷新后有持久化结果时按钮先恢复结果且不会重复恢复', () => {
  const button = {disabled:false,textContent:'',attributes:{},
    setAttribute(name,value){this.attributes[name]=value}};
  const h = harness({
    $: id => id === 'runButton' ? button : {value:'0'},
    latestAgentJob:{status:'idle',result_available:true,result_task_id:'task-1',
      result_completed:2,result_total:2},
    runRequestStartedAt:null,runRequestPhase:'preflight',runResultHandled:new Set(),
    retryFrames:[],retryScope:null,retryMode:'',loadedDataPath:'/data.mcap',topic:'camera0',
    renderAgentProgressPanel() {},
  });
  const text = source('app.js');
  h.run(section(text, 'function pendingRunResult', 'function formatAgentDuration'));
  h.run(section(text, 'function renderRunButton', 'function renderAgentProgress'));
  h.run('renderRunButton()');
  assert.equal(button.textContent, '查看上次结果');
  h.run("markRunResultHandled('task-1');renderRunButton()");
  assert.equal(button.textContent, '开始自动标注');
  h.run("retryFrames=[3,7];retryScope={file:'/data.mcap',topic:'camera0'};retryMode='failed';renderRunButton()");
  assert.equal(button.textContent, '重试 2 个失败帧');
  h.run("retryMode='resume';renderRunButton()");
  assert.equal(button.textContent, '继续 2 个未完成帧');
});

test('中断回执恢复已完成结果并只排队未完成帧', async () => {
  const button = {disabled:false,textContent:'',attributes:{},
    setAttribute(name,value){this.attributes[name]=value}};
  let opened = null, notice = '';
  const h = harness({
    $: id => id === 'runButton' ? button : {value:'0'},
    latestAgentJob:{status:'idle',result_available:true,result_task_id:'interrupted-1'},
    runRequestStartedAt:null,runRequestPhase:'preflight',runRequestFrame:null,runRequestCount:null,
    runResultHandled:new Set(),retryFrames:[],retryScope:null,retryMode:'',
    loadedDataPath:'/data.mcap',topic:'camera0',candidateCountIndex:{},
    async fetch() { return {ok:true,async json(){return {
      task_id:'interrupted-1',status:'running',file:'/data.mcap',topic:'camera0',
      completed:1,frames:[{frame:1,remaining:2,failed:false}],pending_frames:[2,3],failed_frames:[]
    }}}},
    refreshCandidateSummary() {},
    async openCandidateFrame(frame,topic) { opened={frame,topic}; },
    toast(message) { notice=message; },
    renderAgentProgressPanel() {},
  });
  const text = source('app.js');
  h.run(section(text, 'function pendingRunResult', 'function formatAgentDuration'));
  h.run(section(text, 'function renderRunButton', 'function renderAgentProgress'));
  assert.equal(await h.run('recoverRunResult(latestAgentJob)'), true);
  assert.deepEqual(opened, {frame:1,topic:'camera0'});
  assert.equal(h.run('retryFrames.join(",")'), '2,3');
  assert.equal(button.textContent, '继续 2 个未完成帧');
  assert.ok(notice.includes('已完成帧不会重跑'));
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
  assert.equal(element('runProgressPercent').textContent, '约 58%');
  h.run("renderAgentProgressPanel({task_id:'a',status:'running',phase:'retrying',total:1,completed:0,events:[]},20)");
  assert.equal(element('runProgressPercent').textContent, '约 58%');
  assert.equal(element('runProgressTitle').textContent, '服务短暂异常，正在自动重试本帧');
  h.run("renderAgentProgressPanel({task_id:'b',status:'running',phase:'retrying',total:1,completed:0,events:[]},2)");
  assert.equal(element('runProgressPercent').textContent, '约 18%');
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
  assert.equal(calls, 3);
});

test('参考帧描述失效时预检自动生成并重新检查后继续', async () => {
  const calls = [];
  let generated = 0;
  const h = harness({
    runRequestPhase: 'preflight', renderRunButton() {}, toast() {},
    async regenerateReferenceDescription(frame) { generated++; assert.equal(frame, 0); },
    async fetch(url) {
      calls.push(url);
      if (url === '/api/model_readiness') return {ok:true, async json(){return {sam:{ready:true},vlm:{ready:true}};}};
      if (url === '/api/run/data-readiness') return {ok:true, async json(){return {ready:true};}};
      const readinessCalls = calls.filter(item => item === '/api/run/readiness').length;
      return {ok:true, async json(){return readinessCalls === 1
        ? {ready:false,code:'description_stale',reference_frame:0,message:'参考帧 0 的描述已失效，请重新生成'}
        : {ready:true,code:'ready',reference_frame:0};}};
    },
  });
  h.run(section(source('app.js'), 'async function ensureAutoLabelReadiness', "$('runButton').onclick"));
  const result = await h.context.ensureAutoLabelReadiness(null,1,1);
  assert.equal(result.workflow.ready, true);
  assert.equal(generated, 1);
  assert.equal(calls.filter(item => item === '/api/run/readiness').length, 2);
  assert.equal(h.context.runRequestPhase, 'preflight');
});

test('参考描述自动生成失败时保持未启动且不重复提交', async () => {
  let generated = 0;
  const h = harness({
    runRequestPhase:'preflight', renderRunButton() {}, toast() {},
    async regenerateReferenceDescription(){generated++;throw new Error('VLM 暂时繁忙');},
    async fetch(url){return {ok:true,async json(){
      if(url==='/api/model_readiness')return {sam:{ready:true},vlm:{ready:true}};
      if(url==='/api/run/data-readiness')return {ready:true};
      return {ready:false,code:'description_missing',reference_frame:55,message:'请生成描述'};
    }};},
  });
  h.run(section(source('app.js'), 'async function ensureAutoLabelReadiness', "$('runButton').onclick"));
  await assert.rejects(() => h.context.ensureAutoLabelReadiness(null,1,1), /尚未启动推理.*VLM 暂时繁忙/);
  assert.equal(generated,1);
  assert.equal(h.context.runRequestPhase,'preflight');
});

test('非描述类流程错误不会误触发描述生成', async () => {
  let generated = 0;
  const h = harness({
    runRequestPhase:'preflight', renderRunButton() {}, toast() {},
    async regenerateReferenceDescription(){generated++;},
    async fetch(url){return {ok:true,async json(){
      if(url==='/api/model_readiness')return {sam:{ready:true},vlm:{ready:true}};
      if(url==='/api/run/data-readiness')return {ready:true};
      return {ready:false,code:'reference_truth_missing',message:'参考帧尚无人工真值'};
    }};},
  });
  h.run(section(source('app.js'), 'async function ensureAutoLabelReadiness', "$('runButton').onclick"));
  await assert.rejects(() => h.context.ensureAutoLabelReadiness(null,1,1), /参考帧尚无人工真值/);
  assert.equal(generated,0);
});

test('运行前弹窗展示参考描述并提供一键恢复入口', () => {
  const index = source('index.html');
  const guide = source('guide.js');
  const app = source('app.js');
  assert.match(index,/data-readiness="description"/);
  assert.match(guide,/生成描述并继续/);
  assert.match(guide,/window\.regenerateReferenceDescription\(checks\.description\.referenceFrame\)/);
  assert.match(app,/descriptionRequest=\{key:requestKey,promise\}/);
  assert.match(app,/参考描述需要更新，正在自动生成；完成后会继续推理/);
});

test('完成与失败帧重试均复用轻量帧计数且不回读完整状态', () => {
  const handler = section(source('app.js'), "$('runButton').onclick", 'let runStatusPollTimer');
  assert.ok(handler.includes('{frames:requestedFrames,max_rounds:6,defer_agent_review:true}'));
  assert.ok(handler.includes('candidateCountIndex[String(item.frame)]'));
  assert.equal(handler.includes("fetch('/api/state')"), false);
});

test('主自动标注先返回机器候选并用绑定读回完成后台精修', () => {
  const app = source('app.js');
  const refinement = section(app, 'async function refineDeferredRunFrames', 'function invalidateChangedCandidateAudits');
  const handler = section(app, "$('runButton').onclick", 'let runStatusPollTimer');
  assert.match(handler, /defer_agent_review:true/);
  assert.match(handler, /void refineDeferredRunFrames\(data\.frames,jobTopic,jobFile\)/);
  assert.match(handler, /初始机器候选已打开，可立即检查/);
  assert.match(refinement, /readCompactFrameStateSnapshot\(target,sourceTopic,sourcePath\)/);
  assert.match(refinement, /expected_candidate_artifact_sha256:artifact/);
  assert.match(refinement, /expected_frame_sha256:imageHash/);
  assert.match(refinement, /candidate_only!==true/);
  assert.match(refinement, /human_truth!==false/);
  assert.match(refinement, /approved_for_training!==false/);
  assert.equal(refinement.includes("fetch('/api/state')"), false);
});

test('人工修改候选会按唯一任务号取消单帧和批量后台精修', () => {
  const app = source('app.js');
  const cancellation = section(app, 'function newCandidateRefinementId', 'async function auditReferenceCandidateSnapshot');
  const refinement = section(app, 'async function refineReferenceCandidateSnapshot', 'function invalidateChangedCandidateAudits');
  const refresh = section(app, 'function refreshShapeList', 'function draw');
  assert.match(cancellation, /reference_candidates\/refine\/stop/);
  assert.match(cancellation, /entry\.editSnapshot===snapshot/);
  assert.match(cancellation, /refinement_id:entry\.refinementId/);
  assert.match(cancellation, /intent\?\.source!=='machine_candidate'/);
  assert.match(refresh, /cancelChangedCandidateRefinement\(selectedShapes\(\),frame\)/);
  assert.equal((refinement.match(/refinement_id:entry\.refinementId/g) || []).length, 2);
  assert.match(refinement, /entry\.stale/);
  assert.match(refinement, /人工草稿已变化，过时的后台精修已取消/);
  const openDraft = section(app, 'window.prepareReferenceCandidateDraft=async', 'window.prepareReferenceLegacyTruthDraft=async');
  assert.match(openDraft, /candidateAuditFullSnapshot\(manualDrafts\[key\]\)!==candidateAuditFullSnapshot\(candidate\)\)return/);
  assert.ok(openDraft.indexOf('candidateAuditFullSnapshot(manualDrafts[key])!==candidateAuditFullSnapshot(candidate)')
    < openDraft.lastIndexOf('return finishAudit(candidate)'));
});

test('启动前并行检查模型、参考流程与首个请求帧的数据兼容性', async () => {
  const calls = [];
  const h = harness({
    fetch: async (url, options = {}) => {
      calls.push({url, options});
      if (url === '/api/model_readiness') return {ok:true, async json(){return {sam:{ready:true},vlm:{ready:true}};}};
      if (url === '/api/run/readiness') return {ok:true, async json(){return {ready:true};}};
      return {ok:true, async json(){return {ready:true,frame:7,width:640,height:512};}};
    },
  });
  h.run(section(source('app.js'), 'async function ensureAutoLabelReadiness', "$('runButton').onclick"));
  const result = await h.context.ensureAutoLabelReadiness(null,7,4);
  assert.equal(result.data.frame, 7);
  const dataCall = calls.find(item => item.url === '/api/run/data-readiness');
  assert.deepEqual(JSON.parse(dataCall.options.body), {frame:7,count:4});

  h.context.fetch = async url => ({ok:true, async json(){
    return url === '/api/run/data-readiness'
      ? {ready:false,message:'帧 7 无法解码'}
      : url === '/api/model_readiness'
        ? {sam:{ready:true},vlm:{ready:true}}
        : {ready:true};
  }});
  await assert.rejects(() => h.context.ensureAutoLabelReadiness([7,9],7,2), /数据：帧 7 无法解码/);
});

test('多文件新数据集使用一次批量上传请求', () => {
  const text = source('app.js');
  const uploader = section(text, 'function setFileUploadControls', 'function renderFiles');
  const handler = section(text, "$('chooseButton').onclick", "$('openButton').onclick");
  assert.ok(uploader.includes("form.append('files',file,file.name)"));
  assert.ok(uploader.includes("xhr.open('POST','/api/uploads')"));
  assert.ok(uploader.includes('xhr.upload.onprogress'));
  assert.ok(handler.includes("$('cancelUploadButton').onclick=()=>fileUploadRequest?.abort()"));
  assert.equal(handler.includes("fetch('/api/upload'"), false);
});

test('加载新数据后明确显示首帧兼容检查结果', () => {
  const handler = section(source('app.js'), 'function renderFiles', 'async function refreshFiles');
  assert.ok(handler.includes('data.data_compatibility?.ready'));
  assert.ok(handler.includes('首帧兼容检查通过'));
  assert.ok(handler.includes('data.data_compatibility.width'));
  assert.ok(handler.includes('data.data_compatibility.height'));
});

test('大文件上传显示实时进度并可在传输阶段取消', async () => {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {disabled:false,hidden:true,textContent:''});
    return elements.get(id);
  }
  class FakeFormData { append() {} }
  class FakeXHR {
    static last = null;
    constructor() { this.upload={};FakeXHR.last=this; }
    open(method,url) { this.method=method;this.url=url; }
    send() { this.sent=true; }
    abort() { this.onabort(); }
  }
  const h = harness({$:element,FormData:FakeFormData,XMLHttpRequest:FakeXHR});
  h.run(section(source('app.js'), 'let fileManagerBusy=', 'function renderFiles'));
  const pending = h.run("uploadFilesWithProgress([{name:'large.mcap'}])");
  FakeXHR.last.upload.onprogress({lengthComputable:true,loaded:50,total:100});
  assert.equal(element('fileManagerStatus').textContent, '正在上传 1 个文件 · 50%');
  assert.equal(FakeXHR.last.url, '/api/uploads');
  FakeXHR.last.abort();
  await assert.rejects(pending, /已取消上传/);
});

test('按需帧标注严格绑定文件相机帧并只替换目标帧', () => {
  const h = harness();
  h.run(`let topic='/camera0',loadedDataPath='/data/a.mcap',frameLoadGeneration=1;
    let shapes={},truthReviewStatus={},candidates={'2':[{'id':'old'}],'3':[{'id':'keep'}]},agentRuns={'2':{'rounds':1}},candidateReviewBindings={'2':{'old':true}},candidateReviewProposals=[
      {file:'/data/a.mcap',topic:'/camera0',frame:2,proposal_id:'old'},
      {file:'/data/a.mcap',topic:'/camera0',frame:3,proposal_id:'keep'}];
    let truthFrameIndex=new Set(),candidateCountIndex={};`);
  h.run(section(source('app.js'), 'function applyCompactFrameState', 'const viewedFrameStorageKey'));
  const payload = {
    compact_geometry: true, path: '/data/a.mcap', topic: '/camera0', frame: 2,
    truth: {'2': [{label: 'floor'}]},
    truth_review_status: {'2': {reusable_as_reference: true}},
    candidates: {}, agent_runs: {}, candidate_review_bindings: {},
    candidate_review_proposals: [{file: '/data/a.mcap', topic: '/camera0', frame: 2, proposal_id: 'new'}],
    truth_frames: ['2'], candidate_counts: {'2': 0, '3': 5},
  };
  h.context.payload = payload;
  h.run('applyCompactFrameState(payload,2)');
  const result = JSON.parse(h.run(`JSON.stringify({shapes,truthReviewStatus,candidates,agentRuns,
    candidateReviewBindings,candidateReviewProposals,truthFrames:[...truthFrameIndex],candidateCountIndex})`));
  assert.equal(result.shapes['2'][0].label, 'floor');
  assert.equal(result.truthReviewStatus['2'].reusable_as_reference, true);
  assert.equal(result.candidates['2'], undefined);
  assert.deepEqual(result.candidates['3'], [{id: 'keep'}]);
  assert.equal(result.agentRuns['2'], undefined);
  assert.deepEqual(result.truthFrames, ['2']);
  assert.deepEqual(result.candidateCountIndex, {'2': 0, '3': 5});
  assert.deepEqual(result.candidateReviewProposals.map(item => item.proposal_id).sort(), ['keep', 'new']);
  assert.throws(() => h.run(`applyCompactFrameState({...payload,topic:'/camera1'},2)`), /不一致/);
  assert.throws(() => h.run(`applyCompactFrameState({...payload,path:'/data/b.mcap'},2)`), /不一致/);
  assert.throws(() => h.run(`applyCompactFrameState({...payload,frame:3},2)`), /不一致/);
});
