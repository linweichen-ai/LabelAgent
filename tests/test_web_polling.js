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
    frame: 0,
    frameNavigationGeneration: 0,
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

test('参考起稿支持逐类尝试全部已选类别但仍保持候选边界', () => {
  const prep = source('reference-prep.js');
  assert.match(prep, /ALL_SELECTED_LABELS = '__all_selected_labels__'/);
  assert.match(prep, /逐类尝试全部已选推理类别（更慢，易误检）/);
  assert.match(prep, /fresh\.auto_labels \|\| \[\]/);
  assert.match(prep, /系统不会用逐类重跑覆盖/);
  assert.match(prep, /labels: tryAllSelected \? selectedLabels : \[label\]/);
  assert.match(prep, /result\.candidate_only/);
  assert.match(prep, /result\.human_truth !== false/);
});

test('参考起稿提供两阶段补漏但依据实测保持快速模式为默认', () => {
  const prep = source('reference-prep.js');
  const html = source('index.html');
  assert.match(prep, /AUTO_COVERAGE_LABELS = '__auto_visible_with_coverage__'/);
  assert.match(prep, /自动识别当前帧可见类别（推荐，快速）/);
  assert.match(prep, /自动识别 \+ 已选类别补漏（实验，较慢）/);
  assert.match(prep, /selectedLabel = labelSelect\.value \|\| AUTO_VISIBLE_LABELS/);
  assert.match(prep, /probe_missing_selected: probeMissingSelected/);
  assert.match(prep, /recall_probe_labels/);
  assert.match(html, /人工发现明显漏类时使用，但更慢且不保证产生有效轮廓/);
  assert.match(prep, /result\.candidate_only/);
  assert.match(prep, /result\.human_truth !== false/);
});

test('工作台持续显示新旧数据集身份和逐类别覆盖状态', () => {
  const html = source('index.html');
  const app = source('app.js');
  assert.match(html, /id="datasetIdentity"/);
  assert.match(html, /id="datasetIdentitySwitch"/);
  assert.match(app, /function datasetIdentityForPath/);
  assert.match(app, /SFL-CPD15-Y · 批次 A（0920）· 000/);
  assert.match(app, /CVAT Task3 · 斜坡复核集/);
  assert.match(app, /VLM 看见，SAM 漏分/);
  assert.match(app, /点击任一类别可查看当前帧依据/);
  assert.match(app, /function applyInferenceLabelState\(data,refresh=true\)/);
  assert.match(app, /function candidateCoverageEvidence\(/);
  assert.match(app, /action\.dataset\.coverageRepair=label/);
  assert.match(app, /startCandidateCoverageRepair\(label\)/);
  assert.match(app, /不会自动运行模型或写入人工真值/);
  assert.match(app, /applyInferenceLabelState\(data,false\)/);
  assert.match(app, /window\.referenceFrames=data\.reference_frames\|\|\{\};applyInferenceLabelState\(data\)/);
  assert.match(app, /updateDatasetIdentity\(\)/);
});

test('新数据集先原子替换逐类状态并直接打开当前图像相机', () => {
  const app = source('app.js');
  const prep = source('reference-prep.js');
  const packet = source('review-packet.js');
  const load = section(app, 'async function loadVideo', 'async function readCurrentDataScope');
  assert.ok(load.indexOf('candidates=data.candidates||{}') < load.indexOf('applyInferenceLabelState(data,false)'));
  assert.ok(load.indexOf('candidateCountIndex=data.candidate_counts') < load.indexOf('applyInferenceLabelState(data,false)'));
  assert.ok(load.indexOf('refreshTopics()') < load.indexOf('refreshCandidateSummary()'));
  assert.match(app, /if\(catalog\[topic\]\?\.type!=='image'\)return/);
  assert.match(app, /await addWindow\('image',\[topic\],true\)/);
  assert.match(app, /async function addWindow\(type,sources,deferInitialLoad=false\)/);
  assert.match(app, /if\(deferInitialLoad\).*continue/);
  assert.match(app, /pendingDatasetLoad=\{path:file\.path,name:file\.name\};updateDatasetIdentity\(\)/);
  assert.match(app, /finally\{pendingDatasetLoad=null;updateDatasetIdentity\(\)/);
  assert.match(app, /正在读取媒体索引并检查首帧；完成前保留当前工作区/);
  assert.match(app, /compact:true,path:loadedDataPath,file_activation_generation:loadedFileActivationGeneration/);
  assert.match(app, /async function loadVerifiedVideoData\(data\)/);
  assert.match(app, /window\.loadVideoData=loadVerifiedVideoData/);
  assert.match(app, /const before=await readCurrentDataScope\(\)/);
  assert.match(app, /await loadVideo\(data\);const after=await readCurrentDataScope\(\)/);
  assert.match(app, /await loadVerifiedVideoData\(data\)/);
  assert.match(prep, /window\.loadVerifiedVideoData\(data\)/);
  assert.match(packet, /window\.loadVerifiedVideoData\(data\)/);
});

test('过期文件加载回执必须与服务器当前文件和相机同时一致', () => {
  const h = harness();
  h.run(section(source('app.js'), 'function sameLoadedDataScope', 'async function loadVerifiedVideoData'));
  assert.equal(h.run(`sameLoadedDataScope({path:'/a',topic:'c0',file_activation_generation:7},{loaded:true,path:'/a',topic:'c0',file_activation_generation:7})`), true);
  assert.equal(h.run(`sameLoadedDataScope({path:'/a',topic:'c0',file_activation_generation:6},{loaded:true,path:'/a',topic:'c0',file_activation_generation:7})`), false);
  assert.equal(h.run(`sameLoadedDataScope({path:'/a',topic:'c0',file_activation_generation:7},{loaded:true,path:'/b',topic:'c0',file_activation_generation:7})`), false);
  assert.equal(h.run(`sameLoadedDataScope({path:'/a',topic:'c0',file_activation_generation:7},{loaded:true,path:'/a',topic:'c1',file_activation_generation:7})`), false);
  assert.equal(h.run(`sameLoadedDataScope({path:'/a',topic:'c0',file_activation_generation:7},{loaded:false,path:'/a',topic:'c0',file_activation_generation:7})`), false);
});

test('逐类别覆盖区分可见漏分、明确未见和没有可追溯证据', () => {
  const h = harness();
  h.run(section(source('app.js'), 'function candidateCoverageEvidence', 'function showCandidateCoverageDetail'));
  const missing = h.run(`candidateCoverageEvidence('slope',0,true,{evidence:'present',confidence:.91,reason:'ramp visible'},true)`);
  const absent = h.run(`candidateCoverageEvidence('ego',0,true,{evidence:'absent',confidence:.88,reason:'not visible'},false)`);
  const pending = h.run(`candidateCoverageEvidence('other_robot',0,true,undefined,false)`);
  assert.equal(missing.state, 'missing');
  assert.match(missing.detail, /置信度 91%/);
  assert.match(missing.detail, /ramp visible/);
  assert.equal(absent.state, 'absent');
  assert.match(absent.detail, /只是机器判断/);
  assert.equal(pending.state, 'pending');
  assert.match(pending.detail, /不能据此判断画面中没有/);
});

test('人工真值界面提供明确的连续推理入口', () => {
  const html = source('index.html');
  const inspection = source('inspection.js');
  assert.match(html, /id="goToAutoInference"/);
  assert.match(html, /去自动推理 · 设置 N 帧/);
  assert.match(inspection, /从当前帧开始，在下方设置“本次处理 N 帧”/);
  assert.match(inspection, /inferenceShortcut\.hidden=mode==='auto'/);
  assert.match(inspection, /count\?\.focus\(\)/);
  assert.match(inspection, /再点“开始自动标注”/);
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
  assert.match(prep, /\/api\/reference-preparation\?file_activation_generation=/);
  assert.match(prep, /window\.readCompactFrameStateSnapshot\(frameId, expected\.topic, expected\.path,[\s\S]*expected\.file_activation_generation\)/);
  assert.doesNotMatch(prep, /fetch\('\/api\/state'/);
  assert.match(prep, /state\?\.truth_index \|\| state\?\.truth/);
  assert.match(prep, /state\?\.candidate_index \|\| state\?\.candidates/);
});

test('参考准备弹窗为当前空白帧预取可见类别并显示SAM预热状态', () => {
  const prep = source('reference-prep.js');
  const prefetch = prep.slice(prep.indexOf('async function prefetchVisibleDiscovery'),
    prep.indexOf('\n  function render(state)'));
  assert.match(prefetch, /hasTruthFrame\(state, frameId\) \|\| currentFrameCandidate\(state, frameId\)/);
  assert.match(prefetch, /speculativeFetch\('\/api\/reference_discovery\/prefetch'/);
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
  assert.match(dwell, /speculativeFetch\('\/api\/model_readiness'/);
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
  assert.match(adjacent, /speculativeFetch\('\/api\/run\/status'/);
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
    const speculativeFetch = fetch;
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

test('数据文件或相机切换前取消全部推测性预取且调用方先取消再切换', () => {
  const prep = source('reference-prep.js');
  const reviewPacket = source('review-packet.js');
  const app = source('app.js');
  const cancellation = prep.slice(prep.indexOf('function cancelReferencePrefetch'),
    prep.indexOf('\n  function scheduleAdjacentFramePrefetch'));
  assert.match(prep, /const speculativeFetchControllers = new Set\(\)/);
  assert.match(prep, /signal: controller\.signal/);
  assert.match(cancellation, /frameDwellPrefetchGeneration \+= 1/);
  assert.match(cancellation, /clearTimeout\(frameDwellPrefetchTimer\)/);
  assert.match(cancellation, /cancelAdjacentFramePrefetch\(\)/);
  assert.match(cancellation, /controller\.abort\(\)/);
  assert.match(cancellation, /speculativeFetchControllers\.clear\(\)/);
  assert.match(prep, /document\.addEventListener\('labelagent:data-scope-changing', cancelReferencePrefetch\)/);
  assert.match(prep, /window\.cancelReferencePrefetch = cancelReferencePrefetch/);
  assert.match(reviewPacket, /window\.cancelReferencePrefetch\?\.\(\);[\s\S]*\/activate/);
  assert.ok(app.includes("window.cancelReferencePrefetch?.();const data=await api('files/'+file.id+'/load'"));
  assert.match(app, /window\.cancelReferencePrefetch\?\.\(\);const data=await api\('topic'/);
});

test('保存真值和锁定参考帧只读回目标帧且保留强绑定校验', () => {
  const app = source('app.js');
  const lock = app.slice(app.indexOf('async function lockReferenceFrame'), app.indexOf("$('referenceButton').onclick"));
  const save = app.slice(app.indexOf('async function saveCurrentTruth'), app.indexOf("$('truthSave').onclick"));
  assert.match(lock, /readCompactFrameStateSnapshot\(target,targetTopic,targetPath,targetGeneration\)/);
  assert.match(save, /readCompactFrameStateSnapshot\(targetFrame,targetTopic,targetPath,targetGeneration\)/);
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
  h.run("retryFrames=[3,7];retryScope={file:'/data.mcap',topic:'camera0',taskId:'failed-task'};retryMode='failed';renderRunButton()");
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
    runResultHandled:new Set(),runResultAutoRestoreDeferred:new Set(),retryFrames:[],retryScope:null,retryMode:'',
    loadedDataPath:'/data.mcap',topic:'camera0',frameNavigationGeneration:0,candidateCountIndex:{},
    async fetch() { return {ok:true,async json(){return {
      task_id:'interrupted-1',status:'running',file:'/data.mcap',topic:'camera0',
      completed:1,frames:[{frame:1,remaining:2,failed:false}],pending_frames:[2,3],failed_frames:[]
    }}}},
    refreshCandidateSummary() {},
    async openCandidateFrame(frame,topic) { opened={frame,topic}; return true; },
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

test('候选恢复被用户导航取消后保留查看入口且不反复抢回帧', async () => {
  let fetchCount = 0, allowOpen = false, notice = '';
  const job = {status:'completed',result_available:true,result_task_id:'recover-cancelled-1'};
  const h = harness({
    latestAgentJob:job, runResultHandled:new Set(), runResultAutoRestoreDeferred:new Set(),
    retryFrames:[],retryScope:null,retryMode:'',loadedDataPath:'/data.mcap',topic:'camera0',
    frameNavigationGeneration:0,candidateCountIndex:{},
    async fetch() { fetchCount++; return {ok:true,async json(){return {
      task_id:'recover-cancelled-1',status:'completed',file:'/data.mcap',topic:'camera0',
      completed:1,frames:[{frame:10,remaining:2,failed:false}],pending_frames:[],failed_frames:[]
    }}}},
    refreshCandidateSummary() {},
    async openCandidateFrame() { return allowOpen; },
    renderRunButton() {},
    toast(message) { notice=message; },
  });
  const text = source('app.js');
  h.run(section(text, 'function pendingRunResult', 'function formatAgentDuration'));

  assert.equal(await h.run('recoverRunResult(latestAgentJob,{automatic:true})'), false);
  assert.equal(fetchCount, 1);
  assert.equal(h.run("runResultHandled.has('recover-cancelled-1')"), false);
  assert.equal(h.run("runResultAutoRestoreDeferred.has('recover-cancelled-1')"), true);
  assert.equal(h.run('pendingRunResult(latestAgentJob)'), true);
  assert.match(notice, /结果已保留/);

  assert.equal(await h.run('recoverRunResult(latestAgentJob,{automatic:true})'), false);
  assert.equal(fetchCount, 1);

  allowOpen = true;
  assert.equal(await h.run('recoverRunResult(latestAgentJob)'), true);
  assert.equal(fetchCount, 2);
  assert.equal(h.run("runResultHandled.has('recover-cancelled-1')"), true);
  assert.equal(h.run("runResultAutoRestoreDeferred.has('recover-cancelled-1')"), false);
  assert.equal(h.run('pendingRunResult(latestAgentJob)'), false);
  assert.match(text, /recoverRunResult\(job,\{automatic:true\}\)/);
  assert.match(text, /\(latestAgentJob\?\.result_task_id\|\|latestAgentJob\?\.task_id\)===taskId/);
});

test('恢复旧结果期间出现更新任务时不抢帧、不改计数且保留旧候选', async () => {
  let releaseResult, opened = 0, notice = '';
  const blocked = new Promise(resolve => { releaseResult = resolve; });
  const oldJob = {status:'idle',result_available:true,result_task_id:'old-task'};
  const h = harness({
    latestAgentJob:oldJob, runResultHandled:new Set(), runResultAutoRestoreDeferred:new Set(),
    retryFrames:[],retryScope:null,retryMode:'',loadedDataPath:'/data.mcap',topic:'camera0',
    frameNavigationGeneration:7,candidateCountIndex:{},
    async fetch() { await blocked; return {ok:true,async json(){return {
      task_id:'old-task',status:'completed',file:'/data.mcap',topic:'camera0',
      completed:1,frames:[{frame:10,remaining:2,failed:false}],pending_frames:[],failed_frames:[]
    }}}},
    refreshCandidateSummary() {},
    async openCandidateFrame() { opened++; return true; },
    renderRunButton() {},
    toast(message) { notice=message; },
  });
  const text = source('app.js');
  h.run(section(text, 'function pendingRunResult', 'function formatAgentDuration'));
  const recovery = h.run('recoverRunResult(latestAgentJob,{automatic:false})');
  h.run("latestAgentJob={status:'running',task_id:'new-task',result_available:false}");
  releaseResult();
  assert.equal(await recovery, false);
  assert.equal(opened, 0);
  assert.equal(h.run("candidateCountIndex['10'] ?? null"), null);
  assert.equal(h.run("runResultHandled.has('old-task')"), false);
  assert.equal(h.run("runResultAutoRestoreDeferred.has('old-task')"), true);
  assert.match(notice, /新的自动标注任务已经开始/);
});

test('恢复旧结果读取期间的新导航和工作模式优先于旧回执', async () => {
  let releaseResult, opened = 0, notice = '';
  const blocked = new Promise(resolve => { releaseResult = resolve; });
  const job = {status:'idle',result_available:true,result_task_id:'old-task'};
  const h = harness({
    latestAgentJob:job, runResultHandled:new Set(), runResultAutoRestoreDeferred:new Set(),
    retryFrames:[],retryScope:null,retryMode:'',loadedDataPath:'/data.mcap',topic:'camera0',
    frameNavigationGeneration:7,candidateCountIndex:{},
    async fetch() { await blocked; return {ok:true,async json(){return {
      task_id:'old-task',status:'completed',file:'/data.mcap',topic:'camera0',
      completed:1,frames:[{frame:10,remaining:2,failed:false}],pending_frames:[],failed_frames:[]
    }}}},
    refreshCandidateSummary() {},
    async openCandidateFrame() { opened++; return true; },
    renderRunButton() {},
    toast(message) { notice=message; },
  });
  const text = source('app.js');
  h.run(section(text, 'function pendingRunResult', 'function formatAgentDuration'));
  const recovery = h.run('recoverRunResult(latestAgentJob,{automatic:false})');
  h.run("frameNavigationGeneration++;window.workspaceMode='manual'");
  releaseResult();
  assert.equal(await recovery, false);
  assert.equal(opened, 0);
  assert.equal(h.run("candidateCountIndex['10'] ?? null"), null);
  assert.equal(h.run("runResultHandled.has('old-task')"), false);
  assert.equal(h.run("runResultAutoRestoreDeferred.has('old-task')"), true);
  assert.match(notice, /恢复期间您已切换/);
});

test('旧候选图片载入期间出现新任务会返回恢复前帧和工作模式', async () => {
  let restoredFrame = null, notice = '';
  const oldJob = {status:'idle',result_available:true,result_task_id:'old-task'};
  const h = harness({
    frame:74,latestAgentJob:oldJob,runResultHandled:new Set(),runResultAutoRestoreDeferred:new Set(),
    retryFrames:[],retryScope:null,retryMode:'',loadedDataPath:'/data.mcap',topic:'camera0',
    frameNavigationGeneration:7,candidateCountIndex:{},
    async fetch() { return {ok:true,async json(){return {
      task_id:'old-task',status:'completed',file:'/data.mcap',topic:'camera0',
      completed:1,frames:[{frame:10,remaining:2,failed:false}],pending_frames:[],failed_frames:[]
    }}}},
    refreshCandidateSummary() {},
    async openCandidateFrame() {
      h.context.candidateCountIndex['10']=9;
      h.context.frame=10;h.context.workspaceMode='auto';
      h.context.latestAgentJob={status:'running',task_id:'new-task'};
      return true;
    },
    async showFrame(target) { restoredFrame=target;h.context.frame=target;return true; },
    async setWorkspaceMode(mode) { h.context.workspaceMode=mode; },
    renderRunButton() {},
    toast(message) { notice=message; },
  });
  const text = source('app.js');
  h.run(section(text, 'function pendingRunResult', 'function formatAgentDuration'));
  h.context.workspaceMode='manual';
  assert.equal(await h.run('recoverRunResult(latestAgentJob,{automatic:false})'), false);
  assert.equal(restoredFrame, 74);
  assert.equal(h.run('frame'), 74);
  assert.equal(h.run('window.workspaceMode'), 'manual');
  assert.equal(h.run("candidateCountIndex['10'] ?? null"), null);
  assert.equal(h.run("runResultHandled.has('old-task')"), false);
  assert.equal(h.run("runResultAutoRestoreDeferred.has('old-task')"), true);
  assert.match(notice, /已返回您原来的帧和工作模式/);
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
  let renders = 0, fetches = 0;
  class FakeAbortController {
    constructor() { this.signal = {}; }
    abort() { if (this.signal.onabort) this.signal.onabort(); }
  }
  const button = { title: '', removeAttribute() {} };
  const h = harness({
    AbortController: FakeAbortController,
    progressFetching: false,
    progressFetchPromise: null,
    runStatusFailureCount: 0,
    runStatusConnectionMessage: '',
    latestAgentJob: {status: 'running'},
    $() { return button; },
    renderRunButton() { renders++; },
    fetch(url, options) {
      fetches++;
      return new Promise((resolve, reject) => {
        options.signal.onabort = () => {
          const error = new Error('aborted'); error.name = 'AbortError'; reject(error);
        };
      });
    },
  });
  h.run(section(source('app.js'), 'async function fetchAgentStatus', 'async function ensureAutoLabelReadiness'));
  const pending = h.run('refreshAgentProgress()');
  const duplicate = h.run('refreshAgentProgress()');
  assert.equal(fetches, 1);
  await h.advance(8000);
  await Promise.all([pending, duplicate]);
  assert.equal(h.context.progressFetching, false);
  assert.ok(h.context.runStatusConnectionMessage.includes('超过 8 秒'));
  assert.equal(renders, 1);
});

test('旧数据集状态回执不会在新数据集显示查看结果或伪装连接失败', async () => {
  const button = {disabled:false,textContent:'',title:'',attributes:{},
    setAttribute(name,value){this.attributes[name]=value},removeAttribute(name){delete this[name]}};
  let responseIndex = 0, rendered = 0, scheduled = null;
  const responses = [
    {status:'idle',result_available:true,result_task_id:'old-task',
      selected_file:'/old.mcap',selected_topic:'camera0'},
    {status:'idle',result_available:false,selected_file:'/new.mcap',selected_topic:'camera0'},
  ];
  const h = harness({
    loadedDataPath:'/new.mcap',topic:'camera0',latestAgentJob:{status:'idle'},
    progressFetching:false,progressFetchPromise:null,runStatusScopeMismatch:null,
    runStatusConnectionMessage:'',runStatusFailureCount:0,runRequestStartedAt:null,
    bootstrapComplete:true,workspaceRestoreState:'ready',
    $: id => id === 'runButton' ? button : {},
    async fetchAgentStatus(){const job=responses[responseIndex++];return {ok:true,async json(){return job}}},
    renderAgentProgress(job){rendered++;h.context.latestAgentJob=job},
    renderRunButton(){button.disabled=h.context.runStatusScopeMismatch!==null;
      button.textContent=button.disabled?'数据已在其他窗口切换 · 请重新载入':'开始自动标注'},
    scheduleRunStatusPoll(delay){scheduled=delay},pendingRunResult(){return false},
    markWorkspaceReady(){},
  });
  h.run(section(source('app.js'), 'async function refreshAgentProgress', 'async function ensureAutoLabelReadiness'));
  assert.equal(await h.context.refreshAgentProgress(), false);
  assert.equal(rendered, 0);
  assert.equal(button.disabled, true);
  assert.match(button.textContent, /数据已在其他窗口切换/);
  assert.match(button.title, /不会沿用其任务状态/);
  assert.equal(scheduled, 250);
  assert.equal(h.context.runStatusFailureCount, 0);
  assert.equal(await h.context.refreshAgentProgress(), true);
  assert.equal(rendered, 1);
  assert.equal(h.context.runStatusScopeMismatch, null);
  assert.equal(button.title, undefined);
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

test('运行前弹窗与当前数据集相机帧号和处理数量强绑定', () => {
  const guide = source('guide.js');
  assert.match(guide, /function currentWorkbenchScope\(\)/);
  assert.match(guide, /window\.currentWorkbenchFrameState\(\)/);
  assert.match(guide, /String\(state\.path \|\| ''\) === current\.path/);
  assert.match(guide, /String\(state\.topic \|\| ''\) === current\.topic/);
  assert.match(guide, /readinessScope = currentWorkbenchScope\(\)/);
  assert.match(guide, /sameReadinessScope\(readinessScope, currentWorkbenchScope\(\)\)/);
  assert.match(guide, /当前数据或帧已切换，运行前检查已自动失效/);
  assert.match(guide, /byId\('frameNumber'\)\?\.addEventListener\('change', workspaceChanged\)/);
  assert.match(guide, /byId\('taskCount'\)\?\.addEventListener\('change', workspaceChanged\)/);
  assert.match(guide, /document\.addEventListener\('labelagent:frame-state', workspaceChanged\)/);
  assert.match(guide, /document\.addEventListener\('labelagent:workspace-ready', workspaceChanged\)/);
});

test('完成与失败帧重试均复用轻量帧计数且不回读完整状态', () => {
  const handler = section(source('app.js'), "$('runButton').onclick", 'let runStatusPollTimer');
  assert.ok(handler.includes('{...runScope,frames:requestedFrames,expected_result_task_id:retryTaskId,max_rounds:maxRounds,defer_agent_review:false,compare_with_truth:comparison}'));
  assert.ok(handler.includes('candidateCountIndex[String(item.frame)]'));
  assert.equal(handler.includes("fetch('/api/state')"), false);
});

test('失败与中断重试绑定来源任务并在过期时清除旧清单', () => {
  const app = source('app.js');
  const recovery = section(app, 'function pendingRetry', 'function formatAgentDuration');
  const handler = section(app, "$('runButton').onclick", 'let runStatusPollTimer');
  assert.match(recovery, /typeof retryScope\?\.taskId==='string'/);
  assert.match(recovery, /taskId\}/);
  assert.match(handler, /retryTaskId=requestedFrames\?retryScope\.taskId:null/);
  assert.match(handler, /retryScope\.taskId!==retryTaskId/);
  assert.match(handler, /expected_result_task_id:retryTaskId/);
  assert.match(handler, /requestedFrames&&e\.status===409/);
  assert.match(handler, /retryFrames=\[\];retryScope=null;retryMode=''/);
  assert.match(handler, /scheduleRunStatusPoll\(0\)/);
  assert.match(app, /error\.status=response\.status/);
});

test('主推理复查轮数可调且默认六轮', () => {
  const html = source('index.html');
  const app = source('app.js');
  assert.match(html, /id="reviewRounds"[^>]*min="1"[^>]*max="12"[^>]*value="6"/);
  assert.match(app, /复查轮数须为 1～12/);
  assert.match(app, /max_rounds:maxRounds/);
  assert.match(app, /max_rounds:runRequestMaxRounds/);
});

test('主工作台与参考起稿停止操作均绑定当前任务号', () => {
  const app = source('app.js');
  const prep = source('reference-prep.js');
  const handler = section(app, "$('runButton').onclick", 'let runStatusPollTimer');
  const cancel = section(prep, "byId('referencePrepCancel').addEventListener", "dialog.addEventListener('cancel'");
  assert.match(handler, /const stopTaskId=latestAgentJob\.task_id/);
  assert.match(handler, /expected_task_id:stopTaskId/);
  assert.match(handler, /stopped\.task_id!==stopTaskId/);
  assert.doesNotMatch(handler, /api\('run\/stop',\{\}\)/);
  assert.match(prep, /let draftTaskId = null/);
  assert.match(prep, /draftTaskId = status\.task_id/);
  assert.match(cancel, /expected_task_id: stopTaskId/);
  assert.match(cancel, /result\.task_id !== stopTaskId/);
  assert.doesNotMatch(cancel, /body: '\{\}'/);
});

test('参考描述恢复验收按精修唯一编号取消后台精修', () => {
  const tool = fs.readFileSync(path.join(__dirname, '../tools/browser_description_recovery_acceptance.cjs'), 'utf8');
  assert.match(tool, /refinementRequest\.postDataJSON\(\)/);
  assert.match(tool, /\/api\/reference_candidates\/refine\/stop/);
  assert.match(tool, /frame: payload\.frame/);
  assert.match(tool, /topic: payload\.topic/);
  assert.match(tool, /refinement_id: payload\.refinement_id/);
  assert.match(tool, /stopResult\.cancellation_requested, true/);
  assert.doesNotMatch(tool, /fetch\("\/api\/run\/stop"/);
});

test('主自动标注仅在服务器完成 GPT 复查后宣告完成', () => {
  const app = source('app.js');
  const refinement = section(app, 'async function refineDeferredRunFrames', 'function invalidateChangedCandidateAudits');
  const handler = section(app, "$('runButton').onclick", 'let runStatusPollTimer');
  assert.match(handler, /defer_agent_review:false/);
  assert.doesNotMatch(handler, /void refineDeferredRunFrames\(data\.frames,jobTopic,jobFile\)/);
  assert.match(handler, /SAM3、GPT 复查与 Agent 决策均已完成/);
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
  assert.match(openDraft, /candidateAuditFullSnapshot\(manualDrafts\[key\]\)!==candidateAuditFullSnapshot\(candidate\)\)return false/);
  assert.ok(openDraft.indexOf('candidateAuditFullSnapshot(manualDrafts[key])!==candidateAuditFullSnapshot(candidate)')
    < openDraft.lastIndexOf('return finishAudit(candidate)'));
});

test('四种参考起稿只在目标帧真实打开后激活且后台返回服从当前人工作用域', async () => {
  const app = source('app.js');
  const openFrame = section(app, 'async function openFrameForReferencePreparation', 'async function openReferenceFrame');
  const preparation = section(app, 'async function enterReferencePreparationFrame', "$('openReferenceButton').onclick");
  const begin = section(app, 'function beginReferenceFinalize', 'function clearReferenceFinalize');
  const guide = section(app, 'function updateReferenceFinalizeGuide', 'function selectInitialReferenceReviewTarget');
  assert.equal((preparation.match(/await enterReferencePreparationFrame\(/g) || []).length, 4);
  assert.match(preparation, /const active=\(\)=>referencePreparationIsActive\(target,sourceTopic,sourcePath\)/);
  assert.match(preparation, /if\(!active\(\)\)return false/);
  assert.ok(preparation.indexOf('await enterReferencePreparationFrame(target,requestGeneration,sourceTopic,sourcePath)')
    < preparation.indexOf('manualDrafts[key]=preparedDraft'));
  assert.ok(preparation.indexOf('await enterReferencePreparationFrame(target,requestGeneration,sourceTopic,sourcePath)')
    < preparation.indexOf('shapes[key]=preparedTruth'));
  assert.match(begin, /path:loadedDataPath/);
  assert.match(guide, /intent\.path===loadedDataPath/);
  assert.match(openFrame, /const targetFrame=Math\.max/);
  assert.match(openFrame, /const timeline=await prepareTimeline\(topic\)/);
  assert.match(openFrame, /clockTime=\(timeline\.offset\|\|0\)\+\(timeline\.times\[targetFrame\]\?\?0\)/);
  assert.match(openFrame, /await loadStill\(targetFrame\)/);

  let resolveOpen;
  const h = harness({
    openFrameForReferencePreparation() { return new Promise(resolve => { resolveOpen = resolve; }); },
  });
  h.run("let frameNavigationGeneration=3,topic='/camera',loadedDataPath='file-a',frame=5;");
  h.run(section(app, 'async function enterReferencePreparationFrame', 'window.prepareReferenceCandidateDraft=async'));
  const stale = h.run("enterReferencePreparationFrame(10,3,'/camera','file-a')");
  h.run('frameNavigationGeneration=4;frame=20');
  resolveOpen(true);
  await assert.rejects(stale, /本次参考起稿未进入/);

  h.context.openFrameForReferencePreparation = async () => true;
  h.run('frameNavigationGeneration=5;frame=10');
  assert.equal(await h.run("enterReferencePreparationFrame(10,5,'/camera','file-a')"), true);
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

test('自动标注预检后只提交原数据文件相机和导航作用域', () => {
  const app = source('app.js');
  const handler = section(app, "$('runButton').onclick=async", 'let runStatusPollTimer');
  assert.match(handler, /jobTopic=topic,jobFile=loadedDataPath,jobNavigationGeneration=frameNavigationGeneration,jobWorkspaceMode=window\.workspaceMode/);
  assert.match(handler, /topic!==jobTopic\|\|loadedDataPath!==jobFile\|\|frameNavigationGeneration!==jobNavigationGeneration\|\|window\.workspaceMode!==jobWorkspaceMode/);
  assert.match(handler, /预检期间数据集、相机、当前帧或工作模式已经切换，本次自动标注未启动/);
  assert.match(handler, /expected_file:jobFile,expected_topic:jobTopic/);
  assert.ok(handler.indexOf('jobTopic=topic') < handler.indexOf('await ensureAutoLabelReadiness'));
  assert.ok(handler.indexOf('await ensureAutoLabelReadiness') < handler.indexOf("const runScope={expected_file:jobFile,expected_topic:jobTopic}"));
});

test('自动标注完成回执只更新原始数据导航与工作模式作用域', () => {
  const handler = section(source('app.js'), "$('runButton').onclick=async", 'let runStatusPollTimer');
  assert.match(handler, /data\.file!==jobFile\|\|data\.topic!==jobTopic/);
  assert.match(handler, /const sameDataScope=topic===jobTopic&&loadedDataPath===jobFile/);
  assert.match(handler, /const sameOperatorScope=sameDataScope&&frameNavigationGeneration===jobNavigationGeneration&&window\.workspaceMode===jobWorkspaceMode/);
  assert.match(handler, /runResultAutoRestoreDeferred\.add\(data\.task_id\)/);
  assert.match(handler, /自动标注结果已保存到原数据作用域；当前页面保持不变/);
  assert.ok(handler.indexOf('if(!sameOperatorScope)') < handler.indexOf('markRunResultHandled(data.task_id)'));
  assert.ok(handler.indexOf('if(!sameOperatorScope)') < handler.indexOf('currentDescription=data.description'));
  assert.ok(handler.indexOf('if(!sameOperatorScope)') < handler.indexOf('candidateCountIndex[String(item.frame)]'));
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

test('评测模式在原图上优先显示机器候选且禁止覆盖人工真值', () => {
  const h = harness();
  h.run(`let frame=5,shapes={'5':[{'id':'truth'}]},candidates={'5':[{'id':'candidate'}]},
    manualDrafts={},autoDrafts={},truthReviewStatus={'5':{reusable_as_reference:true}};`);
  h.run(section(source('app.js'), 'function selectedShapes', 'const labelVisuals'));
  assert.equal(h.run('selectedShapes()[0].id'), 'candidate');
  assert.equal(h.run('frameAnnotationState(5)'), 'comparison_candidate');
  const app = source('app.js');
  assert.match(app, /compare_with_truth:comparison/);
  assert.match(app, /评测候选；人工真值已存在，禁止用候选覆盖/);
});

test('未通过当前人工门禁的已保存标注仍以待复核图层显示', () => {
  const html = source('index.html');
  const compare = source('compare.js');
  assert.match(html, /id="comparePending"[^>]*checked> 待人工复核标注/);
  assert.match(html, /legendPending[^<]*<\/i>待人工复核标注/);
  assert.match(compare, /canvasLayer\('Pending'/);
  assert.match(compare, /const pending = truthVerified \? \[\] : storedAnnotation/);
  assert.match(compare, /drawShapes\(canvases\.Pending, pending, 'pending'/);
  assert.match(compare, /待人工复核标注，已用黄色虚线叠加显示/);
});

test('拖动工作台高度后所有结果图层会重新贴合原图', () => {
  const compare = source('compare.js');
  assert.match(compare, /new ResizeObserver\(scheduleLayoutRender\)\.observe\(stage\)/);
  assert.match(compare, /new MutationObserver\(scheduleLayoutRender\)\.observe\(baseCanvas/);
  assert.match(compare, /attributeFilter: \['style', 'width', 'height'\]/);
});

test('点提示请求同步单飞、等待工作台就绪并绑定提交时的数据集类别与点位', async () => {
  const app = source('app.js');
  const generate = section(app, 'async function generatePointMask', 'function frameAtTime');
  function setup(restoreState = 'ready') {
    const elements = {labelSelect: {value: 'pallet'}};
    const calls = [], items = [], notices = [];
    let resolveRequest;
    const h = harness({
      $(id) { return elements[id] || (elements[id] = {}); },
      setInterval() { return 1; }, clearInterval() {},
      updatePointPromptCoach() {}, refreshShapeList() {}, draw() {},
      toast(message) { notices.push(message); },
      selectedShapes() { return items; },
      api(route, body) {
        calls.push({route, body: JSON.parse(JSON.stringify(body))});
        return new Promise(resolve => { resolveRequest = resolve; });
      },
    });
    h.run(`let segmentBusy=false,loading=false,playing=false,annotationMode=true,pointMode=true,
      promptPoints=[{x:10,y:20,label:1}],pointPromptNotice='',pointPromptStartedAt=0,
      pointPromptTimer=null,frame=3,topic='/camera',loadedDataPath='file-a',
      loadedFileActivationGeneration=7,
      generatedShapeFrame=null,generatedShapeTopic='',generatedShapeIndex=-1,
      selectedShapeIndex=-1,referenceReviewOriginalOnly=true,pointPromptGeneratedPointCount=0,
      pointPromptRequiresRefinement=false,workspaceRestoreState='${restoreState}';`);
    h.run(generate);
    return {h, elements, calls, items, notices, resolve(value) { resolveRequest(value); }};
  }

  const first = setup();
  const owner = first.h.run('generatePointMask()');
  const duplicate = first.h.run('generatePointMask()');
  assert.equal(first.calls.length, 1);
  first.h.run("promptPoints[0].x=999;$('labelSelect').value='person'");
  first.resolve({path:'file-a',topic:'/camera',frame:3,file_activation_generation:7,points:[[1,1],[2,1],[2,2]],holes:[],parts:[{points:[[1,1],[2,1],[2,2]],holes:[]}],score:.9});
  await Promise.all([owner, duplicate]);
  assert.deepEqual(first.calls[0], {route:'segment_points', body:{frame:3, points:[{x:10,y:20,label:1}],path:'file-a',topic:'/camera',file_activation_generation:7}});
  assert.equal(first.items.length, 1);
  assert.equal(first.items[0].label, 'pallet');
  assert.equal(first.h.run('pointPromptGeneratedPointCount'), 1);
  assert.equal(first.h.run('segmentBusy'), false);
  assert.match(first.h.run('pointPromptNotice'), /提交时类别“pallet”/);

  const stale = setup();
  const pending = stale.h.run('generatePointMask()');
  stale.h.run("loadedDataPath='file-b'");
  stale.resolve({path:'file-a',topic:'/camera',frame:3,file_activation_generation:7,points:[[1,1],[2,1],[2,2]],holes:[],parts:[{points:[[1,1],[2,1],[2,2]],holes:[]}],score:.9});
  await pending;
  assert.equal(stale.items.length, 0);
  assert.ok(stale.notices.includes('文件载入代次、数据集、相机或帧已切换，旧 SAM 结果已安全丢弃'));

  const reloaded = setup();
  const oldLoad = reloaded.h.run('generatePointMask()');
  reloaded.h.run('loadedFileActivationGeneration=8');
  reloaded.resolve({path:'file-a',topic:'/camera',frame:3,file_activation_generation:7,points:[[1,1],[2,1],[2,2]],holes:[],parts:[{points:[[1,1],[2,1],[2,2]],holes:[]}],score:.9});
  await oldLoad;
  assert.equal(reloaded.items.length, 0);
  assert.ok(reloaded.notices.includes('文件载入代次、数据集、相机或帧已切换，旧 SAM 结果已安全丢弃'));

  const restoring = setup('restoring');
  await restoring.h.run('generatePointMask()');
  assert.equal(restoring.calls.length, 0);
  assert.ok(restoring.notices.includes('正在恢复上次工作帧，完成后才能运行 SAM'));

  assert.match(app, /pointMode&&segmentBusy.*本次点位已锁定/);
  assert.match(app, /pointMode&&segmentBusy.*完成后再撤销/);
  assert.match(app, /if\(segmentBusy\).*完成后再清空本帧/);
  assert.match(source('index.html'), /app\.js\?v=188/);
});

test('页面启动先恢复最终工作帧再开放点选推理', () => {
  const app = source('app.js');
  assert.match(app, /workspaceRestoreState='booting'/);
  assert.match(app, /window\.labelAgentWorkspaceReady=window\.labelAgentBootstrapReady\.then/);
  assert.match(app, /workspaceRestoreState='restoring';bootstrapComplete=true;updateDatasetIdentity\(\);updatePointPromptCoach\(\);await refreshAgentProgress\(\);await workspaceReadyDeferred/);
  assert.match(app, /if\(bootstrapComplete&&runRequestStartedAt===null&&pendingRunResult\(job\)\)await recoverRunResult\(job,\{automatic:true\}\);if\(bootstrapComplete&&workspaceRestoreState!=='ready'\)markWorkspaceReady\(\)/);
  assert.match(app, /正在恢复上次工作帧，请稍候/);
  assert.match(app, /button\.textContent=restoring\?'正在恢复工作帧…'/);
  assert.doesNotMatch(app, /bootstrapComplete=true;setTimeout\(refreshAgentProgress,0\)/);
});

test('快速连续翻帧和拖动时间轴始终以最后一次操作为准', async () => {
  const app = source('app.js');
  const timelineResolvers = [], loads = [], notices = [];
  const h = harness({
    stopPlayback() {},
    prepareTimeline() { return new Promise(resolve => timelineResolvers.push(resolve)); },
    async loadStill(target) { loads.push(target); h.context.frame = target; },
    updateControls() {}, updateWindows() {},
    toast(message) { notices.push(message); },
  });
  h.run(`let frameNavigationGeneration=0,topic='/camera',loadedDataPath='file-a',
    topics={'/camera':{count:100}},playback=null,clockTime=0,frame=74;`);
  h.run(section(app, 'function frameAtTime', 'function currentClock'));
  h.run(section(app, 'function beginFrameNavigation', 'async function pauseAtClock'));
  h.run(section(app, 'async function showFrame', 'function refreshTopics'));

  const first = h.run('showFrame(10)');
  const second = h.run('showFrame(20)');
  assert.equal(timelineResolvers.length, 2);
  timelineResolvers[1]({offset:0,times:Array.from({length:100},(_,index)=>index)});
  await h.flush();
  timelineResolvers[0]({offset:0,times:Array.from({length:100},(_,index)=>index)});
  await Promise.all([first, second]);
  assert.deepEqual(loads, [20]);
  assert.equal(h.context.frame, 20);

  h.run("playback=null;loadedDataPath='file-a'");
  const staleScope = h.run('showFrame(30)');
  h.run("loadedDataPath='file-b'");
  timelineResolvers[2]({offset:0,times:Array.from({length:100},(_,index)=>index)});
  await staleScope;
  assert.deepEqual(loads, [20]);
  assert.deepEqual(notices, []);

  h.run("playback=null;loadedDataPath='file-a'");
  const oldTime = h.run('showTime(1)');
  const latestTime = h.run('showTime(2)');
  timelineResolvers[4]({offset:0,times:[0,1,2,3]});
  await h.flush();
  timelineResolvers[3]({offset:0,times:[0,1,2,3]});
  await Promise.all([oldTime, latestTime]);
  assert.deepEqual(loads, [20,2]);
  assert.equal(h.context.frame, 2);

  assert.match(app, /async function loadVideo\(data\)\{\s*invalidateFrameNavigation\(\)/);
  assert.match(app, /function clearLoadedVideo\(\)\{invalidateFrameNavigation\(\)/);
  assert.match(app, /async function activateAnnotation\(view\).*const navigation=beginFrameNavigation\(\)/);
  assert.match(source('index.html'), /app\.js\?v=188/);
});

test('区域编辑结果严格绑定提交时的帧、数据作用域、模式和原轮廓', async () => {
  const app = source('app.js');
  const pendingRequests = [], notices = [];
  const h = harness({
    refreshPointTools() {}, refreshShapeList() {}, draw() {},
    toast(message) { notices.push(message); },
  });
  h.context.itemsByFrame = {
    '3': [{label:'source',points:[[1,1],[10,1],[1,10]]}],
    '4': [{label:'destination',points:[[20,20],[30,20],[20,30]]}],
  };
  h.context.api = (route, body) => new Promise(resolve => pendingRequests.push({route,body,resolve}));
  h.run(`let pointMode=false,annotationMode=true,loading=false,playing=null,
    draft=[[2,2],[8,2],[2,8]],regionMode='add',regionEditGeneration=0,
    selectedShapeIndex=0,frame=3,topic='/camera',loadedDataPath='file-a',loadedFileActivationGeneration=7;
    function selectedShapes(){return itemsByFrame[String(frame)]}
    function candidateAuditFullSnapshot(value){return JSON.stringify(value)}`);
  h.run(section(app, 'async function finish', "$('frameImage').onload"));

  const stale = h.run('finish()');
  assert.equal(pendingRequests.length, 1);
  h.run('frame=4;selectedShapeIndex=0');
  pendingRequests[0].resolve({path:'file-a',topic:'/camera',frame:3,file_activation_generation:7,label:'stale-result',points:[[4,4],[9,4],[4,9]]});
  await stale;
  assert.equal(h.context.itemsByFrame['4'][0].label, 'destination');
  assert.ok(notices.includes('文件载入代次、帧、相机、数据集或模式已切换，旧区域编辑结果已安全丢弃'));

  h.run("frame=3;loadedFileActivationGeneration=7;regionMode='add';draft=[[2,2],[8,2],[2,8]];selectedShapeIndex=0");
  const oldLoad = h.run('finish()');
  assert.equal(pendingRequests.length, 2);
  h.run('loadedFileActivationGeneration=8');
  pendingRequests[1].resolve({path:'file-a',topic:'/camera',frame:3,file_activation_generation:7,label:'old-load-result',points:[[3,3],[7,3],[3,7]]});
  await oldLoad;
  assert.equal(h.context.itemsByFrame['3'][0].label, 'source');

  h.run("loadedFileActivationGeneration=7;regionMode='add';draft=[[2,2],[8,2],[2,8]];selectedShapeIndex=0");
  const older = h.run('finish()');
  const latest = h.run('finish()');
  assert.equal(pendingRequests.length, 4);
  pendingRequests[2].resolve({path:'file-a',topic:'/camera',frame:3,file_activation_generation:7,label:'older-result',points:[[3,3],[7,3],[3,7]]});
  await older;
  assert.equal(h.context.itemsByFrame['3'][0].label, 'source');
  pendingRequests[3].resolve({path:'file-a',topic:'/camera',frame:3,file_activation_generation:7,label:'latest-result',points:[[5,5],[9,5],[5,9]]});
  await latest;
  assert.equal(h.context.itemsByFrame['3'][0].label, 'latest-result');
  assert.equal(pendingRequests[3].body.frame, 3);
  assert.equal(pendingRequests[3].body.operation, 'add');
  assert.equal(pendingRequests[3].body.path, 'file-a');
  assert.equal(pendingRequests[3].body.topic, '/camera');
  assert.equal(pendingRequests[3].body.file_activation_generation, 7);
  assert.equal(JSON.stringify(pendingRequests[3].body.shape), JSON.stringify({label:'source',points:[[1,1],[10,1],[1,10]]}));

  h.run("itemsByFrame['3']=[];regionMode='add';draft=[[2,2],[8,2],[2,8]];selectedShapeIndex=0");
  await h.run('finish()');
  assert.equal(pendingRequests.length, 4);
  assert.ok(notices.includes('当前轮廓不存在，请重新选择后再编辑'));
  assert.match(source('index.html'), /app\.js\?v=188/);
});

test('Local VLM 短暂抖动会自动重试并在弹窗打开时自恢复', () => {
  const prep = source('reference-prep.js');
  assert.match(prep, /for \(let attempt = 0; attempt < 3; attempt \+= 1\)/);
  assert.match(prep, /setTimeout\(resolve, attempt \? 750 : 250\)/);
  assert.match(prep, /readModelReadiness\(generation\)\.catch\(\(\) => scheduleModelReadinessRetry\(\)\)/);
  assert.match(prep, /dialog\.addEventListener\('close', clearModelReadinessRetry\)/);
});
