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

test('ego 点扫描补救在界面中保持高风险人工复核边界', () => {
  const app = source('app.js');
  assert.match(app, /ego_bottom_point_fallback/);
  assert.match(app, /egoFallback\.status==='candidate_generated'/);
  assert.match(app, /文本分割漏检/);
  assert.match(app, /该结果已进入高风险复核，仍须逐个检查后才能成为人工真值/);
});

test('车辆可见漏分补救在界面中保持候选和人工复核边界', () => {
  const app = source('app.js');
  assert.match(app, /visible_vehicle_point_recovery/);
  assert.match(app, /vehicleRecoveryCandidate/);
  assert.match(app, /VLM 看见远处机器，但文本分割漏检/);
  assert.match(app, /routedLabel/);
  assert.match(app, /仍是机器候选/);
  assert.match(app, /必须人工检查边界和类别后才能成为真值/);
});

test('参考帧后台复核结束后立即刷新顶部向导而不继续显示处理中', () => {
  const app = source('app.js');
  const guide = source('guide.js');
  assert.match(app, /signalBackgroundComplete\('reference_background_audit_complete'\)/);
  assert.match(app, /signalBackgroundComplete\('reference_background_refinement_complete'\)/);
  assert.match(app, /labelagent:candidate-state-synced/);
  assert.match(guide, /document\.addEventListener\('labelagent:candidate-state-synced', refreshGuide\)/);
});

test('参考起稿支持快速尝试全部已选类别但仍保持候选边界', () => {
  const prep = source('reference-prep.js');
  assert.match(prep, /ALL_SELECTED_LABELS = '__all_selected_labels__'/);
  assert.match(prep, /快速尝试全部已选推理类别（推荐用于完整覆盖）/);
  assert.match(prep, /fresh\.auto_labels \|\| \[\]/);
  assert.match(prep, /系统不会用逐类重跑覆盖/);
  assert.match(prep, /labels: tryAllSelected \? selectedLabels : \[label\]/);
  assert.match(prep, /deterministic_description: tryAllSelected/);
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
  assert.match(app, /Capture_20261103 · 常熟实采 · 四相机/);
  assert.match(app, /SFL-CPD15-Y · 批次 B（1104）· 问题包/);
  assert.match(app, /问题包抽样/);
  assert.match(app, /SFL-CPD15-Y · 批次 A（0920）· 000/);
  assert.match(app, /CVAT Task3 · 斜坡复核集/);
  assert.match(app, /VLM 看见，SAM 漏分/);
  assert.match(app, /点击任一类别可查看当前帧依据/);
  assert.match(app, /function applyInferenceLabelState\(data,refresh=true\)/);
  assert.match(app, /function candidateCoverageEvidence\(/);
  assert.match(app, /function candidateReviewCountsByLabel\(/);
  assert.match(app, /candidateNeedsReview/);
  assert.match(html, /style\.css\?v=124/);
  assert.match(html, /homepage-theme\.css\?v=5/);
  assert.match(html, /app\.js\?v=217/);
  assert.match(app, /action\.dataset\.coverageRepair=label/);
  assert.match(app, /startCandidateCoverageRepair\(label\)/);
  assert.match(app, /不会自动运行模型或写入人工真值/);
  assert.match(app, /applyInferenceLabelState\(data,false\)/);
  assert.match(app, /window\.referenceFrames=data\.reference_frames\|\|\{\};applyInferenceLabelState\(data\)/);
  assert.match(app, /updateDatasetIdentity\(\)/);
});

test('数据集身份按真实路径区分旧验证集、新实采数据和问题包抽样', () => {
  const h = harness();
  h.run(section(source('app.js'), 'function datasetIdentityForPath', 'function updateDatasetIdentity'));
  const capture = h.context.datasetIdentityForPath('D:\\Apps\\LabelAgent\\work\\datasets\\Capture_20261103\\常熟0920_四相机与点云.mcap');
  const batchB = h.context.datasetIdentityForPath('D:\\Apps\\LabelAgent\\work\\datasets\\SFL_CPD15Y_20261104_smoke\\003-1793759959809514987.png');
  const batchB004 = h.context.datasetIdentityForPath('D:\\Apps\\LabelAgent\\work\\datasets\\SFL_CPD15Y_20261104_smoke\\004-1793760662042842987.png');
  const batchA = h.context.datasetIdentityForPath('D:\\Apps\\LabelAgent\\work\\datasets\\SFL_CPD15Y_20261111\\batchA_0920_000_clean_review.mcap');
  const cvat = h.context.datasetIdentityForPath('D:\\Apps\\LabelAgent\\work\\datasets\\CVAT_Task3_20261028\\CVAT_Task3_SAM3_candidates.mcap');
  assert.equal(capture.badge, '新实采数据');
  assert.equal(capture.name, 'Capture_20261103 · 常熟实采 · 四相机');
  assert.equal(batchB.badge, '问题包抽样');
  assert.equal(batchB.name, 'SFL-CPD15-Y · 批次 B（1104）· 问题包 003 · 单图抽样');
  assert.equal(batchB004.name, 'SFL-CPD15-Y · 批次 B（1104）· 问题包 004 · 单图抽样');
  assert.equal(batchA.badge, '新数据集');
  assert.equal(cvat.badge, '旧验证集');
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

test('逐类别覆盖汇总绑定当前候选的复核风险且按候选去重', () => {
  const h = harness();
  h.run(section(source('app.js'), 'function candidateReviewCountsByLabel', 'function showCandidateCoverageDetail'));
  const run = `({
    candidate_class_audits:[
      {candidate_id:15,bound_current_label:'person',route:'human_review_required',decision:{classification:'uncertain'}},
      {candidate_id:15,bound_current_label:'person',human_review_required:true,decision:{candidate_supported:false}},
      {candidate_id:3,bound_current_label:'ego',route:'accepted',decision:{classification:'supported'}}
    ],
    review_queue:[
      {shape_ids:[14],source_label:'person'},
      {shape_ids:[15],source_label:'person'},
      {source_label:'person'}
    ]
  })`;
  assert.equal(h.run(`candidateReviewCountsByLabel(${run}).get('person')`), 2);
  assert.equal(h.run(`candidateReviewCountsByLabel(${run}).has('ego')`), false);
  const review = h.run(`candidateCoverageEvidence('person',2,true,undefined,true,2)`);
  assert.equal(review.state, 'candidate-review');
  assert.match(review.title, /其中 2 个需重点复核/);
  assert.match(review.detail, /不会据此自动删除或改写真值/);
});

test('结果叠加验收工具跟随当前 Node 运行时寻找 Playwright', () => {
  const tool = fs.readFileSync(path.join(
    __dirname, '../tools/browser_result_overlay_acceptance.cjs'), 'utf8');
  assert.match(tool, /path\.dirname\(process\.execPath\)/);
  assert.match(tool, /node_modules', 'playwright/);
  assert.match(tool, /playwrightCandidates\.find\(candidate => fs\.existsSync\(candidate\)\)/);
  assert.match(tool, /Playwright runtime not found/);
});

test('数据集覆盖验收跟随当前运行时并等待工作区恢复完成', () => {
  const tool = fs.readFileSync(path.join(
    __dirname, '../tools/browser_dataset_coverage_acceptance.cjs'), 'utf8');
  assert.match(tool, /path\.dirname\(process\.execPath\)/);
  assert.match(tool, /playwrightCandidates\.find\(candidate => fs\.existsSync\(candidate\)\)/);
  assert.match(tool, /datasetIdentityKind/);
  assert.match(tool, /新实采数据.*新数据集.*旧验证集.*已加载数据/s);
  assert.match(tool, /button\[data-coverage-label\]/);
  assert.match(tool, /问题包 003/);
  assert.match(tool, /问题包 004/);
});

test('Python 数据集覆盖验收绑定作用域并保存失败收据', () => {
  const tool = fs.readFileSync(path.join(
    __dirname, '../tools/browser_dataset_coverage_acceptance.py'), 'utf8');
  assert.match(tool, /workbench scope changed during/);
  assert.match(tool, /dataset-coverage-browser-failure\.v1/);
  assert.match(tool, /initial_scope/);
  assert.match(tool, /current_scope/);
  assert.match(tool, /truth_write_requests/);
});

test('跨窗口同文件重载浏览器验收拒绝正式端口和真值写入', () => {
  const tool = fs.readFileSync(path.join(
    __dirname, '../tools/browser_cross_window_scope_acceptance.py'), 'utf8');
  assert.match(tool, /refuses the production workbench/);
  assert.match(tool, /parsed\.port in \{None, 8765\}/);
  assert.match(tool, /truth_write_requests/);
  assert.match(tool, /selected_file_activation_generation/);
  assert.match(tool, /switched\["path"\] == initial_scope\["path"\]/);
  assert.match(tool, /button_disabled/);
  assert.match(tool, /已安全暂停/);
});

test('新实采数据端到端验收绑定 Capture 身份而不是模糊的新数据文案', () => {
  const tool = fs.readFileSync(path.join(
    __dirname, '../tools/browser_new_dataset_inference_acceptance.py'), 'utf8');
  assert.match(tool, /identity\["kind"\] == "新实采数据"/);
  assert.match(tool, /Capture_20261103/);
  assert.match(tool, /parser\.add_argument\("--frame"/);
  assert.match(tool, /parser\.add_argument\("--coverage-probe"/);
  assert.match(tool, /select_option\(\s*"__auto_visible_with_coverage__"/);
  assert.match(tool, /candidates\?\.\[String\(expected\)\]/);
  assert.match(tool, /"frame": args\.frame/);
  assert.match(tool, /reference draft request failed/);
  assert.match(tool, /reference draft returned a non-JSON success response/);
  assert.match(tool, /details_open/);
  assert.match(tool, /VLM observed_label/);
});

test('候选类别复核默认显示中文摘要并折叠原始模型依据', () => {
  const app = source('app.js');
  const style = source('style.css');
  assert.match(app, /function candidateClassAuditDisplay/);
  assert.match(app, /通过项已收起/);
  assert.match(app, /必须人工确认/);
  assert.match(app, /candidateAuditDetails/);
  assert.match(app, /查看全部模型原始依据/);
  assert.match(style, /\.candidateAuditDetails summary/);
});

test('人工真值界面提供明确的连续推理入口', () => {
  const html = source('index.html');
  const inspection = source('inspection.js');
  assert.match(html, /id="goToAutoInference"/);
  assert.match(html, /连续推理 · N 帧/);
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
  assert.match(adjacent, /window\.labelAgentRunStatusSnapshot \|\| null/);
  assert.match(adjacent, /window\.labelAgentRunStatusObservedAtMs \|\| 0/);
  assert.match(adjacent, /snapshotAge >= 0 && snapshotAge <= 2000/);
  assert.match(adjacent, /status\.selected_file === state\.path/);
  assert.match(adjacent, /status\.selected_topic === state\.topic/);
  assert.match(adjacent, /selected_file_activation_generation/);
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
          return {status: statusReads === 1 ? 'running' : 'idle',
            selected_file: 'dataset.mcap', selected_topic: '/camera',
            selected_file_activation_generation: 7};
        }};
      }
      return {ok: true, async json() { return {prefetched: true}; }};
    },
  });
  h.context.currentWorkbenchFrameState = () => ({
    path: 'dataset.mcap', topic: '/camera', file_activation_generation: 7,
    labels: ['floor'],
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
  const state = {path: 'dataset.mcap', topic: '/camera', file_activation_generation: 7,
    labels: ['floor'],
    frame_count: 40, truth_index: {}, candidate_index: {}};
  h.context.state = state;
  h.run('scheduleAdjacentFramePrefetch(state, 30)');
  await h.advance(2999);
  assert.equal(calls.length, 0);
  await h.advance(1);
  assert.deepEqual(calls.map(item => item.url), ['/api/run/status']);
  h.context.window.labelAgentRunStatusObservedAtMs = Date.now() - 5000;
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

test('相邻帧预取复用主工作台运行快照且繁忙转空闲后不重复读取状态', async () => {
  const prep = source('reference-prep.js');
  const adjacent = prep.slice(prep.indexOf('function cancelAdjacentFramePrefetch'),
    prep.indexOf('\n  function render(state)'));
  const calls = [];
  const h = harness({
    frameValue: 30,
    async fetch(url, options = {}) {
      calls.push({url, options});
      return {ok: true, async json() { return {prefetched: true}; }};
    },
  });
  h.context.window.labelAgentRunStatusSnapshot = {status: 'running',
    selected_file: 'dataset.mcap', selected_topic: '/camera',
    selected_file_activation_generation: 7};
  h.context.window.labelAgentRunStatusObservedAtMs = Date.now();
  h.context.currentWorkbenchFrameState = () => ({
    path: 'dataset.mcap', topic: '/camera', file_activation_generation: 7,
    labels: ['floor'],
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
  const state = {path: 'dataset.mcap', topic: '/camera', file_activation_generation: 7,
    labels: ['floor'],
    frame_count: 40, truth_index: {}, candidate_index: {}};
  h.context.state = state;
  h.run('scheduleAdjacentFramePrefetch(state, 30)');
  await h.advance(3000);
  assert.deepEqual(calls, []);
  h.context.window.labelAgentRunStatusSnapshot = {status: 'idle',
    selected_file: 'dataset.mcap', selected_topic: '/camera',
    selected_file_activation_generation: 7};
  h.context.window.labelAgentRunStatusObservedAtMs = Date.now();
  await h.advance(5000);
  assert.deepEqual(calls.map(item => item.url), ['/api/reference_discovery/prefetch']);
  assert.deepEqual(JSON.parse(calls[0].options.body), {
    frame: 31, expected_path: 'dataset.mcap', expected_topic: '/camera'});
});

test('相邻帧预取不信任陈旧或错作用域的空闲快照', async () => {
  const prep = source('reference-prep.js');
  const adjacent = prep.slice(prep.indexOf('function cancelAdjacentFramePrefetch'),
    prep.indexOf('\n  function render(state)'));
  const calls = [];
  const h = harness({
    frameValue: 30,
    async fetch(url, options = {}) {
      calls.push({url, options});
      if (url === '/api/run/status') return {ok: true, async json() {
        return {status: 'running', selected_file: 'dataset.mcap',
          selected_topic: '/camera', selected_file_activation_generation: 7};
      }};
      return {ok: true, async json() { return {prefetched: true}; }};
    },
  });
  h.context.window.labelAgentRunStatusSnapshot = {status: 'idle',
    selected_file: 'other.mcap', selected_topic: '/camera',
    selected_file_activation_generation: 7};
  h.context.window.labelAgentRunStatusObservedAtMs = Date.now() - 5000;
  h.context.currentWorkbenchFrameState = () => ({
    path: 'dataset.mcap', topic: '/camera', file_activation_generation: 7,
    labels: ['floor'],
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
  const state = {path: 'dataset.mcap', topic: '/camera', file_activation_generation: 7,
    labels: ['floor'], frame_count: 40, truth_index: {}, candidate_index: {}};
  h.context.state = state;
  h.run('scheduleAdjacentFramePrefetch(state, 30)');
  await h.advance(3000);
  assert.deepEqual(calls.map(item => item.url), ['/api/run/status']);
  assert.equal(calls.some(item => item.url === '/api/reference_discovery/prefetch'), false);
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

test('引导台复用主工作台运行快照且并发刷新只补读一次', async () => {
  const calls = [];
  const h = harness({
    latest: { state: null, job: {status: 'idle'} }, latestRunStatus: 'idle',
    guidePollTimer: null, guideDeferredRefreshTimer: null, guideRefreshBusy: false,
    guideStatusRefreshPending: false, guideLastRefreshAt: 0,
    guideWorkspaceReady: true,
    GUIDE_REFRESH_COOLDOWN_MS: 1000,
    async fetch(url) {
      calls.push(url);
      return {ok: true, async json() { return url === '/api/state/summary'
        ? {loaded: false} : {status: 'idle'}; }};
    },
    render() {}, renderUnavailable() {}, scheduleGuidePoll() {},
  });
  h.context.window.labelAgentRunStatusSnapshot = {status: 'idle'};
  const text = source('guide.js');
  h.run(section(text, '  function deferGuideRefresh(', '\n  function scheduleGuidePoll'));
  await h.run('refreshGuide()');
  assert.deepEqual(calls, ['/api/state/summary']);
  await h.run('refreshGuide()');
  assert.deepEqual(calls, ['/api/state/summary']);
  assert.equal(h.timers.size, 1);
  await h.advance(1000);
  assert.deepEqual(calls, ['/api/state/summary', '/api/state/summary']);
  assert.match(text, /await window\.labelAgentInitialRunStatusReady/);
  h.context.guideRefreshBusy = true;
  await h.run('refreshGuide()');
  assert.equal(h.context.guideStatusRefreshPending, true);
});

test('引导台在工作区恢复完成前合并启动事件且只首次读取一次摘要', async () => {
  const calls = [];
  const h = harness({
    latest: { state: null, job: {status: 'idle'} }, latestRunStatus: 'idle',
    guidePollTimer: null, guideDeferredRefreshTimer: null, guideRefreshBusy: false,
    guideStatusRefreshPending: false, guideLastRefreshAt: 0,
    guideWorkspaceReady: false, GUIDE_REFRESH_COOLDOWN_MS: 1000,
    async fetch(url) {
      calls.push(url);
      return {ok: true, async json() { return {loaded: false}; }};
    },
    render() {}, renderUnavailable() {}, scheduleGuidePoll() {},
  });
  h.context.window.labelAgentRunStatusSnapshot = {status: 'idle'};
  const text = source('guide.js');
  h.run(section(text, '  function deferGuideRefresh(', '\n  function scheduleGuidePoll'));
  await h.run('refreshGuide()');
  await h.run('refreshGuide()');
  assert.deepEqual(calls, []);
  assert.equal(h.context.guideStatusRefreshPending, true);
  h.context.guideWorkspaceReady = true;
  h.context.guideStatusRefreshPending = false;
  await h.run('refreshGuide()');
  assert.deepEqual(calls, ['/api/state/summary']);
});

test('隐藏工作台暂停版本探针且切回可见时立即补查', async () => {
  const calls = [];
  const h = harness({
    currentUiVersion: '169',
    showUiVersionNotice() {},
    async fetch(url) {
      calls.push(url);
      return {ok: true, async text() {
        return '<meta name="labelagent-ui-version" content="169">';
      }};
    },
  });
  const text = source('guide.js');
  h.run(section(text, '  async function checkUiVersion()', '\n  function objectKeys'));
  h.context.document.hidden = true;
  await h.run('checkUiVersion()');
  assert.deepEqual(calls, []);
  h.context.document.hidden = false;
  await h.run('checkUiVersion()');
  assert.equal(calls.length, 1);
  assert.match(calls[0], /^\/\?ui_version_probe=/);
});

test('引导台直接消费任务事件且只在任务结束时同步摘要', async () => {
  let refreshes = 0, renders = 0;
  const h = harness({
    latest: {state: {loaded: true}, job: {status: 'idle'}}, latestRunStatus: 'idle',
    guidePollTimer: null, guideRefreshBusy: false,
    stateMatchesCurrentWorkspace() { return true; },
    render() { renders++; },
    async refreshGuide() { refreshes++; },
  });
  const text = source('guide.js');
  h.run(section(text, '  function scheduleGuidePoll(', '\n  function showWorkbench'));
  h.run(section(text, "  document.addEventListener('labelagent:job-status'", "  byId('manualModeButton')"));
  await h.emit('labelagent:job-status', { status: 'running' });
  assert.equal(renders, 1);
  assert.equal(refreshes, 0);
  assert.equal(h.timers.size, 1);
  await h.advance(14999);
  assert.equal(refreshes, 0);
  await h.advance(1);
  assert.equal(refreshes, 1);
  await h.emit('labelagent:job-status', { status: 'completed' });
  assert.equal(renders, 2);
  assert.equal(refreshes, 2);
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

test('高风险类别复核进度区分批量、串行、人工与实际重试', () => {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      textContent: '', style: {}, hidden: false, attributes: {},
      className: '', classList: { values: new Set(), toggle(name, enabled) {
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
  h.run("renderAgentProgressPanel({task_id:'audit',status:'running',phase:'candidate_class_audit',total:1,completed:0,frame:0,audit_completed:3,audit_total:3,audit_batch_completed:2,audit_sequential_completed:1,audit_review_required:1,audit_retry_count:1,events:[]},4)");
  assert.equal(element('runProgressTitle').textContent, '隔离复核高风险类别候选');
  assert.ok(element('runProgressDetail').textContent.includes('高风险类别已复核 3/3'));
  assert.ok(element('runProgressDetail').textContent.includes('批量 2 · 串行 1'));
  assert.ok(element('runProgressDetail').textContent.includes('需人工 1 · 实际重试 1'));
});

test('人工检查区实时显示后台精修且不冒充人工真值', () => {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {textContent:'轮廓草稿待人工确认',dataset:{baseStatus:'轮廓草稿待人工确认'}});
    return elements.get(id);
  }
  const h = harness({$:element,loadedDataPath:'dataset-a',topic:'/camera3',frame:101});
  const text = source('app.js');
  h.run("let latestAgentJob={status:'idle'};let referencePreparationIntent={source:'machine_candidate',path:'dataset-a',topic:'/camera3',frame:101};");
  h.run(section(text, 'function formatAgentDuration', 'function estimatedAgentPhaseWeight'));
  h.run(section(text, 'function referenceBackgroundProgressText', 'function updateReferenceFinalizeGuide'));
  h.run("refreshReferenceBackgroundProgress({status:'running',phase:'agent_refining',frame:101,phase_elapsed_seconds:7})");
  assert.ok(element('referenceFinalizeStatus').textContent.includes('初始候选已显示'));
  assert.ok(element('referenceFinalizeStatus').textContent.includes('可先逐个检查'));
  assert.ok(element('referenceFinalizeStatus').textContent.includes('不会写入人工真值'));
});

test('人工检查区显示高风险类别复核计数并在任务结束后恢复基础提示', () => {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {textContent:'轮廓草稿待人工确认',dataset:{baseStatus:'轮廓草稿待人工确认'}});
    return elements.get(id);
  }
  const h = harness({$:element,loadedDataPath:'dataset-a',topic:'/camera2',frame:24});
  const text = source('app.js');
  h.run("let latestAgentJob={status:'idle'};let referencePreparationIntent={source:'machine_candidate',path:'dataset-a',topic:'/camera2',frame:24};");
  h.run(section(text, 'function formatAgentDuration', 'function estimatedAgentPhaseWeight'));
  h.run(section(text, 'function referenceBackgroundProgressText', 'function updateReferenceFinalizeGuide'));
  h.run("refreshReferenceBackgroundProgress({status:'running',phase:'candidate_class_audit',frame:24,phase_elapsed_seconds:13,audit_completed:2,audit_total:5})");
  assert.ok(element('referenceFinalizeStatus').textContent.includes('后台复核 2/5'));
  h.run("refreshReferenceBackgroundProgress({status:'completed',phase:'completed',frame:24})");
  assert.equal(element('referenceFinalizeStatus').textContent, '轮廓草稿待人工确认');
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
  h.context.modelHealthState = {ready:false,sam:{ready:false},vlm:{ready:true}};
  h.run('renderRunButton()');
  assert.equal(button.disabled, true);
  assert.equal(button.textContent, '模型暂不可用');
  assert.match(button.title, /SAM3=不可用.*Qwen8B=正常/);
  h.context.latestAgentJob = {status:'running',task_id:'active-1',round:1,max_rounds:6};
  h.run('renderRunButton()');
  assert.equal(button.disabled, false);
  assert.match(button.textContent, /停止自动标注/);
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

test('同一数据重新载入后的旧状态回执会安全暂停且不会伪装连接失败', async () => {
  const button = {disabled:false,textContent:'',title:'',attributes:{},
    setAttribute(name,value){this.attributes[name]=value},removeAttribute(name){delete this[name]}};
  let responseIndex = 0, rendered = 0, scheduled = null, toasts = [];
  const responses = [
    {status:'idle',result_available:true,result_task_id:'old-task',
      selected_file:'/new.mcap',selected_topic:'camera0',selected_file_activation_generation:18},
    {status:'idle',result_available:true,result_task_id:'old-task',
      selected_file:'/new.mcap',selected_topic:'camera0',selected_file_activation_generation:18},
    {status:'idle',result_available:false,selected_file:'/new.mcap',selected_topic:'camera0',
      selected_file_activation_generation:17},
  ];
  const h = harness({
    loadedDataPath:'/new.mcap',topic:'camera0',loadedFileActivationGeneration:17,
    latestAgentJob:{status:'idle'},
    progressFetching:false,progressFetchPromise:null,runStatusScopeMismatch:null,
    runStatusConnectionMessage:'',runStatusFailureCount:0,runRequestStartedAt:null,
    bootstrapComplete:true,workspaceRestoreState:'ready',
    $: id => id === 'runButton' ? button : {},
    async fetchAgentStatus(){const job=responses[responseIndex++];return {ok:true,async json(){return job}}},
    renderAgentProgress(job){rendered++;h.context.latestAgentJob=job},
    renderRunButton(){button.disabled=h.context.runStatusScopeMismatch!==null;
      button.textContent=button.disabled?'数据已切换 · 请点右上“切换数据”同步':'开始自动标注'},
    scheduleRunStatusPoll(delay){scheduled=delay},pendingRunResult(){return false},
    toast(message){toasts.push(message)},
    markWorkspaceReady(){},
  });
  h.run(section(source('app.js'), 'async function refreshAgentProgress', 'async function ensureAutoLabelReadiness'));
  assert.equal(await h.context.refreshAgentProgress(), false);
  assert.equal(rendered, 0);
  assert.equal(button.disabled, true);
  assert.match(button.textContent, /切换数据/);
  assert.match(button.title, /不会沿用其任务状态/);
  assert.deepEqual(toasts, ['数据已在另一窗口切换或重新载入；本页已安全暂停，请点右上“切换数据”同步']);
  assert.equal(scheduled, 1000);
  assert.equal(h.context.runStatusFailureCount, 0);
  assert.equal(h.context.window.labelAgentRunStatusSnapshot.status, 'idle');
  assert.equal(await h.context.refreshAgentProgress(), false);
  assert.deepEqual(toasts, ['数据已在另一窗口切换或重新载入；本页已安全暂停，请点右上“切换数据”同步']);
  assert.equal(await h.context.refreshAgentProgress(), true);
  assert.equal(rendered, 1);
  assert.equal(h.context.runStatusScopeMismatch, null);
  assert.equal(scheduled, 15000);
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

test('页面已打开后模型掉线会禁用新任务并在恢复时提示一次', async () => {
  const classes = new Set();
  const badge = {lastChild:{textContent:''},title:'',classList:{
    add(...names){for(const name of names)classes.add(name)},
    remove(...names){for(const name of names)classes.delete(name)},
  }};
  const notices = [];
  const replies = [
    {ready:false,sam:{ready:false},vlm:{ready:true}},
    {ready:true,sam:{ready:true},vlm:{ready:true}},
  ];
  const h = harness({
    $: id => id === 'modelHealthBadge' ? badge : null,
    async fetch(){const value=replies.shift();return {ok:true,async json(){return value}}},
    toast(message){notices.push(message)},renderRunButton(){},updatePointPromptCoach(){},
    document:{getElementById(id){return id === 'modelHealthBadge' ? badge : null},dispatchEvent(){}},CustomEvent:function(name,options){return {name,...options}},
    modelHealthState:{ready:true,sam:{ready:true},vlm:{ready:true}},
    modelHealthFailureCount:0,modelHealthNoticeKey:'ready',
  });
  const text = source('app.js');
  h.run(section(text, 'function modelHealthUiAvailable', 'function scheduleRunStatusPoll'));
  assert.equal(await h.context.refreshModelHealth(), false);
  assert.equal(h.context.modelHealthState.ready, false);
  assert.equal(badge.lastChild.textContent, 'SAM3离线');
  assert.deepEqual(notices, ['模型连接异常：SAM3 不可用，Qwen8B 正常；页面正在自动重试']);
  assert.equal(await h.context.refreshModelHealth(), true);
  assert.equal(badge.lastChild.textContent, '服务器模型正常');
  assert.deepEqual(notices, [
    '模型连接异常：SAM3 不可用，Qwen8B 正常；页面正在自动重试',
    '服务器模型已恢复，可以继续标注',
  ]);
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

test('运行前弹窗与正式预检按帧数检查保存空间', () => {
  const index = source('index.html');
  const guide = source('guide.js');
  const app = source('app.js');
  assert.match(index,/data-readiness="storage"/);
  assert.match(guide,/api\/run\/readiness\?count=/);
  assert.match(guide,/保存空间不足，尚未启动推理/);
  assert.match(guide,/storage\.max_safe_frames/);
  assert.match(app,/data\.storage\?\.warning/);
  assert.match(app,/保存空间偏低，建议只运行少量帧/);
});

test('参考帧起稿在低空间或非 JSON 服务异常时明确失败且不伪装为模型结果', () => {
  const source = fs.readFileSync(path.join(__dirname, '../source/web/reference-prep.js'), 'utf8');
  assert.match(source, /response\.text\(\)/);
  assert.match(source, /JSON\.parse\(responseText\)/);
  assert.match(source, /参考帧服务返回异常（HTTP/);
  assert.match(source, /result\.code === 'storage_low'/);
  assert.match(source, /保存空间不足，尚未启动推理/);
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
  assert.match(guide, /document\.addEventListener\('labelagent:workspace-ready', \(\) => \{/);
  assert.match(guide, /guideWorkspaceReady = true;[\s\S]*workspaceChanged\(\)/);
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

test('可见车辆恢复候选在前后端都必须完成类别盲审', () => {
  const app = source('app.js');
  const required = section(app, 'function candidateClassAuditRequired', 'function newCandidateRefinementId');
  assert.match(required, /shape\?\.source==='sam_vlm_visible_vehicle_point_recovery'/);
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
  assert.match(openDraft, /candidateAuditFullSnapshot\(manualDrafts\[key\]\)!==candidateAuditFullSnapshot\(candidate\)\)\{signalBackgroundComplete\('reference_background_refinement_terminal'\);return false\}/);
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

test('文件移除绑定当前工作区列表版本和目标条目且旧回执不卸载新页面', async () => {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      children: [], textContent: '', replaceChildren(...items) { this.children = items; },
      append(...items) { this.children.push(...items); },
    });
    return elements.get(id);
  }
  function node(tag) {
    return {tag,children:[],className:'',textContent:'',title:'',disabled:false,
      append(...items) { this.children.push(...items); }};
  }
  let resolveDelete, clearCount = 0, status = '';
  const calls = [];
  const latestLibrary = {files: [], library_state_sha256: 'b'.repeat(64)};
  const h = harness({
    document: {hidden:false,addEventListener(){},createElement:node},
    $(id) { return element(id); },
    datasetIdentityForPath() { return {kind:'other',badge:'数据',name:'样例'}; },
    updateDatasetIdentity() {},
    fileStatus(message) { status = message; },
    clearLoadedVideo() { clearCount += 1; },
    async loadVideo() {},
    fetch(url, options={}) {
      calls.push({url,options});
      if (options.method === 'DELETE') {
        return new Promise(resolve => { resolveDelete = body => resolve({
          ok:true,status:200,json:async()=>body,
        }); });
      }
      return Promise.resolve({ok:true,status:200,json:async()=>latestLibrary});
    },
  });
  h.run(`let fileManagerBusy=false,fileLibraryStateSha256='',loadedDataPath='/data/a.mcap',
    topic='/camera0',loadedFileActivationGeneration=7,pendingDatasetLoad=null;`);
  h.run(section(source('app.js'), 'function applyFileLibrary', 'function clearLoadedVideo'));
  const file = {id:'file-a',path:'/data/a.mcap',name:'a.mcap',size:1024,
    available:true,loaded:true,entry_sha256:'e'.repeat(64)};
  h.context.initialLibrary = {files:[file],library_state_sha256:'a'.repeat(64)};
  h.run('applyFileLibrary(initialLibrary)');
  const row = element('fileList').children[0];
  const remove = row.children[row.children.length - 1];
  const pending = remove.onclick();
  await h.flush();
  assert.equal(calls[0].url, '/api/files/file-a');
  assert.equal(calls[0].options.method, 'DELETE');
  assert.deepEqual(JSON.parse(calls[0].options.body), {
    expected_path:'/data/a.mcap',expected_topic:'/camera0',
    expected_file_activation_generation:7,
    expected_library_state_sha256:'a'.repeat(64),
    expected_file_entry_sha256:'e'.repeat(64),expected_loaded:true,
  });
  h.run('loadedFileActivationGeneration=8');
  resolveDelete({files:[],library_state_sha256:'c'.repeat(64),unloaded:true,
    removed_library_state_sha256:'a'.repeat(64),
    removed_file_entry_sha256:'e'.repeat(64),removed_file_id:'file-a',
    disk_file_deleted:false,file_activation_generation:7,
    previous_path:'/data/a.mcap',previous_topic:'/camera0',path:'',topic:null});
  await pending;
  await h.flush();
  assert.equal(clearCount, 0);
  assert.match(status, /当前页面已切换|已从列表移除/);
  assert.equal(h.run('fileLibraryStateSha256'), 'b'.repeat(64));
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

test('结果图层默认使用较细边界并保留手动调节', () => {
  const html = source('index.html');
  const compare = source('compare.js');
  assert.match(html, /id="boundaryWidth"[^>]*min="1"[^>]*max="10"[^>]*value="2"/);
  assert.match(html, /compare\.js\?v=68/);
  assert.match(compare, /controls\.boundaryWidth\.value \|\| 2/);
  assert.match(compare, /Math\.max\(2, canvas\.width \/ 900 \* boundary\)/);
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
      queueMicrotask(callback) { Promise.resolve().then(callback); },
      requestAnimationFrame() { return 1; },
      updatePointPromptCoach() {}, refreshShapeList() {}, draw() {},
      toast(message) { notices.push(message); },
      selectedShapes() { return items; },
      api(route, body) {
        calls.push({route, body: JSON.parse(JSON.stringify(body))});
        return new Promise(resolve => { resolveRequest = resolve; });
      },
    });
    h.run(`let segmentBusy=false,loading=false,playing=false,annotationMode=true,pointMode=true,
      promptPoints=[{x:10,y:20,label:1}],promptBox=null,pointPromptNotice='',pointPromptStartedAt=0,
      pointPromptTimer=null,frame=3,topic='/camera',loadedDataPath='file-a',
      loadedFileActivationGeneration=7,
      generatedShapeFrame=null,generatedShapeTopic='',generatedShapeIndex=-1,
      selectedShapeIndex=-1,referenceReviewOriginalOnly=true,pointPromptGeneratedPointCount=0,
      pointPromptGeneratedSignature='',pointPromptQueuedRerun=false,canvasDrawFrame=null,
      pointPromptRequiresRefinement=false,workspaceRestoreState='${restoreState}';`);
    h.run(section(app, 'function pointSetSignature', 'function draw'));
    h.run(generate);
    return {h, elements, calls, items, notices, resolve(value) { resolveRequest(value); }};
  }

  const first = setup();
  const owner = first.h.run('generatePointMask()');
  const duplicate = first.h.run('generatePointMask()');
  assert.equal(first.calls.length, 1);
  first.h.run("$('labelSelect').value='person'");
  first.resolve({path:'file-a',topic:'/camera',frame:3,file_activation_generation:7,points:[[1,1],[2,1],[2,2]],holes:[],parts:[{points:[[1,1],[2,1],[2,2]],holes:[]}],score:.9});
  await Promise.all([owner, duplicate]);
  assert.deepEqual(first.calls[0], {route:'segment_points', body:{frame:3, points:[{x:10,y:20,label:1}],box:null,path:'file-a',topic:'/camera',file_activation_generation:7}});
  assert.equal(first.items.length, 1);
  assert.equal(first.items[0].label, 'pallet');
  assert.equal(first.h.run('pointPromptGeneratedPointCount'), 1);
  assert.equal(first.h.run('segmentBusy'), false);
  assert.match(first.h.run('pointPromptNotice'), /提交时类别“pallet”/);

  const coalesced = setup();
  const initial = coalesced.h.run('generatePointMask()');
  coalesced.h.run("promptPoints.push({x:30,y:40,label:1});promptChangedDuringRequest();promptPoints.push({x:50,y:60,label:0});promptChangedDuringRequest();");
  assert.equal(coalesced.calls.length, 1);
  coalesced.resolve({path:'file-a',topic:'/camera',frame:3,file_activation_generation:7,points:[[1,1],[2,1],[2,2]],holes:[],parts:[{points:[[1,1],[2,1],[2,2]],holes:[]}],score:.9});
  await initial;
  await coalesced.h.flush();
  assert.equal(coalesced.calls.length, 2);
  assert.deepEqual(coalesced.calls[1].body.points, [{x:10,y:20,label:1},{x:30,y:40,label:1},{x:50,y:60,label:0}]);
  coalesced.resolve({path:'file-a',topic:'/camera',frame:3,file_activation_generation:7,points:[[5,5],[8,5],[8,8]],holes:[],parts:[{points:[[5,5],[8,5],[8,8]],holes:[]}],score:.9});
  await coalesced.h.flush();
  assert.equal(coalesced.calls.length, 2);
  assert.equal(coalesced.items.length, 1);
  assert.equal(coalesced.h.run('pointPromptGeneratedPointCount'), 3);

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

  assert.match(app, /pointPromptQueuedRerun=true/);
  assert.match(app, /旧轮廓已丢弃，正在合并最新提示/);
  assert.match(app, /if\(segmentBusy\).*完成后再清空本帧/);
  assert.match(source('index.html'), /app\.js\?v=217/);
});

test('页面启动先恢复最终工作帧再开放点选推理', () => {
  const app = source('app.js');
  assert.match(app, /workspaceRestoreState='booting'/);
  assert.match(app, /window\.labelAgentWorkspaceReady=window\.labelAgentBootstrapReady\.then/);
  assert.match(app, /function sharedRunStatusMatchesCurrentScope/);
  assert.match(app, /workspaceRestoreState='restoring';bootstrapComplete=true;updateDatasetIdentity\(\);updatePointPromptCoach\(\);await window\.labelAgentInitialRunStatusReady;if\(!sharedRunStatusMatchesCurrentScope\(\)\)await refreshAgentProgress\(\);else if\(workspaceRestoreState!=='ready'\)markWorkspaceReady\(\);await workspaceReadyDeferred/);
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
  assert.match(app, /function clearLoadedVideo\(currentGeneration=0\)\{invalidateFrameNavigation\(\)/);
  assert.match(app, /async function activateAnnotation\(view\).*const navigation=beginFrameNavigation\(\)/);
  assert.match(source('index.html'), /app\.js\?v=217/);
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
  assert.match(source('index.html'), /app\.js\?v=217/);
});

test('类别与推理类别保存绑定文件载入代次和类别状态版本', async () => {
  const app = source('app.js'), calls = [], pending = [], applied = [], notices = [];
  const elements = {
    labelsSave: {}, labelsInput: {value:'a,b'},
    autoLabelsSave: {disabled:false},
    autoLabelOptions: {querySelectorAll:()=>[{checked:true,value:'b'}]},
  };
  const h = harness({
    $(id) { return elements[id]; },
    api(route, body) { calls.push({route,body}); return new Promise(resolve=>pending.push(resolve)); },
    applyInferenceLabelState(data) { applied.push(JSON.parse(JSON.stringify(data))); },
    toast(message) { notices.push(message); },
  });
  h.run(`let loadedDataPath='file-a',topic='/camera',loadedFileActivationGeneration=7,
    labelStateSha256='${'a'.repeat(64)}',currentDescription={ready:true};`);
  h.run(section(app, "$('labelsSave').onclick", "$('autoLabelsAll').onclick"));
  h.run(section(app, "$('autoLabelsSave').onclick", 'async function lockReferenceFrame'));

  const stale = h.run("$('labelsSave').onclick()");
  assert.deepEqual(JSON.parse(JSON.stringify(calls[0])), {route:'labels',body:{labels:['a','b'],path:'file-a',topic:'/camera',file_activation_generation:7,expected_label_state_sha256:'a'.repeat(64)}});
  h.run('loadedFileActivationGeneration=8');
  pending[0]({labels:['a','b'],auto_labels:['a'],reference_only_labels:[],label_state_sha256:'b'.repeat(64),path:'file-a',topic:'/camera',file_activation_generation:7});
  await stale;
  assert.equal(applied.length, 0);
  assert.ok(notices.includes('类别保存期间数据文件、相机或载入代次已变化，拒绝更新当前页面'));

  h.run("loadedFileActivationGeneration=7;labelStateSha256='"+'c'.repeat(64)+"'");
  const inference = h.run("$('autoLabelsSave').onclick()");
  assert.equal(calls[1].route, 'auto_labels');
  assert.equal(calls[1].body.expected_label_state_sha256, 'c'.repeat(64));
  pending[1]({labels:['b'],project_labels:['a','b'],auto_labels:['b'],reference_only_labels:['a'],label_state_sha256:'d'.repeat(64),path:'file-a',topic:'/camera',file_activation_generation:7});
  await inference;
  assert.equal(applied.length, 1);
  assert.deepEqual(applied[0].labels, ['a','b']);
  assert.deepEqual(applied[0].auto_labels, ['b']);
  assert.equal(h.run('currentDescription'), null);
  assert.match(source('index.html'), /app\.js\?v=217/);
});

test('模型配置保存绑定工作区载入代次和配置版本', async () => {
  const app = source('app.js'), calls = [], pending = [], notices = [];
  const dialog = {closed:false,close(){this.closed=true}};
  const elements = {
    configSave: {}, llmUrl: {value:'http://localhost/v1'},
    llmModel: {value:'model'}, llmKey: {value:'local'}, config: dialog,
    expertEnabled: {checked:true}, expertProvider: {value:'openai_responses'},
    expertUrl: {value:'https://expert.example/v1'},
    expertModel: {value:'expert-vision'},
    expertKeyEnv: {value:'LABELAGENT_EXPERT_API_KEY'},
    expertTimeout: {value:'120'}, expertShadowEnabled: {checked:true},
    expertRoundThreshold: {value:'2'}, expertScoreThreshold: {value:'3'},
    expertConfidenceThreshold: {value:'65'}, expertEdgeMargin: {value:'3'},
    expertSmallCandidate: {value:'1'}, expertSamFilterThreshold: {value:'3'},
    expertSamMaskIou: {value:'70'}, expertSamAreaChange: {value:'25'},
    expertPriorityFrames: {value:'12, 46, 12'},
  };
  const h = harness({
    $(id) { return elements[id]; },
    api(route, body) { calls.push({route,body}); return new Promise(resolve=>pending.push(resolve)); },
    toast(message) { notices.push(message); },
  });
  h.run(`let loadedDataPath='file-a',topic='/camera',loadedFileActivationGeneration=7,
    configStateSha256='${'a'.repeat(64)}',workbenchConfig={llm_model:'old'},
    currentDescription={ready:true};`);
  h.run(section(app, "$('configSave').onclick", 'let currentDescription'));

  const stale = h.run("$('configSave').onclick()");
  assert.deepEqual(JSON.parse(JSON.stringify(calls[0])), {route:'config',body:{
    llm_url:'http://localhost/v1',llm_model:'model',llm_key:'local',
    llm_expert_enabled:true,llm_expert_provider:'openai_responses',
    llm_expert_url:'https://expert.example/v1',llm_expert_model:'expert-vision',
    llm_expert_key_env:'LABELAGENT_EXPERT_API_KEY',
    llm_expert_timeout_seconds:120,llm_expert_shadow_routing_enabled:true,
    llm_expert_shadow_round_threshold:2,llm_expert_shadow_score_threshold:3,
    llm_expert_shadow_low_confidence_percent:65,
    llm_expert_shadow_edge_margin_percent:3,
    llm_expert_shadow_small_candidate_percent:1,
    llm_expert_shadow_sam_filter_threshold:3,
    llm_expert_shadow_sam_mask_iou_percent:70,
    llm_expert_shadow_sam_area_change_percent:25,
    llm_expert_shadow_priority_frames:[12,46],
    expected_path:'file-a',expected_topic:'/camera',
    expected_file_activation_generation:7,expected_config_state_sha256:'a'.repeat(64)}});
  h.run('loadedFileActivationGeneration=8');
  pending[0]({saved:true,path:'file-a',topic:'/camera',file_activation_generation:7,
    previous_config_state_sha256:'a'.repeat(64),config_state_sha256:'b'.repeat(64),
    config:{llm_model:'model'}});
  await stale;
  assert.equal(h.run('configStateSha256'), 'a'.repeat(64));
  assert.equal(dialog.closed, false);
  assert.ok(notices.includes('配置保存期间数据文件、相机、载入代次或配置版本已变化，拒绝更新当前页面'));

  h.run("loadedFileActivationGeneration=7;configStateSha256='"+'c'.repeat(64)+"'");
  const current = h.run("$('configSave').onclick()");
  pending[1]({saved:true,path:'file-a',topic:'/camera',file_activation_generation:7,
    previous_config_state_sha256:'c'.repeat(64),config_state_sha256:'d'.repeat(64),
    config:{llm_model:'model'}});
  await current;
  assert.equal(h.run('configStateSha256'), 'd'.repeat(64));
  assert.equal(h.run('workbenchConfig.llm_model'), 'model');
  assert.equal(h.run('currentDescription'), null);
  assert.equal(dialog.closed, true);
});

test('专家模型配置界面不接受原始密钥且明确影子模式不上传', () => {
  const html = source('index.html'), app = source('app.js');
  assert.match(html, /id="expertKeyEnv"/);
  assert.doesNotMatch(html, /id="expertKey"/);
  assert.match(html, /只记录困难帧分流建议（不上传图片）/);
  assert.match(app, /llm_expert_key_env:\$\('expertKeyEnv'\)\.value/);
  assert.doesNotMatch(app, /llm_expert_key:\$\('expert/);
  assert.match(app, /请勿在工作台粘贴密钥/);
  assert.match(html, /综合分数阈值/);
  assert.match(html, /人工重点复核帧/);
});

test('清除机器结果绑定文件代次及候选和过程记录哈希并以服务器读回为准', async () => {
  const inspection = source('inspection.js');
  function setup() {
    const notices = [], requests = [], pending = [], readbacks = [], applied = [];
    const button = {disabled:false,textContent:'清除本帧候选'};
    const h = harness({
      $(id) { return id==='clearAutoResult'?button:{}; },
      fetch(url, options) { requests.push({url,options}); return new Promise(resolve=>pending.push(resolve)); },
      async readCompactFrameStateSnapshot(...args) { readbacks.push(args); return {compact_geometry:true,path:'file-a',topic:'/camera',frame:3,file_activation_generation:7,candidates:{},agent_runs:{},candidate_review_bindings:{}}; },
      applyCompactFrameState(data, ...scope) { applied.push({data,scope}); },
      resetPrompts() {}, refreshShapeList() {}, draw() {}, updateReferenceStatus() {},
      renderInspection() {}, async refreshInspection() {}, async refreshAgentProgress() {},
      toast(message) { notices.push(message); },
    });
    h.run(`let frame=3,topic='/camera',loadedDataPath='file-a',loadedFileActivationGeneration=7,
      candidates={'3':[{}]},agentRuns={'3':{}},autoDrafts={'3':[]},candidateReviewBindings={'3':{}},candidateCountIndex={'3':1},draft=[],
      inspectionData={clear_auto_result_binding:{path:'file-a',topic:'/camera',frame:3,file_activation_generation:7,candidate_artifact_sha256:'${'a'.repeat(64)}',agent_run_sha256:'${'b'.repeat(64)}'},trace:[],steps:[],frames:[3]};`);
    h.run(section(inspection, "$('clearAutoResult').onclick", "window.setWorkspaceMode('manual',false)"));
    return {h,button,notices,requests,pending,readbacks,applied};
  }

  const current = setup(), operation = current.h.run("$('clearAutoResult').onclick()");
  const submitted = JSON.parse(current.requests[0].options.body);
  assert.equal(submitted.path, 'file-a');
  assert.equal(submitted.file_activation_generation, 7);
  assert.equal(submitted.expected_candidate_artifact_sha256, 'a'.repeat(64));
  assert.equal(submitted.expected_agent_run_sha256, 'b'.repeat(64));
  current.pending[0]({ok:true,json:async()=>({path:'file-a',topic:'/camera',frame:3,file_activation_generation:7,removed_candidate_artifact_sha256:'a'.repeat(64),removed_agent_run_sha256:'b'.repeat(64),truth_preserved:true})});
  await operation;
  assert.equal(current.readbacks.length, 1);
  assert.equal(current.applied.length, 1);
  assert.ok(current.notices.includes('帧 3 的自动结果已清除，人工真值保留'));

  const stale = setup(), old = stale.h.run("$('clearAutoResult').onclick()");
  stale.h.run('loadedFileActivationGeneration=8');
  stale.pending[0]({ok:true,json:async()=>({path:'file-a',topic:'/camera',frame:3,file_activation_generation:7,removed_candidate_artifact_sha256:'a'.repeat(64),removed_agent_run_sha256:'b'.repeat(64),truth_preserved:false})});
  await old;
  assert.equal(stale.readbacks.length, 0);
  assert.equal(stale.applied.length, 0);
  assert.ok(stale.notices.some(message=>message.includes('载入代次')));
  assert.match(source('index.html'), /inspection\.js\?v=109/);
});

test('栈板边界优化只改当前强绑定机器候选并完成服务器读回', async () => {
  const inspection = source('inspection.js'), notices = [], requests = [], pending = [], applied = [];
  const button = {disabled:false,textContent:'优化本帧栈板边界'};
  const h = harness({
    $(id) { return id==='cleanupPalletGeometry'?button:{}; },
    fetch(url, options) { requests.push({url,options}); return new Promise(resolve=>pending.push(resolve)); },
    async readCompactFrameStateSnapshot() { return {compact_geometry:true,path:'file-a',topic:'/camera',frame:3,file_activation_generation:7,candidates:{'3':[{}]},agent_runs:{'3':{}},candidate_review_bindings:{}}; },
    applyCompactFrameState(data, ...scope) { applied.push({data,scope}); },
    resetPrompts() {}, refreshShapeList() {}, draw() {}, updateReferenceStatus() {},
    async refreshInspection() {}, async refreshAgentProgress() {}, renderInspection() {},
    toast(message) { notices.push(message); },
  });
  h.run(`let frame=3,topic='/camera',loadedDataPath='file-a',loadedFileActivationGeneration=7,draft=[],inspectionVersion='',inspectionData={clear_auto_result_binding:{path:'file-a',topic:'/camera',frame:3,file_activation_generation:7,candidate_artifact_sha256:'${'a'.repeat(64)}',agent_run_sha256:'${'b'.repeat(64)}'}};`);
  h.run(section(inspection, "$('cleanupPalletGeometry').onclick", "window.setWorkspaceMode('manual',false)"));
  const operation = h.run("$('cleanupPalletGeometry').onclick()");
  assert.equal(requests[0].url, '/api/candidates/pallet_cleanup/3');
  const submitted = JSON.parse(requests[0].options.body);
  assert.equal(submitted.path, 'file-a');
  assert.equal(submitted.topic, '/camera');
  assert.equal(submitted.file_activation_generation, 7);
  assert.equal(submitted.expected_candidate_artifact_sha256, 'a'.repeat(64));
  assert.equal(submitted.expected_agent_run_sha256, 'b'.repeat(64));
  pending[0]({ok:true,json:async()=>({path:'file-a',topic:'/camera',frame:3,file_activation_generation:7,prior_candidate_artifact_sha256:'a'.repeat(64),prior_agent_run_sha256:'b'.repeat(64),candidate_only:true,human_truth:false,approved_for_training:false,approved_for_deployment:false,changed:true,adjusted_candidate_count:2})});
  await operation;
  assert.equal(applied.length, 1);
  assert.ok(notices.some(message=>message.includes('已优化 2 个栈板候选边界')));
  assert.equal(button.textContent, '优化本帧栈板边界');
  assert.match(source('index.html'), /id="cleanupPalletGeometry"/);
  assert.match(source('index.html'), /inspection\.js\?v=109/);
});

test('Local VLM 短暂抖动会自动重试并在弹窗打开时自恢复', () => {
  const prep = source('reference-prep.js');
  assert.match(prep, /for \(let attempt = 0; attempt < 3; attempt \+= 1\)/);
  assert.match(prep, /setTimeout\(resolve, attempt \? 750 : 250\)/);
  assert.match(prep, /readModelReadiness\(generation\)\.catch\(\(\) => scheduleModelReadinessRetry\(\)\)/);
  assert.match(prep, /dialog\.addEventListener\('close', clearModelReadinessRetry\)/);
});

test('专业简约模式默认收起向导且保留用户展开选择', () => {
  const guide = source('guide.js'), html = source('index.html'), theme = source('homepage-theme.css');
  assert.match(guide, /function recalled\(key, fallback = false\)/);
  assert.match(guide, /setGuideCompact\(recalled\('labelagentGuideCompactV165', true\), false\)/);
  assert.match(html, /连续推理 · N 帧/);
  assert.match(html, /homepage-theme\.css\?v=5/);
  assert.match(theme, /Professional compact layer/);
  assert.match(theme, /\.tutorialButton[\s\S]*display: none/);
});
