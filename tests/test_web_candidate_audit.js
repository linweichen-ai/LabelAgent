const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const source = fs.readFileSync(path.join(__dirname, '../source/web/app.js'), 'utf8');

// 在独立上下文运行生产函数；不加载页面、不访问服务或修改项目 state。
function context() {
  const start = source.indexOf('const candidateAuditSnapshots=');
  const end = source.indexOf('function referenceReviewScope', start);
  const scope = vm.createContext({frame: 3, topic: '/camera', loadedDataPath: 'file-a',
    loadedFileActivationGeneration: 7,
    readCompactFrameStateSnapshot: async target => {
      const response = await scope.fetch(`/api/frame_state/${target}`);
      return response.json();
    },
    agentRuns: {'3': {candidate_class_audits: [{candidate_id: 1}], review_queue: [
      {source: 'candidate_class_audit'}, {source: 'geometry'}]}},
    referenceReviewedShapeKeys: new Set(['checked']), referenceOriginalSweepVisited: true,
    referenceOpenSetReviewed: true, referenceDismissedMissingLabels: new Set(['person']),
    referenceFinalizeNotice: ''});
  vm.runInContext(source.slice(start, end), scope);
  return scope;
}
const candidates = () => [{id: 1, label: 'ego', score: .8, points: [[0, 0], [9, 0], [0, 9]]}];

test('同一快照不清除显式人工检查', () => {
  const ctx = context(); ctx.items = candidates();
  vm.runInContext('candidateAuditSnapshots.set(candidateAuditScope(), {snapshot:candidateAuditSnapshot(items), generation:0});', ctx);
  assert.equal(vm.runInContext('invalidateChangedCandidateAudits(items)', ctx), false);
  assert.equal(ctx.referenceReviewedShapeKeys.size, 1);
});

for (const [name, mutate] of [
  ['修改类别', items => items[0].label = 'person'],
  ['修改几何', items => items[0].points[0][0] = 1],
  ['新增候选', items => items.push({...items[0], id: 2})],
  ['删除候选', items => items.pop()],
]) test(`${name}使旧盲审及人工逐项进度失效`, () => {
  const ctx = context(); ctx.items = candidates();
  vm.runInContext('candidateAuditSnapshots.set(candidateAuditScope(), {snapshot:candidateAuditSnapshot(items), generation:0});', ctx);
  mutate(ctx.items);
  assert.equal(vm.runInContext('invalidateChangedCandidateAudits(items)', ctx), true);
  assert.equal(ctx.agentRuns['3'].candidate_class_audits.length, 0);
  assert.equal(ctx.agentRuns['3'].review_queue.length, 1);
  assert.equal(ctx.referenceReviewedShapeKeys.size, 0);
  assert.equal(ctx.referenceOpenSetReviewed, false);
});

test('文件、话题和帧共同隔离审核', () => {
  const ctx = context();
  const original = vm.runInContext('candidateAuditScope()', ctx);
  for (const expression of ["candidateAuditScope(4)", "candidateAuditScope(3,'/other')", "candidateAuditScope(3,'/camera','file-b')"])
    assert.notEqual(vm.runInContext(expression, ctx), original);
});

test('箱体形和极细长栈板进入隔离复核而普通侧视栈板保持快速路径', () => {
  const ctx = context();
  ctx.solid = {id: 8, label: 'pallet', points: [[0,0],[149,0],[149,99],[0,99]]};
  ctx.normal = {id: 9, label: 'pallet', points: [[0,0],[239,0],[239,59],[0,59]]};
  ctx.extremeThin = {id: 11, label: 'pallet', points: [[0,0],[199,0],[199,11],[0,11]]};
  ctx.malformed = {id: 10, label: 'pallet', points: [[0,0],[1,1]]};
  assert.equal(vm.runInContext('candidateClassAuditRequired(solid)', ctx), true);
  assert.equal(vm.runInContext('candidateClassAuditRequired(normal)', ctx), false);
  assert.equal(vm.runInContext('candidateClassAuditRequired(extremeThin)', ctx), true);
  assert.equal(vm.runInContext('candidateClassAuditRequired(malformed)', ctx), true);
});

for (const scenario of ['success', 'failure', 'stale', 'readback', 'fingerprint', 'image', 'cache-hit']) test(`异步复审 ${scenario}`, async () => {
  const ctx = context();
  ctx.candidates = {'3': candidates()}; ctx.candidateReviewBindings = {};
  const clone = value => JSON.parse(JSON.stringify(value));
  const binding = {candidate_artifact_sha256: 'artifact', candidates: [{candidate_id: 1, candidate_fingerprint_sha256: 'fingerprint', candidate_fingerprint: 'fingerprint', candidate_sha256: 'full'}]};
  const version = {schema_version:'labelagent.candidate_class_audit_version.v1',config_sha256:'c'.repeat(64),protocol_sha256:'d'.repeat(64),labels_sha256:'e'.repeat(64)};
  const audit = {candidate_id: 1, bound_current_label:'ego', candidate_sha256: 'fingerprint', source_image_sha256: '01', candidate_artifact_sha256: 'artifact', schema_version:'labelagent.blind_candidate_class_audit.v1',audit_version:version,prompt_sha256:'a'.repeat(64),audit_sha256:'b'.repeat(64),human_truth:false,decision:{classification:'known_label'},route:'candidate_preserved'};
  const agent = {candidate_class_audits: [audit],candidate_class_audit_version:version,candidate_class_audit_status:'complete'};
  const state = {path:'file-a', topic:'/camera', candidates:clone(ctx.candidates), candidate_review_bindings:{'3':binding}, candidate_class_audit_version:version,agent_runs:{'3':{}}};
  let reads = 0;
  ctx.crypto = {subtle:{digest: async () => Uint8Array.from([1]).buffer}};
  const expectedAgent=clone(agent);expectedAgent.candidate_artifact_sha256='artifact';expectedAgent.input_sha256='01';
  if(scenario==='cache-hit')state.agent_runs['3']=clone(expectedAgent);
  else state.agent_runs['3']={candidate_class_audits:[]};
  let audits=0;
  ctx.fetch = async url => url.startsWith('/api/frame/') ? {ok:true,arrayBuffer:async()=>new ArrayBuffer(0)} : {ok:true,json:async()=>{reads++;const value=clone(state);if(scenario==='readback'&&reads>1)value.candidates['3'][0].label='person';if(scenario==='cache-hit'&&reads>1)value.agent_runs['3']=clone(value.agent_runs['3']);return value}};
  ctx.api = async (route, body) => {
    audits++;assert.equal(route,'reference_candidates/audit');assert.equal(body.expected_frame_sha256,'01');assert.equal(body.expected_candidate_artifact_sha256,'artifact');assert.equal(body.candidate_id,undefined);
    if(scenario==='failure')throw Error('模型失败');
    if(scenario==='stale')ctx.candidates['3'][0].label='person';
    const result={frame:3,topic:'/camera',candidates:clone(state.candidates['3']),agent:clone(expectedAgent),binding:clone(binding),input_sha256:'01',candidate_artifact_sha256:'artifact',candidate_only:true,human_truth:false};
    if(scenario==='fingerprint')result.agent.candidate_class_audits[0].candidate_sha256='wrong';
    if(scenario==='image')result.input_sha256='wrong';
    state.agent_runs['3']=clone(result.agent);
    return result;
  };
  const promise=vm.runInContext("auditReferenceCandidateSnapshot(3,candidates['3'])",ctx);
  if(scenario==='success') {const result=await promise;assert.equal(audits,1);assert.equal(result.candidate_class_audits.length,1);assert.equal(ctx.referenceReviewedShapeKeys.size,1)}
  else if(scenario==='cache-hit'){const result=await promise;assert.equal(audits,0);assert.equal(result.candidate_class_audits.length,1)}
  else {await assert.rejects(promise);assert.equal(audits,scenario==='failure'?1:1)}
});

for (const scenario of ['命中','箱体栈板','多候选','多候选一坏','疑似','冲突','artifact','image','fingerprint','类别','重复','缺失','失败','过期','配置','协议','缺版本','run-artifact','run-image','版本schema','版本hash','缺决策','异常决策','重复ID','非整数ID','原图读取期间候选变化','读回审核变化']) test(`严格缓存复用 ${scenario}`,async()=>{
 const ctx=context(),clone=value=>JSON.parse(JSON.stringify(value)),version={schema_version:'labelagent.candidate_class_audit_version.v1',config_sha256:'c'.repeat(64),protocol_sha256:'d'.repeat(64),labels_sha256:'e'.repeat(64)};
 ctx.candidates={'3':candidates()};ctx.candidateReviewBindings={};
 if(scenario==='箱体栈板')ctx.candidates['3']=[{id:8,label:'pallet',points:[[0,0],[149,0],[149,99],[0,99]]}];
 if(['多候选','多候选一坏','重复ID'].includes(scenario))ctx.candidates['3'].push({...candidates()[0],id:scenario==='重复ID'?1:2,label:'person'},{...candidates()[0],id:3,label:'floor'});
 if(scenario==='非整数ID')ctx.candidates['3'][0].id=1.5;
 const binding={candidate_artifact_sha256:'artifact',candidates:ctx.candidates['3'].map(shape=>({candidate_id:shape.id,candidate_fingerprint_sha256:'fp'+shape.id,candidate_fingerprint:'fp'+shape.id,candidate_sha256:'full'+shape.id}))};
 const audits=ctx.candidates['3'].filter(shape=>shape.label!=='floor').map(shape=>({candidate_id:shape.id,bound_current_label:shape.label,candidate_sha256:'fp'+shape.id,candidate_artifact_sha256:'artifact',source_image_sha256:'01',audit_version:clone(version),schema_version:'labelagent.blind_candidate_class_audit.v1',prompt_sha256:'a'.repeat(64),audit_sha256:'b'.repeat(64),human_truth:false,decision:{classification:'known_label'},route:'candidate_preserved'}));
 const valid={candidate_class_audits:audits,candidate_class_audit_version:clone(version),candidate_class_audit_status:'complete',candidate_artifact_sha256:'artifact',input_sha256:'01'},cached=clone(valid),row=cached.candidate_class_audits[0];
 if(scenario==='artifact')row.candidate_artifact_sha256='wrong';if(scenario==='image')row.source_image_sha256='wrong';if(scenario==='fingerprint')row.candidate_sha256='wrong';if(scenario==='类别')row.bound_current_label='person';if(scenario==='重复'||scenario==='重复ID')cached.candidate_class_audits.push(clone(row));if(scenario==='缺失')cached.candidate_class_audits=[];if(scenario==='失败')row.error='模型失败';if(scenario==='过期')cached.candidate_class_audit_status='stale';if(scenario==='配置')cached.candidate_class_audit_version.config_sha256='changed';if(scenario==='协议')row.audit_version.protocol_sha256='changed';if(scenario==='缺版本')delete cached.candidate_class_audit_version;if(scenario==='run-artifact')cached.candidate_artifact_sha256='wrong';if(scenario==='run-image')cached.input_sha256='wrong';if(scenario==='版本schema')cached.candidate_class_audit_version.schema_version='v0';if(scenario==='版本hash')cached.candidate_class_audit_version.protocol_sha256='bad';if(scenario==='缺决策')delete row.decision;if(scenario==='异常决策')row.decision={classification:'error'};if(scenario==='多候选一坏')cached.candidate_class_audits[1].candidate_sha256='wrong';
 if(['疑似','冲突'].includes(scenario)){row.route='human_review_required';row.decision={classification:scenario==='疑似'?'uncertain':'conflict'}}
 const state={path:'file-a',topic:'/camera',candidates:clone(ctx.candidates),candidate_review_bindings:{'3':binding},candidate_class_audit_version:version,agent_runs:{'3':cached}};let posts=0,reads=0;
 ctx.crypto={subtle:{digest:async()=>Uint8Array.from([1]).buffer}};
 ctx.fetch=async url=>url.startsWith('/api/frame/')?{ok:true,arrayBuffer:async()=>{if(scenario==='原图读取期间候选变化')ctx.candidates['3'][0].label='person';return new ArrayBuffer(0)}}:{ok:true,json:async()=>{reads++;const value=clone(state);if(scenario==='读回审核变化'&&reads>1)value.agent_runs['3'].candidate_class_audits[0].route='human_review_required';return value}};
 ctx.api=async()=>{posts++;state.agent_runs['3']=clone(valid);return {frame:3,topic:'/camera',candidate_only:true,human_truth:false,candidates:clone(state.candidates['3']),binding:clone(binding),agent:clone(valid),input_sha256:'01',candidate_artifact_sha256:'artifact'}};
 const promise=vm.runInContext("auditReferenceCandidateSnapshot(3,candidates['3'])",ctx);
 if(['原图读取期间候选变化','读回审核变化','重复ID','非整数ID'].includes(scenario)){await assert.rejects(promise);assert.equal(posts,['重复ID','非整数ID'].includes(scenario)?1:0)}else{const result=await promise;assert.equal(posts,['命中','箱体栈板','多候选','疑似','冲突'].includes(scenario)?0:1);assert.equal(result.candidate_class_audits.length,audits.length);assert.equal(ctx.referenceReviewedShapeKeys.size,1);assert.equal(ctx.referenceOpenSetReviewed,true)}
});

for (const scenario of ['success','frame','file','topic','generation','truth','hash','receipt','readback-frame','readback-file','readback-hash','readback-hash-mismatch','readback-unverified','readback-match-false','readback-match-missing','receipt-hash-missing']) test(`保存真值延迟响应 ${scenario}`, async () => {
  const ctx=context(),items=candidates(),snapshot=JSON.stringify(items);
  Object.assign(ctx,{annotationMode:true,loading:false,playing:false,shapes:{},manualDrafts:{'3':items,'9':[{label:'保留'}]},autoDrafts:{},candidates:{'3':items,'9':[{label:'保留'}]},truthReviewStatus:{},truthFrameIndex:new Set(),candidateCountIndex:{'3':1,'9':1},referencePreparationIntent:null,generatedShapeFrame:null,generatedShapeTopic:'',selectedShapes:()=>items,refreshShapeList(){},updateReferenceStatus(){},draw(){},toast(){}});
  const result={saved:1,review:{file:'file-a',topic:'/camera',frame:3,truth_sha256:'truth-hash'},readback:{verified:true,truth_sha256_matches:true}};
  if(scenario==='readback-hash')result.readback.truth_sha256='truth-hash';if(scenario==='readback-hash-mismatch')result.readback.truth_sha256='wrong';if(scenario==='readback-match-false')result.readback.truth_sha256_matches=false;if(scenario==='readback-match-missing')delete result.readback.truth_sha256_matches;if(scenario==='readback-unverified')result.readback.verified=false;if(scenario==='receipt-hash-missing')delete result.review.truth_sha256;
  ctx.window={workspaceMode:'manual',commitHumanReviewedTruth:async request=>{assert.notEqual(request.shapes,items);assert.equal(request.fileActivationGeneration,7);if(scenario==='frame')ctx.frame=9;if(scenario==='file')ctx.loadedDataPath='file-b';if(scenario==='topic')ctx.topic='/other';if(scenario==='generation')ctx.loadedFileActivationGeneration=8;if(scenario==='receipt')result.review.frame=9;return result}};
  ctx.fetch=async()=>({ok:true,json:async()=>{const value={path:'file-a',topic:'/camera',file_activation_generation:7,truth:{'3':JSON.parse(snapshot)},truth_review_status:{'3':{reusable_as_reference:true,truth_sha256:scenario==='hash'?'wrong':'truth-hash'}}};if(scenario==='truth')value.truth['3'][0].label='wrong';if(scenario==='readback-frame')ctx.frame=9;if(scenario==='readback-file')ctx.loadedDataPath='file-b';return value}});
  const start=source.indexOf('async function saveCurrentTruth('),end=source.indexOf("$('truthSave').onclick",start);vm.runInContext(source.slice(start,end),ctx);
  if(scenario==='success'||scenario==='readback-hash'){await vm.runInContext('saveCurrentTruth()',ctx);assert.equal(JSON.stringify(ctx.shapes['3']),snapshot);assert.equal(ctx.manualDrafts['3'],undefined)}
  else {await assert.rejects(vm.runInContext('saveCurrentTruth()',ctx));assert.equal(ctx.shapes['3'],undefined);assert.equal(ctx.manualDrafts['3'],items);assert.equal(ctx.candidates['9'][0].label,'保留')}
});

for(const scenario of ['success','frame','file','generation','content','hash']) test(`参考帧锁定读回 ${scenario}`,async()=>{
 const ctx=context(),truth=candidates();Object.assign(ctx,{shapes:{'3':truth},truthReviewStatus:{'3':{truth_sha256:'truth-hash'}},window:{},updateReferenceStatus(){}});
 ctx.api=async(route,body)=>{assert.equal(body.expected_file_activation_generation,7);if(scenario==='frame')ctx.frame=9;if(scenario==='file')ctx.loadedDataPath='other';if(scenario==='generation')ctx.loadedFileActivationGeneration=8};
 ctx.fetch=async()=>({ok:true,json:async()=>({path:'file-a',topic:'/camera',file_activation_generation:7,reference_frames:{'/camera':[3]},truth:{'3':scenario==='content'?[]:truth},truth_review_status:{'3':{truth_sha256:scenario==='hash'?'wrong':'truth-hash',reusable_as_reference:true}}})});
 const start=source.indexOf('async function lockReferenceFrame('),end=source.indexOf("$('referenceButton').onclick",start);vm.runInContext(source.slice(start,end),ctx);
 if(scenario==='success')await vm.runInContext('lockReferenceFrame()',ctx);else await assert.rejects(vm.runInContext('lockReferenceFrame()',ctx));
});

test('直接调用finalize不能绕过逐个检查门禁', async () => {
  const start = source.indexOf('async function finalizeReferencePreparation()');
  const end = source.indexOf("$('referenceFinalizePrimary').onclick", start);
  let submissions = 0;
  const ctx = vm.createContext({referencePreparationIntent: {topic: '/camera', frame: 3},
    topic: '/camera', frame: 3, referenceFinalizeBusy: false,
    updateReferenceFinalizeGuide() {}, $() {return {disabled: true}}, toast() {},
    saveCurrentTruth() {submissions++}, truthIsVerified() {return false}});
  vm.runInContext(source.slice(start, end), ctx);
  await vm.runInContext('finalizeReferencePreparation()', ctx);
  assert.equal(submissions, 0);
});
