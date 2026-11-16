const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');
const source = fs.readFileSync(path.join(__dirname, '../source/web/app.js'), 'utf8');
function declaration(name) {
  const start = source.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `未找到真实前端函数 ${name}`);
  const next = source.indexOf('\nfunction ', start + 1);
  return source.slice(start, next < 0 ? source.length : next);
}
function context() {
  const sandbox = vm.createContext({});
  vm.runInContext(`let frame=3,topic='camera',loadedDataPath='one.mcap',agentRuns={},referenceReviewedShapeKeys=new Set(['checked']),referenceOriginalSweepVisited=true,referenceOpenSetReviewed=true,referenceDismissedMissingLabels=new Set(['person']),referenceFinalizeNotice='';const candidateAuditSnapshots=new Map();`, sandbox);
  for (const name of ['referenceReviewKeys', 'candidateAuditScope', 'candidateAuditSnapshot', 'invalidateChangedCandidateAudits', 'referenceReviewScope']) vm.runInContext(declaration(name), sandbox);
  return sandbox;
}
const candidate = () => ({id: 1, label: 'person', points: [[0, 0], [10, 0], [0, 10]]});
test('人工检查和机器审核作用域均隔离数据文件、topic和frame', () => {
  const sandbox = context();
  sandbox.items = [candidate()];
  const scopes = vm.runInContext('[referenceReviewScope(items),candidateAuditScope()]', sandbox);
  for (const change of ["loadedDataPath='two.mcap'", "topic='other'", 'frame=9']) {
    vm.runInContext(change, sandbox);
    const changed = vm.runInContext('[referenceReviewScope(items),candidateAuditScope()]', sandbox);
    assert.notEqual(changed[0], scopes[0]);
    assert.notEqual(changed[1], scopes[1]);
  }
});
for (const [name, mutate] of [
  ['新增', items => items.push({...candidate(), id: 2})],
  ['删除', items => items.pop()],
  ['改类', items => {items[0].label = 'ego';}],
  ['改几何', items => {items[0].points[0][0] = 1;}],
]) test(`${name}候选使整集合旧审核失效并重置人工进度`, () => {
  const sandbox = context();
  sandbox.items = [candidate()];
  vm.runInContext("candidateAuditSnapshots.set(candidateAuditScope(),{snapshot:candidateAuditSnapshot(items),generation:0,stale:false});agentRuns['3']={candidate_class_audits:[{candidate_id:1}],review_queue:[{source:'candidate_class_audit'},{source:'other'}]};", sandbox);
  mutate(sandbox.items);
  assert.equal(vm.runInContext('invalidateChangedCandidateAudits(items)', sandbox), true);
  assert.equal(vm.runInContext("agentRuns['3'].candidate_class_audits.length", sandbox), 0);
  assert.equal(vm.runInContext("agentRuns['3'].review_queue.length", sandbox), 1);
  assert.equal(vm.runInContext("agentRuns['3'].candidate_class_audit_status", sandbox), 'stale');
  assert.equal(vm.runInContext('referenceReviewedShapeKeys.size', sandbox), 0);
  assert.equal(vm.runInContext('referenceOpenSetReviewed', sandbox), false);
  assert.equal(vm.runInContext('candidateAuditSnapshots.get(candidateAuditScope()).generation', sandbox), 1);
  assert.equal(vm.runInContext('invalidateChangedCandidateAudits(items)', sandbox), false);
});
test('保持候选内容只选择轮廓不会撤销人工进度', () => {
  const sandbox = context();
  sandbox.items = [candidate()];
  vm.runInContext('candidateAuditSnapshots.set(candidateAuditScope(),{snapshot:candidateAuditSnapshot(items),generation:0,stale:false})', sandbox);
  assert.equal(vm.runInContext('invalidateChangedCandidateAudits(items)', sandbox), false);
  assert.equal(vm.runInContext('referenceReviewedShapeKeys.size', sandbox), 1);
});
