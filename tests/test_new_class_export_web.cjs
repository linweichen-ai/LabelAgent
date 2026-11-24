const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../source/web/app.js'), 'utf8');

function context(payload, ok = true) {
  const start = source.indexOf('async function downloadNewClassObservations');
  const end = source.indexOf('window.downloadNewClassObservations', start);
  assert.ok(start >= 0 && end > start);
  const clicks = [], messages = [], revoked = [];
  const scope = vm.createContext({
    Blob,
    URL: {createObjectURL: () => 'blob:new-class', revokeObjectURL: value => revoked.push(value)},
    document: {createElement: () => ({click() { clicks.push(this.download); }})},
    fetch: async () => ({ok, json: async () => payload}),
    setTimeout: callback => callback(),
    toast: message => messages.push(message),
  });
  vm.runInContext(source.slice(start, end), scope);
  return {scope, clicks, messages, revoked};
}

test('疑似新类观察下载保持只读文件名并恢复按钮', async () => {
  const item = context({summary: {observation_count: 2}}), button = {disabled: false};
  await item.scope.downloadNewClassObservations(button);
  assert.deepEqual(item.clicks, ['labelagent-new-class-observations.json']);
  assert.equal(item.revoked[0], 'blob:new-class');
  assert.match(item.messages[0], /已导出 2 帧/);
  assert.equal(button.disabled, false);
});

test('空观察不会下载并给出明确提示', async () => {
  const item = context({summary: {observation_count: 0}}), button = {disabled: false};
  await item.scope.downloadNewClassObservations(button);
  assert.deepEqual(item.clicks, []);
  assert.match(item.messages[0], /没有可导出/);
  assert.equal(button.disabled, false);
});
