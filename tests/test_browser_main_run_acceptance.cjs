const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

function toolSource() {
  return fs.readFileSync(path.join(__dirname, '../tools/browser_main_run_acceptance.cjs'), 'utf8');
}

test('主按钮浏览器验收等待换帧稳定并容忍启动时窗口恢复竞态', () => {
  const source = toolSource();
  assert.match(source, /for \(let attempt = 0; attempt < 3 && !frameSelected; attempt \+= 1\)/);
  assert.match(source, /new URL\(image\.src\)\.pathname === `\/api\/frame\/\$\{expected\}`/);
  assert.match(source, /await page\.waitForTimeout\(300\)/);
  assert.match(source, /assert\.equal\(Number\(await page\.locator\("#frameNumber"\)\.inputValue\(\)\), frame\)/);
});

test('主按钮浏览器验收在预检拒绝时立即报告而不是等待五分钟', () => {
  const source = toolSource();
  const runRequestBlock = source.slice(
    source.indexOf('const runRequestPromise ='),
    source.indexOf('const preflightFailurePromise ='),
  );
  assert.match(source, /const preflightFailurePromise = page\.waitForFunction/);
  assert.match(runRequestBlock, /item\.method\(\) === "POST"/);
  assert.doesNotMatch(runRequestBlock, /item\.request\(\)\.method\(\)/);
  assert.match(source, /timeout: 60000/);
  assert.match(source, /Promise\.race\(\[runRequestPromise, preflightFailurePromise\]\)/);
  assert.match(source, /await runOutcome\.runRequest\.response\(\)/);
  assert.match(source, /automatic-label preflight rejected the run/);
});
