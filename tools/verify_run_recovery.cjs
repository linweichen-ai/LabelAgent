// Read-only browser acceptance for a persisted run result after service restart.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
let chromium;
try { ({chromium} = require('playwright-core')); }
catch { ({chromium} = require('playwright')); }

const base = process.argv[2] || 'http://127.0.0.1:8765';
const output = process.argv[3];
if (!output) throw Error('Usage: node verify_run_recovery.cjs BASE_URL OUTPUT_DIRECTORY');

(async () => {
  fs.mkdirSync(output, {recursive:true});
  const browser = await chromium.launch({
    executablePath:'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    headless:true,
  });
  const report = {page_errors:[],console_errors:[],failed_requests:[],responses:[]};
  try {
    const page = await browser.newPage({viewport:{width:1600,height:1000}});
    page.on('pageerror', error => report.page_errors.push(error.message));
    page.on('console', message => {
      if (message.type() === 'error') report.console_errors.push(message.text());
    });
    page.on('requestfailed', request => report.failed_requests.push({
      url:request.url(),error:request.failure()?.errorText,
    }));
    page.on('response', response => {
      const pathname = new URL(response.url()).pathname;
      if (['/api/bootstrap','/api/run/status','/api/run/result','/api/state'].includes(pathname)) {
        report.responses.push({pathname,status:response.status()});
      }
    });
    const started=Date.now();
    await page.goto(base,{waitUntil:'networkidle'});
    await page.waitForFunction(() => window.labelAgentBootstrapReady && bootstrapComplete);
    await page.waitForFunction(() => annotationMode && !loading && frame === 30);
    report.recovery_ms=Date.now()-started;
    report.recovered=await page.evaluate(() => ({
      frame,topic,annotationMode,compactGeometryMode,
      candidate_count:(candidates[String(frame)]||[]).length,
      button:document.querySelector('#runButton').textContent,
      result_available:latestAgentJob.result_available,
      handled:[...runResultHandled],
    }));
    assert.equal(report.recovered.frame,30);
    assert.equal(report.recovered.candidate_count,10);
    assert.equal(report.recovered.compactGeometryMode,true);
    assert.equal(report.recovered.button,'开始自动标注');
    assert.equal(report.recovered.result_available,false);
    assert.equal(report.recovered.handled.length,1);
    assert.equal(report.responses.filter(item => item.pathname === '/api/run/result').length,1);
    assert.equal(report.responses.filter(item => item.pathname === '/api/state').length,0);
    assert.deepEqual(report.page_errors,[]);
    assert.deepEqual(report.console_errors,[]);
    assert.deepEqual(report.failed_requests,[]);
    await page.screenshot({path:path.join(output,'run-recovery-after-restart-round23.png'),fullPage:true});
    report.passed=true;
  } catch (error) {
    report.error=error.stack;
    throw error;
  } finally {
    fs.writeFileSync(path.join(output,'run-recovery-browser-round23.json'),JSON.stringify(report,null,2));
    await browser.close();
  }
  console.log(JSON.stringify(report,null,2));
})().catch(error=>{console.error(error.stack);process.exitCode=1});
