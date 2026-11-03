// Requires Playwright and an installed Edge browser. Does not confirm/save truth.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');
const base = process.argv[2] || 'http://127.0.0.1:8765';
const output = process.argv[3];
if (!output) throw Error('Usage: node verify_candidates.cjs BASE_URL OUTPUT_DIRECTORY');
(async () => {
  fs.mkdirSync(output, {recursive:true});
  const browser = await chromium.launch({channel:'msedge',headless:true});
  const report = {checks:[],page_errors:[],saved_as_truth:false};
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1000}});
    page.setDefaultTimeout(60000);
    page.on('pageerror',error=>report.page_errors.push(error.message));
    await page.goto(base,{waitUntil:'networkidle'});
    await page.waitForFunction(()=>Object.keys(topics).length===4);
    async function openReview() {
      await page.locator('#addWindowButton').click();
      await page.locator('#windowType').selectOption('image');
      await page.locator('#windowSources input').evaluateAll(inputs=>{
        for (const input of inputs) input.checked=input.value==='/capture/camera_1/image';
      });
      await page.locator('#addWindowForm button[type="submit"]').click();
      await page.waitForFunction(()=>[...extraWindows.values()].some(view=>view.sources[0]==='/capture/camera_1/image'&&view.layers[0]?.video?.readyState>=1));
      const view=page.locator('#windowGrid > .sensorWindow').filter({has:page.locator('.windowHeader > span',{hasText:'图像 · /capture/camera_1/image'})});
      await view.locator('button[title="进入标注模式"]').click();
      await page.waitForFunction(()=>annotationMode&&!loading&&imageWidth===1920);
      await page.locator('#autoModeButton').click();
      await page.waitForFunction(()=>window.workspaceMode==='auto');
    }
    await openReview();
    for (const index of [0,76,151]) {
      await page.locator('#frameNumber').fill(String(index));
      await page.locator('#frameNumber').press('Tab');
      await page.waitForFunction(index=>frame===index&&!loading,index);
      const value=await page.evaluate(()=>({frame,shapes:selectedShapes().length,
        integer_ids:selectedShapes().every(item=>Number.isInteger(item.id)),
        labels:[...new Set(selectedShapes().map(item=>item.label))],
        truth_frames:Object.keys(shapes).length}));
      assert.ok(value.shapes>0);
      assert.equal(value.truth_frames,0);
      assert.equal(value.integer_ids,true);
      assert.ok(value.labels.every(label=>['pallet','person'].includes(label)));
      assert.equal(await page.locator('#saveCandidate').isEnabled(),true);
      report.checks.push(value);
      if(index===76) await page.screenshot({path:path.join(output,'sam3_candidate_review.png'),fullPage:true});
    }
    const before=await page.evaluate(()=>candidates['0'].length);
    await page.reload({waitUntil:'networkidle'});
    await page.waitForFunction(()=>Object.keys(topics).length===4);
    await openReview();
    assert.equal(await page.evaluate(()=>candidates['0'].length),before);
    report.refresh_restores_candidates=true;
    assert.deepEqual(report.page_errors,[]);
    report.passed=true;
  } catch(error) {
    report.error=error.stack;
    throw error;
  } finally {
    fs.writeFileSync(path.join(output,'candidate_browser_receipt.json'),JSON.stringify(report,null,2));
    await browser.close();
  }
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
