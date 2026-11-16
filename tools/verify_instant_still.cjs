// Browser acceptance for instant still-image annotation and playback-on-demand.
// This does not save truth, candidates, or training data.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
let chromium;
try { ({chromium} = require('playwright-core')); }
catch { ({chromium} = require('playwright')); }

const base = process.argv[2] || 'http://127.0.0.1:8765';
const output = process.argv[3];
const selectedTopic = process.argv[4] || '/capture/camera_3/image';
const playbackFile = process.argv[5];
if (!output || !playbackFile) {
  throw Error('Usage: node verify_instant_still.cjs BASE_URL OUTPUT_DIRECTORY TOPIC PLAYBACK_MP4');
}

(async () => {
  fs.mkdirSync(output, {recursive:true});
  const report = {
    topic:selectedTopic,
    playback_file:playbackFile,
    playback_existed_before:fs.existsSync(playbackFile),
    requests:[], page_errors:[], console_errors:[], failed_requests:[],
  };
  assert.equal(report.playback_existed_before, false, 'acceptance topic must not have a cached MP4');
  const browser = await chromium.launch({
    executablePath:'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    headless:true,
  });
  try {
    const page = await browser.newPage({viewport:{width:1600,height:1000}});
    page.setDefaultTimeout(90000);
    page.on('pageerror', error => report.page_errors.push(error.message));
    page.on('console', message => {
      if (message.type() === 'error') report.console_errors.push(message.text());
    });
    page.on('requestfailed', request => report.failed_requests.push({
      url:request.url(), error:request.failure()?.errorText,
    }));
    page.on('response', response => {
      const pathname = new URL(response.url()).pathname;
      if (['/api/timeline','/api/playback','/api/topic'].includes(pathname)
          || pathname.startsWith('/api/frame/')) {
        report.requests.push({pathname,status:response.status(),at:Date.now()});
      }
    });

    await page.goto(base, {waitUntil:'networkidle'});
    await page.waitForFunction(() => window.labelAgentBootstrapReady && Object.keys(topics).length > 0);
    await page.locator('#addWindowButton').click();
    await page.locator('#windowType').selectOption('image');
    await page.locator('#windowSources input').evaluateAll((inputs, topic) => {
      for (const input of inputs) input.checked = input.value === topic;
    }, selectedTopic);
    let started = Date.now();
    await page.locator('#addWindowForm button[type="submit"]').click();
    await page.waitForFunction(topic => {
      const view = [...extraWindows.values()].find(item => item.sources[0] === topic);
      return view?.layers[0]?.metadata?.times?.length > 0
        && view.layers[0].image?.complete && view.layers[0].image?.naturalWidth > 0;
    }, selectedTopic);
    report.window_still_ready_ms = Date.now() - started;
    report.after_window = await page.evaluate(topic => {
      const view = [...extraWindows.values()].find(item => item.sources[0] === topic);
      const layer = view.layers[0];
      return {status:view.status.textContent, image_hidden:layer.image.hidden,
        image_width:layer.image.naturalWidth, video_hidden:layer.video.hidden,
        playback_ready:layer.metadata.playback_ready, count:layer.metadata.count};
    }, selectedTopic);
    assert.equal(report.requests.filter(item => item.pathname === '/api/playback').length, 0);
    assert.equal(fs.existsSync(playbackFile), false, 'opening a still window must not create MP4');
    assert.equal(report.after_window.image_hidden, false);
    assert.equal(report.after_window.video_hidden, true);

    const view = page.locator('#windowGrid > .sensorWindow').filter({
      has:page.locator('.windowHeader > span', {hasText:`图像 · ${selectedTopic}`}),
    });
    started = Date.now();
    await view.locator('button[title="进入标注模式"]').click();
    await page.waitForFunction(() => annotationMode && !loading && compactGeometryMode);
    report.annotation_ready_ms = Date.now() - started;
    report.annotation = await page.evaluate(() => ({
      topic, frame, annotationMode, compactGeometryMode,
      image_width:document.querySelector('#frameImage').naturalWidth,
      image_hidden:document.querySelector('#frameImage').hidden,
      playback_hidden:document.querySelector('#playbackVideo').hidden,
    }));
    assert.equal(report.annotation.topic, selectedTopic);
    assert.ok(report.annotation.image_width > 0);
    assert.equal(report.annotation.image_hidden, false);
    assert.equal(report.annotation.playback_hidden, true);
    assert.equal(report.requests.filter(item => item.pathname === '/api/playback').length, 0);
    assert.equal(fs.existsSync(playbackFile), false, 'entering annotation must not create MP4');

    started = Date.now();
    await page.evaluate(() => showFrame(101));
    await page.waitForFunction(() => frame === 101 && !loading);
    report.frame_101_ready_ms = Date.now() - started;
    assert.equal(report.requests.filter(item => item.pathname === '/api/playback').length, 0);
    assert.equal(fs.existsSync(playbackFile), false, 'frame navigation must not create MP4');
    await page.screenshot({path:path.join(output,'instant-still-before-play-round22.png'),fullPage:true});

    started = Date.now();
    await page.locator('#playButton').click();
    await page.waitForFunction(() => playing === true);
    report.explicit_play_ready_ms = Date.now() - started;
    report.playback_requests_after_click = report.requests.filter(item => item.pathname === '/api/playback').length;
    report.playback_existed_after_click = fs.existsSync(playbackFile);
    assert.ok(report.playback_requests_after_click > 0);
    assert.equal(report.playback_existed_after_click, true);
    await page.locator('#playButton').click();
    await page.waitForFunction(() => !playing && !loading);
    report.after_stop = await page.evaluate(() => ({
      image_hidden:document.querySelector('#frameImage').hidden,
      playback_hidden:document.querySelector('#playbackVideo').hidden,
    }));
    assert.equal(report.after_stop.image_hidden, false);
    assert.equal(report.after_stop.playback_hidden, true);

    assert.deepEqual(report.page_errors, []);
    assert.deepEqual(report.console_errors, []);
    report.unexpected_failed_requests = report.failed_requests.filter(item =>
      !(item.error === 'net::ERR_ABORTED' && new URL(item.url).pathname.startsWith('/api/frame/')));
    assert.deepEqual(report.unexpected_failed_requests, []);
    report.passed = true;
  } catch (error) {
    report.error = error.stack;
    throw error;
  } finally {
    fs.writeFileSync(path.join(output,'instant-still-browser-round22.json'),JSON.stringify(report,null,2));
    await browser.close();
  }
  console.log(JSON.stringify(report,null,2));
})().catch(error => {
  console.error(error.stack);
  process.exitCode = 1;
});
