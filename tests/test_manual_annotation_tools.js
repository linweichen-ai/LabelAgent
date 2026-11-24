const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const webRoot = path.join(__dirname, '../source/web');
const app = fs.readFileSync(path.join(webRoot, 'app.js'), 'utf8');
const html = fs.readFileSync(path.join(webRoot, 'index.html'), 'utf8');
const css = fs.readFileSync(path.join(webRoot, 'style.css'), 'utf8');

function geometryContext() {
  const start = app.indexOf('function pointSetSignature');
  const end = app.indexOf('function scheduleCanvasDraw', start);
  assert.ok(start >= 0 && end > start, 'manual geometry helpers are present');
  const context = vm.createContext({ promptPoints: [] });
  vm.runInContext(app.slice(start, end), context);
  return context;
}

test('rectangle drag produces a normalized four-corner polygon and closed preview', () => {
  const context = geometryContext();
  const rectangle = vm.runInContext('rectanglePoints({x:40,y:30},{x:10,y:70})', context);
  assert.deepEqual(JSON.parse(JSON.stringify(rectangle)), [[10,30],[40,30],[40,70],[10,70]]);
  const preview = vm.runInContext('rectanglePreviewPoints({x:40,y:30},{x:10,y:70})', context);
  assert.equal(preview.length, 5);
  assert.deepEqual(JSON.parse(JSON.stringify(preview[0])), JSON.parse(JSON.stringify(preview[4])));
});

test('ellipse drag produces a smooth closed polygon inside the dragged box', () => {
  const context = geometryContext();
  const ellipse = vm.runInContext('ellipsePoints({x:10,y:20},{x:50,y:60},40)', context);
  assert.equal(ellipse.length, 40);
  assert.ok(ellipse.every(([x, y]) => x >= 10 && x <= 50 && y >= 20 && y <= 60));
  const preview = vm.runInContext("dragPreviewPoints({x:10,y:20},{x:50,y:60},'ellipse')", context);
  assert.equal(preview.length, 41);
  assert.deepEqual(JSON.parse(JSON.stringify(preview[0])), JSON.parse(JSON.stringify(preview.at(-1))));
});

test('point signature detects coordinate, polarity, and ordering changes', () => {
  const context = geometryContext();
  const first = vm.runInContext("pointSetSignature([{x:10.2,y:20.7,label:1},{x:30,y:40,label:0}])", context);
  const sameRounded = vm.runInContext("pointSetSignature([{x:10,y:21,label:1},{x:30,y:40,label:0}])", context);
  const moved = vm.runInContext("pointSetSignature([{x:11,y:21,label:1},{x:30,y:40,label:0}])", context);
  const flipped = vm.runInContext("pointSetSignature([{x:10,y:21,label:0},{x:30,y:40,label:0}])", context);
  assert.equal(first, sameRounded);
  assert.notEqual(first, moved);
  assert.notEqual(first, flipped);
});

test('manual toolbar exposes persistent rectangle and documented shortcuts', () => {
  assert.match(html, /id="rectangleButton"[^>]*（R）/);
  assert.match(html, /id="ellipseButton"[^>]*（O）/);
  assert.match(html, /id="freehandButton"[^>]*（L）/);
  assert.match(html, /id="freehandSensitivity"/);
  assert.match(html, /id="polygonButton"[^>]*（V）/);
  assert.match(html, /id="pointButton"[^>]*（P）/);
  assert.match(html, /id="negativeButton"[^>]*（N）/);
  assert.match(html, /id="eraseRegionButton"[^>]*（E）/);
  assert.match(html, /id="undoButton"[^>]*Ctrl\+Z/);
  assert.match(html, /id="finishButton"[^>]*Enter/);
  assert.match(app, /manualTool=tool/);
  assert.match(app, /\$\('rectangleButton'\)\.onclick=\(\)=>activateShapeTool\('rectangle'\)/);
  assert.match(app, /\$\('ellipseButton'\)\.onclick=\(\)=>activateShapeTool\('ellipse'\)/);
});

test('freehand lasso converts a dragged boundary into a closed segmentation polygon', () => {
  assert.match(app, /function activateFreehandTool\(\)/);
  assert.match(app, /freehandDrag=\{pointerId:event\.pointerId,points:\[point\]\}/);
  assert.match(app, /freehandDrag\.points\.push\(point\)/);
  assert.match(app, /getCoalescedEvents\?\.\(\)\|\|\[event\]/);
  assert.match(app, /freehandImagePoint\(sample,event\.currentTarget\)/);
  assert.match(app, /function freehandMinimumDistance\(\)/);
  assert.match(app, /items\.push\(\{label:\$\('labelSelect'\)\.value,points\}\)/);
  assert.match(app, /event\.key\.toLowerCase\(\)==='l'/);
});

test('canvas uses pointer capture and requestAnimationFrame-coalesced preview', () => {
  assert.match(app, /onpointerdown=/);
  assert.match(app, /onpointermove=/);
  assert.match(app, /onpointerup=finishRectangleDrag/);
  assert.match(app, /setPointerCapture\?\./);
  assert.match(app, /requestAnimationFrame\(\(\)=>\{canvasDrawFrame=null;draw\(\)\}\)/);
  assert.match(app, /if\(canvas\.width!==imageWidth\)canvas\.width=imageWidth/);
  assert.match(app, /if\(canvas\.height!==imageHeight\)canvas\.height=imageHeight/);
  assert.doesNotMatch(app, /canvas\.width=imageWidth;canvas\.height=imageHeight/);
  assert.match(css, /\.imageStage canvas\{touch-action:none\}/);
  assert.doesNotMatch(app, /\$\('canvas'\)\.onclick=/);
});

test('SAM accepts point edits while busy and coalesces them into latest-only rerun', () => {
  assert.match(app, /function promptChangedDuringRequest\(\)\{if\(segmentBusy\)\{pointPromptQueuedRerun=true/);
  assert.match(app, /if\(pointSetSignature\(\)!==requestSignature\)\{pointPromptQueuedRerun=true/);
  assert.match(app, /rerun=pointPromptQueuedRerun&&latestSignature!==requestSignature/);
  assert.match(app, /pointPromptQueuedRerun=false;if\(rerun\)/);
  assert.match(app, /queueMicrotask\(generatePointMask\)/);
  assert.doesNotMatch(app, /pointMode&&segmentBusy\)\{toast\('SAM 正在生成，本次点位已锁定/);
});

test('keyboard mappings cover tools, finish, undo, and draft cancellation', () => {
  for (const [key, button] of [['v','polygonButton'],['r','rectangleButton'],['p','pointButton'],['n','negativeButton'],['e','eraseRegionButton']]) {
    assert.match(app, new RegExp(`key==='${key}'\\)\\{\\$\\('${button}'\\)\\.click\\(\\)`));
  }
  assert.match(app, /e\.key==='Enter'[^}]*finish\(\)/);
  assert.match(app, /e\.key\.toLowerCase\(\)==='z'[^}]*undoCurrentInput\(\)/);
  assert.match(app, /e\.key==='Escape'[^}]*cancelCurrentDraft\(\)/);
  assert.match(app, /event\.key\.toLowerCase\(\)==='o'[^}]*activateShapeTool\('ellipse'\)/);
});

test('polygon can finish by double click and ellipse uses pointer capture', () => {
  assert.match(app, /addEventListener\('dblclick'[^\n]*manualTool!=='polygon'/);
  assert.match(app, /rectangleDrag=\{pointerId:event\.pointerId,start:point,current:point,tool:'ellipse'\}/);
  assert.match(app, /ellipsePoints\(drag\.start,drag\.current\)/);
});
