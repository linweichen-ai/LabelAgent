(() => {
  const byId = id => document.getElementById(id);
  const dock = byId('compareDock');
  const stage = byId('stage');
  const baseCanvas = byId('canvas');
  const frameImage = byId('frameImage');
  const divider = byId('compareDivider');
  if (!dock || !stage || !baseCanvas || !frameImage) return;

  const controls = {
    original: byId('compareOriginal'), candidate: byId('compareCandidate'), pending: byId('comparePending'), truth: byId('compareTruth'),
    selected: byId('compareSelectedOnly'), labels: byId('compareLabels'), fill: byId('compareFill'),
    boundaries: byId('compareBoundaries'), traversable: byId('compareTraversable'),
    candidateOpacity: byId('candidateOpacity'), truthOpacity: byId('truthOpacity'),
    boundaryWidth: byId('boundaryWidth'), wipe: byId('wipePosition'),
  };
  const status = byId('compareStatus');
  const selectionInfo = byId('compareSelectionInfo');
  const modeButtons = [...dock.querySelectorAll('[data-compare-mode]')];
  const canvases = {};
  let state = null;
  let selected = null;
  let mode = 'overlay';

  function canvasLayer(name, className, zIndex) {
    let canvas = byId(`compare${name}Canvas`);
    if (!canvas) {
      canvas = document.createElement('canvas');
      canvas.id = `compare${name}Canvas`;
      canvas.className = `compareLayer ${className}`;
      canvas.style.zIndex = String(zIndex);
      stage.insertBefore(canvas, divider);
    }
    canvases[name] = canvas;
    return canvas;
  }

  canvasLayer('Candidate', 'compareCandidateLayer', 2);
  canvasLayer('Pending', 'comparePendingLayer', 3);
  canvasLayer('Truth', 'compareTruthLayer', 4);
  canvasLayer('Traversable', 'compareTraversableLayer', 5);
  canvasLayer('Difference', 'compareDifferenceLayer', 6);

  function syncCanvas(canvas) {
    const width = baseCanvas.width || frameImage.naturalWidth;
    const height = baseCanvas.height || frameImage.naturalHeight;
    if (!width || !height) return false;
    canvas.width = width;
    canvas.height = height;
    for (const property of ['left', 'top', 'width', 'height']) {
      canvas.style[property] = baseCanvas.style[property];
    }
    return true;
  }

  function allPoints(shape) {
    const parts = shape?.parts?.length ? shape.parts : [shape];
    return parts.flatMap(part => part?.points || []);
  }

  function bounds(shape) {
    const points = allPoints(shape);
    if (!points.length) return null;
    const xs = points.map(point => Number(point[0]));
    const ys = points.map(point => Number(point[1]));
    return {
      x1: Math.min(...xs), y1: Math.min(...ys), x2: Math.max(...xs), y2: Math.max(...ys),
    };
  }

  function traceShape(context, shape) {
    const parts = shape?.parts?.length ? shape.parts : [shape];
    context.beginPath();
    for (const part of parts) {
      const points = part?.points || [];
      if (points.length < 3) continue;
      points.forEach(([x, y], index) => index ? context.lineTo(x, y) : context.moveTo(x, y));
      context.closePath();
      for (const hole of part?.holes || []) {
        hole.forEach(([x, y], index) => index ? context.lineTo(x, y) : context.moveTo(x, y));
        context.closePath();
      }
    }
  }

  function colorFor(shape, source) {
    if (source === 'pending') return '#f59e0b';
    const label = String(shape?.label || '').toLowerCase();
    if (label === 'floor' || label === '地面') return '#3b82f6';
    if (label === 'slope' || label === 'ramp' || label === '斜坡') return '#f59e0b';
    return source === 'truth' ? '#d96adf' : '#22b985';
  }

  function drawShapes(canvas, shapes, source, opacity) {
    if (!syncCanvas(canvas)) return;
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    const boundary = Number(controls.boundaryWidth.value || 4);
    shapes.forEach((shape, index) => {
      const color = colorFor(shape, source);
      traceShape(context, shape);
      if (controls.fill.checked) {
        context.save();
        context.globalAlpha = opacity;
        context.fillStyle = color;
        context.fill('evenodd');
        context.restore();
      }
      if (controls.boundaries.checked) {
        context.strokeStyle = color;
        context.lineWidth = Math.max(2, canvas.width / 900 * boundary);
        context.setLineDash(source === 'pending' ? [Math.max(8, canvas.width / 150), Math.max(5, canvas.width / 240)] : []);
        context.stroke();
        context.setLineDash([]);
      }
      if (controls.labels.checked) {
        const box = bounds(shape);
        if (!box) return;
        const prefix = source === 'pending' ? '待复核 ' : '';
        const text = `${prefix}${index + 1} ${shape.label || 'unknown'}`;
        const fontSize = Math.max(14, canvas.width / 95);
        context.font = `700 ${fontSize}px Arial, sans-serif`;
        context.lineWidth = Math.max(3, fontSize / 4);
        context.strokeStyle = '#07161d';
        context.fillStyle = '#fff';
        context.strokeText(text, box.x1 + 5, Math.max(fontSize + 4, box.y1 - 6));
        context.fillText(text, box.x1 + 5, Math.max(fontSize + 4, box.y1 - 6));
      }
    });
  }

  function rasterize(shapes, width, height) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.fillStyle = '#fff';
    shapes.forEach(shape => {
      traceShape(context, shape);
      context.fill('evenodd');
    });
    return context.getImageData(0, 0, width, height).data;
  }

  function drawDifference(canvas, candidates, truth) {
    if (!syncCanvas(canvas)) return;
    const candidateMask = rasterize(candidates, canvas.width, canvas.height);
    const truthMask = rasterize(truth, canvas.width, canvas.height);
    const context = canvas.getContext('2d');
    const output = context.createImageData(canvas.width, canvas.height);
    for (let index = 0; index < output.data.length; index += 4) {
      const candidate = candidateMask[index + 3] > 0;
      const reference = truthMask[index + 3] > 0;
      if (!candidate && !reference) continue;
      if (candidate && reference) {
        output.data[index] = 34; output.data[index + 1] = 197; output.data[index + 2] = 94;
      } else if (candidate) {
        output.data[index] = 239; output.data[index + 1] = 68; output.data[index + 2] = 68;
      } else {
        output.data[index] = 59; output.data[index + 1] = 130; output.data[index + 2] = 246;
      }
      output.data[index + 3] = 145;
    }
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.putImageData(output, 0, 0);
  }

  function traversableShapes(candidates, truth) {
    const source = candidates.length ? candidates : truth;
    return source.filter(shape => ['floor', 'slope', 'ramp', '地面', '斜坡'].includes(String(shape?.label || '').toLowerCase()));
  }

  function drawTraversable(canvas, shapes) {
    if (!syncCanvas(canvas)) return;
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = 'rgba(34, 197, 94, .30)';
    context.strokeStyle = '#4ade80';
    context.lineWidth = Math.max(3, canvas.width / 700);
    shapes.forEach(shape => {
      traceShape(context, shape);
      context.fill('evenodd');
      context.stroke();
    });
  }

  function setVisibility(name, visible) {
    canvases[name].hidden = !visible;
  }

  function applyMode(hasTruth) {
    const position = mode === 'split' ? 50 : Number(controls.wipe.value || 50);
    byId('wipeControl').hidden = mode !== 'wipe';
    divider.hidden = !['split', 'wipe'].includes(mode);
    divider.style.left = `${position}%`;
    const candidate = canvases.Candidate;
    const pending = canvases.Pending;
    candidate.style.clipPath = '';
    pending.style.clipPath = '';
    if (mode === 'split') candidate.style.clipPath = 'inset(0 0 0 50%)';
    if (mode === 'split') pending.style.clipPath = 'inset(0 0 0 50%)';
    if (mode === 'wipe') candidate.style.clipPath = `inset(0 ${100 - position}% 0 0)`;
    if (mode === 'wipe') pending.style.clipPath = `inset(0 ${100 - position}% 0 0)`;
    const standard = mode === 'overlay';
    setVisibility('Candidate', (standard || mode === 'split' || mode === 'wipe') && controls.candidate.checked);
    setVisibility('Pending', (standard || mode === 'split' || mode === 'wipe') && controls.pending.checked);
    setVisibility('Truth', standard && controls.truth.checked);
    setVisibility('Traversable', standard && controls.traversable.checked);
    setVisibility('Difference', mode === 'difference' && hasTruth);
  }

  function updateOutputs() {
    controls.candidateOpacity.nextElementSibling.textContent = `${controls.candidateOpacity.value}%`;
    controls.truthOpacity.nextElementSibling.textContent = `${controls.truthOpacity.value}%`;
    controls.boundaryWidth.nextElementSibling.textContent = controls.boundaryWidth.value;
    controls.wipe.nextElementSibling.textContent = `${controls.wipe.value}%`;
  }

  function currentFrame() {
    return Number(byId('frameNumber')?.value || 0);
  }

  function render() {
    if (!state || document.body.dataset.workspace !== 'auto') {
      Object.values(canvases).forEach(canvas => { canvas.hidden = true; });
      divider.hidden = true;
      baseCanvas.style.opacity = '';
      return;
    }
    baseCanvas.style.opacity = '0';
    frameImage.style.opacity = controls.original.checked ? '1' : '0';
    const frame = currentFrame();
    const allCandidates = state.candidates?.[String(frame)] || [];
    const storedAnnotation = state.truth?.[String(frame)] || [];
    const truthVerified = state.truth_review_status?.[String(frame)]?.reusable_as_reference === true;
    const truth = truthVerified ? storedAnnotation : [];
    const pending = truthVerified ? [] : storedAnnotation;
    const candidates = controls.selected.checked && selected?.frame === frame
      ? [selected.shape] : allCandidates;
    const hasTruth = truth.length > 0;
    const differenceButton = modeButtons.find(button => button.dataset.compareMode === 'difference');
    differenceButton.disabled = !hasTruth;
    differenceButton.title = hasTruth ? '查看候选与人工真值的像素差异' : '当前帧没有人工像素真值';
    if (mode === 'difference' && !hasTruth) {
      mode = 'overlay';
      modeButtons.forEach(button => button.classList.toggle('selected', button.dataset.compareMode === mode));
    }
    drawShapes(canvases.Candidate, candidates, 'candidate', Number(controls.candidateOpacity.value) / 100);
    drawShapes(canvases.Pending, pending, 'pending', Number(controls.candidateOpacity.value) / 100);
    drawShapes(canvases.Truth, truth, 'truth', Number(controls.truthOpacity.value) / 100);
    drawTraversable(canvases.Traversable, traversableShapes(allCandidates, pending.length ? pending : truth));
    if (hasTruth && mode === 'difference') drawDifference(canvases.Difference, candidates, truth);
    applyMode(hasTruth);
    if (!hasTruth && pending.length) {
      status.textContent = `第 ${frame} 帧：${allCandidates.length} 个机器候选，另有 ${pending.length} 个待人工复核标注，已用黄色虚线叠加显示；它们不作为人工真值参与差异或质量判断。`;
    } else if (!hasTruth) {
      status.textContent = `第 ${frame} 帧：${allCandidates.length} 个机器候选，暂无人工像素真值；差异查看和质量指标不可用。`;
    } else if (mode === 'difference') {
      status.textContent = '差异可视化：绿色为重合区域、红色为候选独有、蓝色为真值独有；该图不等同于 IoU 或质量结论。';
    } else if (mode === 'split') {
      status.textContent = `左右对比：左侧原图，右侧 ${allCandidates.length} 个机器候选、${pending.length} 个待复核标注；人工真值 ${truth.length} 个。`;
    } else if (mode === 'wipe') {
      status.textContent = `拖动对比：分割线 ${controls.wipe.value}%，可移动滑杆检查候选边界。`;
    } else {
      status.textContent = `普通叠加：机器候选 ${allCandidates.length} 个，待人工复核标注 ${pending.length} 个，人工真值 ${truth.length} 个。floor 与 slope 分色显示。`;
    }
    updateOutputs();
  }

  function resetView() {
    stage.style.transform = '';
    stage.style.transformOrigin = 'center center';
    byId('compareResetView').disabled = true;
  }

  function focusSelected() {
    if (!selected?.shape || !baseCanvas.width || !baseCanvas.height) return;
    const box = bounds(selected.shape);
    if (!box) return;
    const widthRatio = Math.max((box.x2 - box.x1) / baseCanvas.width, .05);
    const heightRatio = Math.max((box.y2 - box.y1) / baseCanvas.height, .05);
    const scale = Math.min(3, Math.max(1.25, .72 / Math.max(widthRatio, heightRatio)));
    const x = ((box.x1 + box.x2) / 2 / baseCanvas.width) * 100;
    const y = ((box.y1 + box.y2) / 2 / baseCanvas.height) * 100;
    stage.style.transformOrigin = `${x}% ${y}%`;
    stage.style.transform = `scale(${scale})`;
    byId('compareResetView').disabled = false;
  }

  modeButtons.forEach(button => button.addEventListener('click', () => {
    if (button.disabled) return;
    mode = button.dataset.compareMode;
    modeButtons.forEach(item => item.classList.toggle('selected', item === button));
    render();
  }));
  Object.values(controls).forEach(control => control?.addEventListener('input', render));
  Object.values(controls).forEach(control => control?.addEventListener('change', render));
  byId('compareFocusSelected')?.addEventListener('click', focusSelected);
  byId('compareResetView')?.addEventListener('click', resetView);
  byId('compareResetView').disabled = true;
  byId('compareDockToggle')?.addEventListener('click', () => {
    const compact = dock.classList.toggle('compact');
    byId('compareDockToggle').textContent = compact ? '展开' : '收起';
    byId('compareDockToggle').setAttribute('aria-expanded', String(!compact));
    localStorage.setItem('labelagentCompareCompact', compact ? '1' : '0');
  });
  document.addEventListener('labelagent:review-state', event => {
    state = event.detail?.state || null;
    render();
  });
  document.addEventListener('labelagent:candidate-selected', event => {
    selected = event.detail || null;
    byId('compareFocusSelected').disabled = !selected;
    if (!selected) {
      selectionInfo.textContent = '尚未选中候选';
      controls.selected.checked = false;
      resetView();
    } else {
      const box = bounds(selected.shape);
      selectionInfo.textContent = box
        ? `已选 ${selected.shape.label || 'unknown'} · bbox ${Math.round(box.x1)},${Math.round(box.y1)}–${Math.round(box.x2)},${Math.round(box.y2)}`
        : `已选 ${selected.shape.label || 'unknown'}`;
    }
    render();
  });
  byId('frameNumber')?.addEventListener('change', () => {
    selected = null;
    controls.selected.checked = false;
    selectionInfo.textContent = '尚未选中候选';
    resetView();
    window.setTimeout(render, 180);
  });
  frameImage.addEventListener('load', () => window.setTimeout(render, 40));
  window.addEventListener('resize', () => window.setTimeout(render, 40));
  // The lower workbench panel can be resized without changing the browser
  // window. Keep every comparison canvas locked to the source-image canvas
  // when that happens; otherwise an old, smaller overlay can float above the
  // newly resized image.
  const scheduleLayoutRender = () => window.requestAnimationFrame(() => render());
  new ResizeObserver(scheduleLayoutRender).observe(stage);
  new MutationObserver(scheduleLayoutRender).observe(baseCanvas, {
    attributes: true, attributeFilter: ['style', 'width', 'height'],
  });
  new MutationObserver(render).observe(document.body, { attributes: true, attributeFilter: ['data-workspace'] });

  if (localStorage.getItem('labelagentCompareCompact') === '0') {
    dock.classList.remove('compact');
    byId('compareDockToggle').textContent = '收起';
    byId('compareDockToggle').setAttribute('aria-expanded', 'true');
  }
  updateOutputs();
  window.labelAgentBootstrapReady?.then(() => {
    state = window.currentWorkbenchFrameState?.() || null;
    render();
  }).catch(() => {
    status.textContent = '暂时无法读取候选与真值；缺少证据时不会显示为可比较。';
  });
})();
