(() => {
  const byId = id => document.getElementById(id);
  const panel = byId('candidateReviewPanel');
  if (!panel) return;

  const list = byId('candidateReviewList');
  const stats = byId('candidateReviewStats');
  const context = byId('candidateReviewContext');
  const labelFilter = byId('candidateLabelFilter');
  const thresholdSelect = byId('candidateConfidenceThreshold');
  const filterButtons = [...panel.querySelectorAll('[data-review-filter]')];
  const draftStorageKey = 'labelagentCandidateReviewDraftsV8';
  let state = null;
  let refreshBusy = false;
  let refreshPending = false;
  let renderSignature = '';
  let activeFilter = 'all';
  let selectedIndex = null;

  const operationNames = {
    split: '拆分', merge: '合并', resegment: '重新分割', reclassify: '修改类别',
    drop: '删除误检', no_op: '保持不变', sam_points: 'SAM 局部修复',
    add_region: '补标', erase_region: '擦除', relabel: '修改类别',
  };
  const draftNames = {
    accepted_draft: '暂存接受', needs_edit: '需要修改', rejected_draft: '暂存驳回',
    hard_case: '疑难样本', pending: '待复核',
  };

  function readDrafts() {
    try { return JSON.parse(sessionStorage.getItem(draftStorageKey) || '{}'); } catch (_) { return {}; }
  }

  function writeDrafts(value) {
    try { sessionStorage.setItem(draftStorageKey, JSON.stringify(value)); } catch (_) {}
    document.dispatchEvent(new CustomEvent('labelagent:review-drafts-changed'));
  }

  function stable(value) {
    if (Array.isArray(value)) return value.map(stable);
    if (!value || typeof value !== 'object') return value;
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, stable(value[key])]));
  }

  function seed(shape) {
    const payload = JSON.stringify(stable({
      id: shape?.id, label: shape?.label, score: shape?.score,
      points: shape?.points, parts: shape?.parts, holes: shape?.holes,
    }));
    let hash = 2166136261;
    for (let index = 0; index < payload.length; index += 1) {
      hash ^= payload.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return (hash >>> 0).toString(16).padStart(8, '0');
  }

  async function sha256(shape) {
    const payload = JSON.stringify(stable({
      id: shape?.id, label: shape?.label, score: shape?.score,
      points: shape?.points, parts: shape?.parts, holes: shape?.holes,
    }));
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(payload));
    return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('');
  }

  function candidateKey(topic, frame, shape, index) {
    return `${topic || 'none'}|${frame}|${shape?.id ?? index + 1}|${seed(shape)}`;
  }

  function referencedIds(value, found = new Set()) {
    if (Array.isArray(value)) {
      value.forEach(item => referencedIds(item, found));
      return found;
    }
    if (!value || typeof value !== 'object') return found;
    for (const [key, item] of Object.entries(value)) {
      if (['shape_id', 'kept_id', 'removed_id'].includes(key) && Number.isFinite(Number(item))) {
        found.add(Number(item));
      } else if (['shape_ids', 'result_ids', 'removed_ids'].includes(key) && Array.isArray(item)) {
        item.filter(id => Number.isFinite(Number(id))).forEach(id => found.add(Number(id)));
      } else if (typeof item === 'object') referencedIds(item, found);
    }
    return found;
  }

  function operationName(value) {
    const operation = typeof value === 'string' ? value : value?.operation;
    return operationNames[operation] || operation || '修改记录';
  }

  function runMetadata(run, shape) {
    const id = Number(shape?.id);
    const applied = Array.isArray(run?.applied) ? run.applied : [];
    const related = applied.filter(item => referencedIds(item).has(id));
    const actions = [...new Set(related.map(item => operationName(item.operation || item)))];
    const queue = Array.isArray(run?.review_queue) ? run.review_queue : [];
    const queueIds = referencedIds(queue);
    const queueLabels = new Set(queue.map(item => String(item?.label || '')).filter(Boolean));
    const frameConflict = queue.length > 0 && queueIds.size === 0 && queueLabels.size === 0;
    return {
      modified: related.length > 0,
      actions,
      conflict: queueIds.has(id) || queueLabels.has(String(shape?.label || '')) || frameConflict,
      frameConflict,
      converged: run?.converged === true,
      hasRun: Boolean(run),
    };
  }

  function polygonArea(points) {
    if (!Array.isArray(points) || points.length < 3) return 0;
    let area = 0;
    for (let index = 0; index < points.length; index += 1) {
      const current = points[index];
      const next = points[(index + 1) % points.length];
      area += Number(current?.[0]) * Number(next?.[1]) - Number(next?.[0]) * Number(current?.[1]);
    }
    return Math.abs(area) / 2;
  }

  function shapeGeometry(shape, imageWidth, imageHeight) {
    const parts = Array.isArray(shape?.parts) && shape.parts.length ? shape.parts : [shape];
    const points = parts.flatMap(part => Array.isArray(part?.points) ? part.points : []);
    const valid = points.filter(point => Array.isArray(point) && Number.isFinite(Number(point[0])) && Number.isFinite(Number(point[1])));
    if (!valid.length) return { bounds: null, areaRatio: null, flags: ['invalid_geometry'] };
    const xs = valid.map(point => Number(point[0]));
    const ys = valid.map(point => Number(point[1]));
    const bounds = { left: Math.min(...xs), top: Math.min(...ys), right: Math.max(...xs), bottom: Math.max(...ys) };
    const area = parts.reduce((total, part) => {
      const outer = polygonArea(part?.points || []);
      const holes = (part?.holes || []).reduce((sum, hole) => sum + polygonArea(hole), 0);
      return total + Math.max(0, outer - holes);
    }, 0);
    const imageArea = Number(imageWidth) * Number(imageHeight);
    const areaRatio = imageArea > 0 ? area / imageArea : null;
    const flags = [];
    if (area <= 0 || valid.length < 3) flags.push('invalid_geometry');
    if (areaRatio !== null && areaRatio < 0.00008) flags.push('tiny_fragment');
    if (areaRatio !== null && areaRatio > 0.75) flags.push('near_full_frame');
    if (imageArea > 0 && (bounds.left <= 1 || bounds.top <= 1 || bounds.right >= imageWidth - 2 || bounds.bottom >= imageHeight - 2)) flags.push('touches_edge');
    return { bounds, areaRatio, flags };
  }

  function boundsIou(first, second) {
    if (!first || !second) return 0;
    const intersection = Math.max(0, Math.min(first.right, second.right) - Math.max(first.left, second.left))
      * Math.max(0, Math.min(first.bottom, second.bottom) - Math.max(first.top, second.top));
    const firstArea = Math.max(0, first.right - first.left) * Math.max(0, first.bottom - first.top);
    const secondArea = Math.max(0, second.right - second.left) * Math.max(0, second.bottom - second.top);
    const union = firstArea + secondArea - intersection;
    return union > 0 ? intersection / union : 0;
  }

  function geometryReview(shapes, imageWidth, imageHeight) {
    const geometries = shapes.map(shape => shapeGeometry(shape, imageWidth, imageHeight));
    const signatures = shapes.map(shape => JSON.stringify(stable({
      label: shape?.label, points: shape?.points, parts: shape?.parts, holes: shape?.holes,
    })));
    for (let first = 0; first < shapes.length; first += 1) {
      for (let second = first + 1; second < shapes.length; second += 1) {
        const sameLabel = String(shapes[first]?.label || '') === String(shapes[second]?.label || '');
        const overlap = boundsIou(geometries[first].bounds, geometries[second].bounds);
        if (signatures[first] === signatures[second] || (sameLabel && overlap >= 0.9)) {
          geometries[first].flags.push('possible_duplicate');
          geometries[second].flags.push('possible_duplicate');
        } else if (!sameLabel && overlap >= 0.85) {
          geometries[first].flags.push('cross_label_overlap');
          geometries[second].flags.push('cross_label_overlap');
        }
      }
    }
    geometries.forEach(item => { item.flags = [...new Set(item.flags)]; });
    return geometries;
  }

  function candidateInfo(shape, index, run, labels, threshold, topic, frame, drafts, geometry) {
    const metadata = runMetadata(run, shape);
    const score = Number(shape?.score);
    const hasScore = Number.isFinite(score);
    const label = String(shape?.label || 'unknown');
    const unknown = !labels.includes(label) || /unknown|new[_ -]?label/i.test(label);
    const low = hasScore && score < threshold;
    const key = candidateKey(topic, frame, shape, index);
    const draft = drafts[key]?.status || 'pending';
    const geometryFlags = geometry?.flags || [];
    const geometryRisk = geometryFlags.length > 0;
    const riskScore = (metadata.conflict ? 100 : 0) + (unknown ? 90 : 0) + (low ? 80 : 0)
      + (geometryFlags.includes('invalid_geometry') ? 75 : 0)
      + (geometryFlags.includes('possible_duplicate') || geometryFlags.includes('cross_label_overlap') ? 65 : 0)
      + (geometryFlags.includes('near_full_frame') ? 55 : 0)
      + (geometryFlags.includes('tiny_fragment') ? 45 : 0)
      + (geometryFlags.includes('touches_edge') ? 15 : 0);
    return {
      shape, index, key, label, score, hasScore, low, unknown, draft,
      geometryFlags, geometryRisk, areaRatio: geometry?.areaRatio ?? null, riskScore,
      risk: low || unknown || metadata.conflict || geometryRisk,
      ...metadata,
    };
  }

  window.labelAgentCandidateRiskOrder = (shapes, run, labels, threshold, imageWidth, imageHeight) => {
    const safeShapes = Array.isArray(shapes) ? shapes : [];
    const geometries = geometryReview(safeShapes, imageWidth, imageHeight);
    return safeShapes.map((shape, index) => candidateInfo(
      shape, index, run, Array.isArray(labels) ? labels : [], Number(threshold) || 0.65,
      '', 0, {}, geometries[index],
    )).filter(info => info.risk).map(info => ({
      index: info.index, riskScore: info.riskScore, geometryFlags: info.geometryFlags,
      low: info.low, unknown: info.unknown, conflict: info.conflict,
    })).sort((first, second) => second.riskScore - first.riskScore || first.index - second.index);
  };
  document.dispatchEvent(new CustomEvent('labelagent:risk-engine-ready'));

  function clearHighlight() {
    byId('reviewHighlightCanvas')?.remove();
    selectedIndex = null;
    document.dispatchEvent(new CustomEvent('labelagent:candidate-selected', { detail: null }));
  }

  function highlightCandidate(info) {
    selectedIndex = info.index;
    list.querySelectorAll('.candidateReviewItem').forEach(item => {
      item.classList.toggle('selected', Number(item.dataset.index) === info.index);
    });
    document.querySelectorAll('#shapeList .shapeItem')[info.index]?.click();
    const source = byId('canvas');
    const stage = byId('stage');
    if (!source || !stage || !source.width || !source.height) return;
    let overlay = byId('reviewHighlightCanvas');
    if (!overlay) {
      overlay = document.createElement('canvas');
      overlay.id = 'reviewHighlightCanvas';
      overlay.className = 'reviewHighlightCanvas';
      stage.append(overlay);
    }
    overlay.width = source.width;
    overlay.height = source.height;
    for (const name of ['left', 'top', 'width', 'height']) overlay.style[name] = source.style[name];
    const draw = overlay.getContext('2d');
    draw.clearRect(0, 0, overlay.width, overlay.height);
    draw.fillStyle = 'rgba(255, 191, 46, .18)';
    draw.strokeStyle = '#ffca45';
    draw.lineWidth = Math.max(4, overlay.width / 420);
    const parts = info.shape?.parts?.length ? info.shape.parts : [info.shape];
    for (const part of parts) {
      const points = part?.points || [];
      if (points.length < 3) continue;
      draw.beginPath();
      points.forEach(([x, y], pointIndex) => pointIndex ? draw.lineTo(x, y) : draw.moveTo(x, y));
      draw.closePath();
      for (const hole of part?.holes || []) {
        hole.forEach(([x, y], pointIndex) => pointIndex ? draw.lineTo(x, y) : draw.moveTo(x, y));
        draw.closePath();
      }
      draw.fill('evenodd');
      draw.stroke();
    }
    stage.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    document.dispatchEvent(new CustomEvent('labelagent:candidate-selected', {
      detail: { index: info.index, shape: info.shape, frame: Number(byId('frameNumber')?.value || 0) },
    }));
  }

  function setDraft(info, status) {
    const drafts = readDrafts();
    drafts[info.key] = {
      status,
      recorded_at: new Date().toISOString(),
      candidate_seed: info.key.split('|').at(-1),
      local_only: true,
    };
    writeDrafts(drafts);
    render();
  }

  function chip(text, tone = '') {
    const node = document.createElement('span');
    node.className = `candidateRiskChip ${tone}`.trim();
    node.textContent = text;
    return node;
  }

  function statCard(label, value, tone = '') {
    const node = document.createElement('span');
    node.className = `candidateStat ${tone}`.trim();
    node.innerHTML = `<b>${value}</b><small>${label}</small>`;
    return node;
  }

  function matches(info) {
    if (labelFilter.value !== 'all' && info.label !== labelFilter.value) return false;
    if (activeFilter === 'risk') return info.risk;
    if (activeFilter === 'geometry') return info.geometryRisk;
    if (activeFilter === 'low') return info.low;
    if (activeFilter === 'conflict') return info.conflict;
    if (activeFilter === 'unknown') return info.unknown;
    if (activeFilter === 'modified') return info.modified;
    if (activeFilter === 'pending') return info.draft === 'pending';
    return true;
  }

  function updateLabels(infos) {
    const selected = labelFilter.value;
    const labels = [...new Set(infos.map(info => info.label))].sort();
    labelFilter.replaceChildren(new Option('全部类别', 'all'));
    labels.forEach(label => labelFilter.append(new Option(label, label)));
    labelFilter.value = labels.includes(selected) ? selected : 'all';
  }

  function renderEmpty(message, detail) {
    list.replaceChildren();
    const empty = document.createElement('div');
    empty.className = 'candidateReviewEmpty';
    empty.innerHTML = `<span aria-hidden="true">◎</span><strong>${message}</strong><small>${detail}</small>`;
    list.append(empty);
  }

  function renderItem(info) {
    const item = document.createElement('article');
    item.className = `candidateReviewItem${selectedIndex === info.index ? ' selected' : ''}`;
    item.dataset.index = String(info.index);
    item.tabIndex = 0;
    const head = document.createElement('div');
    head.className = 'candidateItemHead';
    const identity = document.createElement('div');
    identity.innerHTML = `<span class="candidateNumber">${info.index + 1}</span><span><strong>${info.label}</strong><small>候选 ID ${info.shape?.id ?? info.index + 1}</small></span>`;
    const status = document.createElement('span');
    status.className = `candidateDraftStatus ${info.draft}`;
    status.textContent = draftNames[info.draft] || '待复核';
    head.append(identity, status);

    const risks = document.createElement('div');
    risks.className = 'candidateRiskRow';
    if (info.unknown) risks.append(chip('未知 / 新类别', 'danger'));
    if (info.low) risks.append(chip('低置信', 'warning'));
    if (info.conflict) risks.append(chip('模型冲突', 'danger'));
    const geometryLabels = {
      invalid_geometry: ['无效轮廓', 'danger'], tiny_fragment: ['疑似碎片', 'warning'],
      near_full_frame: ['疑似全图', 'danger'], touches_edge: ['贴边', 'warning'],
      possible_duplicate: ['疑似重复', 'danger'], cross_label_overlap: ['跨类重叠', 'danger'],
    };
    info.geometryFlags.forEach(flag => {
      const [label, tone] = geometryLabels[flag] || [flag, 'warning'];
      risks.append(chip(label, tone));
    });
    if (info.modified) risks.append(chip(`Agent：${info.actions.join('、') || '已修改'}`, 'modified'));
    if (!info.risk && !info.modified) risks.append(chip('未发现显式风险', 'safe'));

    const evidence = document.createElement('dl');
    evidence.className = 'candidateEvidence';
    const source = info.hasRun ? 'SAM → VLM → Agent' : info.hasScore ? 'SAM 候选（无 Agent 记录）' : '导入候选（来源版本未记录）';
    const confidence = info.hasScore ? `${(info.score * 100).toFixed(1)}%` : '未记录';
    const areaRatio = info.areaRatio === null ? '尺寸待确认' : `${(info.areaRatio * 100).toFixed(info.areaRatio < 0.01 ? 3 : 1)}%`;
    evidence.innerHTML = `<div><dt>置信度</dt><dd>${confidence}</dd></div><div><dt>面积占比</dt><dd>${areaRatio}</dd></div><div><dt>来源</dt><dd>${source}</dd></div><div><dt>Agent</dt><dd>${info.modified ? info.actions.join('、') : info.hasRun ? (info.converged ? '复核完成 · 无局部修改' : '流程未收敛') : '无记录'}</dd></div><div><dt>候选 SHA</dt><dd><code>计算中…</code></dd></div>`;

    const actions = document.createElement('div');
    actions.className = 'candidateDraftActions';
    const definitions = [
      ['accepted_draft', '暂存接受'], ['needs_edit', '需要修改'],
      ['rejected_draft', '暂存驳回'], ['hard_case', '标记疑难'],
    ];
    definitions.forEach(([value, label]) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.reviewDraft = value;
      button.textContent = label;
      button.classList.toggle('selected', info.draft === value);
      if (value === 'accepted_draft' && info.unknown) {
        button.disabled = true;
        button.title = '未知 / 新类别必须先完成人工类别审核，不能直接接受';
      }
      button.addEventListener('click', event => {
        event.stopPropagation();
        setDraft(info, value);
      });
      actions.append(button);
    });
    const localNote = document.createElement('small');
    localNote.textContent = '仅本地复核草稿';
    actions.append(localNote);

    item.append(head, risks, evidence, actions);
    item.addEventListener('click', () => highlightCandidate(info));
    item.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        highlightCandidate(info);
      }
    });
    sha256(info.shape).then(hash => {
      if (item.isConnected) {
        const code = item.querySelector('code');
        code.textContent = `${hash.slice(0, 12)}…`;
        code.title = hash;
      }
    }).catch(() => {
      const code = item.querySelector('code');
      if (code) code.textContent = '无法计算';
    });
    return item;
  }

  function render() {
    if (!state) return;
    const frame = Number(byId('frameNumber')?.value || 0);
    const topic = state.topic || '';
    const shapes = state.candidates?.[String(frame)] || [];
    const run = state.agent_runs?.[String(frame)];
    const imageNode = byId('frameImage');
    const signature = JSON.stringify([frame, topic, shapes, run, state.labels, activeFilter,
      labelFilter.value, thresholdSelect.value, readDrafts(), imageNode?.naturalWidth, imageNode?.naturalHeight]);
    if (signature === renderSignature) return;
    renderSignature = signature;
    const labels = Array.isArray(state.labels) ? state.labels : [];
    const threshold = Number(thresholdSelect.value || 0.65);
    const drafts = readDrafts();
    const image = byId('frameImage');
    const geometries = geometryReview(shapes, image?.naturalWidth || 0, image?.naturalHeight || 0);
    const infos = shapes.map((shape, index) => candidateInfo(
      shape, index, run, labels, threshold, topic, frame, drafts, geometries[index],
    ));
    updateLabels(infos);
    const counts = {
      total: infos.length,
      risk: infos.filter(info => info.risk).length,
      low: infos.filter(info => info.low).length,
      conflict: infos.filter(info => info.conflict).length,
      unknown: infos.filter(info => info.unknown).length,
      modified: infos.filter(info => info.modified).length,
      geometry: infos.filter(info => info.geometryRisk).length,
      pending: infos.filter(info => info.draft === 'pending').length,
    };
    panel.classList.toggle('noCandidates', infos.length === 0);
    stats.replaceChildren(
      statCard('候选总数', counts.total), statCard('需关注', counts.risk, counts.risk ? 'warning' : 'safe'),
      statCard('低置信', counts.low), statCard('模型冲突', counts.conflict),
      statCard('未知类别', counts.unknown), statCard('Agent 修改', counts.modified),
      statCard('轮廓风险', counts.geometry, counts.geometry ? 'warning' : 'safe'),
      statCard('待复核', counts.pending, counts.pending ? 'pending' : 'safe'),
    );
    context.textContent = `${topic || '未选择相机'} · 第 ${frame} 帧 · ${counts.total} 个候选`;
    const filtered = infos.filter(matches).sort((first, second) => second.riskScore - first.riskScore || first.index - second.index);
    if (!infos.length) {
      clearHighlight();
      renderEmpty('当前帧没有机器候选', '运行完整 LabelAgent 或切换到已有候选的帧后，这里会显示复核清单。');
    } else if (!filtered.length) {
      clearHighlight();
      renderEmpty('没有符合当前筛选的候选', '可以切换到“全部”或调整类别与置信度阈值。');
    } else {
      list.replaceChildren(...filtered.map(renderItem));
    }
  }

  async function refresh() {
    if (refreshBusy) { refreshPending = true; return; }
    refreshBusy = true;
    try {
      const current = window.currentWorkbenchFrameState?.();
      const frame = Number(byId('frameNumber')?.value || 0);
      if (!current) throw new Error('workbench state unavailable');
      if (current.path && current.topic) {
        if (typeof window.readCompactFrameStateSnapshot !== 'function') throw new Error('frame state unavailable');
        const frameState = await window.readCompactFrameStateSnapshot(frame, current.topic, current.path);
        state = {...current, ...frameState, labels: current.labels || [], config: current.config || {}};
      } else state = current;
      panel.classList.remove('unavailable');
      render();
      document.dispatchEvent(new CustomEvent('labelagent:review-state', { detail: { state } }));
    } catch (_) {
      state = null;
      panel.classList.add('unavailable');
      panel.classList.add('noCandidates');
      stats.replaceChildren();
      context.textContent = '状态不可用';
      renderSignature = '';
      renderEmpty('暂时无法读取候选状态', '请确认本机服务在线后刷新页面；缺少证据时不会显示为已复核。');
    } finally {
      refreshBusy = false;
      if (refreshPending) {
        refreshPending = false;
        if (!document.hidden) await refresh();
      }
    }
  }

  function adoptFrameState(next) {
    if (!next) return;
    state = next;
    panel.classList.remove('unavailable');
    render();
    document.dispatchEvent(new CustomEvent('labelagent:review-state', { detail: { state } }));
  }

  filterButtons.forEach(button => button.addEventListener('click', () => {
    activeFilter = button.dataset.reviewFilter;
    filterButtons.forEach(item => item.classList.toggle('selected', item === button));
    render();
  }));
  labelFilter.addEventListener('change', render);
  thresholdSelect.value = localStorage.getItem('labelagentCandidateThreshold') || '0.65';
  thresholdSelect.addEventListener('change', () => {
    localStorage.setItem('labelagentCandidateThreshold', thresholdSelect.value);
    render();
  });
  byId('candidateReviewToggle')?.addEventListener('click', () => {
    const compact = panel.classList.toggle('compact');
    byId('candidateReviewToggle').textContent = compact ? '展开复核台' : '收起复核台';
    byId('candidateReviewToggle').setAttribute('aria-expanded', String(!compact));
    localStorage.setItem('labelagentCandidateReviewCompact', compact ? '1' : '0');
  });
  document.addEventListener('labelagent:frame-state', event => adoptFrameState(event.detail?.state));
  byId('frameNumber')?.addEventListener('change', () => window.setTimeout(() => adoptFrameState(window.currentWorkbenchFrameState?.()), 180));
  byId('autoModeButton')?.addEventListener('click', () => window.setTimeout(() => adoptFrameState(window.currentWorkbenchFrameState?.()), 120));
  window.addEventListener('resize', () => {
    if (selectedIndex === null || !state) return;
    const frame = Number(byId('frameNumber')?.value || 0);
    const shape = state.candidates?.[String(frame)]?.[selectedIndex];
    if (shape) highlightCandidate(candidateInfo(
      shape, selectedIndex, state.agent_runs?.[String(frame)], state.labels || [],
      Number(thresholdSelect.value || 0.65), state.topic || '', frame, readDrafts(),
    ));
  });
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) refresh();
  });

  if (localStorage.getItem('labelagentCandidateReviewCompact') === '1') {
    panel.classList.add('compact');
    byId('candidateReviewToggle').textContent = '展开复核台';
    byId('candidateReviewToggle').setAttribute('aria-expanded', 'false');
  }
  window.labelAgentBootstrapReady?.then(() => adoptFrameState(window.currentWorkbenchFrameState?.())).catch(() => {
    panel.classList.add('unavailable');
  });
  let reviewPollTimer = null;
  let latestJobStatus = 'idle';
  let latestJobTaskId = null;
  let latestCompletedFrames = 0;
  function scheduleReviewPoll(delay) {
    if (reviewPollTimer) clearTimeout(reviewPollTimer);
    if (document.hidden) return;
    // Frame completion events trigger an immediate refresh below.  This timer
    // is only a recovery heartbeat, so a long VLM round does not repeatedly
    // download and parse the multi-megabyte full state.
    const interval = 30000;
    reviewPollTimer = window.setTimeout(async () => {
      reviewPollTimer = null;
      if (window.workspaceMode === 'auto') await refresh();
      scheduleReviewPoll();
    }, delay ?? interval);
  }
  document.addEventListener('labelagent:job-status', event => {
    const previousStatus = latestJobStatus;
    const detail = event.detail || {};
    latestJobStatus = detail.status || 'idle';
    if (detail.task_id !== latestJobTaskId) {
      latestJobTaskId = detail.task_id || null;
      latestCompletedFrames = 0;
    }
    const completed = Number(detail.completed || 0);
    const frameCompleted = completed > latestCompletedFrames;
    latestCompletedFrames = Math.max(latestCompletedFrames, completed);
    if (window.workspaceMode !== 'auto') return;
    if (['running', 'stopping'].includes(latestJobStatus)) {
      if (!['running', 'stopping'].includes(previousStatus) || frameCompleted) scheduleReviewPoll(0);
    } else {
      scheduleReviewPoll(0);
    }
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      if (reviewPollTimer) clearTimeout(reviewPollTimer);
      reviewPollTimer = null;
    } else {
      refresh();
      scheduleReviewPoll(30000);
    }
  });
  scheduleReviewPoll();
})();
