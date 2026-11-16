(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const panel = byId('batchReviewPanel');
  if (!panel) return;
  const queueHost = byId('batchReviewQueue');
  const draftStorageKey = 'labelagentCandidateReviewDraftsV8';
  let state = null;
  let activeFilter = 'all';
  let visibleQueue = [];

  function drafts() {
    try { return JSON.parse(sessionStorage.getItem(draftStorageKey) || '{}'); } catch (_) { return {}; }
  }
  function stable(value) {
    if (Array.isArray(value)) return value.map(stable);
    if (!value || typeof value !== 'object') return value;
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, stable(value[key])]));
  }
  function seed(shape) {
    const payload = JSON.stringify(stable({id: shape?.id, label: shape?.label, score: shape?.score,
      points: shape?.points, parts: shape?.parts, holes: shape?.holes}));
    let hash = 2166136261;
    for (let index = 0; index < payload.length; index += 1) { hash ^= payload.charCodeAt(index); hash = Math.imul(hash, 16777619); }
    return (hash >>> 0).toString(16).padStart(8, '0');
  }
  function candidateKey(topic, frame, shape, index) {
    return `${topic || 'none'}|${frame}|${shape?.id ?? index + 1}|${seed(shape)}`;
  }
  function normalizeSplit(value) {
    const text = String(value || '').trim().toLowerCase().replace(/[_-]/g, '');
    if (['train', 'train72', 'training'].includes(text)) return 'train';
    if (['test', 'test24', 'evaluation', 'eval'].includes(text)) return 'test';
    if (['val', 'validation'].includes(text)) return 'val';
    if (!text || ['none', 'unknown', 'unassigned'].includes(text)) return 'unassigned';
    return 'invalid';
  }
  function frameSplit(shapes, run) {
    const values = [state?.config?.dataset_split, run?.dataset_split, run?.split];
    for (const shape of shapes) values.push(shape?.dataset_split, shape?.split);
    const normalized = new Set(values.filter(value => value !== undefined && value !== null).map(normalizeSplit));
    normalized.delete('unassigned');
    if (normalized.has('invalid') || normalized.size > 1) return 'mixed';
    return normalized.values().next().value || 'unassigned';
  }
  function persistentStatus(frame) {
    return Object.values(state?.candidate_review_status || {}).find(item => Number(item?.frame) === Number(frame)
      && item?.topic === state?.topic) || null;
  }
  function frameInfo(frame, shapes, allDrafts) {
    const run = state?.agent_runs?.[String(frame)] || null;
    const known = Array.isArray(state?.labels) ? state.labels : [];
    const threshold = Number(byId('candidateConfidenceThreshold')?.value || localStorage.getItem('labelagentCandidateThreshold') || 0.65);
    const statuses = shapes.map((shape, index) => allDrafts[candidateKey(state?.topic, frame, shape, index)]?.status || 'pending');
    const unknown = shapes.filter(shape => {
      const label = String(shape?.label || 'unknown');
      return !known.includes(label) || /unknown|new[_ -]?label/i.test(label);
    }).length;
    const low = shapes.filter(shape => Number.isFinite(Number(shape?.score)) && Number(shape.score) < threshold).length;
    const hard = statuses.filter(status => status === 'hard_case').length;
    const reviewed = statuses.filter(status => status !== 'pending').length;
    const status = persistentStatus(frame);
    const conflict = Boolean(run?.failed || run?.error || run?.review_queue?.length || status?.remediation_required);
    const qcFailure = Boolean(run?.quality_failure || run?.qc_failed || run?.stop_reason === 'quality_failure');
    const split = frameSplit(shapes, run);
    let priority = 3;
    let reason = '普通待复核';
    if (unknown) { priority = 0; reason = `${unknown} 个未知 / 新类别`; }
    else if (conflict || qcFailure) { priority = 1; reason = qcFailure ? 'QC 失败' : '模型冲突 / 需修正'; }
    else if (low) { priority = 2; reason = `${low} 个低置信候选`; }
    if (reviewed === shapes.length && shapes.length) { priority = 4; reason = '本次会话已暂存'; }
    return {frame: Number(frame), shapes, total: shapes.length, reviewed, pending: shapes.length - reviewed,
      unknown, low, hard, conflict, qcFailure, split, priority, reason};
  }
  function buildQueue() {
    const allDrafts = drafts();
    return Object.entries(state?.candidates || {})
      .filter(([, shapes]) => Array.isArray(shapes) && shapes.length)
      .map(([frame, shapes]) => frameInfo(frame, shapes, allDrafts))
      .sort((a, b) => a.priority - b.priority || b.hard - a.hard || a.frame - b.frame);
  }
  function matches(item) {
    if (activeFilter === 'priority') return item.priority < 3;
    if (activeFilter === 'unknown') return item.unknown > 0;
    if (activeFilter === 'conflict') return item.conflict || item.qcFailure;
    if (activeFilter === 'low') return item.low > 0;
    if (activeFilter === 'hard') return item.hard > 0;
    if (activeFilter === 'pending') return item.pending > 0;
    if (activeFilter === 'reviewed') return item.reviewed === item.total;
    return true;
  }
  function stat(label, value, tone = '') {
    const node = document.createElement('span');
    node.className = `batchReviewStat ${tone}`.trim();
    node.innerHTML = `<b>${value}</b><small>${label}</small>`;
    return node;
  }
  function splitMeta(split) {
    return {
      train: ['train72 / train', '训练候选 · 晋升前禁回流'],
      test: ['test24 / test', '仅评测 · 永不训练'],
      val: ['validation', '仅评测 · 永不训练'],
      mixed: ['split 混入', '已阻止训练回流'],
      unassigned: ['split 未分配', '已阻止训练回流'],
    }[split] || ['split 无效', '已阻止训练回流'];
  }
  function overallSplit(items) {
    const splits = new Set(items.map(item => item.split));
    if (!splits.size) return normalizeSplit(state?.config?.dataset_split);
    if (splits.size > 1 || splits.has('mixed')) return 'mixed';
    return [...splits][0];
  }
  function go(frame) {
    const input = byId('frameNumber');
    if (!input) return;
    input.value = String(frame);
    input.dispatchEvent(new Event('change', {bubbles: true}));
    queueHost.querySelector(`[data-batch-frame="${frame}"]`)?.scrollIntoView({behavior: 'smooth', block: 'nearest'});
  }
  function navigate(delta) {
    if (!visibleQueue.length) return;
    const current = Number(byId('frameNumber')?.value || 0);
    let index = visibleQueue.findIndex(item => item.frame === current);
    if (index < 0) index = delta > 0 ? -1 : 0;
    go(visibleQueue[(index + delta + visibleQueue.length) % visibleQueue.length].frame);
  }
  function jumpHard() {
    const queue = buildQueue().filter(item => item.hard > 0);
    if (!queue.length) { window.labelAgentToast?.('当前会话还没有标记疑难样本'); return; }
    const current = Number(byId('frameNumber')?.value || 0);
    const next = queue.find(item => item.frame > current) || queue[0];
    go(next.frame);
  }
  function queueItem(item) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `batchQueueItem${Number(byId('frameNumber')?.value || 0) === item.frame ? ' current' : ''}`;
    button.dataset.batchFrame = String(item.frame);
    const split = splitMeta(item.split);
    button.innerHTML = `<span class="batchQueueFrame">${item.frame}</span><span class="batchQueueCopy"><strong>${item.reason}</strong><small>${item.reviewed}/${item.total} 个候选已暂存 · ${split[1]}</small><span class="batchQueueMeta"><span class="${item.priority < 3 ? 'risk' : item.reviewed === item.total ? 'reviewed' : ''}">优先级 ${item.priority + 1}</span><span class="split-${item.split}">${split[0]}</span>${item.hard ? `<span>${item.hard} 个疑难</span>` : ''}</span></span>`;
    button.addEventListener('click', () => go(item.frame));
    return button;
  }
  function render() {
    if (!state) return;
    const queue = buildQueue();
    visibleQueue = queue.filter(matches);
    const totalCandidates = queue.reduce((sum, item) => sum + item.total, 0);
    const reviewedCandidates = queue.reduce((sum, item) => sum + item.reviewed, 0);
    const percent = totalCandidates ? Math.round(reviewedCandidates / totalCandidates * 100) : 0;
    const split = overallSplit(queue);
    const splitInfo = splitMeta(split);
    const badge = byId('batchSplitBadge');
    badge.className = `batchSplitBadge ${split}`;
    badge.textContent = splitInfo[0];
    badge.title = splitInfo[1];
    byId('batchProgressPercent').textContent = `${percent}%`;
    byId('batchProgressBar').setAttribute('aria-valuenow', String(percent));
    byId('batchProgressBar').querySelector('span').style.width = `${percent}%`;
    byId('batchProgressTitle').textContent = queue.length ? `${reviewedCandidates} / ${totalCandidates} 个候选已暂存复核` : '等待机器候选';
    const recovered = Object.keys(drafts()).length;
    byId('batchProgressDetail').textContent = recovered ? `已恢复本次浏览器会话的 ${recovered} 条复核草稿 · ${splitInfo[1]}` : `暂无会话草稿 · ${splitInfo[1]}`;
    byId('batchReviewContext').textContent = queue.length ? `${queue.length} 帧候选 · 未知类与冲突优先` : '当前数据没有机器候选';
    const riskFrames = queue.filter(item => item.priority < 3).length;
    const hardFrames = queue.filter(item => item.hard).length;
    const completedFrames = queue.filter(item => item.total && item.reviewed === item.total).length;
    byId('batchReviewStats').replaceChildren(
      stat('候选帧', queue.length), stat('高风险帧', riskFrames, riskFrames ? 'danger' : 'safe'),
      stat('未知类帧', queue.filter(item => item.unknown).length, 'warning'),
      stat('冲突 / QC', queue.filter(item => item.conflict || item.qcFailure).length, 'warning'),
      stat('疑难样本帧', hardFrames), stat('已暂存完成', completedFrames, completedFrames ? 'safe' : ''),
    );
    if (!visibleQueue.length) {
      queueHost.innerHTML = `<div class="batchReviewEmpty">${queue.length ? '当前筛选下没有候选帧' : '运行 LabelAgent 或导入候选后，这里会按风险自动排队。'}</div>`;
    } else queueHost.replaceChildren(...visibleQueue.map(queueItem));
    byId('batchPrevious').disabled = !visibleQueue.length;
    byId('batchNext').disabled = !visibleQueue.length;
    byId('batchJumpHard').disabled = !hardFrames;
  }
  async function refresh() {
    try {
      const response = await fetch('/api/state', {cache: 'no-store'});
      if (!response.ok) throw new Error('state unavailable');
      state = await response.json();
      panel.classList.remove('unavailable');
      render();
    } catch (_) {
      state = null; panel.classList.add('unavailable');
      byId('batchReviewContext').textContent = '状态不可用';
      queueHost.innerHTML = '<div class="batchReviewEmpty">缺少实时证据，批量队列已停止。</div>';
    }
  }
  panel.querySelectorAll('[data-batch-filter]').forEach(button => button.addEventListener('click', () => {
    activeFilter = button.dataset.batchFilter;
    panel.querySelectorAll('[data-batch-filter]').forEach(item => item.classList.toggle('selected', item === button));
    render();
  }));
  byId('batchPrevious').addEventListener('click', () => navigate(-1));
  byId('batchNext').addEventListener('click', () => navigate(1));
  byId('batchJumpHard').addEventListener('click', jumpHard);
  byId('batchReviewToggle').addEventListener('click', () => {
    const compact = panel.classList.toggle('compact');
    byId('batchReviewToggle').textContent = compact ? '展开队列' : '收起队列';
    byId('batchReviewToggle').setAttribute('aria-expanded', String(!compact));
    localStorage.setItem('labelagentBatchReviewCompact', compact ? '1' : '0');
  });
  document.addEventListener('labelagent:review-state', event => { state = event.detail?.state || state; render(); });
  document.addEventListener('labelagent:review-drafts-changed', render);
  byId('frameNumber')?.addEventListener('change', () => setTimeout(render, 100));
  byId('candidateConfidenceThreshold')?.addEventListener('change', render);
  document.addEventListener('keydown', event => {
    if (window.workspaceMode !== 'auto' || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target?.tagName) || document.querySelector('dialog[open]')) return;
    if (event.key.toLowerCase() === 'j') { event.preventDefault(); navigate(1); }
    if (event.key.toLowerCase() === 'k') { event.preventDefault(); navigate(-1); }
    if (event.key.toLowerCase() === 'h') { event.preventDefault(); jumpHard(); }
  });
  if (localStorage.getItem('labelagentBatchReviewCompact') === '1') {
    panel.classList.add('compact'); byId('batchReviewToggle').textContent = '展开队列'; byId('batchReviewToggle').setAttribute('aria-expanded', 'false');
  }
  refresh();
})();
