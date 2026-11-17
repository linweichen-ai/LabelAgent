(() => {
  const AUTO_VISIBLE_LABELS = '__auto_visible_labels__';
  const byId = id => document.getElementById(id);
  const dialog = byId('referencePrepDialog');
  if (!dialog) return;

  let latestState = null;
  let preparationGeneration = 0;
  let preparationStateReady = true;
  let draftBusy = false;
  let draftTimer = null;
  let draftStartedAt = 0;
  let draftLastPhaseKey = '';
  let draftPhaseChangedAt = 0;
  let draftProgressFailures = 0;
  let modelReadiness = null;
  let registeredSources = [];
  let sourceSwitchBusy = false;
  let recommendationTargetId = '';
  let discoveryPrefetch = {key: '', status: 'idle', completedAt: 0, samReady: false};
  const DISCOVERY_PREFETCH_REUSE_MS = 5 * 60 * 1000;
  const clone = value => JSON.parse(JSON.stringify(value));
  const currentFrame = () => Number.parseInt(byId('frameNumber')?.value || '0', 10) || 0;

  function syncEditorLabel(label) {
    const select = byId('labelSelect');
    if (!select || ![...select.options].some(item => item.value === label)) return false;
    select.value = label;
    select.dispatchEvent(new Event('change', {bubbles: true}));
    return select.value === label;
  }

  function existingAnnotationMessage(state, frameId) {
    return state?.truth_review_status?.[String(frameId)]?.reusable_as_reference === true
      ? '当前帧已有已核验人工真值，请直接复用或另选空白帧'
      : '当前帧已有历史标注，请选择“载入历史标注并复核”';
  }

  function option(select, value, label) {
    const item = document.createElement('option');
    item.value = String(value);
    item.textContent = label;
    select.append(item);
  }

  function fillFrameSelect(select, entries, emptyLabel, describe = null, preferredFrame = null) {
    select.replaceChildren();
    if (!entries.length) {
      option(select, '', emptyLabel);
      select.disabled = true;
      return;
    }
    select.disabled = false;
    for (const [frameId, items] of entries) {
      const classes = frameItemLabels(items);
      const detail = typeof describe === 'function' ? describe(frameId, items) : '';
      option(select, frameId, `帧 ${frameId} · ${frameItemCount(items)} 个轮廓${classes.length ? ` · ${classes.slice(0, 3).join(' / ')}` : ''}${detail}`);
    }
    if (preferredFrame !== null && entries.some(([frameId]) => String(frameId) === String(preferredFrame))) {
      select.value = String(preferredFrame);
    }
  }

  function frameItemCount(items) {
    return Array.isArray(items) ? items.length : Number(items?.count || 0);
  }

  function frameItemLabels(items) {
    return Array.isArray(items)
      ? [...new Set(items.map(item => item?.label).filter(Boolean))]
      : Array.isArray(items?.labels) ? [...new Set(items.labels.filter(Boolean))] : [];
  }

  function frameEntries(record) {
    return Object.entries(record || {})
      .filter(([, items]) => frameItemCount(items) > 0)
      .sort((a, b) => Number(a[0]) - Number(b[0]));
  }

  function currentFrameCandidate(state = latestState, frameId = currentFrame()) {
    const items = state?.candidates?.[String(frameId)] ?? state?.candidate_index?.[String(frameId)];
    return frameItemCount(items) > 0 ? items : null;
  }

  function hasTruthFrame(state, frameId) {
    const key = String(frameId);
    return Object.hasOwn(state?.truth || {}, key) || Object.hasOwn(state?.truth_index || {}, key);
  }

  function updateAutoDraftAction() {
    const button = byId('runReferenceAutoDraft');
    const hint = byId('fullPipelineHint');
    if (!button || !hint) return;
    const existing = currentFrameCandidate();
    const prefetchStatus = discoveryPrefetch.key === discoveryPrefetchKey()
      ? discoveryPrefetch.status : 'idle';
    if (draftBusy) {
      button.textContent = '全流程已开始 · 无需再点图片';
      hint.textContent = '本轮请求已经提交；请查看上方实时阶段和计时，不要重复点击。';
    } else if (existing) {
      button.textContent = `已有 ${frameItemCount(existing)} 个候选 · 直接载入复核`;
      hint.textContent = '将复用已有轮廓并轻量复审高风险候选类别，不重新分割；审核通过绑定校验后载入，仍须逐个人工检查。';
    } else {
      button.textContent = '立即运行整帧 SAM + VLM 全流程';
      hint.textContent = prefetchStatus === 'warming'
        ? '正在后台预备当前图片的 VLM 类别识别，随后会在空闲时预备整图 SAM；现在直接运行也会合并相同请求。'
        : prefetchStatus === 'ready'
          ? discoveryPrefetch.samReady
            ? '当前图片的 VLM 类别识别与整图 SAM 已预备；缓存仍有效时，正式运行会直接复用，再继续几何过滤与 Agent。'
            : '当前图片的 VLM 类别识别已预备；SAM 当时忙碌或不可预热，正式运行会直接复用类别结果后继续分割。'
          : '无需再点图片；系统会自动执行 VLM 识别 → SAM 分割 → Agent 修正 → VLM 复核，通常约 1～3 分钟。';
    }
  }

  function updateRecommendation() {
    const text = byId('referencePrepRecommendationText');
    const button = byId('referenceRecommendedAction');
    if (!text || !button) return;
    const state = latestState;
    const loaded = preparationStateReady && Boolean(state?.loaded && state?.topic);
    const truthEntries = frameEntries(state?.truth_index || state?.truth);
    const truthStatus = state?.truth_review_status || {};
    const verified = truthEntries.filter(([frameId]) => truthStatus[frameId]?.reusable_as_reference === true);
    const legacy = truthEntries.filter(([frameId]) => truthStatus[frameId]?.reusable_as_reference !== true);
    const candidates = frameEntries(state?.candidate_index || state?.candidates).filter(([frameId]) => !hasTruthFrame(state, frameId));
    const currentCandidate = candidates.find(([frameId]) => String(frameId) === String(currentFrame()));
    let source = '', message = '请先加载图像或 MCAP，随后会自动选择最省事的参考帧路线。';
    let target = '', actionLabel = '按推荐继续';
    if (loaded && verified.length) {
      source = 'existing'; target = 'useExistingTruth'; actionLabel = '直接采用';
      message = `最快：直接采用 ${verified.length} 帧已核验人工真值中的一帧，无需重画。`;
    } else if (loaded && legacy.length) {
      source = 'existing'; target = 'useLegacyTruthDraft'; actionLabel = '载入并复核';
      message = `优先复用：当前数据有 ${legacy.length} 帧历史标注，载入后检查边界即可，不必从零画。`;
    } else if (loaded && currentCandidate) {
      source = 'sam'; target = 'useCandidateDraft'; actionLabel = '复用当前帧候选';
      message = `当前帧已有 ${frameItemCount(currentCandidate[1])} 个机器候选，可复用并轻量复审高风险类别；不会重新运行 SAM，之后仍须逐个检查。`;
    } else if (loaded && registeredSources.length) {
      source = 'existing'; target = 'loadRegisteredAnnotationSource'; actionLabel = '切换并复核';
      message = `当前帧 ${currentFrame()} 没有已核验真值或机器候选；另有 ${registeredSources.length} 份含标注数据可先切换复核${candidates.length ? `，当前数据另有 ${candidates.length} 帧机器候选也可在下方选择` : ''}。`;
    } else if (loaded && candidates.length) {
      source = 'sam'; target = 'useCandidateDraft'; actionLabel = '复用机器候选';
      message = `当前帧 ${currentFrame()} 没有候选；当前数据另有 ${candidates.length} 帧机器候选，可在下方选择一帧作为可编辑草稿。`;
    } else if (loaded && modelReadiness?.ready && byId('referenceAutoLabel')?.value) {
      source = 'sam'; target = 'runReferenceAutoDraft'; actionLabel = 'SAM + VLM 一键起稿';
      message = '推荐自动起稿：SAM 生成轮廓，VLM 判断类别与漏误标，Agent 修正后交给您确认。';
    } else if (loaded) {
      source = 'manual'; target = 'startManualReference'; actionLabel = '开始手工绘制';
      message = '模型或可复用标注暂不可用，先用多边形绘制，保存前仍会经过人工门禁。';
    }
    recommendationTargetId = target;
    text.textContent = message;
    button.textContent = actionLabel;
    button.disabled = !target || draftBusy || sourceSwitchBusy;
    document.querySelectorAll('.referenceSourceCard').forEach(card => card.classList.toggle('recommended', card.dataset.referenceSource === source));
  }

  function renderAnnotationSources(value) {
    registeredSources = Array.isArray(value?.sources) ? value.sources : [];
    const select = byId('registeredAnnotationSource');
    const button = byId('loadRegisteredAnnotationSource');
    const detail = byId('registeredAnnotationSourceDetail');
    const selected = select.value;
    select.replaceChildren();
    if (!registeredSources.length) {
      option(select, '', '暂无其他含标注的数据');
      select.disabled = true;
    } else {
      select.disabled = sourceSwitchBusy;
      for (const source of registeredSources) {
        const counts = source.legacy_source_counts || {};
        option(select, source.id, `${source.name} · ${source.frame_count} 帧 / ${source.shape_count} 个轮廓 · 人工来源 ${counts.manual || 0} / 自动来源 ${counts.auto || 0}`);
      }
      if (registeredSources.some(source => source.id === selected)) select.value = selected;
    }
    const active = registeredSources.find(source => source.id === select.value);
    if (active) {
      const counts = active.legacy_source_counts || {};
      detail.textContent = `来源构成：人工来源 ${counts.manual || 0}、自动来源 ${counts.auto || 0}、未知来源 ${counts.unknown || 0}。这里只是历史来源记录，不等于当前人工真值；载入后仍须人工复核。`;
    } else {
      detail.textContent = '切换前会保存当前文件的候选；外部标注仍按历史草稿处理，必须重新人工确认。';
    }
    button.disabled = !preparationStateReady || sourceSwitchBusy || draftBusy || !registeredSources.length;
    button.textContent = sourceSwitchBusy ? '正在安全切换…' : '切换并查看已有标注';
    updateRecommendation();
  }

  async function readAnnotationSources(generation = preparationGeneration) {
    const response = await fetch('/api/files/annotation-sources', {cache: 'no-store'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '无法读取其他已有标注数据');
    if (generation !== preparationGeneration) return data;
    renderAnnotationSources(data);
    return data;
  }

  function updateModelButtons() {
    const loaded = Boolean(latestState?.loaded && latestState?.topic);
    const label = byId('referenceAutoLabel')?.value;
    const existing = currentFrameCandidate();
    byId('runReferenceAutoDraft').disabled = draftBusy || !loaded || (!existing && (!label || !modelReadiness?.ready));
    byId('startSamDraft').disabled = draftBusy || !loaded || !modelReadiness?.sam?.ready;
    updateAutoDraftAction();
  }

  function readinessText(service, kind) {
    if (!service) return '检查中';
    if (service.ready && kind === 'sam') return service.model_loaded ? '可用 · 已加载' : '可用 · 按需加载';
    if (service.ready) return `模型匹配 · ${service.model || '可用'}`;
    return service.error || '不可用';
  }

  function renderReadiness(value) {
    modelReadiness = value;
    for (const [id, kind] of [['referenceSamReadiness', 'sam'], ['referenceVlmReadiness', 'vlm']]) {
      const element = byId(id);
      const service = value?.[kind];
      element.classList.toggle('ready', Boolean(service?.ready));
      element.classList.toggle('blocked', Boolean(service && !service.ready));
      const detail = readinessText(service, kind);
      element.querySelector('small').textContent = detail;
      element.title = detail;
    }
    byId('referenceModelReadiness').classList.toggle('checking', !value);
    updateModelButtons();
    updateRecommendation();
  }

  async function readModelReadiness(generation = preparationGeneration) {
    if (generation === preparationGeneration) renderReadiness(null);
    const response = await fetch('/api/model_readiness', {cache: 'no-store'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '模型就绪检查失败');
    if (generation === preparationGeneration) renderReadiness(data);
    return data;
  }

  function discoveryPrefetchKey(state = latestState, frameId = currentFrame()) {
    if (!state?.loaded || !state?.path || !state?.topic || !(state.labels || []).length) return '';
    return JSON.stringify([state.path, state.topic, Number(frameId), state.labels]);
  }

  async function maybePrefetchVisibleDiscovery(generation = preparationGeneration) {
    const state = latestState;
    const frameId = currentFrame();
    const key = discoveryPrefetchKey(state, frameId);
    if (generation !== preparationGeneration || !preparationStateReady || !modelReadiness?.ready
        || byId('referenceAutoLabel')?.value !== AUTO_VISIBLE_LABELS || !key
        || hasTruthFrame(state, frameId) || currentFrameCandidate(state, frameId)) return;
    if (discoveryPrefetch.key === key && discoveryPrefetch.status === 'warming') return;
    if (discoveryPrefetch.key === key && discoveryPrefetch.status === 'ready'
        && Date.now() - discoveryPrefetch.completedAt < DISCOVERY_PREFETCH_REUSE_MS) return;
    discoveryPrefetch = {key, status: 'warming', completedAt: 0, samReady: false};
    updateAutoDraftAction();
    try {
      const response = await fetch('/api/reference_discovery/prefetch', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({frame: frameId, expected_path: state.path,
          expected_topic: state.topic}),
      });
      const data = await response.json();
      if (generation !== preparationGeneration || discoveryPrefetch.key !== key) return;
      discoveryPrefetch = response.ok && data.prefetched === true
        && data.schema_version === 'labelagent.reference-discovery-prefetch.v1'
        && data.input_sha256 && data.discovery_sha256
        ? {key, status: 'ready', completedAt: Date.now(),
          samReady: data.sam_prefetch?.prefetched === true}
        : {key, status: 'idle', completedAt: 0, samReady: false};
    } catch (_) {
      if (generation === preparationGeneration && discoveryPrefetch.key === key) {
        discoveryPrefetch = {key, status: 'idle', completedAt: 0, samReady: false};
      }
    } finally {
      if (generation === preparationGeneration) updateAutoDraftAction();
    }
  }

  function render(state) {
    latestState = state;
    const loaded = Boolean(state?.loaded && state?.topic);
    const truthEntries = frameEntries(state?.truth_index || state?.truth);
    const truthStatus = state?.truth_review_status || {};
    const verifiedTruthEntries = truthEntries.filter(([frameId]) => truthStatus[frameId]?.reusable_as_reference === true);
    const legacyTruthEntries = truthEntries.filter(([frameId]) => truthStatus[frameId]?.reusable_as_reference !== true);
    const candidateEntries = frameEntries(state?.candidate_index || state?.candidates)
      .filter(([frameId]) => !hasTruthFrame(state, frameId));
    const preferredFrame = currentFrame();
    fillFrameSelect(byId('existingTruthFrame'), verifiedTruthEntries, '暂无通过当前门禁的人工真值',
      () => ' · 人工回执已核验', preferredFrame);
    fillFrameSelect(byId('legacyTruthFrame'), legacyTruthEntries, '暂无需要重新确认的历史标注', frameId => {
      const counts = truthStatus[frameId]?.legacy_source_counts || {};
      const parts = [];
      if (counts.manual) parts.push(`来源人工 ${counts.manual}`);
      if (counts.auto) parts.push(`来源自动 ${counts.auto}`);
      if (counts.unknown) parts.push(`来源未知 ${counts.unknown}`);
      return parts.length ? ` · ${parts.join(' / ')}` : ' · 未绑定当前人工回执';
    }, preferredFrame);
    fillFrameSelect(byId('candidateReferenceFrame'), candidateEntries, '当前数据没有可精修的机器候选', null, preferredFrame);
    const labelSelect = byId('referenceAutoLabel');
    const selectedLabel = labelSelect.value || AUTO_VISIBLE_LABELS;
    labelSelect.replaceChildren();
    option(labelSelect, AUTO_VISIBLE_LABELS, '自动识别当前帧全部可见类别（推荐）');
    for (const label of state?.labels || []) option(labelSelect, label, label);
    labelSelect.value = [...labelSelect.options].some(item => item.value === selectedLabel)
      ? selectedLabel : AUTO_VISIBLE_LABELS;
    byId('useExistingTruth').disabled = !loaded || !verifiedTruthEntries.length;
    byId('useLegacyTruthDraft').disabled = !loaded || !legacyTruthEntries.length;
    byId('useCandidateDraft').disabled = !loaded || !candidateEntries.length;
    updateModelButtons();
    byId('startManualReference').disabled = !loaded;
    const reference = state?.reference_frames?.[state?.topic]?.[0];
    const frameId = currentFrame();
    byId('referencePrepCurrent').textContent = loaded
      ? `当前 ${state.topic} · 帧 ${frameId}${reference === undefined ? ' · 尚未设置参考帧' : ` · 已有参考帧 ${reference}`}`
      : '尚未加载图像或 MCAP 数据';
    byId('referencePrepSummary').innerHTML = loaded
      ? `<strong>已找到 ${verifiedTruthEntries.length} 帧已核验人工真值、${legacyTruthEntries.length} 帧历史标注、${candidateEntries.length} 帧机器候选。</strong> 历史/混合来源标注可省去重画，但仍要重新人工确认。`
      : '<strong>请先加载图像或 MCAP。</strong> 数据载入后，这里会自动列出可复用的人工标注和机器候选。';
    updateRecommendation();
  }

  async function readPreparationState() {
    const response = await fetch('/api/reference-preparation', { cache: 'no-store' });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '无法读取参考帧目录');
    if (data.schema_version !== 'labelagent.reference-preparation.v1' || data.compact_geometry !== true) {
      throw new Error('参考帧目录版本不兼容，已停止准备');
    }
    return data;
  }

  async function readFrameSnapshot(frameId, expected = latestState) {
    if (typeof window.readCompactFrameStateSnapshot !== 'function') throw new Error('单帧按需读取接口不可用，请刷新页面');
    if (!expected?.loaded || !expected?.topic || !expected?.path) throw new Error('请先加载图像或 MCAP');
    const data = await window.readCompactFrameStateSnapshot(frameId, expected.topic, expected.path);
    if (data.path !== expected.path || data.topic !== expected.topic || Number(data.frame) !== Number(frameId)) {
      throw new Error('单帧内容与参考目录不一致，已拒绝采用');
    }
    return data;
  }

  async function openDialog() {
    const generation = ++preparationGeneration;
    preparationStateReady = false;
    latestState = null;
    const dependentButtons = ['useExistingTruth', 'useLegacyTruthDraft', 'useCandidateDraft', 'runReferenceAutoDraft', 'startSamDraft', 'startManualReference', 'loadRegisteredAnnotationSource'];
    for (const id of dependentButtons) byId(id).disabled = true;
    byId('referencePrepSummary').textContent = '正在检查已有人工标注和机器候选……';
    renderReadiness(null);
    dialog.showModal();
    const [stateResult, readinessResult, sourceResult] = await Promise.allSettled([
      readPreparationState().then(state => {
        if (generation !== preparationGeneration) return state;
        preparationStateReady = true;
        render(state);
        return state;
      }, error => {
        if (generation === preparationGeneration) {
          latestState = null;
          byId('referencePrepSummary').innerHTML = `<strong>状态读取失败，已停止准备。</strong> ${error.message}`;
          for (const id of dependentButtons) byId(id).disabled = true;
        }
        throw error;
      }), readModelReadiness(generation), readAnnotationSources(generation),
    ]);
    if (generation !== preparationGeneration) return;
    if (readinessResult.status === 'rejected') {
      renderReadiness({ready: false, sam: {ready: false, error: '就绪检查失败'}, vlm: {ready: false, error: readinessResult.reason.message}});
    }
    if (sourceResult.status === 'rejected') {
      renderAnnotationSources({sources: []});
      byId('registeredAnnotationSource').title = sourceResult.reason.message;
    }
    if (stateResult.status === 'rejected' || !preparationStateReady) {
      preparationStateReady = false;
      for (const id of dependentButtons) byId(id).disabled = true;
    }
    if (stateResult.status === 'fulfilled' && readinessResult.status === 'fulfilled'
        && preparationStateReady) void maybePrefetchVisibleDiscovery(generation);
  }

  function setDraftBusy(busy, title = '') {
    draftBusy = busy;
    const progress = byId('referenceDraftProgress');
    progress.hidden = !busy;
    dialog.setAttribute('aria-busy', String(busy));
    if (title) byId('referenceDraftProgressTitle').textContent = title;
    updateAutoDraftAction();
    byId('runReferenceAutoDraft').setAttribute('aria-busy', String(busy));
    for (const id of ['useExistingTruth', 'useLegacyTruthDraft', 'useCandidateDraft', 'runReferenceAutoDraft', 'startSamDraft', 'startManualReference', 'referencePrepClose', 'refreshReferenceReadiness', 'loadRegisteredAnnotationSource']) {
      byId(id).disabled = busy;
    }
    byId('referencePrepCancel').textContent = busy ? '停止生成' : '暂不准备';
    if (!busy && draftTimer) { clearInterval(draftTimer); draftTimer = null; }
    if (!busy) {
      draftLastPhaseKey = '';
      draftPhaseChangedAt = 0;
      draftProgressFailures = 0;
    }
    if (!busy) updateModelButtons();
    if (!busy) renderAnnotationSources({sources: registeredSources});
    updateRecommendation();
    if (busy) requestAnimationFrame(() => {
      progress.scrollIntoView({behavior: 'smooth', block: 'center'});
      progress.focus({preventScroll: true});
    });
  }

  async function updateDraftProgress() {
    const elapsed = Math.max(0, Math.round((Date.now() - draftStartedAt) / 1000));
    byId('referenceDraftElapsed').textContent = `${elapsed} 秒`;
    try {
      const response = await fetch('/api/run/status', {cache: 'no-store'});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const status = await response.json();
      draftProgressFailures = 0;
      const statusStartedAt = Number(status.started_at || 0) * 1000;
      if (!Number.isFinite(statusStartedAt) || statusStartedAt < draftStartedAt - 2000) {
        draftLastPhaseKey = `awaiting_current_run|${draftStartedAt}`;
        draftPhaseChangedAt = draftStartedAt;
        byId('referenceDraftProgressTitle').textContent = '请求已提交 · 等待后台确认本轮任务';
        byId('referenceDraftProgressDetail').textContent = '上一轮运行状态已忽略；本轮 SAM + VLM 请求正在进入后台，请不要重复点击。';
        return;
      }
      const phaseKey = [status.status, status.phase, status.round, status.shape_count].join('|');
      if (phaseKey !== draftLastPhaseKey) {
        draftLastPhaseKey = phaseKey;
        draftPhaseChangedAt = Date.now();
      }
      const unchanged = Math.max(0, Math.round((Date.now() - draftPhaseChangedAt) / 1000));
      const phaseNames = {discovering: 'VLM 正在识别当前帧可见类别', discovering_retry: 'VLM 首次标签列表不完整 · 正在自动重试', describing: 'VLM 正在理解类别并生成提示词', generating: 'SAM 正在生成初始轮廓',
        observing: 'VLM 正在检查候选', editing: 'Agent 正在修正候选', edited: 'Agent 已完成一轮修正', review_deferred: '初始候选已生成', agent_refining: 'Agent 正在后台精修候选',
        finished: 'Agent 候选已生成', auditing: 'VLM 正在复核高风险候选', committing: '正在安全保存机器候选', completed: '草稿生成完成', stopping: '正在安全停止',
        stopped: '草稿生成已停止', failed: '草稿生成失败'};
      const phase = status.phase || status.status || '准备中';
      byId('referenceDraftProgressTitle').textContent = phaseNames[phase] || 'SAM + VLM + Agent 正在运行';
      const details = [`当前阶段：${phase}`, `第 ${status.round || 0} 轮`, `${status.shape_count || 0} 个候选`];
      if (phase === 'auditing' && Number(status.audit_total || 0) > 0) details.push(`高风险复核 ${status.audit_completed || 0}/${status.audit_total}`);
      if (status.error) details.push(`原因：${status.error}`);
      else if (elapsed >= 90 && unchanged >= 60) details.push('该阶段仍在计算，并非没有点击；可用下方“停止生成”安全取消');
      else details.push('首次加载模型通常需要 30～90 秒');
      byId('referenceDraftProgressDetail').textContent = `${details.join(' · ')}。`;
    } catch (_) {
      draftProgressFailures += 1;
      if (draftProgressFailures >= 3) {
        byId('referenceDraftProgressTitle').textContent = '暂时无法确认后台进度';
        byId('referenceDraftProgressDetail').textContent = '进度接口连续读取失败，请不要重复点击；可用下方“停止生成”安全结束后重试。';
      }
    }
  }

  async function runAutoDraft() {
    const frameId = currentFrame();
    const label = byId('referenceAutoLabel').value;
    const autoDetectVisible = label === AUTO_VISIBLE_LABELS;
    const fresh = await readPreparationState();
    if (!fresh.loaded) throw new Error('请先加载图像或 MCAP');
    if (latestState && (fresh.path !== latestState.path || fresh.topic !== latestState.topic)) throw new Error('数据作用域已变化，请重新打开参考帧准备');
    if (hasTruthFrame(fresh, frameId)) throw new Error(existingAnnotationMessage(fresh, frameId));
    const existing = currentFrameCandidate(fresh, frameId);
    if (existing) {
      const frameState = await readFrameSnapshot(frameId, fresh);
      const candidate = frameState.candidates?.[String(frameId)];
      if (!Array.isArray(candidate) || !candidate.length) throw new Error('当前帧候选已发生变化，请重新打开参考帧准备');
      window.syncWorkbenchCandidateState?.(frameState);
      dialog.close();
      await window.prepareReferenceCandidateDraft?.(frameId, clone(candidate), clone(frameState.agent_runs?.[String(frameId)] || {}));
      window.labelAgentToast?.(`当前帧已有 ${candidate.length} 个机器候选，已直接载入；已完成高风险类别轻量复审，未重新分割，未写入人工真值，仍须逐个检查`);
      return;
    }
    if (!autoDetectVisible && (!label || !(fresh.labels || []).includes(label))) throw new Error('请选择有效的参考类别');
    if (autoDetectVisible && !(fresh.labels || []).length) throw new Error('当前标签表为空，无法自动识别');
    const readiness = await readModelReadiness();
    if (!readiness.ready) throw new Error(`模型尚未就绪：SAM ${readinessText(readiness.sam, 'sam')}；VLM ${readinessText(readiness.vlm, 'vlm')}`);
    draftStartedAt = Date.now();
    draftLastPhaseKey = '';
    draftPhaseChangedAt = draftStartedAt;
    draftProgressFailures = 0;
    setDraftBusy(true, autoDetectVisible ? '请求已提交 · VLM 正在识别可见类别…' : '请求已提交 · 正在启动 SAM 与 VLM…');
    byId('referenceDraftProgressDetail').textContent = autoDetectVisible
      ? '后台已开始：先识别当前帧可见类别，再依次运行 SAM、Agent 与高风险候选复核；当前 640×512 流程通常约 30～90 秒，请不要重复点击。'
      : '后台已确认开始一键起稿；将依次运行 SAM、VLM 与 Agent，当前 640×512 流程通常约 20～60 秒，请不要重复点击。';
    draftTimer = setInterval(updateDraftProgress, 1000);
    await updateDraftProgress();
    try {
      const response = await fetch('/api/reference_draft', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(autoDetectVisible
          ? {frame: frameId, auto_detect_visible: true, defer_agent_review: true, defer_class_audit: true}
          : {frame: frameId, labels: [label], defer_agent_review: true, defer_class_audit: true}),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || '参考帧机器草稿生成失败');
      if (result.stopped) throw new Error('已停止生成，未写入人工真值');
      if (!result.candidate_only || result.human_truth !== false) throw new Error('服务器未明确返回 candidate_only，已停止采用');
      if (!Array.isArray(result.candidates) || !result.candidates.length) throw new Error('SAM/VLM 没有生成可精修轮廓');
      const readback = await window.fetchCompactFrameState?.(frameId);
      if (!readback) throw new Error('单帧候选读回接口不可用，已停止载入');
      const persistedCandidates = readback.candidates?.[String(frameId)];
      const canonical = value => Array.isArray(value) ? '[' + value.map(canonical).join(',') + ']'
        : value && typeof value === 'object' ? '{' + Object.keys(value).sort().map(key => JSON.stringify(key) + ':' + canonical(value[key])).join(',') + '}' : JSON.stringify(value);
      if (currentFrame() !== frameId || readback.path !== fresh.path || readback.topic !== fresh.topic
          || !Array.isArray(persistedCandidates) || canonical(persistedCandidates) !== canonical(result.candidates)
          || canonical(readback.agent_runs?.[String(frameId)]) !== canonical(result.agent)
          || !result.candidate_artifact_sha256
          || readback.candidate_review_bindings?.[String(frameId)]?.candidate_artifact_sha256 !== result.candidate_artifact_sha256) {
        throw new Error('服务器读回的候选内容或数据绑定与本次结果不一致，已停止载入');
      }
      setDraftBusy(false);
      dialog.close();
      await window.prepareReferenceCandidateDraft?.(frameId, clone(persistedCandidates), clone(readback.agent_runs?.[String(frameId)] || {}));
      const visibleLabels = Array.isArray(result.labels) ? result.labels : [];
      const labelSynced = visibleLabels.length === 1 && syncEditorLabel(visibleLabels[0]);
      const uncertain = result.visible_label_discovery?.uncertain_labels || [];
      const unknown = result.visible_label_discovery?.unknown_candidates || [];
      const conflicts = result.visible_label_discovery?.registry_conflicts || [];
      const reviewHint = [...(uncertain.length ? [`${uncertain.length} 类不确定`] : []),
        ...(unknown.length ? [`${unknown.length} 个疑似新类`] : []),
        ...(conflicts.length ? [`${conflicts.length} 个标签别名冲突`] : [])].join('、');
      window.labelAgentToast?.(`已生成 ${result.candidates.length} 个机器草稿${visibleLabels.length ? `（${visibleLabels.join('、')}）` : ''}；${labelSynced ? '编辑类别已同步，' : ''}${reviewHint ? `${reviewHint}已送人工复核，` : ''}${result.agent_review_deferred ? 'Agent 正在后台精修，可先检查轮廓；' : result.candidate_class_audit_deferred ? '高风险类别正在后台复核，可先检查轮廓；' : ''}请精修后人工确认`);
    } catch (error) {
      setDraftBusy(false);
      const progress = byId('referenceDraftProgress');
      progress.hidden = false;
      byId('referenceDraftProgressTitle').textContent = '本次没有生成候选 · 可以重新尝试';
      byId('referenceDraftProgressDetail').textContent = `失败原因：${error.message}。未写入人工真值或参考帧；请按提示调整类别后重新运行。`;
      progress.scrollIntoView({behavior: 'smooth', block: 'center'});
      progress.focus({preventScroll: true});
      throw error;
    } finally {
      if (draftBusy) setDraftBusy(false);
    }
  }

  async function useExisting() {
    const frameId = Number(byId('existingTruthFrame').value);
    const index = await readPreparationState();
    const fresh = await readFrameSnapshot(frameId, index);
    const truth = fresh.truth?.[String(frameId)];
    if (!Array.isArray(truth) || !truth.length) throw new Error('所选人工标注已发生变化，请重新选择');
    if (fresh.truth_review_status?.[String(frameId)]?.reusable_as_reference !== true) {
      throw new Error('该帧缺少与当前轮廓绑定的人工确认回执，请改用“载入历史标注并复核”');
    }
    const response = await fetch('/api/reference_frame', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({frame: frameId, expected_topic: index.topic, expected_path: index.path}),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '设置参考帧失败');
    const readback = await readFrameSnapshot(frameId, index);
    if (readback.reference_frames?.[index.topic]?.[0] !== frameId) throw new Error('服务器读回未确认参考帧，已停止');
    window.referenceFrames = readback.reference_frames || {};
    dialog.close();
    await window.openFrameForReferencePreparation?.(frameId);
    document.dispatchEvent(new CustomEvent('labelagent:reference-updated', { detail: { frame: frameId, source: 'existing_human_truth' } }));
    window.labelAgentToast?.(`已采用帧 ${frameId} 的人工真值作为参考帧`);
  }

  async function useLegacyTruth() {
    const frameId = Number(byId('legacyTruthFrame').value);
    const index = await readPreparationState();
    const fresh = await readFrameSnapshot(frameId, index);
    const truth = fresh.truth?.[String(frameId)];
    const status = fresh.truth_review_status?.[String(frameId)] || {};
    if (!Array.isArray(truth) || !truth.length) throw new Error('所选历史标注已发生变化，请重新选择');
    if (status.reusable_as_reference === true) throw new Error('该帧已经通过当前人工门禁，请直接采用已核验真值');
    dialog.close();
    await window.prepareReferenceLegacyTruthDraft?.(frameId, clone(truth), clone(status));
  }

  async function useCandidateFrame(frameId) {
    const index = await readPreparationState();
    const fresh = await readFrameSnapshot(frameId, index);
    if (Object.hasOwn(fresh.truth || {}, String(frameId))) throw new Error(existingAnnotationMessage(fresh, frameId));
    const candidate = fresh.candidates?.[String(frameId)];
    if (!Array.isArray(candidate) || !candidate.length) throw new Error('所选机器候选已发生变化，请重新选择');
    window.syncWorkbenchCandidateState?.(fresh);
    dialog.close();
    const agentRun = fresh.agent_runs?.[String(frameId)] || {};
    await window.prepareReferenceCandidateDraft?.(frameId, clone(candidate), clone(agentRun));
  }

  async function useCandidate() {
    return useCandidateFrame(Number(byId('candidateReferenceFrame').value));
  }

  async function startSam() {
    const frameId = currentFrame();
    const fresh = await readPreparationState();
    if (!fresh.loaded) throw new Error('请先加载图像或 MCAP');
    if (hasTruthFrame(fresh, frameId)) throw new Error(existingAnnotationMessage(fresh, frameId));
    const readiness = await readModelReadiness();
    if (!readiness.sam?.ready) throw new Error(`SAM 尚未就绪：${readinessText(readiness.sam, 'sam')}`);
    dialog.close();
    await window.prepareReferenceSamDraft?.(frameId);
  }

  async function startManual() {
    const frameId = currentFrame();
    const fresh = await readPreparationState();
    if (!fresh.loaded) throw new Error('请先加载图像或 MCAP');
    if (hasTruthFrame(fresh, frameId)) throw new Error(existingAnnotationMessage(fresh, frameId));
    dialog.close();
    await window.prepareReferenceManualDraft?.(frameId);
  }

  async function loadRegisteredAnnotationSource() {
    const fileId = byId('registeredAnnotationSource').value;
    const source = registeredSources.find(item => item.id === fileId);
    if (!source) throw new Error('请选择仍然可用的已有标注数据');
    if (typeof window.loadVideoData !== 'function') throw new Error('当前页面版本不支持安全切换，请刷新后重试');
    sourceSwitchBusy = true;
    renderAnnotationSources({sources: registeredSources});
    try {
      const response = await fetch(`/api/files/${encodeURIComponent(fileId)}/load`, {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}',
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || '切换已有标注数据失败');
      await window.loadVideoData(data);
      const [freshState, freshSources] = await Promise.all([readPreparationState(), readAnnotationSources()]);
      render(freshState);
      renderAnnotationSources(freshSources);
      window.labelAgentToast?.(`已切换到 ${source.name}；请选择历史帧并复核，不需要从零重画`);
    } finally {
      sourceSwitchBusy = false;
      renderAnnotationSources({sources: registeredSources});
    }
  }

  function safely(action) {
    return async () => {
      try { await action(); }
      catch (error) { window.labelAgentToast?.(error.message); if (!dialog.open) await openDialog(); }
    };
  }

  async function startAutoDraftFromGuide() {
    if (draftBusy) throw new Error('当前机器草稿正在生成，请等待本轮完成');
    if (!dialog.open) await openDialog();
    const fresh = await readPreparationState();
    const frameId = currentFrame();
    const currentCandidate = currentFrameCandidate(fresh, frameId);
    if (currentCandidate) {
      await useCandidateFrame(frameId);
      window.labelAgentToast?.(`当前帧已有 ${frameItemCount(currentCandidate)} 个机器候选，已轻量复审并载入；未重新运行 SAM，仍需逐个检查`);
      return;
    }
    await runAutoDraft();
  }

  byId('useExistingTruth').addEventListener('click', safely(useExisting));
  byId('useLegacyTruthDraft').addEventListener('click', safely(useLegacyTruth));
  byId('useCandidateDraft').addEventListener('click', safely(useCandidate));
  byId('loadRegisteredAnnotationSource').addEventListener('click', safely(loadRegisteredAnnotationSource));
  byId('referenceRecommendedAction').addEventListener('click', () => {
    const target = recommendationTargetId && byId(recommendationTargetId);
    if (target && !target.disabled) target.click();
  });
  byId('registeredAnnotationSource').addEventListener('change', () => renderAnnotationSources({sources: registeredSources}));
  byId('runReferenceAutoDraft').addEventListener('click', safely(runAutoDraft));
  byId('startSamDraft').addEventListener('click', safely(startSam));
  byId('referenceAutoLabel').addEventListener('change', () => {
    updateModelButtons();
    if (byId('referenceAutoLabel').value === AUTO_VISIBLE_LABELS) {
      void maybePrefetchVisibleDiscovery(preparationGeneration);
    }
  });
  byId('refreshReferenceReadiness').addEventListener('click', safely(readModelReadiness));
  byId('startManualReference').addEventListener('click', safely(startManual));
  byId('referencePrepClose').addEventListener('click', () => { if (!draftBusy) dialog.close(); });
  byId('referencePrepCancel').addEventListener('click', async () => {
    if (!draftBusy) { dialog.close(); return; }
    try {
      const response = await fetch('/api/run/stop', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
      if (!response.ok) throw new Error('停止请求失败');
      byId('referenceDraftProgressTitle').textContent = '正在安全停止…';
    } catch (error) { window.labelAgentToast?.(error.message); }
  });
  dialog.addEventListener('cancel', event => { event.preventDefault(); if (!draftBusy) dialog.close(); });
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  window.openReferencePreparation = openDialog;
  window.loadCurrentFrameCandidateForReference = frameId => useCandidateFrame(Number(frameId));
  window.startReferenceAutoDraft = safely(startAutoDraftFromGuide);
})();
