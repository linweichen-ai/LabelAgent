(() => {
  const byId = id => document.getElementById(id);
  const guide = byId('workflowGuide');
  if (!guide) return;

  const title = byId('guideTitle');
  const detail = byId('guideDetail');
  const action = byId('guideAction');
  const autoDraftAction = byId('guideAutoDraft');
  const badges = {
    data: byId('guideDataBadge'),
    candidate: byId('guideCandidateBadge'),
    truth: byId('guideTruthBadge'),
  };
  const steps = Object.fromEntries(
    [...guide.querySelectorAll('[data-guide-step]')].map(node => [node.dataset.guideStep, node]),
  );
  let latest = { action: 'load', state: null, job: null };
  let autoDraftSubmitting = false;
  let guidePollTimer = null;
  let guideRefreshBusy = false;
  let guideStatusRefreshPending = false;
  let latestRunStatus = 'idle';
  const tutorial = byId('tutorialCoach');
  const tutorialSteps = [
    {
      target: '.workspaceModes', label: '模式选择',
      title: '先认识两种工作模式',
      text: '“人工标注 / 真值”用于手工绘制和确认；“自动标注 / 候选”用于查看 SAM、VLM 和 Agent 的机器结果。',
      tip: '机器候选不会因为显示在画面上就自动成为人工真值。',
    },
    {
      target: '.playbackRight', label: '相机帧号',
      title: '选择需要处理的图像帧',
      text: '右侧数字是当前帧号。输入数字后按 Tab 或点击输入框外，画面才会真正切换。',
      tip: '帧号和“本次处理帧数”是两个不同的输入框。',
    },
    {
      target: '#workflowGuide', label: '状态向导',
      title: '跟随顶部状态向导',
      text: '向导会根据数据、参考帧、候选和真值的实际状态，告诉你下一步该做什么。',
      tip: '状态缺失或接口异常时会显示“未知”，不会冒充已完成。',
    },
    {
      target: '#prepareReference', label: '参考帧准备', mode: 'auto',
      title: '参考帧不必从零手画',
      text: '优先复用已核验标注或当前帧机器候选；没有合适候选时，也可以直接运行整帧 SAM + VLM + Agent，得到可编辑草稿。',
      tip: '机器草稿不会自动成为真值，最后仍要由你检查边界和类别并确认。',
    },
    {
      target: '#pointButton', label: '单目标 SAM', mode: 'manual',
      title: '点提示有“进入模式”和“真正提交”两步',
      text: '点击“SAM 点选”只会进入点选模式；随后还要在图片里的目标内部点一下。出现绿色提示点和“正在生成”计时，才代表请求已经提交。',
      tip: '如果既没有绿色点也没有计时，就是尚未开始，不是模型速度慢。整帧起稿则点击一次就会直接运行。',
    },
    {
      target: '#truth', label: '候选精修', mode: 'manual',
      title: '机器候选载入后按风险逐个检查',
      text: '系统会优先选中高风险轮廓；使用“检查通过并下一个”连续复核，刷新页面后会从下一个未检查轮廓继续。',
      tip: 'VLM 提示的漏标、冲突或不确定类别必须明确处理，不能只看轮廓数量。',
    },
    {
      target: '#taskCount', parent: true, label: '单帧数量', mode: 'auto',
      title: '第一次只处理 1 帧',
      text: '“本次处理”表示从当前帧开始连续处理多少帧。首次演示保持 1，确认流程正常后再扩大。',
      tip: '页面已默认设为 1，避免误跑整段数据。',
    },
    {
      target: '#runButton', label: '启动完整流程', mode: 'auto',
      title: '启动 SAM → VLM → Agent',
      text: '开始后，SAM 先产生分割候选，VLM 检查类别和漏误标，Agent 再按判断执行修改或保持。',
      tip: '第一次加载模型可能需要约一分钟，请观察进度，不要重复点击。',
    },
    {
      target: '#runProgress', label: '运行进度', mode: 'auto',
      title: '查看当前运行到了哪一步',
      text: '这里会显示当前帧、复查轮次、候选数量、耗时，以及 SAM、VLM、Agent 的执行阶段。',
      tip: 'HTTP 成功只代表接口可用，最终仍要查看候选边界和类别。',
    },
    {
      target: '.inspectionStages', label: '过程证据', mode: 'auto',
      title: '逐阶段查看模型看到了什么',
      text: '运行后可依次查看模型输入、SAM 分割、VLM 复查、Agent 决策和最终候选，方便定位错误来源。',
      tip: '即使 Agent 显示 done，也不代表候选已经正确。',
    },
    {
      target: '#saveCandidate', label: '人工真值门禁', mode: 'auto',
      title: '最后一步必须由人确认',
      text: '只有检查完类别、边界、漏标和多标后，才可以把当前机器候选保存为人工真值。',
      tip: '只是查看效果时不要保存；点击后页面还会再次确认。',
    },
  ];
  let tutorialIndex = 0;
  let tutorialRestoreFocus = false;
  let tutorialRestoreCompact = false;
  const quickActions = byId('quickActionsDialog');
  const currentUiVersion = document.querySelector('meta[name="labelagent-ui-version"]')?.content || '';

  function showUiVersionNotice(version) {
    const notice = byId('uiVersionNotice');
    if (!notice) return;
    notice.hidden = false;
    notice.querySelector('strong').textContent = `页面已有新版 ${version || ''}，当前旧页面可能导致 SAM 按钮看起来没有反应`;
  }

  async function checkUiVersion() {
    try {
      const response = await fetch(`/?ui_version_probe=${Date.now()}`, {cache: 'no-store'});
      if (!response.ok) return;
      const html = await response.text();
      const match = html.match(/<meta\s+name=["']labelagent-ui-version["']\s+content=["']([^"']+)["']/i);
      const remoteVersion = match?.[1] || '';
      if (remoteVersion && currentUiVersion && remoteVersion !== currentUiVersion) showUiVersionNotice(remoteVersion);
    } catch (_) {
      // Version checks are advisory and must never interrupt annotation work.
    }
  }

  function objectKeys(value) {
    return value && typeof value === 'object' ? Object.keys(value) : [];
  }

  function setBadge(node, text, tone = 'neutral') {
    node.textContent = text;
    node.className = `guideBadge ${tone}`;
  }

  function setStep(name, state) {
    const node = steps[name];
    if (!node) return;
    node.classList.toggle('done', state === 'done');
    node.classList.toggle('active', state === 'active');
    node.classList.toggle('blocked', state === 'blocked');
    node.setAttribute('aria-current', state === 'active' ? 'step' : 'false');
  }

  function compactTopic(topic) {
    if (!topic) return '未选择相机';
    return topic.replace(/^\/capture\//, '').replace(/\/image$/, '');
  }

  function render(state, job) {
    latest.state = state;
    latest.job = job;
    const loaded = Boolean(state?.loaded);
    const topic = state?.topic || '';
    const topicReadiness = state?.topic_readiness || {};
    const currentTopicReadiness = topicReadiness[topic] || null;
    const readyTopicCount = Object.values(topicReadiness).filter(item => item?.ready).length;
    const imageTopicCount = Object.keys(topicReadiness).length;
    const frame = Number(byId('frameNumber')?.value || 0);
    const reference = state?.reference_frames?.[topic]?.[0];
    const candidateKeys = objectKeys(state?.candidate_counts);
    const truthKeys = (state?.truth_frames || []).map(String);
    const verifiedTruthSet = new Set((state?.verified_truth_frames || []).map(String));
    const verifiedTruthKeys = truthKeys.filter(key => verifiedTruthSet.has(key));
    const legacyTruthKeys = truthKeys.filter(key => !verifiedTruthSet.has(key));
    const currentCandidateCount = Number(state?.candidate_counts?.[String(frame)] || 0);
    const hasCandidate = currentCandidateCount > 0;
    const referenceReviewActive = hasCandidate && byId('referenceFinalizeGuide')?.hidden === false;
    const hasStoredAnnotation = truthKeys.includes(String(frame));
    const hasTruth = hasStoredAnnotation && verifiedTruthSet.has(String(frame));
    const hasLegacyTruth = hasStoredAnnotation && !hasTruth;
    const running = ['running', 'stopping'].includes(job?.status);
    const currentFrameAutoDraftRunning = running && Number(job?.frame) === frame && Number(job?.total || 1) === 1;
    const autoDraftBusy = autoDraftSubmitting || currentFrameAutoDraftRunning;
    const showDirectAutoDraft = loaded && reference === undefined && !hasCandidate && !hasStoredAnnotation
      && (!running || currentFrameAutoDraftRunning);
    autoDraftAction.hidden = !showDirectAutoDraft;
    autoDraftAction.disabled = !showDirectAutoDraft || autoDraftBusy;
    autoDraftAction.setAttribute('aria-busy', String(autoDraftBusy));
    autoDraftAction.textContent = autoDraftBusy ? '已提交 · 查看实时进度' : '一键 SAM + VLM 起稿';

    setBadge(badges.data, loaded ? '数据已载入' : '尚未载入数据', loaded ? 'success' : 'warning');
    setBadge(
      badges.candidate,
      candidateKeys.length ? `机器候选 ${candidateKeys.length} 帧` : '暂无机器候选',
      candidateKeys.length ? 'candidate' : 'neutral',
    );
    const truthBadgeText = verifiedTruthKeys.length
      ? `人工真值 ${verifiedTruthKeys.length} 帧${legacyTruthKeys.length ? ` · 历史标注 ${legacyTruthKeys.length} 帧待复核` : ''}`
      : legacyTruthKeys.length
        ? `历史标注 ${legacyTruthKeys.length} 帧 · 待人工复核`
        : '暂无人工真值';
    setBadge(badges.truth, truthBadgeText, verifiedTruthKeys.length ? 'success' : legacyTruthKeys.length ? 'warning' : 'neutral');

    setStep('frame', loaded ? 'done' : 'active');
    setStep('reference', !loaded ? 'blocked' : reference !== undefined ? 'done' : 'active');
    setStep('run', !loaded || reference === undefined || currentTopicReadiness?.ready === false ? 'blocked' : hasCandidate || hasTruth ? 'done' : 'active');
    setStep('review', !loaded || reference === undefined ? 'blocked' : hasTruth ? 'done' : hasCandidate || hasLegacyTruth ? 'active' : 'blocked');

    detail.textContent = loaded
      ? `${compactTopic(topic)} · 第 ${frame} 帧${currentCandidateCount ? ` · 当前帧候选 ${currentCandidateCount} 个` : ''}${imageTopicCount ? ` · 相机就绪 ${readyTopicCount}/${imageTopicCount}` : ''} · 首次演示建议只处理 1 帧`
      : '先加载一段图像或 MCAP 数据，再开始单帧演示。';

    if (!loaded) {
      title.textContent = '第一步：载入需要查看的数据';
      action.textContent = '加载数据';
      latest.action = 'load';
    } else if (running) {
      title.textContent = `模型正在处理第 ${job?.frame ?? frame} 帧，请等待结果`;
      action.textContent = '查看运行进度';
      latest.action = 'progress';
    } else if (reference === undefined) {
      title.textContent = referenceReviewActive
        ? `已载入 ${currentCandidateCount} 个候选，正在逐个检查`
        : hasLegacyTruth
        ? '下一步：复核当前历史标注并准备参考帧'
        : hasCandidate
          ? `当前帧已有 ${currentCandidateCount} 个机器候选，可直接载入精修`
          : '下一步：可直接一键自动起稿，完成后只需检查确认';
      action.textContent = referenceReviewActive ? '继续检查当前候选' : hasLegacyTruth ? '载入历史标注并复核' : hasCandidate ? '复用当前帧候选' : '准备参考帧';
      latest.action = referenceReviewActive ? 'reference_review' : hasCandidate && !hasLegacyTruth ? 'reference_candidate' : 'reference';
    } else if (currentTopicReadiness && !currentTopicReadiness.description_ready) {
      title.textContent = '参考帧已锁定，还需生成一次该相机的场景描述';
      action.textContent = '打开参考帧并生成描述';
      latest.action = 'description';
    } else if (hasCandidate && !hasTruth) {
      title.textContent = '机器候选已生成，等待人工检查边界和类别';
      action.textContent = '查看候选并复核';
      latest.action = 'review';
    } else if (hasTruth && Number(reference) === frame) {
      title.textContent = '参考帧已锁定，下一步运行一帧完整 LabelAgent';
      action.textContent = '前往下一待处理帧';
      latest.action = 'next_unprocessed';
    } else if (hasTruth) {
      title.textContent = '当前帧已有人工作出的真值记录';
      action.textContent = '查看人工真值';
      latest.action = 'truth';
    } else if (hasLegacyTruth) {
      title.textContent = '当前帧有历史标注，但尚未通过本次人工门禁';
      action.textContent = '载入历史标注并复核';
      latest.action = 'legacy';
    } else {
      title.textContent = '参考帧已准备，可以运行一帧完整 LabelAgent';
      action.textContent = '准备单帧运行';
      latest.action = 'run';
    }
  }

  function renderUnavailable() {
    latest.state = null;
    latest.job = null;
    title.textContent = '暂时无法读取平台状态';
    detail.textContent = '请确认本机服务在线后刷新页面；缺少证据时不会显示为已完成。';
    setBadge(badges.data, '状态未知', 'danger');
    setBadge(badges.candidate, '候选未知', 'danger');
    setBadge(badges.truth, '真值未知', 'danger');
    Object.keys(steps).forEach(name => setStep(name, 'blocked'));
    action.textContent = '重新读取状态';
    latest.action = 'refresh';
  }

  async function refreshGuide() {
    if (guideRefreshBusy) return;
    guideRefreshBusy = true;
    try {
      const [stateResponse, jobResponse] = await Promise.all([
        fetch('/api/state/summary', { cache: 'no-store' }),
        fetch('/api/run/status', { cache: 'no-store' }),
      ]);
      if (!stateResponse.ok || !jobResponse.ok) throw new Error('status unavailable');
      const state = await stateResponse.json();
      const job = await jobResponse.json();
      latestRunStatus = job.status || 'idle';
      render(state, job);
    } catch (_) {
      renderUnavailable();
    } finally {
      guideRefreshBusy = false;
      if (guideStatusRefreshPending && !document.hidden) {
        guideStatusRefreshPending = false;
        refreshGuide();
      } else {
        guideStatusRefreshPending = false;
        scheduleGuidePoll();
      }
    }
  }

  function scheduleGuidePoll(delay) {
    if (guidePollTimer) clearTimeout(guidePollTimer);
    if (document.hidden) return;
    const interval = ['running', 'stopping'].includes(latestRunStatus) ? 3000 : 15000;
    guidePollTimer = window.setTimeout(refreshGuide, delay ?? interval);
  }

  function showWorkbench(mode = 'auto') {
    byId(mode === 'manual' ? 'manualModeButton' : 'autoModeButton')?.click();
    const count = byId('taskCount');
    if (count) count.value = '1';
    byId('workbench')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function remember(key, value) {
    try { window.localStorage.setItem(key, value ? '1' : '0'); } catch (_) {}
  }

  function recalled(key) {
    try { return window.localStorage.getItem(key) === '1'; } catch (_) { return false; }
  }

  function setGuideCompact(compact, persist = true) {
    guide.classList.toggle('compact', compact);
    byId('guideToggle').textContent = compact ? '展开步骤' : '收起步骤';
    byId('guideToggle').setAttribute('aria-expanded', String(!compact));
    if (persist) remember('labelagentGuideCompact', compact);
  }

  function setFocusMode(enabled, persist = true) {
    document.body.classList.toggle('focusMode', enabled);
    const button = byId('focusModeButton');
    button.classList.toggle('selected', enabled);
    button.setAttribute('aria-pressed', String(enabled));
    button.innerHTML = enabled ? '<span aria-hidden="true">↙</span>退出专注' : '<span aria-hidden="true">⛶</span>专注视图';
    if (persist) remember('labelagentFocusMode', enabled);
  }

  function setDarkTheme(enabled, persist = true) {
    document.body.classList.toggle('darkTheme', enabled);
    const button = byId('themeButton');
    button.classList.toggle('selected', enabled);
    button.setAttribute('aria-pressed', String(enabled));
    button.innerHTML = enabled ? '<span aria-hidden="true">☀</span>浅色' : '<span aria-hidden="true">☾</span>夜间';
    if (persist) remember('labelagentDarkTheme', enabled);
  }

  function setCanvasMode(enabled, persist = true) {
    document.body.classList.toggle('workbenchCollapsed', enabled);
    const button = byId('canvasModeButton');
    button.classList.toggle('selected', enabled);
    button.setAttribute('aria-pressed', String(enabled));
    button.innerHTML = enabled ? '<span aria-hidden="true">▤</span>展开操作台' : '<span aria-hidden="true">▣</span>画布优先';
    if (persist) remember('labelagentCanvasMode', enabled);
  }

  function resetLayout() {
    setFocusMode(false);
    setCanvasMode(false);
    setGuideCompact(false);
    window.setTimeout(() => window.dispatchEvent(new Event('resize')), 30);
  }

  function frameLimit() {
    const match = byId('frameTotal')?.textContent?.match(/\d+/g);
    return match?.length ? Number(match.at(-1)) : 0;
  }

  function moveFrame(delta) {
    const input = byId('frameNumber');
    if (!input || !latest.state?.loaded) return;
    const current = Number(input.value) || 0;
    const target = Math.max(0, Math.min(frameLimit(), current + delta));
    if (target === current) return;
    input.value = String(target);
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function nextUnprocessedFrame(state, start) {
    const maximum = frameLimit();
    if (!Number.isInteger(maximum) || maximum < 0) return null;
    const truth = new Set((state?.truth_frames || []).map(String));
    const candidates = new Set(Object.keys(state?.candidate_counts || {}));
    for (let offset = 1; offset <= maximum + 1; offset += 1) {
      const target = (start + offset) % (maximum + 1);
      if (!truth.has(String(target)) && !candidates.has(String(target))) return target;
    }
    return null;
  }

  function openSingleFrameReadiness() {
    showWorkbench('auto');
    window.setTimeout(() => byId('runButton')?.click(), 220);
  }

  const readinessDialog = byId('runReadinessDialog');
  let allowRunOnce = false;
  let readinessScope = null;

  function currentWorkbenchScope() {
    const state = typeof window.currentWorkbenchFrameState === 'function'
      ? window.currentWorkbenchFrameState()
      : null;
    return {
      path: String(state?.path || ''),
      topic: String(state?.topic || ''),
      frame: Number(byId('frameNumber')?.value),
      count: Number(byId('taskCount')?.value),
    };
  }

  function sameReadinessScope(left, right) {
    return Boolean(left && right
      && left.path === right.path
      && left.topic === right.topic
      && left.frame === right.frame
      && left.count === right.count);
  }

  function stateMatchesCurrentWorkspace(state) {
    const current = currentWorkbenchScope();
    if (!current.path || !current.topic) return true;
    return Boolean(state?.loaded
      && String(state.path || '') === current.path
      && String(state.topic || '') === current.topic);
  }

  function closeRunReadiness() {
    if (!readinessDialog?.open) return;
    if (typeof readinessDialog.close === 'function') readinessDialog.close();
    else readinessDialog.removeAttribute('open');
  }

  function runReadiness() {
    const state = latest.state || {};
    const scopeMatches = stateMatchesCurrentWorkspace(state);
    const current = currentWorkbenchScope();
    const topic = scopeMatches ? (state.topic || '') : current.topic;
    const frame = Number(byId('frameNumber')?.value);
    const count = Number(byId('taskCount')?.value);
    const maximum = frameLimit();
    const topicStatus = scopeMatches ? (state.topic_readiness?.[topic] || {}) : {};
    const referenceFrame = scopeMatches ? state.reference_frames?.[topic]?.[0] : undefined;
    const hasReference = Number.isInteger(referenceFrame);
    const descriptionReady = hasReference && topicStatus.description_ready === true;
    const descriptionFail = topicStatus.code === 'description_stale'
      ? `帧 ${referenceFrame} 的描述已过期`
      : hasReference ? `帧 ${referenceFrame} 尚未生成描述` : '需先准备人工参考帧';
    return {
      data: {
        ok: Boolean(state.loaded && topic && scopeMatches),
        pass: '当前数据已载入',
        fail: current.path && current.topic ? '数据集状态正在同步，请稍候重试' : '尚未载入数据',
      },
      reference: {
        ok: hasReference,
        pass: '参考帧已准备', fail: '缺少人工参考帧',
      },
      description: {
        ok: descriptionReady,
        pass: `帧 ${referenceFrame} 的描述可用`, fail: descriptionFail,
        referenceFrame,
      },
      frame: {
        ok: Number.isInteger(frame) && frame >= 0 && frame <= maximum,
        pass: `第 ${frame} 帧有效`, fail: `请输入 0～${maximum} 的整数`,
      },
      count: {
        ok: Number.isInteger(count) && count >= 1 && count <= 100,
        pass: `将处理 ${count} 帧`, fail: '请输入 1～100 帧',
      },
    };
  }

  function renderRunReadinessChecks(checks) {
    let passed = 0;
    for (const [name, result] of Object.entries(checks)) {
      const item = readinessDialog?.querySelector(`[data-readiness="${name}"]`);
      if (!item) continue;
      item.classList.toggle('passed', result.ok === true);
      item.classList.toggle('blocked', result.ok === false && result.pending !== true);
      item.querySelector('b').textContent = result.pending ? '检查中' : result.ok ? result.pass : result.fail;
      if (result.ok === true) passed += 1;
    }
    const ready = passed === Object.keys(checks).length;
    byId('runReadinessSummary').textContent = ready
      ? `检查通过：将从第 ${byId('frameNumber')?.value || 0} 帧开始处理 ${byId('taskCount')?.value || 1} 帧。`
      : Object.values(checks).some(result => result.pending)
        ? '正在核对保存空间，确认不会在任务中途因磁盘不足而失败。'
        : `还有 ${Object.keys(checks).length - passed} 项未就绪，请先补齐后再运行。`;
    byId('runReadinessConfirm').disabled = !ready;
    return ready;
  }

  async function openRunReadiness(event) {
    if (allowRunOnce) {
      allowRunOnce = false;
      return;
    }
    if (byId('runButton')?.getAttribute('aria-busy') === 'true') return;
    event.preventDefault();
    event.stopImmediatePropagation();
    const checks = {...runReadiness(), storage: {
      ok: false, pending: true, pass: '', fail: '正在检查保存空间',
    }};
    readinessScope = currentWorkbenchScope();
    renderRunReadinessChecks(checks);
    const prepare = byId('runReadinessPrepare');
    prepare.hidden = checks.reference.ok && checks.description.ok;
    prepare.textContent = checks.reference.ok ? '生成描述并继续' : '去准备参考帧';
    if (!readinessDialog.open) readinessDialog.showModal();
    const checkedScope = readinessScope;
    try {
      const response = await fetch(`/api/run/readiness?count=${encodeURIComponent(checkedScope.count)}`, {cache: 'no-store'});
      const result = await response.json();
      if (!sameReadinessScope(checkedScope, currentWorkbenchScope())) return;
      const storage = result?.storage;
      checks.storage = storage ? {
        ok: storage.ready === true,
        pass: storage.warning
          ? `空间偏低但本次可运行 · 剩余 ${storage.free_mib} MiB · 当前最多 ${storage.max_safe_frames} 帧`
          : `空间充足 · 剩余 ${storage.free_mib} MiB`,
        fail: storage.message || '保存空间不足，尚未启动推理',
      } : {ok: false, pass: '', fail: '无法确认保存空间，尚未启动推理'};
    } catch (error) {
      if (!sameReadinessScope(checkedScope, currentWorkbenchScope())) return;
      checks.storage = {ok: false, pass: '', fail: `保存空间检查失败：${error.message}`};
    }
    renderRunReadinessChecks(checks);
  }

  function openQuickActions() {
    if (!quickActions || quickActions.open) return;
    if (typeof quickActions.showModal === 'function') quickActions.showModal();
    else quickActions.setAttribute('open', '');
    byId('quickActionsButton')?.setAttribute('aria-expanded', 'true');
    window.setTimeout(() => quickActions.querySelector('.quickAction')?.focus(), 20);
  }

  function closeQuickActions() {
    if (!quickActions?.open) return;
    if (typeof quickActions.close === 'function') quickActions.close();
    else quickActions.removeAttribute('open');
    byId('quickActionsButton')?.setAttribute('aria-expanded', 'false');
  }

  function runQuickAction(name) {
    closeQuickActions();
    if (name === 'manual') byId('manualModeButton')?.click();
    if (name === 'auto') byId('autoModeButton')?.click();
    if (name === 'tutorial') openTutorial();
    if (name === 'files') byId('openFilesDialog')?.click();
    if (name === 'window') byId('addWindowButton')?.click();
    if (name === 'focus') setFocusMode(!document.body.classList.contains('focusMode'));
    if (name === 'theme') setDarkTheme(!document.body.classList.contains('darkTheme'));
    if (name === 'canvas') setCanvasMode(!document.body.classList.contains('workbenchCollapsed'));
    if (name === 'reset') resetLayout();
    if (name === 'guide') {
      setFocusMode(false);
      setGuideCompact(false);
      guide.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  function clearTutorialTarget() {
    document.querySelectorAll('.tutorialTarget').forEach(node => {
      node.classList.remove('tutorialTarget');
      delete node.dataset.tutorialLabel;
    });
  }

  function tutorialTarget(step) {
    const target = document.querySelector(step.target);
    return step.parent ? target?.parentElement : target;
  }

  function renderTutorialStep(index) {
    tutorialIndex = Math.max(0, Math.min(tutorialSteps.length - 1, index));
    const step = tutorialSteps[tutorialIndex];
    if (step.mode) byId(`${step.mode}ModeButton`)?.click();
    clearTutorialTarget();
    window.setTimeout(() => {
      const target = tutorialTarget(step);
      if (target) {
        target.classList.add('tutorialTarget');
        target.dataset.tutorialLabel = step.label;
        target.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
      }
    }, step.mode ? 180 : 20);
    byId('tutorialProgress').textContent = `操作教程 ${tutorialIndex + 1} / ${tutorialSteps.length}`;
    byId('tutorialTitle').textContent = step.title;
    byId('tutorialText').textContent = step.text;
    byId('tutorialTip').textContent = step.tip;
    byId('tutorialPrevious').disabled = tutorialIndex === 0;
    byId('tutorialNext').textContent = tutorialIndex === tutorialSteps.length - 1 ? '完成教程' : '下一步';
  }

  function openTutorial() {
    tutorialRestoreFocus = document.body.classList.contains('focusMode');
    tutorialRestoreCompact = guide.classList.contains('compact');
    setFocusMode(false, false);
    setGuideCompact(false, false);
    tutorial.hidden = false;
    document.body.classList.add('tutorialActive');
    renderTutorialStep(0);
  }

  function closeTutorial() {
    tutorial.hidden = true;
    document.body.classList.remove('tutorialActive');
    clearTutorialTarget();
    setFocusMode(tutorialRestoreFocus, false);
    setGuideCompact(tutorialRestoreCompact, false);
    byId('tutorialButton')?.focus();
  }

  action.addEventListener('click', async () => {
    if (latest.action === 'load') byId('openFilesDialog')?.click();
    else if (latest.action === 'reference_candidate') {
      const frame = Number(byId('frameNumber')?.value || 0);
      try {
        if (typeof window.loadCurrentFrameCandidateForReference !== 'function') throw new Error('参考帧助手尚未加载');
        await window.loadCurrentFrameCandidateForReference(frame);
      } catch (error) {
        showWorkbench('auto');
        byId('prepareReference')?.click();
        window.labelAgentToast?.(`${error.message}；已打开参考帧助手，请重新选择`);
      }
    }
    else if (latest.action === 'reference_review') {
      byId('referenceFinalizeGuide')?.scrollIntoView({behavior: 'smooth', block: 'center'});
      byId('referenceReviewApproveNext')?.focus({preventScroll: true});
      window.labelAgentToast?.('已返回当前候选检查位置；请处理风险提示并逐个确认');
    }
    else if (latest.action === 'reference') {
      showWorkbench('auto');
      byId('prepareReference')?.click();
    } else if (latest.action === 'description') {
      const reference = latest.state?.reference_frames?.[latest.state?.topic]?.[0];
      if (!Number.isInteger(reference) || typeof window.regenerateReferenceDescription !== 'function') {
        showWorkbench('auto');
        byId('openReferenceButton')?.click();
        window.labelAgentToast?.('无法自动定位参考帧，已打开参考帧页面');
      } else {
        action.disabled = true;
        action.textContent = '正在生成参考描述…';
        try {
          await window.regenerateReferenceDescription(reference);
          window.labelAgentToast?.('参考描述已更新，可以开始自动标注');
        } catch (error) {
          window.labelAgentToast?.(error.message);
        } finally {
          await refreshGuide();
        }
      }
    } else if (latest.action === 'next_unprocessed') {
      const input = byId('frameNumber');
      const current = Number(input?.value) || 0;
      const target = nextUnprocessedFrame(latest.state, current);
      if (target === null) {
        window.labelAgentToast?.('当前数据没有未处理帧；请查看已有候选或载入新数据');
      } else {
        showWorkbench('auto');
        input.value = String(target);
        input.dispatchEvent(new Event('change', {bubbles: true}));
        window.setTimeout(() => byId('runButton')?.click(), 260);
      }
    } else if (latest.action === 'run') openSingleFrameReadiness();
    else if (latest.action === 'truth' || latest.action === 'legacy') showWorkbench('manual');
    else if (latest.action === 'refresh') refreshGuide();
    else showWorkbench('auto');
  });

  autoDraftAction.addEventListener('click', async () => {
    if (autoDraftAction.disabled) return;
    autoDraftSubmitting = true;
    autoDraftAction.disabled = true;
    autoDraftAction.setAttribute('aria-busy', 'true');
    autoDraftAction.textContent = '已提交 · 查看实时进度';
    try {
      if (typeof window.startReferenceAutoDraft !== 'function') throw new Error('参考帧自动起稿尚未加载，请强制刷新页面后重试');
      await window.startReferenceAutoDraft();
    } catch (error) {
      window.labelAgentToast?.(error.message);
    } finally {
      autoDraftSubmitting = false;
      await refreshGuide();
    }
  });

  byId('saveCandidate')?.addEventListener('click', event => {
    const approved = window.confirm(
      '这一步会把当前机器候选写入人工真值。\n\n请确认你已经检查：类别、边界、漏标和多标。\n如果只是查看效果，请选择“取消”。',
    );
    if (!approved) {
      event.preventDefault();
      event.stopImmediatePropagation();
    }
  }, true);

  byId('clearAutoResult')?.addEventListener('click', event => {
    const approved = window.confirm('确定只清除当前帧的机器候选和过程记录吗？人工真值不会被清除。');
    if (!approved) {
      event.preventDefault();
      event.stopImmediatePropagation();
    }
  }, true);

  byId('tutorialButton')?.addEventListener('click', openTutorial);
  byId('reloadLatestUi')?.addEventListener('click', () => window.location.reload());
  byId('previousFrameButton')?.addEventListener('click', () => moveFrame(-1));
  byId('nextFrameButton')?.addEventListener('click', () => moveFrame(1));
  byId('runButton')?.addEventListener('click', openRunReadiness, true);
  byId('runReadinessClose')?.addEventListener('click', closeRunReadiness);
  byId('runReadinessCancel')?.addEventListener('click', closeRunReadiness);
  byId('runReadinessPrepare')?.addEventListener('click', async () => {
    const button = byId('runReadinessPrepare');
    if (!sameReadinessScope(readinessScope, currentWorkbenchScope())) {
      closeRunReadiness();
      window.labelAgentToast?.('数据集、相机、帧号或处理数量已经变化，请重新点击开始');
      refreshGuide();
      return;
    }
    const checks = runReadiness();
    if (!checks.reference.ok) {
      closeRunReadiness();
      byId('prepareReference')?.click();
      return;
    }
    if (button.disabled) return;
    button.disabled = true;
    button.textContent = '正在生成描述…';
    try {
      if (typeof window.regenerateReferenceDescription !== 'function') throw new Error('描述生成组件尚未加载，请强制刷新页面后重试');
      await window.regenerateReferenceDescription(checks.description.referenceFrame);
      await refreshGuide();
      closeRunReadiness();
      allowRunOnce = true;
      byId('runButton')?.click();
    } catch (error) {
      window.labelAgentToast?.(error.message);
    } finally {
      button.disabled = false;
      const current = runReadiness();
      button.textContent = current.reference.ok ? '生成描述并继续' : '去准备参考帧';
    }
  });
  byId('runReadinessConfirm')?.addEventListener('click', () => {
    if (!sameReadinessScope(readinessScope, currentWorkbenchScope())) {
      closeRunReadiness();
      window.labelAgentToast?.('数据集、相机、帧号或处理数量已经变化，本次未启动；请重新点击开始');
      refreshGuide();
      return;
    }
    closeRunReadiness();
    allowRunOnce = true;
    byId('runButton')?.click();
    window.requestAnimationFrame(() => {
      const progress = byId('runProgress');
      progress?.scrollIntoView({behavior: 'auto', block: 'center'});
      progress?.focus({preventScroll: true});
    });
  });
  readinessDialog?.addEventListener('cancel', event => {
    event.preventDefault();
    closeRunReadiness();
  });
  readinessDialog?.addEventListener('click', event => {
    if (event.target === readinessDialog) closeRunReadiness();
  });
  byId('quickActionsButton')?.addEventListener('click', openQuickActions);
  byId('quickActionsClose')?.addEventListener('click', closeQuickActions);
  quickActions?.addEventListener('cancel', event => {
    event.preventDefault();
    closeQuickActions();
  });
  quickActions?.addEventListener('click', event => {
    if (event.target === quickActions) closeQuickActions();
  });
  quickActions?.querySelectorAll('[data-quick-action]').forEach(button => {
    button.addEventListener('click', () => runQuickAction(button.dataset.quickAction));
  });
  byId('guideToggle')?.addEventListener('click', () => setGuideCompact(!guide.classList.contains('compact')));
  byId('focusModeButton')?.addEventListener('click', () => setFocusMode(!document.body.classList.contains('focusMode')));
  byId('canvasModeButton')?.addEventListener('click', () => setCanvasMode(!document.body.classList.contains('workbenchCollapsed')));
  byId('themeButton')?.addEventListener('click', () => setDarkTheme(!document.body.classList.contains('darkTheme')));
  byId('emptyAddWindow')?.addEventListener('click', () => byId('addWindowButton')?.click());
  byId('tutorialClose')?.addEventListener('click', closeTutorial);
  byId('tutorialPrevious')?.addEventListener('click', () => renderTutorialStep(tutorialIndex - 1));
  byId('tutorialNext')?.addEventListener('click', () => {
    if (tutorialIndex === tutorialSteps.length - 1) closeTutorial();
    else renderTutorialStep(tutorialIndex + 1);
  });
  byId('tutorialPrepare')?.addEventListener('click', () => {
    closeTutorial();
    showWorkbench('auto');
  });
  document.addEventListener('keydown', event => {
    const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)
      || document.activeElement?.isContentEditable;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      if (quickActions?.open) closeQuickActions();
      else openQuickActions();
      return;
    }
    if (readinessDialog?.open) return;
    if (quickActions?.open) return;
    if (!tutorial.hidden) {
      if (event.key === 'Escape') closeTutorial();
      if (event.key === 'ArrowRight') byId('tutorialNext')?.click();
      if (event.key === 'ArrowLeft') byId('tutorialPrevious')?.click();
      return;
    }
    if (typing || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === '?') openTutorial();
    if (event.key === '1') byId('manualModeButton')?.click();
    if (event.key === '2') byId('autoModeButton')?.click();
    if (event.key.toLowerCase() === 'g') setGuideCompact(!guide.classList.contains('compact'));
    if (event.key.toLowerCase() === 'f') setFocusMode(!document.body.classList.contains('focusMode'));
    if (event.key.toLowerCase() === 'b') setCanvasMode(!document.body.classList.contains('workbenchCollapsed'));
    if (event.key.toLowerCase() === 't') setDarkTheme(!document.body.classList.contains('darkTheme'));
  });

  byId('frameNumber').title = '输入帧号后按 Tab 或点击输入框外完成切换';
  byId('prepareReference').title = '准备人工参考帧；不会自动保存真值';
  byId('runButton').title = '运行当前帧的 SAM、VLM 与 Agent 完整流程';
  byId('saveCandidate').title = '人工检查后，将机器候选保存为真值';

  const workspaceChanged = () => {
    if (latest.state && !stateMatchesCurrentWorkspace(latest.state)) latest.state = null;
    if (readinessDialog?.open && !sameReadinessScope(readinessScope, currentWorkbenchScope())) {
      closeRunReadiness();
      window.labelAgentToast?.('当前数据或帧已切换，运行前检查已自动失效');
    }
    if (guideRefreshBusy) guideStatusRefreshPending = true;
    else refreshGuide();
  };
  byId('frameNumber')?.addEventListener('change', workspaceChanged);
  byId('taskCount')?.addEventListener('change', workspaceChanged);
  document.addEventListener('labelagent:frame-state', workspaceChanged);
  document.addEventListener('labelagent:workspace-ready', workspaceChanged);
  document.addEventListener('labelagent:frame-changed', refreshGuide);
  document.addEventListener('labelagent:candidate-state-synced', refreshGuide);
  document.addEventListener('labelagent:reference-draft-loaded', refreshGuide);
  document.addEventListener('labelagent:reference-draft-cleared', refreshGuide);
  document.addEventListener('labelagent:job-status', event => {
    const status = event.detail?.status || 'idle';
    if (status === latestRunStatus) return;
    latestRunStatus = status;
    if (!document.hidden) {
      if (guideRefreshBusy) guideStatusRefreshPending = true;
      else refreshGuide();
    }
    scheduleGuidePoll();
  });
  byId('manualModeButton')?.addEventListener('click', () => setTimeout(refreshGuide, 50));
  byId('autoModeButton')?.addEventListener('click', () => setTimeout(refreshGuide, 50));
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      if (guidePollTimer) clearTimeout(guidePollTimer);
      guidePollTimer = null;
    } else {
      refreshGuide();
      checkUiVersion();
    }
  });
  window.addEventListener('focus', () => { refreshGuide(); checkUiVersion(); });
  window.addEventListener('pageshow', refreshGuide);

  setGuideCompact(recalled('labelagentGuideCompact'), false);
  setFocusMode(recalled('labelagentFocusMode'), false);
  setCanvasMode(recalled('labelagentCanvasMode'), false);
  setDarkTheme(recalled('labelagentDarkTheme'), false);
  refreshGuide();
  checkUiVersion();
  scheduleGuidePoll();
  window.setInterval(checkUiVersion, 30000);
})();
