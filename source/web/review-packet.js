(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const dialog = byId('reviewPacketDialog');
  const core = window.LabelAgentReviewPacketCore;
  if (!dialog || !core) return;

  const select = byId('reviewPacketSelect');
  const activateButton = byId('reviewPacketActivate');
  const statusBox = byId('reviewPacketStatus');
  const body = byId('reviewPacketBody');
  const queue = byId('reviewPacketQueue');
  const dock = byId('reviewPacketDock');
  const assistantBox = byId('reviewPacketAssistant');
  const reviewerInput = byId('reviewPacketReviewer');
  const decisionSelect = byId('reviewPacketDecision');
  const decisionNote = byId('reviewPacketDecisionNote');
  const decisionSubmit = byId('reviewPacketDecisionSubmit');
  let catalog = [];
  let packet = null;
  let completed = new Map();
  let activeFilter = 'all';
  let currentIndex = -1;
  let baseline = null;
  let busy = false;
  let decisionStartedAt = null;
  let decisionKey = null;

  const decisionLabels = {
    slope_present_and_mask_acceptable: '有斜坡 · 当前掩码可接受',
    slope_present_mask_needs_correction: '有斜坡 · 掩码需要修正',
    no_slope_present: '当前画面没有斜坡',
    uncertain: '不确定 · 留待进一步复核',
  };

  function makeDecisionKey() {
    if (globalThis.crypto?.randomUUID) return `slope-review-${crypto.randomUUID()}`;
    return `slope-review-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  async function sha256Json(value) {
    const bytes = new TextEncoder().encode(core.stableJson(value));
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    return [...new Uint8Array(digest)].map(item => item.toString(16).padStart(2, '0')).join('');
  }

  async function request(path, options = {}) {
    const response = await fetch(path, {cache: 'no-store', ...options});
    const raw = await response.text();
    let data;
    try { data = JSON.parse(raw); }
    catch { throw new Error(`专项复核接口返回异常（HTTP ${response.status}）`); }
    if (!response.ok) throw new Error(data.error || `专项复核请求失败（HTTP ${response.status}）`);
    return data;
  }

  function setStatus(message, tone = '') {
    statusBox.textContent = message;
    statusBox.className = `reviewPacketStatus ${tone}`.trim();
  }

  function setBusy(value, message = '') {
    busy = value;
    activateButton.disabled = value || !catalog.length;
    byId('reviewPacketRefresh').disabled = value;
    select.disabled = value || !catalog.length;
    decisionSubmit.disabled = value || currentIndex < 0
      || !reviewerInput.value.trim() || !decisionSelect.value;
    if (message) setStatus(message, value ? 'busy' : '');
  }

  function packetLabel(item) {
    const summary = item.summary || {};
    return `${item.title || item.packet_id} · ${summary.positive_proxy_frames || 0} 正 / ${summary.negative_cohort_frames || 0} 负`;
  }

  function matches(row) {
    const done = completed.has(Number(row.frame));
    if (activeFilter === 'complete') return done;
    if (activeFilter === 'pending') return !done;
    if (activeFilter === 'positive_proxy' || activeFilter === 'negative_cohort') return row.cohort === activeFilter;
    return true;
  }

  function cohortLabel(row) {
    return row.cohort === 'positive_proxy' ? '斜坡正候选' : '困难负样本';
  }

  function renderDock() {
    if (!packet || currentIndex < 0 || !packet.rows?.[currentIndex]) { dock.hidden = true; return; }
    const row = packet.rows[currentIndex];
    const receipt = completed.get(Number(row.frame));
    const done = Boolean(receipt);
    dock.hidden = false;
    byId('reviewPacketDockTitle').textContent = `斜坡专项 · 第 ${row.frame} 帧 · ${currentIndex + 1}/${packet.rows.length}`;
    byId('reviewPacketDockDetail').textContent = receipt?.type === 'slope_scoped_decision'
      ? `已签署专项判断：${decisionLabels[receipt.decision] || receipt.decision}；未写入整帧真值或训练集`
      : done ? '已核对原图哈希并具备当前完整人工真值回执'
        : '原图哈希已核对；请签署专项判断，或修正轮廓后走完整真值门禁';
    const proposal = row.assistant_proposal;
    assistantBox.textContent = proposal
      ? `助手预审（${proposal.confidence_band === 'high' ? '较高把握' : proposal.confidence_band === 'medium' ? '中等把握' : '低把握'}）：${decisionLabels[proposal.proposed_decision] || proposal.proposed_decision}。${proposal.rationale} 仅供参考，不会自动代选。`
      : '当前帧没有助手预审建议；请独立作出人工判断。';
    byId('reviewPacketDockPrevious').disabled = busy || packet.rows.length < 2;
    byId('reviewPacketDockNext').disabled = busy || packet.rows.length < 2;
    decisionSubmit.disabled = busy || !reviewerInput.value.trim() || !decisionSelect.value;
  }

  function rowNode(row, index) {
    const receipt = completed.get(Number(row.frame));
    const done = Boolean(receipt);
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `reviewPacketItem ${row.cohort === 'positive_proxy' ? 'positive' : 'negative'}${done ? ' complete' : ''}${index === currentIndex ? ' current' : ''}`;
    button.dataset.packetFrame = String(row.frame);
    const frame = document.createElement('span');
    frame.className = 'reviewPacketFrame';
    frame.textContent = done ? '✓' : String(row.frame);
    const copy = document.createElement('span');
    copy.className = 'reviewPacketCopy';
    const title = document.createElement('strong');
    title.textContent = `第 ${row.frame} 帧 · ${cohortLabel(row)}`;
    const detail = document.createElement('small');
    detail.textContent = receipt?.type === 'slope_scoped_decision'
      ? decisionLabels[receipt.decision] || receipt.decision
      : done ? '当前完整人工真值回执已验证'
        : `${row.candidate_count} 个斜坡候选 · 等待人工复核`;
    const route = document.createElement('span');
    route.textContent = done ? '专项完成，但不会自动回流训练'
      : row.assistant_proposal
        ? `助手建议：${decisionLabels[row.assistant_proposal.proposed_decision] || row.assistant_proposal.proposed_decision}`
        : `来源：${row.target_provenance || '未验证保存标注'}`;
    copy.append(title, detail, route);
    button.append(frame, copy);
    button.addEventListener('click', () => openRow(index));
    return button;
  }

  function render() {
    const rows = packet?.rows || [];
    const visible = rows.map((row, index) => ({row, index})).filter(item => matches(item.row));
    const done = rows.filter(row => completed.has(Number(row.frame))).length;
    const percent = rows.length ? Math.round(done / rows.length * 100) : 0;
    byId('reviewPacketProgressTitle').textContent = `${done} / ${rows.length} 帧已专项复核`;
    byId('reviewPacketProgressDetail').textContent = rows.length
      ? `${rows.length - done} 帧待处理 · 只认可显式人工决定、精确原图哈希和不可变审计回执`
      : '等待选择复核包';
    byId('reviewPacketProgressPercent').textContent = `${percent}%`;
    byId('reviewPacketProgressBar').setAttribute('aria-valuenow', String(percent));
    byId('reviewPacketProgressBar').querySelector('span').style.width = `${percent}%`;
    if (visible.length) queue.replaceChildren(...visible.map(item => rowNode(item.row, item.index)));
    else {
      const empty = document.createElement('div');
      empty.className = 'reviewPacketBoundary';
      empty.textContent = rows.length ? '当前筛选下没有帧。' : '当前复核包没有可用帧。';
      queue.replaceChildren(empty);
    }
    const badge = byId('reviewPacketBadge');
    badge.hidden = !rows.length;
    badge.textContent = String(rows.length - done);
    byId('openReviewPacketsButton').classList.toggle('ready', Boolean(rows.length));
    body.hidden = !rows.length;
    renderDock();
  }

  async function refreshCompletion() {
    if (!packet?.ready_to_navigate) { completed = new Map(); render(); return; }
    const [preparation, reviewData, decisionData] = await Promise.all([
      request('/api/reference-preparation'),
      request(`/api/human_reviews?topic=${encodeURIComponent(packet.topic)}`),
      request(`/api/review_packets/${encodeURIComponent(packet.packet_id)}/decisions`),
    ]);
    if (decisionData.manifest_sha256 !== packet.manifest_sha256) {
      throw new Error('专项复核决定索引与当前清单不一致');
    }
    completed = core.completionMap(packet, preparation, reviewData.reviews || [],
      decisionData.decisions || []);
    render();
  }

  async function loadPacket(packetId, refreshReceipts = true) {
    const data = await request(`/api/review_packets/${encodeURIComponent(packetId)}`);
    packet = data;
    completed = new Map();
    currentIndex = -1;
    baseline = null;
    decisionStartedAt = null;
    decisionKey = null;
    render();
    if (packet.ready_to_navigate && refreshReceipts) await refreshCompletion();
    setStatus(packet.ready_to_navigate
      ? '复核数据与 topic 已就绪；打开任一帧时还会逐帧核对原图 SHA-256。'
      : '复核包已读取；点击“校验并载入复核数据”会先保护当前文件会话，再切换到绑定数据。',
      packet.ready_to_navigate ? 'ready' : '');
  }

  async function refreshCatalog() {
    setBusy(true, '正在读取并校验本机专项复核包…');
    try {
      const data = await request('/api/review_packets');
      if (data.schema_version !== 'labelagent.review-packet-index.v1') throw new Error('专项复核包索引版本不兼容');
      catalog = data.packets || [];
      select.replaceChildren();
      for (const item of catalog) {
        const option = document.createElement('option');
        option.value = item.packet_id;
        option.textContent = packetLabel(item);
        select.append(option);
      }
      if (!catalog.length) {
        packet = null; completed = new Map(); render();
        setStatus(data.rejected_packets
          ? `没有可用复核包；${data.rejected_packets} 个本机清单未通过安全校验。`
          : '尚未安装本机专项复核包。', 'error');
        return;
      }
      const ready = catalog.find(item => item.ready_to_navigate) || catalog[0];
      select.value = ready.packet_id;
      await loadPacket(ready.packet_id);
      if (data.rejected_packets) setStatus(`${statusBox.textContent} 另有 ${data.rejected_packets} 个清单被拒绝。`, statusBox.classList.contains('ready') ? 'ready' : '');
    } catch (error) {
      packet = null; completed = new Map(); render(); setStatus(error.message, 'error');
    } finally { setBusy(false); }
  }

  async function activateSelected() {
    if (busy || !select.value) return;
    setBusy(true, '正在核对 MCAP 内容哈希并切换复核数据；不会写入真值或训练集…');
    try {
      window.cancelReferencePrefetch?.();
      const data = await request(`/api/review_packets/${encodeURIComponent(select.value)}/activate`, {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}',
      });
      const activation = data.review_packet_activation || {};
      if (activation.mcap_sha256_verified !== true || activation.topic_verified !== true
          || activation.human_truth_written !== false || activation.training_data_written !== false) {
        throw new Error('专项复核激活读回不完整，已停止进入队列');
      }
      if (typeof window.loadVerifiedVideoData !== 'function') throw new Error('工作台安全数据加载器不可用');
      await window.loadVerifiedVideoData(data);
      packet = {...data.review_packet, ready_to_navigate: true, active_file: true, active_topic: true};
      completed = new Map(); currentIndex = -1; baseline = null;
      await refreshCompletion();
      setStatus('MCAP 内容哈希、topic 和工作台读回均已通过；请选择一帧开始人工复核。', 'ready');
    } catch (error) { setStatus(error.message, 'error'); }
    finally { setBusy(false); render(); }
  }

  function currentDraftIsDirty() {
    if (!baseline || typeof window.reviewPacketWorkbenchSnapshot !== 'function') return false;
    return core.dirtyDraft(baseline, window.reviewPacketWorkbenchSnapshot());
  }

  async function ensureSafeNavigation() {
    if (!currentDraftIsDirty()) return true;
    return window.confirm('当前帧有尚未通过人工门禁保存的修改。继续会放弃这些本地修改并打开另一帧；确定继续吗？');
  }

  async function openRow(index) {
    if (busy || !packet?.rows?.[index]) return;
    if (!packet.ready_to_navigate) { await activateSelected(); if (!packet?.ready_to_navigate) return; }
    if (!await ensureSafeNavigation()) return;
    const row = packet.rows[index];
    setBusy(true, `正在校验第 ${row.frame} 帧原图哈希…`);
    try {
      const binding = await request('/api/human_review_binding', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({frame: row.frame, topic: packet.topic}),
      });
      if (String(binding.frame_sha256 || '').toLowerCase() !== row.source_frame_sha256) {
        throw new Error(`第 ${row.frame} 帧原图 SHA-256 与复核包不一致，已拒绝打开`);
      }
      if (binding.topic !== packet.topic || Number(binding.frame) !== Number(row.frame)) {
        throw new Error(`第 ${row.frame} 帧绑定读回与复核包作用域不一致`);
      }
      if (typeof window.openFrameForReferencePreparation !== 'function') throw new Error('工作台帧导航不可用');
      await window.openFrameForReferencePreparation(row.frame);
      const snapshot = window.reviewPacketWorkbenchSnapshot?.();
      if (!snapshot || core.normalizePath(snapshot.path) !== core.normalizePath(packet.mcap_path)
          || snapshot.topic !== packet.topic || Number(snapshot.frame) !== Number(row.frame)) {
        throw new Error('打开后工作台文件、topic 或帧号读回不一致');
      }
      baseline = {...snapshot, shapes_json: core.stableJson(snapshot.shapes || [])};
      currentIndex = index;
      decisionStartedAt = new Date().toISOString();
      decisionKey = makeDecisionKey();
      reviewerInput.value = sessionStorage.getItem('labelagentReviewerDeclaredId') || '';
      decisionSelect.value = '';
      decisionNote.value = '';
      dialog.close();
      render();
      window.labelAgentToast?.(`第 ${row.frame} 帧原图哈希已通过；可签署斜坡专项判断，修正轮廓则走完整真值门禁`);
    } catch (error) { setStatus(error.message, 'error'); window.labelAgentToast?.(error.message); }
    finally { setBusy(false); renderDock(); }
  }

  async function navigate(direction) {
    if (!packet?.rows?.length || busy) return;
    const start = currentIndex < 0 ? (direction < 0 ? 0 : -1) : currentIndex;
    const next = core.adjacentPendingIndex(packet.rows, completed, start, direction);
    if (next >= 0) await openRow(next);
  }

  async function submitScopedDecision() {
    if (busy || currentIndex < 0 || !packet?.rows?.[currentIndex]) return;
    const row = packet.rows[currentIndex];
    const reviewer = reviewerInput.value.trim();
    const decision = decisionSelect.value;
    if (!reviewer) { window.labelAgentToast?.('请填写复核人自报标识'); reviewerInput.focus(); return; }
    if (!row.allowed_human_decisions?.includes(decision)) {
      window.labelAgentToast?.('请选择当前复核包允许的人工判断'); decisionSelect.focus(); return;
    }
    if (currentDraftIsDirty()) {
      window.labelAgentToast?.('当前轮廓已经修改；请先通过完整真值门禁保存，或撤销修改后再签署专项判断');
      return;
    }
    const snapshot = window.reviewPacketWorkbenchSnapshot?.();
    if (!snapshot || core.normalizePath(snapshot.path) !== core.normalizePath(packet.mcap_path)
        || snapshot.topic !== packet.topic || Number(snapshot.frame) !== Number(row.frame)) {
      window.labelAgentToast?.('当前工作台作用域已变化，请重新打开该帧');
      return;
    }
    const slopeGeometry = (snapshot.shapes || []).filter(shape =>
      String(shape?.label || '').trim().toLowerCase() === 'slope');
    const slopeGeometrySha256 = await sha256Json(slopeGeometry);
    sessionStorage.setItem('labelagentReviewerDeclaredId', reviewer);
    setBusy(true, '正在绑定原图、斜坡候选几何并写入专项人工决定；不会写入整帧真值或训练集…');
    try {
      const data = await request(`/api/review_packets/${encodeURIComponent(packet.packet_id)}/decisions`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          frame: row.frame, decision,
          expected_manifest_sha256: packet.manifest_sha256,
          expected_frame_sha256: row.source_frame_sha256,
          expected_slope_geometry_sha256: slopeGeometrySha256,
          expected_assistant_proposal_sha256: row.assistant_proposal?.proposal_sha256 || null,
          review: {reviewer_declared_id: reviewer,
            started_at: decisionStartedAt || new Date().toISOString(),
            note: decisionNote.value.trim(),
            decision_idempotency_key: decisionKey || makeDecisionKey()},
        }),
      });
      const receipt = data.decision_receipt || {};
      if (data.readback?.verified !== true || data.human_truth_written !== false
          || data.training_data_written !== false || receipt.frame !== row.frame
          || receipt.frame_sha256 !== row.source_frame_sha256
          || receipt.slope_geometry_sha256 !== slopeGeometrySha256
          || receipt.decision !== decision || receipt.human_truth !== false
          || receipt.approved_for_training !== false) {
        throw new Error('专项人工决定写入后读回不完整');
      }
      decisionStartedAt = new Date().toISOString();
      decisionKey = makeDecisionKey();
      await refreshCompletion();
      setStatus(`第 ${row.frame} 帧专项人工决定已读回；没有写入整帧真值或训练集。`, 'ready');
      window.labelAgentToast?.(`第 ${row.frame} 帧专项判断已签署；可进入下一帧`);
    } catch (error) {
      setStatus(error.message, 'error');
      window.labelAgentToast?.(error.message);
    } finally { setBusy(false); renderDock(); }
  }

  byId('openReviewPacketsButton').addEventListener('click', async () => {
    dialog.showModal();
    if (!catalog.length) await refreshCatalog();
    else render();
  });
  byId('reviewPacketClose').addEventListener('click', () => dialog.close());
  byId('reviewPacketRefresh').addEventListener('click', refreshCatalog);
  activateButton.addEventListener('click', activateSelected);
  select.addEventListener('change', async () => {
    if (busy || !select.value) return;
    setBusy(true, '正在读取复核包帧清单…');
    try { await loadPacket(select.value); }
    catch (error) { setStatus(error.message, 'error'); }
    finally { setBusy(false); }
  });
  dialog.querySelectorAll('[data-packet-filter]').forEach(button => button.addEventListener('click', () => {
    activeFilter = button.dataset.packetFilter;
    dialog.querySelectorAll('[data-packet-filter]').forEach(item => item.classList.toggle('selected', item === button));
    render();
  }));
  byId('reviewPacketDockQueue').addEventListener('click', () => { dialog.showModal(); render(); });
  byId('reviewPacketDockPrevious').addEventListener('click', () => navigate(-1));
  byId('reviewPacketDockNext').addEventListener('click', () => navigate(1));
  reviewerInput.addEventListener('input', renderDock);
  decisionSelect.addEventListener('change', renderDock);
  decisionSubmit.addEventListener('click', submitScopedDecision);
  document.addEventListener('labelagent:human-truth-saved', async event => {
    if (!packet || core.normalizePath(event.detail?.path) !== core.normalizePath(packet.mcap_path)
        || event.detail?.topic !== packet.topic) return;
    try {
      await refreshCompletion();
      const snapshot = window.reviewPacketWorkbenchSnapshot?.();
      if (snapshot) baseline = {...snapshot, shapes_json: core.stableJson(snapshot.shapes || [])};
      setStatus(`第 ${event.detail.frame} 帧正式人工回执已读回；可以进入下一帧。`, 'ready');
    } catch (error) { setStatus(`真值已保存，但专项队列读回失败：${error.message}`, 'error'); }
  });
  dialog.addEventListener('cancel', event => { event.preventDefault(); dialog.close(); });
  refreshCatalog();
})();
