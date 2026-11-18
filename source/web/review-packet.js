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
  let catalog = [];
  let packet = null;
  let completed = new Map();
  let activeFilter = 'all';
  let currentIndex = -1;
  let baseline = null;
  let busy = false;

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
    const done = completed.has(Number(row.frame));
    dock.hidden = false;
    byId('reviewPacketDockTitle').textContent = `斜坡专项 · 第 ${row.frame} 帧 · ${currentIndex + 1}/${packet.rows.length}`;
    byId('reviewPacketDockDetail').textContent = done
      ? '已核对原图哈希并具备当前人工签名回执'
      : '原图哈希已核对；仍需人工检查并点击“保存当前帧真值”';
    byId('reviewPacketDockPrevious').disabled = busy || packet.rows.length < 2;
    byId('reviewPacketDockNext').disabled = busy || packet.rows.length < 2;
  }

  function rowNode(row, index) {
    const done = completed.has(Number(row.frame));
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
    detail.textContent = done ? '当前人工回执已验证' : `${row.candidate_count} 个斜坡候选 · 等待人工复核`;
    const route = document.createElement('span');
    route.textContent = done ? '不会自动回流训练' : `来源：${row.target_provenance || '未验证保存标注'}`;
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
    byId('reviewPacketProgressTitle').textContent = `${done} / ${rows.length} 帧已正式复核`;
    byId('reviewPacketProgressDetail').textContent = rows.length
      ? `${rows.length - done} 帧待处理 · 只认可当前人工门禁、精确原图哈希和已提交事务`
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
    const [preparation, reviewData] = await Promise.all([
      request('/api/reference-preparation'),
      request(`/api/human_reviews?topic=${encodeURIComponent(packet.topic)}`),
    ]);
    completed = core.completionMap(packet, preparation, reviewData.reviews || []);
    render();
  }

  async function loadPacket(packetId, refreshReceipts = true) {
    const data = await request(`/api/review_packets/${encodeURIComponent(packetId)}`);
    packet = data;
    completed = new Map();
    currentIndex = -1;
    baseline = null;
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
      if (typeof window.loadVideoData !== 'function') throw new Error('工作台数据加载器不可用');
      await window.loadVideoData(data);
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
      dialog.close();
      render();
      window.labelAgentToast?.(`第 ${row.frame} 帧原图哈希已通过；请人工检查轮廓，确认后点击“保存当前帧真值”`);
    } catch (error) { setStatus(error.message, 'error'); window.labelAgentToast?.(error.message); }
    finally { setBusy(false); renderDock(); }
  }

  async function navigate(direction) {
    if (!packet?.rows?.length || busy) return;
    const start = currentIndex < 0 ? (direction < 0 ? 0 : -1) : currentIndex;
    const next = core.adjacentPendingIndex(packet.rows, completed, start, direction);
    if (next >= 0) await openRow(next);
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
