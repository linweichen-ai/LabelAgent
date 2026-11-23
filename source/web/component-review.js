(() => {
  const byId = id => document.getElementById(id);
  const dialog = byId('componentReviewDialog');
  const openButton = byId('componentReviewOpen');
  if (!dialog || !openButton || !window.LabelAgentComponentReviewCore) return;

  const core = window.LabelAgentComponentReviewCore;
  const select = byId('componentReviewSelect');
  const statusBox = byId('componentReviewStatus');
  const body = byId('componentReviewBody');
  const grid = byId('componentReviewGrid');
  const filters = [...dialog.querySelectorAll('[data-component-filter]')];
  const storageKey = 'labelagentComponentReviewDraftsV1';
  const pageSize = 4;
  let catalog = [];
  let packet = null;
  let activeFilter = 'all';
  let page = 0;
  let loading = false;
  let loadedImages = new Set();
  let failedImages = new Set();

  function readDrafts() {
    try { return JSON.parse(sessionStorage.getItem(storageKey) || '{}'); } catch (_) { return {}; }
  }

  function writeDrafts(value) {
    try { sessionStorage.setItem(storageKey, JSON.stringify(value)); } catch (_) {}
  }

  function setStatus(message, tone = '') {
    statusBox.className = `componentReviewStatus ${tone}`.trim();
    statusBox.textContent = message;
  }

  function setBusy(value) {
    loading = value;
    byId('componentReviewLoad').disabled = value || !select.value;
    byId('componentReviewRefresh').disabled = value;
  }

  async function requestJson(url) {
    const response = await fetch(url, {method: 'GET', cache: 'no-store'});
    const value = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(value.error || `请求失败（HTTP ${response.status}）`);
    return value;
  }

  function stat(label, value, tone = '') {
    const node = document.createElement('span');
    node.className = `componentReviewStat ${tone}`.trim();
    const number = document.createElement('b');
    const copy = document.createElement('small');
    number.textContent = String(value);
    copy.textContent = label;
    node.append(number, copy);
    return node;
  }

  function saveDraft(row, decision) {
    if (!packet || !loadedImages.has(row.component_id)
        || failedImages.has(row.component_id)) return;
    const drafts = readDrafts();
    const key = core.draftKey(packet.manifest_sha256, row.component_id);
    if (!decision) delete drafts[key];
    else drafts[key] = {
      status: decision,
      recorded_at: new Date().toISOString(),
      session_only: true,
      local_only: packet.evidence_mode === 'local_only',
      evidence_loaded: true,
      manifest_sha256: packet.manifest_sha256,
      image_sha256: row.focused_image_sha256,
    };
    writeDrafts(drafts);
    render();
  }

  function draftName(value) {
    return {
      likely_fill_ok: '倾向可补', uncertain: '不确定',
      likely_reject: '倾向驳回',
    }[value] || '待判断';
  }

  function renderCard(row, drafts) {
    const draft = core.rowDraft(packet, row, drafts);
    const card = document.createElement('article');
    card.className = `componentReviewCard ${row.policy_queue} ${packet.evidence_mode || ''}`.trim();

    const head = document.createElement('div');
    head.className = 'componentReviewCardHead';
    const identity = document.createElement('div');
    const title = document.createElement('strong');
    const meta = document.createElement('small');
    title.textContent = row.component_id;
    meta.textContent = `${packet.display_name} · 排名 ${row.review_rank} · ${row.policy_queue === 'priority_review' ? '优先复核' : '延后复核'}`;
    identity.append(title, meta);
    const badge = document.createElement('span');
    badge.className = `componentDraftBadge ${draft?.status || 'pending'}`;
    badge.textContent = draftName(draft?.status);
    head.append(identity, badge);

    const figure = document.createElement('figure');
    const image = document.createElement('img');
    image.loading = 'lazy';
    image.decoding = 'async';
    image.alt = `${row.component_id} 聚焦复核图`;
    image.src = row.image_url;
    const caption = document.createElement('figcaption');
    caption.textContent = row.policy_reason;
    figure.append(image, caption);

    const actions = document.createElement('div');
    actions.className = 'componentReviewActions';
    const definitions = [
      ['likely_fill_ok', '倾向可补'], ['uncertain', '不确定'],
      ['likely_reject', '倾向驳回'], ['', '清除草稿'],
    ];
    const buttons = [];
    definitions.forEach(([value, label]) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = label;
      button.disabled = true;
      button.classList.toggle('selected', Boolean(value) && draft?.status === value);
      button.addEventListener('click', () => saveDraft(row, value));
      buttons.push(button);
      actions.append(button);
    });
    image.addEventListener('load', () => {
      loadedImages.add(row.component_id);
      failedImages.delete(row.component_id);
      buttons.forEach(button => { button.disabled = false; });
    }, {once: true});
    image.addEventListener('error', () => {
      loadedImages.delete(row.component_id);
      failedImages.add(row.component_id);
      card.classList.add('assetError');
      caption.textContent = '图片校验或读取失败；本项草稿按钮已禁用。';
      buttons.forEach(button => { button.disabled = true; });
      badge.className = 'componentDraftBadge pending';
      badge.textContent = '图片不可用';
      renderSummary();
    }, {once: true});

    card.append(head, figure, actions);
    return card;
  }

  function effectiveDrafts() {
    const drafts = readDrafts();
    if (!packet || !failedImages.size) return drafts;
    const usable = {...drafts};
    packet.rows.forEach(row => {
      if (failedImages.has(row.component_id)) {
        delete usable[core.draftKey(packet.manifest_sha256, row.component_id)];
      }
    });
    return usable;
  }

  function renderSummary() {
    if (!packet) return;
    const summary = core.summarize(packet, effectiveDrafts());
    const percent = summary.total ? Math.round(summary.reviewed / summary.total * 100) : 0;
    byId('componentReviewProgressTitle').textContent = `${summary.reviewed} / ${summary.total} 项已暂存`;
    byId('componentReviewProgressDetail').textContent = `仅当前会话 · 待判断 ${summary.pending} · 清单 ${packet.manifest_sha256.slice(0, 12)}…`;
    byId('componentReviewProgressPercent').textContent = `${percent}%`;
    byId('componentReviewStats').replaceChildren(
      stat('优先复核', packet.priority_review, packet.priority_review ? 'warning' : ''),
      stat('延后复核', packet.deferred_manual_review),
      stat('倾向可补', summary.likely_fill_ok, 'safe'),
      stat('不确定', summary.uncertain, 'warning'),
      stat('倾向驳回', summary.likely_reject, 'danger'),
    );
  }

  function render() {
    if (!packet) return;
    const drafts = effectiveDrafts();
    const rows = core.filterRows(packet, drafts, activeFilter);
    const pageCount = Math.max(1, Math.ceil(rows.length / pageSize));
    page = Math.min(page, pageCount - 1);
    const visible = rows.slice(page * pageSize, (page + 1) * pageSize);
    renderSummary();
    grid.replaceChildren(...visible.map(row => renderCard(row, drafts)));
    if (!visible.length) {
      const empty = document.createElement('div');
      empty.className = 'componentReviewEmpty';
      empty.textContent = '当前筛选没有组件。';
      grid.append(empty);
    }
    byId('componentReviewPage').textContent = `第 ${page + 1} / ${pageCount} 页 · ${rows.length} 项`;
    byId('componentReviewPrevious').disabled = loading || page <= 0;
    byId('componentReviewNext').disabled = loading || page >= pageCount - 1;
    setStatus(`已校验 ${packet.review_count} 项聚焦证据；只读图片，判断仅存于当前浏览器会话。`, 'ready');
  }

  async function loadSelected() {
    if (!select.value || loading) return;
    setBusy(true);
    setStatus('正在校验清单和每张聚焦图片…', 'checking');
    try {
      const value = await requestJson(`/api/component_review_packs/${encodeURIComponent(select.value)}`);
      if (!['labelagent.ego_tail_focused_review_pack.v1',
            'labelagent.component-review-pack.v2'].includes(value.schema_version)
          || value.read_only_packet !== true || value.session_drafts_only !== true
          || value.submits_human_truth !== false || value.training_data_writes !== false) {
        throw new Error('复核包隔离边界不完整，已拒绝载入');
      }
      packet = value;
      loadedImages = new Set();
      failedImages = new Set();
      page = 0;
      body.hidden = false;
      render();
    } catch (error) {
      packet = null;
      body.hidden = true;
      setStatus(error.message || '复核包载入失败', 'error');
    } finally {
      setBusy(false);
    }
  }

  async function refreshCatalog(silent = false) {
    if (loading) return;
    setBusy(true);
    if (!silent) setStatus('正在检查本机复核包…', 'checking');
    try {
      const value = await requestJson('/api/component_review_packs');
      if (value.schema_version !== 'labelagent.component-review-pack-index.v1'
          || value.human_truth_writes !== false
          || value.training_data_writes !== false
          || value.automatic_decisions !== false) {
        throw new Error('复核包索引隔离边界不完整');
      }
      catalog = Array.isArray(value.packets) ? value.packets : [];
      const previous = select.value;
      select.replaceChildren();
      catalog.forEach(item => {
        const option = new Option(`${item.display_name || item.packet_id} · ${item.review_count} 项`, item.packet_id);
        select.append(option);
      });
      select.value = catalog.some(item => item.packet_id === previous)
        ? previous : catalog[0]?.packet_id || '';
      byId('componentReviewBadge').textContent = catalog.length
        ? String(catalog.reduce((sum, item) => sum + item.review_count, 0)) : '0';
      openButton.classList.toggle('ready', catalog.length > 0);
      if (!silent) setStatus(catalog.length
        ? `发现 ${catalog.length} 个只读复核包。`
        : '没有发现可用的组件证据复核包。', catalog.length ? 'ready' : 'empty');
    } catch (error) {
      catalog = [];
      select.replaceChildren();
      byId('componentReviewBadge').textContent = '!';
      openButton.classList.remove('ready');
      if (!silent) setStatus(error.message || '无法读取复核包', 'error');
    } finally {
      setBusy(false);
    }
  }

  openButton.addEventListener('click', async () => {
    dialog.showModal();
    await refreshCatalog();
    if (catalog.length) await loadSelected();
  });
  byId('componentReviewClose').addEventListener('click', () => dialog.close());
  byId('componentReviewRefresh').addEventListener('click', () => refreshCatalog());
  byId('componentReviewLoad').addEventListener('click', loadSelected);
  byId('componentReviewPrevious').addEventListener('click', () => { page -= 1; render(); });
  byId('componentReviewNext').addEventListener('click', () => { page += 1; render(); });
  filters.forEach(button => button.addEventListener('click', () => {
    activeFilter = button.dataset.componentFilter;
    filters.forEach(item => item.classList.toggle('selected', item === button));
    page = 0;
    render();
  }));
  refreshCatalog(true);
})();
