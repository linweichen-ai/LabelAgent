(() => {
  const byId = id => document.getElementById(id);
  const dialog = byId('candidateComparisonDialog');
  const openButton = byId('candidateComparisonOpen');
  const core = window.LabelAgentCandidateComparisonCore;
  if (!dialog || !openButton || !core) return;

  const packSelect = byId('candidateComparisonPack');
  const rowSelect = byId('candidateComparisonRow');
  const content = byId('candidateComparisonContent');
  const storageKey = 'labelagentCandidateComparisonPreferencesV1';
  let packet = null;
  let row = null;
  let scope = null;
  let loadedImages = new Set();
  let currentImageLoaded = false;
  let preferenceButtons = [];
  let requestGeneration = 0;

  function readPreferences() {
    try {
      const value = JSON.parse(sessionStorage.getItem(storageKey) || '{}');
      return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
    } catch (_) { return {}; }
  }
  function writePreferences(value) {
    try { sessionStorage.setItem(storageKey, JSON.stringify(value)); } catch (_) {}
  }
  function setStatus(message, tone = '') {
    const node = byId('candidateComparisonStatus');
    node.className = `candidateComparisonStatus ${tone}`.trim();
    node.textContent = message;
  }
  async function getJson(url) {
    const response = await fetch(url, {method: 'GET', cache: 'no-store'});
    const value = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(value.error || `请求失败（HTTP ${response.status}）`);
    return value;
  }
  function workspace() {
    const state = window.currentWorkbenchFrameState?.() || {};
    const binding = state.candidate_review_bindings?.[String(state.frame)];
    const candidate = binding?.candidates?.find(item =>
      item.candidate_id === row?.current_candidate_id);
    return {path: state.path, topic: state.topic, frame: state.frame,
      file_activation_generation: state.file_activation_generation,
      current_candidate_artifact_sha256: binding?.candidate_artifact_sha256,
      current_candidate_fingerprint: candidate?.candidate_fingerprint};
  }
  function scopeUrl() {
    const current = workspace();
    const query = new URLSearchParams({path: current.path || '', topic: current.topic || '',
      frame: String(current.frame), file_activation_generation: String(current.file_activation_generation),
      expected_manifest_sha256: packet.manifest_sha256});
    return `/api/candidate_comparison_packs/${encodeURIComponent(packet.packet_id)}/${encodeURIComponent(row.comparison_id)}/scope?${query}`;
  }
  function metricText(metrics) {
    const entries = Object.entries(metrics || {});
    return entries.length ? entries.map(([key, value]) => `${key}: ${value}`).join(' · ') : '无附加指标';
  }
  function preferenceFor(variant) {
    const value = core.binding(packet, row, variant);
    return {...value, local_only: true, read_only_preview: true,
      scope_verified: true, recorded_at: new Date().toISOString()};
  }
  function selectVariant(variant) {
    if (!scope || !core.scopeMatches(scope, packet, row, workspace())
        || !currentImageLoaded || !loadedImages.has(variant.alternative_id)) return;
    const preferences = readPreferences();
    preferences[core.preferenceKey(packet, row)] = preferenceFor(variant);
    writePreferences(preferences);
    render();
  }
  function clearPreference() {
    const preferences = readPreferences();
    delete preferences[core.preferenceKey(packet, row)];
    writePreferences(preferences);
    render();
  }
  function preview(titleText, asset, meta, variant = null) {
    const imageGeneration = requestGeneration;
    const card = document.createElement('article');
    card.className = 'candidateComparisonCard';
    const title = document.createElement('strong');
    title.textContent = titleText;
    const image = document.createElement('img');
    image.alt = `${titleText}只读预览`;
    image.src = asset.image_url;
    image.loading = 'lazy';
    image.decoding = 'async';
    const detail = document.createElement('small');
    detail.textContent = meta;
    card.append(title, image, detail);
    if (variant) {
      const button = document.createElement('button');
      button.type = 'button';
      const selected = core.selectedPreference(packet, row, readPreferences());
      button.textContent = selected?.alternative_id === variant.alternative_id
        ? '已设为会话查看偏好' : '设为会话查看偏好';
      button.classList.toggle('selected', selected?.alternative_id === variant.alternative_id);
      button.disabled = !scope || !currentImageLoaded || !loadedImages.has(variant.alternative_id);
      button.addEventListener('click', () => selectVariant(variant));
      preferenceButtons.push([button, variant.alternative_id]);
      image.addEventListener('load', () => {
        if (imageGeneration !== requestGeneration) return;
        loadedImages.add(variant.alternative_id);
        preferenceButtons.forEach(([item, id]) => {
          item.disabled = !scope || !currentImageLoaded || !loadedImages.has(id);
        });
      }, {once: true});
      image.addEventListener('error', () => {
        if (imageGeneration !== requestGeneration) return;
        loadedImages.delete(variant.alternative_id); button.disabled = true;
        detail.textContent = '图片载入失败；本候选不可选择。';
      }, {once: true});
      card.append(button);
    } else {
      image.addEventListener('load', () => {
        if (imageGeneration !== requestGeneration) return;
        currentImageLoaded = true;
        preferenceButtons.forEach(([button, id]) => {
          button.disabled = !scope || !loadedImages.has(id);
        });
      }, {once: true});
      image.addEventListener('error', () => {
        if (imageGeneration !== requestGeneration) return;
        currentImageLoaded = false;
        detail.textContent = '当前候选图片载入失败；所有会话偏好已禁用。';
        preferenceButtons.forEach(([button]) => { button.disabled = true; });
      }, {once: true});
    }
    return card;
  }
  function render() {
    if (!packet || !row || !scope) { content.replaceChildren(); return; }
    preferenceButtons = [];
    const header = document.createElement('div');
    header.className = 'candidateComparisonSummary';
    const text = document.createElement('div');
    const title = document.createElement('strong');
    title.textContent = `帧 ${row.frame} · 候选 ${String(row.current_candidate_id)}`;
    const note = document.createElement('small');
    note.textContent = `清单 ${packet.manifest_sha256.slice(0, 12)}… · 已绑定当前原图和候选几何`;
    text.append(title, note);
    const clear = document.createElement('button');
    clear.type = 'button'; clear.textContent = '清除会话偏好'; clear.addEventListener('click', clearPreference);
    header.append(text, clear);
    const grid = document.createElement('div');
    grid.className = 'candidateComparisonGrid';
    grid.append(preview('当前候选', row.current,
      `像素 ${row.current.pixel_count ?? '—'} · ${metricText(row.current.metrics)}`));
    row.variants.forEach(variant => grid.append(preview(
      variant.kind === 'consensus' ? `共识 · ${variant.label || variant.alternative_id}` : `替代 · ${variant.label || variant.alternative_id}`,
      variant.preview,
      `策略 ${variant.policy || '—'} · 像素 ${variant.pixel_count} · ${metricText(variant.metrics)}`,
      variant)));
    content.replaceChildren(header, grid);
  }
  async function verifySelected(generation = ++requestGeneration) {
    if (!packet || !row) return;
    const selectedPacket = packet;
    const selectedRow = row;
    const url = scopeUrl();
    scope = null; loadedImages = new Set(); currentImageLoaded = false; content.replaceChildren();
    setStatus('正在校验当前文件、相机、帧、原图和候选绑定…', 'checking');
    try {
      const value = await getJson(url);
      if (!core.requestStillCurrent(generation, requestGeneration,
          selectedPacket.packet_id, packet?.packet_id,
          selectedRow.comparison_id, row?.comparison_id)) return;
      if (!core.scopeMatches(value, selectedPacket, selectedRow, workspace())) throw new Error('作用域读回不一致');
      scope = value; render();
      setStatus('作用域已锁定。这里只比较预览；不会采用候选或写入人工真值。', 'ready');
    } catch (error) {
      if (!core.requestStillCurrent(generation, requestGeneration,
          selectedPacket.packet_id, packet?.packet_id,
          selectedRow.comparison_id, row?.comparison_id)) return;
      scope = null; content.replaceChildren();
      setStatus(error.message || '当前作用域校验失败', 'error');
    }
  }
  async function loadPack(generation = ++requestGeneration) {
    if (!packSelect.value) return;
    const selectedPacketId = packSelect.value;
    scope = null; content.replaceChildren();
    setStatus('正在读取并校验候选对比包…', 'checking');
    try {
      const value = await getJson(`/api/candidate_comparison_packs/${encodeURIComponent(selectedPacketId)}`);
      if (!core.requestStillCurrent(generation, requestGeneration,
          selectedPacketId, packSelect.value, '', '')) return;
      if (value.schema_version !== 'labelagent.candidate-comparison-pack.v1'
          || value.read_only_packet !== true || value.session_preferences_only !== true
          || value.human_truth_writes !== false || value.training_data_writes !== false
          || value.automatic_approval !== false) throw new Error('候选对比包隔离边界不完整');
      packet = value;
      rowSelect.replaceChildren(...value.rows.map(item => new Option(
        `帧 ${item.frame} · 候选 ${String(item.current_candidate_id)} · ${item.comparison_id}`,
        item.comparison_id)));
      row = value.rows[0] || null;
      if (row) await verifySelected(generation);
    } catch (error) {
      if (generation !== requestGeneration || selectedPacketId !== packSelect.value) return;
      packet = null; row = null; rowSelect.replaceChildren();
      setStatus(error.message || '候选对比包载入失败', 'error');
    }
  }
  async function refreshCatalog() {
    const generation = ++requestGeneration;
    setStatus('正在检查本机候选对比包…', 'checking');
    try {
      const value = await getJson('/api/candidate_comparison_packs');
      if (generation !== requestGeneration) return;
      if (value.schema_version !== 'labelagent.candidate-comparison-pack-index.v1'
          || value.read_only !== true || value.session_preferences_only !== true
          || value.human_truth_writes !== false || value.training_data_writes !== false
          || value.automatic_approval !== false) throw new Error('候选对比索引隔离边界不完整');
      packSelect.replaceChildren(...value.packets.map(item =>
        new Option(`${item.packet_id} · ${item.comparison_count} 项`, item.packet_id)));
      byId('candidateComparisonBadge').textContent = String(value.packets.length);
      openButton.classList.toggle('ready', value.packets.length > 0);
      if (value.packets.length) await loadPack(generation);
      else setStatus('没有发现可用的候选对比包。', 'empty');
    } catch (error) {
      if (generation !== requestGeneration) return;
      packSelect.replaceChildren(); rowSelect.replaceChildren();
      byId('candidateComparisonBadge').textContent = '!';
      openButton.classList.remove('ready');
      setStatus(error.message || '无法读取候选对比包', 'error');
    }
  }
  function workspaceChanged() {
    if (!dialog.open || !packet || !row) return;
    if (!scope || !core.scopeMatches(scope, packet, row, workspace())) {
      requestGeneration += 1;
      scope = null; loadedImages = new Set(); currentImageLoaded = false; content.replaceChildren();
      setStatus('当前文件、相机、帧或候选已变化；旧对比已失效，请重新校验。', 'error');
    }
  }
  openButton.addEventListener('click', async () => { dialog.showModal(); await refreshCatalog(); });
  byId('candidateComparisonClose').addEventListener('click', () => {
    requestGeneration += 1; scope = null; content.replaceChildren(); dialog.close();
  });
  byId('candidateComparisonRefresh').addEventListener('click', refreshCatalog);
  byId('candidateComparisonLoad').addEventListener('click', loadPack);
  packSelect.addEventListener('change', () => {
    requestGeneration += 1; scope = null; content.replaceChildren();
    setStatus('已切换对比包；请点击“载入”重新校验。', 'checking');
  });
  rowSelect.addEventListener('change', async () => {
    row = packet?.rows.find(item => item.comparison_id === rowSelect.value) || null;
    await verifySelected();
  });
  document.addEventListener('labelagent:frame-state', workspaceChanged);
  byId('frameNumber')?.addEventListener('change', () => window.setTimeout(workspaceChanged, 0));
})();
