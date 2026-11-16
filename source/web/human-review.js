(() => {
  'use strict';

  const byId = id => document.getElementById(id);
  const dialog = byId('humanReviewDialog');
  if (!dialog) return;

  const reviewerInput = byId('humanReviewerDeclaredId');
  const noteInput = byId('humanReviewNote');
  const bindingBox = byId('humanReviewBinding');
  const summary = byId('humanReviewDecisionSummary');
  const warning = byId('humanReviewWarning');
  const emptyRow = byId('confirmedEmptyRow');
  const emptyCheck = byId('confirmedEmptyCheck');
  const commitButton = byId('humanReviewCommit');
  const rejectButton = byId('humanReviewReject');
  const cancelButton = byId('humanReviewCancel');
  const closeButton = byId('humanReviewClose');
  let active = null;

  function shortHash(value) {
    return value ? `${value.slice(0, 10)}…${value.slice(-8)}` : '无';
  }

  function makeDecisionKey() {
    if (globalThis.crypto?.randomUUID) return `human-review-${crypto.randomUUID()}`;
    return `human-review-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  async function post(path, body) {
    const response = await fetch(`/api/${path}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const raw = await response.text();
    let data;
    try { data = JSON.parse(raw); }
    catch { throw new Error(`人工复核接口返回异常（HTTP ${response.status}）`); }
    if (!response.ok) {
      const error = new Error(data.error || `人工复核失败（HTTP ${response.status}）`);
      error.staleReview = Boolean(data.stale_review);
      throw error;
    }
    return data;
  }

  function reviewPayload() {
    const reviewer = reviewerInput.value.trim();
    if (!reviewer) throw new Error('请填写复核人自报标识');
    sessionStorage.setItem('labelagentReviewerDeclaredId', reviewer);
    const review = {
      reviewer_declared_id: reviewer,
      started_at: active.startedAt,
      note: noteInput.value.trim(),
      decision_idempotency_key: active.decisionKey
    };
    const status = active.binding.review_status;
    if (status?.status === 'rejected' && status.review_id) {
      review.remediation_of_review_id = status.review_id;
    }
    return review;
  }

  function bindingPayload() {
    const binding = active.binding;
    const payload = {
      topic: binding.topic,
      expected_frame_sha256: binding.frame_sha256,
      expected_candidate_sha256: binding.candidate_sha256,
      expected_dataset_id: binding.dataset_id,
      expected_dataset_version: binding.dataset_version,
      expected_camera: binding.camera,
      expected_dataset_split: binding.dataset_split,
      expected_split_policy: binding.split_policy,
      expected_training_return_allowed: binding.training_return_allowed
    };
    if (binding.candidate_origin === 'browser_sam_point_draft') {
      payload.draft_candidate = active.request.shapes;
      payload.draft_source = 'sam_point_prompt';
    }
    return payload;
  }

  function setBusy(busy, message = '') {
    const reviewerReady = Boolean(reviewerInput.value.trim());
    commitButton.disabled = busy || active?.unknownBlocked || !reviewerReady;
    rejectButton.disabled = busy || !active?.binding?.candidate_sha256 || !reviewerReady;
    cancelButton.disabled = busy;
    closeButton.disabled = busy;
    reviewerInput.setAttribute('aria-invalid', String(Boolean(active) && !reviewerReady));
    if (message) warning.textContent = message;
    else if (active && !reviewerReady) warning.textContent = '请先填写复核人自报标识；当前身份只记录为 self_declared，不代表已经登录认证。';
    else if (active?.baseWarning) warning.textContent = active.baseWarning;
  }

  function finish(value, error) {
    const pending = active;
    active = null;
    if (dialog.open) dialog.close();
    if (!pending) return;
    if (error) pending.reject(error);
    else pending.resolve(value);
  }

  async function commit() {
    if (!active) return;
    try {
      if (!active.request.shapes.length && !emptyCheck.checked) {
        throw new Error('空标注必须勾选“确认当前帧确实没有目标”');
      }
      if (active.unknownBlocked) {
        throw new Error('未知或新类别只能进入 ontology 人工复核，不能直接保存为真值');
      }
      setBusy(true, '正在由服务器重新校验图像、候选和数据版本……');
      const payload = {
        frame: active.request.frame,
        shapes: active.request.shapes,
        review: reviewPayload(),
        ...bindingPayload()
      };
      if (!active.request.shapes.length) {
        payload.confirmed_empty = true;
        payload.decision = 'confirmed_empty';
      }
      const result = await post('truth', payload);
      if (!result.readback?.verified) throw new Error('服务器保存后读回校验未通过');
      finish(result);
    } catch (error) {
      setBusy(false, error.staleReview
        ? '当前图像或候选已经变化，请关闭后重新复核。'
        : error.message);
    }
  }

  async function rejectCandidate() {
    if (!active) return;
    try {
      if (!active.binding.candidate_sha256) throw new Error('当前帧没有可驳回的机器候选');
      setBusy(true, '正在驳回候选并生成不可变复核记录……');
      const result = await post('human_review', {
        frame: active.request.frame,
        review: reviewPayload(),
        ...bindingPayload()
      });
      if (!result.readback?.verified || !result.rejected) {
        throw new Error('候选驳回后的服务器读回校验未通过');
      }
      finish({...result, saved: 0, rejected: true});
    } catch (error) {
      setBusy(false, error.staleReview
        ? '当前图像或候选已经变化，请关闭后重新复核。'
        : error.message);
    }
  }

  function cancel() {
    if (!active) return;
    finish(null, new Error('已取消保存，未写入任何人工真值'));
  }

  async function openGate(request) {
    if (active) throw new Error('已有人工复核窗口正在处理');
    const startedAt = new Date().toISOString();
    const bindingRequest = {frame: request.frame, topic: request.topic};
    if (request.candidateSource === 'sam_point_prompt') {
      bindingRequest.draft_candidate = request.shapes;
      bindingRequest.draft_source = 'sam_point_prompt';
    }
    const binding = await post('human_review_binding', bindingRequest);
    const unknownBlocked = request.shapes.some(shape => {
      const label = String(shape?.label || '').trim().toLowerCase();
      return label.startsWith('unknown') || label === 'new_label_candidate';
    });
    const hasCandidate = Boolean(binding.candidate_sha256);
    const hasShapes = request.shapes.length > 0;
    reviewerInput.value = sessionStorage.getItem('labelagentReviewerDeclaredId') || '';
    noteInput.value = '';
    emptyCheck.checked = false;
    emptyRow.hidden = hasShapes;
    bindingBox.innerHTML = `
      <span><b>图像 SHA-256</b>${shortHash(binding.frame_sha256)}</span>
      <span><b>候选 SHA-256</b>${shortHash(binding.candidate_sha256)}</span>
      <span><b>候选来源</b>${binding.candidate_origin === 'browser_sam_point_draft' ? 'SAM 点提示草稿' : binding.candidate_origin === 'server_state' ? '服务器机器候选' : '纯手工绘制'}</span>
      <span><b>数据版本</b>${binding.dataset_version}</span>
      <span><b>相机</b>${binding.camera}</span>
      <span><b>数据分区</b>${binding.dataset_split}</span>
      <span><b>回流策略</b>${binding.split_policy}</span>`;
    const decision = !hasShapes ? '确认空标注'
      : hasCandidate ? '接受或修正机器候选' : '保存人工绘制轮廓';
    summary.innerHTML = `<b>${decision}</b><span>${request.shapes.length} 个轮廓 · 第 ${request.frame} 帧</span>`;
    const baseWarning = unknownBlocked
      ? '检测到 unknown/new_label_candidate：禁止直接晋升真值，只能驳回并进入类别审核。'
      : binding.dataset_split === 'test' || binding.dataset_split === 'val'
        ? '该帧只能保存为评测真值，training_return_allowed=false，禁止回流训练。'
        : binding.dataset_split === 'unassigned'
          ? '该帧尚未分配 train/test；可保存本地人工真值，但禁止直接回流训练。'
          : '提交前会再次校验图像、候选、数据集版本、split 和相机；内容变化时将拒绝保存。';
    return new Promise((resolve, reject) => {
      active = {request, binding, startedAt, decisionKey: makeDecisionKey(),
        unknownBlocked, baseWarning, resolve, reject};
      commitButton.textContent = hasShapes ? '人工确认并保存' : '确认空标注并保存';
      rejectButton.hidden = !hasCandidate;
      setBusy(false);
      dialog.showModal();
      reviewerInput.focus();
    });
  }

  commitButton.addEventListener('click', commit);
  rejectButton.addEventListener('click', rejectCandidate);
  cancelButton.addEventListener('click', cancel);
  closeButton.addEventListener('click', cancel);
  reviewerInput.addEventListener('input', () => setBusy(false));
  dialog.addEventListener('cancel', event => { event.preventDefault(); cancel(); });

  window.commitHumanReviewedTruth = openGate;
})();
