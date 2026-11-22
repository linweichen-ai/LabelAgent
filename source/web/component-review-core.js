(function(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.LabelAgentComponentReviewCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function() {
  const decisions = new Set(['likely_fill_ok', 'uncertain', 'likely_reject']);

  function draftKey(manifestSha256, componentId) {
    return `${String(manifestSha256 || '')}|${String(componentId || '')}`;
  }

  function validDraft(packet, row, value) {
    return Boolean(value && decisions.has(value.status) && value.local_only === true
      && value.manifest_sha256 === packet.manifest_sha256
      && value.image_sha256 === row.focused_image_sha256);
  }

  function rowDraft(packet, row, drafts) {
    const value = drafts?.[draftKey(packet.manifest_sha256, row.component_id)];
    return validDraft(packet, row, value) ? value : null;
  }

  function filterRows(packet, drafts, filter) {
    const rows = Array.isArray(packet?.rows) ? packet.rows : [];
    return rows.filter(row => {
      const draft = rowDraft(packet, row, drafts);
      if (filter === 'priority') return row.policy_queue === 'priority_review';
      if (filter === 'deferred') return row.policy_queue === 'deferred_manual_review';
      if (filter === 'pending') return !draft;
      if (filter === 'reviewed') return Boolean(draft);
      return true;
    });
  }

  function summarize(packet, drafts) {
    const rows = Array.isArray(packet?.rows) ? packet.rows : [];
    const values = rows.map(row => rowDraft(packet, row, drafts)).filter(Boolean);
    return {
      total: rows.length,
      reviewed: values.length,
      pending: rows.length - values.length,
      likely_fill_ok: values.filter(value => value.status === 'likely_fill_ok').length,
      uncertain: values.filter(value => value.status === 'uncertain').length,
      likely_reject: values.filter(value => value.status === 'likely_reject').length,
    };
  }

  return {draftKey, validDraft, rowDraft, filterRows, summarize};
});
