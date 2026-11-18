((root, factory) => {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.LabelAgentReviewPacketCore = api;
})(typeof globalThis === 'object' ? globalThis : this, () => {
  'use strict';
  const acceptedDecisions = new Set(['manual_annotation', 'accept_candidate', 'correct_candidate', 'confirmed_empty']);
  const scopedSlopeDecisions = new Set([
    'slope_present_and_mask_acceptable', 'slope_present_mask_needs_correction',
    'no_slope_present', 'uncertain',
  ]);

  function normalizePath(value) {
    return String(value || '').replaceAll('/', '\\').replace(/\\+$/, '').toLowerCase();
  }

  function stable(value) {
    if (Array.isArray(value)) return value.map(stable);
    if (!value || typeof value !== 'object') return value;
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, stable(value[key])]));
  }

  function stableJson(value) {
    return JSON.stringify(stable(value));
  }

  function completionMap(packet, preparation, reviews, slopeDecisions = []) {
    const completed = new Map();
    if (!packet || !preparation || normalizePath(preparation.path) !== normalizePath(packet.mcap_path)
        || preparation.topic !== packet.topic) return completed;
    const byFrame = new Map();
    for (const review of reviews || []) {
      if (review?.schema_version !== 'labelagent.human-review.v2'
          || !acceptedDecisions.has(review?.decision)
          || review?.topic !== packet.topic
          || normalizePath(review?.file) !== normalizePath(packet.mcap_path)) continue;
      byFrame.set(Number(review.frame), review);
    }
    const scopedByFrame = new Map();
    for (const decision of slopeDecisions || []) {
      if (decision?.schema_version !== 'labelagent.slope-review-decision.v1'
          || decision?.packet_id !== packet.packet_id
          || decision?.manifest_sha256 !== packet.manifest_sha256
          || !scopedSlopeDecisions.has(decision?.decision)
          || decision?.topic !== packet.topic
          || normalizePath(decision?.file) !== normalizePath(packet.mcap_path)
          || decision?.human_decision !== true || decision?.human_truth !== false
          || decision?.approved_for_training !== false
          || decision?.training_return_allowed !== false) continue;
      scopedByFrame.set(Number(decision.frame), decision);
    }
    for (const row of packet.rows || []) {
      const scoped = scopedByFrame.get(Number(row.frame));
      if (scoped?.decision_id
          && (row.allowed_human_decisions || []).includes(scoped.decision)
          && String(scoped.frame_sha256 || '').toLowerCase()
            === String(row.source_frame_sha256 || '').toLowerCase()
          && /^[a-f0-9]{64}$/.test(String(scoped.slope_geometry_sha256 || ''))) {
        completed.set(Number(row.frame), {
          type: 'slope_scoped_decision',
          decision_id: scoped.decision_id,
          decision: scoped.decision,
          reviewed_at: scoped.decided_at || null,
          reviewer_declared_id: scoped.reviewer_declared_id || null,
        });
        continue;
      }
      const status = preparation.truth_review_status?.[String(row.frame)];
      const review = byFrame.get(Number(row.frame));
      if (status?.current_human_review_verified === true
          && status?.reusable_as_reference === true
          && review?.review_id && status.review_id === review.review_id
          && String(review.frame_sha256 || '').toLowerCase() === String(row.source_frame_sha256 || '').toLowerCase()) {
        completed.set(Number(row.frame), {
          type: 'full_frame_truth',
          review_id: review.review_id,
          reviewed_at: review.reviewed_at || null,
          reviewer_declared_id: review.reviewer_declared_id || null,
        });
      }
    }
    return completed;
  }

  function dirtyDraft(baseline, snapshot) {
    if (!baseline || !snapshot) return false;
    if (normalizePath(baseline.path) !== normalizePath(snapshot.path)
        || baseline.topic !== snapshot.topic || Number(baseline.frame) !== Number(snapshot.frame)) return false;
    return snapshot.truth_verified !== true && baseline.shapes_json !== stableJson(snapshot.shapes || []);
  }

  function adjacentPendingIndex(rows, completed, currentIndex, direction) {
    if (!Array.isArray(rows) || !rows.length) return -1;
    const step = direction < 0 ? -1 : 1;
    for (let offset = 1; offset <= rows.length; offset += 1) {
      const index = (currentIndex + step * offset + rows.length * 2) % rows.length;
      if (!completed.has(Number(rows[index].frame))) return index;
    }
    return (currentIndex + step + rows.length) % rows.length;
  }

  return {normalizePath, stableJson, completionMap, dirtyDraft, adjacentPendingIndex};
});
