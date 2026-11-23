(function(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.LabelAgentCandidateComparisonCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function() {
  function binding(packet, row, variant) {
    return {
      manifest_sha256: packet?.manifest_sha256 || '',
      comparison_id: row?.comparison_id || '',
      source_frame_sha256: row?.source_frame_sha256 || '',
      current_candidate_artifact_sha256: row?.current_candidate_artifact_sha256 || '',
      current_candidate_fingerprint: row?.current_candidate_fingerprint || '',
      current_image_sha256: row?.current?.image_sha256 || '',
      current_mask_sha256: row?.current?.mask_sha256 || '',
      alternative_id: variant?.alternative_id || '',
      variant_mask_sha256: variant?.mask_sha256 || '',
      variant_image_sha256: variant?.preview?.image_sha256 || '',
    };
  }

  function preferenceKey(packet, row) {
    const current = binding(packet, row, null);
    return [current.manifest_sha256, current.comparison_id,
      current.source_frame_sha256, current.current_candidate_artifact_sha256,
      current.current_candidate_fingerprint, current.current_image_sha256,
      current.current_mask_sha256].join('|');
  }

  function validPreference(packet, row, value) {
    if (!value || value.local_only !== true || value.read_only_preview !== true
        || value.scope_verified !== true) return false;
    const variant = (row?.variants || []).find(item =>
      item.alternative_id === value.alternative_id);
    if (!variant) return false;
    const expected = binding(packet, row, variant);
    return Object.keys(expected).every(key => value[key] === expected[key]);
  }

  function selectedPreference(packet, row, preferences) {
    const value = preferences?.[preferenceKey(packet, row)];
    return validPreference(packet, row, value) ? value : null;
  }

  function scopeMatches(scope, packet, row, workspace) {
    return Boolean(scope?.scope_verified === true && scope?.read_only === true
      && scope?.human_truth_writes === false && scope?.training_data_writes === false
      && scope?.automatic_approval === false
      && scope.manifest_sha256 === packet?.manifest_sha256
      && scope.comparison_id === row?.comparison_id
      && scope.source_frame_sha256 === row?.source_frame_sha256
      && scope.current_candidate_artifact_sha256 === row?.current_candidate_artifact_sha256
      && scope.current_candidate_fingerprint === row?.current_candidate_fingerprint
      && scope.current_candidate_artifact_sha256 === workspace?.current_candidate_artifact_sha256
      && scope.current_candidate_fingerprint === workspace?.current_candidate_fingerprint
      && scope.topic === workspace?.topic
      && Number(scope.frame) === Number(workspace?.frame)
      && Number(scope.file_activation_generation) === Number(workspace?.file_activation_generation));
  }

  function requestStillCurrent(expectedGeneration, currentGeneration,
      expectedPacketId, currentPacketId, expectedComparisonId, currentComparisonId) {
    return expectedGeneration === currentGeneration
      && expectedPacketId === currentPacketId
      && expectedComparisonId === currentComparisonId;
  }

  return {binding, preferenceKey, validPreference, selectedPreference, scopeMatches,
    requestStillCurrent};
});
