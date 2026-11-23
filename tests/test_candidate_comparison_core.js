const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const core = require('../source/web/candidate-comparison-core.js');

const packet = {manifest_sha256: 'a'.repeat(64)};
const row = {
  comparison_id: 'generic', source_frame_sha256: 'b'.repeat(64),
  current_candidate_artifact_sha256: 'c'.repeat(64),
  current_candidate_fingerprint: 'd'.repeat(64),
  current: {image_sha256: '1'.repeat(64), mask_sha256: '2'.repeat(64)},
  variants: [{alternative_id: 'alt', mask_sha256: 'e'.repeat(64),
    preview: {image_sha256: 'f'.repeat(64)}}],
};

function preference() {
  return {...core.binding(packet, row, row.variants[0]), local_only: true,
    read_only_preview: true, scope_verified: true};
}

test('session preference is bound to every manifest, frame, candidate and variant hash', () => {
  const value = preference();
  assert.equal(core.validPreference(packet, row, value), true);
  for (const field of ['manifest_sha256', 'source_frame_sha256',
    'current_candidate_artifact_sha256', 'current_candidate_fingerprint',
    'current_image_sha256', 'current_mask_sha256',
    'variant_mask_sha256', 'variant_image_sha256']) {
    const changed = {...value, [field]: '0'.repeat(64)};
    assert.equal(core.validPreference(packet, row, changed), false, field);
  }
  assert.equal(core.validPreference(packet, row, {...value, local_only: false}), false);
  assert.equal(core.validPreference(packet, row, {...value, scope_verified: false}), false);
});

test('scope must match all server and current workspace bindings', () => {
  const workspace = {topic: '/camera', frame: 7, file_activation_generation: 12,
    current_candidate_artifact_sha256: row.current_candidate_artifact_sha256,
    current_candidate_fingerprint: row.current_candidate_fingerprint};
  const scope = {scope_verified: true, read_only: true, human_truth_writes: false,
    training_data_writes: false, automatic_approval: false,
    manifest_sha256: packet.manifest_sha256, comparison_id: row.comparison_id,
    source_frame_sha256: row.source_frame_sha256,
    current_candidate_artifact_sha256: row.current_candidate_artifact_sha256,
    current_candidate_fingerprint: row.current_candidate_fingerprint,
    topic: '/camera', frame: 7, file_activation_generation: 12};
  assert.equal(core.scopeMatches(scope, packet, row, workspace), true);
  for (const field of ['manifest_sha256', 'source_frame_sha256',
    'current_candidate_artifact_sha256', 'current_candidate_fingerprint']) {
    assert.equal(core.scopeMatches({...scope, [field]: '0'.repeat(64)}, packet, row, workspace), false);
  }
  assert.equal(core.scopeMatches({...scope, frame: 8}, packet, row, workspace), false);
  assert.equal(core.scopeMatches({...scope, file_activation_generation: 13}, packet, row, workspace), false);
  assert.equal(core.scopeMatches(scope, packet, row,
    {...workspace, current_candidate_artifact_sha256: '0'.repeat(64)}), false);
  assert.equal(core.scopeMatches(scope, packet, row,
    {...workspace, current_candidate_fingerprint: '0'.repeat(64)}), false);
});

test('late catalog, packet or scope responses are rejected by request generation and identity', () => {
  assert.equal(core.requestStillCurrent(4, 4, 'pack-a', 'pack-a', 'row-a', 'row-a'), true);
  assert.equal(core.requestStillCurrent(3, 4, 'pack-a', 'pack-a', 'row-a', 'row-a'), false);
  assert.equal(core.requestStillCurrent(4, 4, 'pack-a', 'pack-b', 'row-a', 'row-a'), false);
  assert.equal(core.requestStillCurrent(4, 4, 'pack-a', 'pack-a', 'row-a', 'row-b'), false);
});

test('comparison UI has no write request or truth adoption control', () => {
  const ui = fs.readFileSync(path.join(__dirname, '../source/web/candidate-comparison.js'), 'utf8');
  const html = fs.readFileSync(path.join(__dirname, '../source/web/index.html'), 'utf8');
  assert.match(html, /id="candidateComparisonOpen"/);
  assert.match(html, /严格只读/);
  assert.match(html, /candidate-comparison-core\.js\?v=2/);
  assert.match(html, /candidate-comparison\.js\?v=2/);
  assert.doesNotMatch(ui, /method:\s*['"]POST/);
  assert.doesNotMatch(ui, /\/api\/(truth|human_review)/);
  assert.doesNotMatch(html, /id="candidateComparison[^" ]*(Apply|Accept|Commit)/i);
  assert.match(ui, /sessionStorage/);
  assert.match(ui, /requestGeneration/);
  assert.match(ui, /function invalidateView\(\{clearCatalog = false\} = \{\}\)/);
  assert.match(ui, /async function refreshCatalog\(\) \{[\s\S]*?invalidateView\(\{clearCatalog: true\}\);[\s\S]*?await getJson\('\/api\/candidate_comparison_packs'\)/);
  assert.match(ui, /catch \(error\) \{[\s\S]*?invalidateView\(\{clearCatalog: true\}\);[\s\S]*?candidateComparisonBadge/);
  assert.match(ui, /labelagent:frame-state/);
  const app = fs.readFileSync(path.join(__dirname, '../source/web/app.js'), 'utf8');
  assert.match(app, /currentWorkbenchFrameState=\(\)=>\(\{[^}]*topic,frame,file_activation_generation/);
});
