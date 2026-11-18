const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const core = require('../source/web/review-packet-core.js');

function packet() {
  return {
    packet_id: 'slope-round', manifest_sha256: 'f'.repeat(64),
    mcap_path: 'D:\\data\\review.mcap', topic: '/camera',
    rows: [
      {frame: 7, source_frame_sha256: 'a'.repeat(64),
        allowed_human_decisions: ['no_slope_present', 'uncertain']},
      {frame: 9, source_frame_sha256: 'b'.repeat(64),
        allowed_human_decisions: ['no_slope_present', 'uncertain']},
    ],
  };
}

test('only exact current-contract receipt and frame hash count as complete', () => {
  const value = packet();
  const preparation = {path: 'd:/data/review.mcap', topic: '/camera', truth_review_status: {
    '7': {current_human_review_verified: true, reusable_as_reference: true, review_id: 'review-7'},
    '9': {current_human_review_verified: true, reusable_as_reference: true, review_id: 'review-9'},
  }};
  const reviews = [
    {schema_version: 'labelagent.human-review.v2', review_id: 'review-7', decision: 'correct_candidate',
      file: 'D:\\data\\review.mcap', topic: '/camera', frame: 7, frame_sha256: 'a'.repeat(64)},
    {schema_version: 'labelagent.human-review.v2', review_id: 'review-9', decision: 'manual_annotation',
      file: 'D:\\data\\review.mcap', topic: '/camera', frame: 9, frame_sha256: 'wrong'},
  ];
  const completed = core.completionMap(value, preparation, reviews);
  assert.deepEqual([...completed.keys()], [7]);
});

test('different file or topic cannot reuse receipts', () => {
  const value = packet();
  const status = {'7': {current_human_review_verified: true, reusable_as_reference: true, review_id: 'r'}};
  const review = [{schema_version: 'labelagent.human-review.v2', review_id: 'r', decision: 'manual_annotation',
    file: value.mcap_path, topic: value.topic, frame: 7, frame_sha256: 'a'.repeat(64)}];
  assert.equal(core.completionMap(value, {path: 'other.mcap', topic: value.topic, truth_review_status: status}, review).size, 0);
  assert.equal(core.completionMap(value, {path: value.mcap_path, topic: '/other', truth_review_status: status}, review).size, 0);
});

test('scoped slope receipt completes queue without claiming full truth or training', () => {
  const value = packet();
  const preparation = {path: value.mcap_path, topic: value.topic, truth_review_status: {}};
  const scoped = [{
    schema_version: 'labelagent.slope-review-decision.v1',
    packet_id: value.packet_id, manifest_sha256: value.manifest_sha256,
    decision_id: 'd'.repeat(64), decision: 'no_slope_present',
    file: value.mcap_path, topic: value.topic, frame: 7,
    frame_sha256: 'a'.repeat(64), slope_geometry_sha256: 'c'.repeat(64),
    reviewer_declared_id: 'human', human_decision: true, human_truth: false,
    approved_for_training: false, training_return_allowed: false,
  }];
  const completed = core.completionMap(value, preparation, [], scoped);
  assert.deepEqual([...completed.keys()], [7]);
  assert.equal(completed.get(7).type, 'slope_scoped_decision');
  assert.equal(completed.get(7).decision, 'no_slope_present');
  scoped[0].approved_for_training = true;
  assert.equal(core.completionMap(value, preparation, [], scoped).size, 0);
});

test('dirty guard blocks only unsaved edits in the same scope', () => {
  const baseline = {path: 'D:\\data\\review.mcap', topic: '/camera', frame: 7,
    shapes_json: core.stableJson([{label: 'slope', points: [[0, 0], [1, 0], [0, 1]]}])};
  const same = {path: 'd:/data/review.mcap', topic: '/camera', frame: 7, truth_verified: false,
    shapes: [{points: [[0, 0], [1, 0], [0, 1]], label: 'slope'}]};
  assert.equal(core.dirtyDraft(baseline, same), false);
  same.shapes[0].label = 'floor';
  assert.equal(core.dirtyDraft(baseline, same), true);
  same.truth_verified = true;
  assert.equal(core.dirtyDraft(baseline, same), false);
  same.truth_verified = false; same.frame = 9;
  assert.equal(core.dirtyDraft(baseline, same), false);
});

test('next navigation skips completed rows and wraps', () => {
  const rows = [{frame: 1}, {frame: 2}, {frame: 3}];
  const completed = new Map([[2, {}]]);
  assert.equal(core.adjacentPendingIndex(rows, completed, 0, 1), 2);
  assert.equal(core.adjacentPendingIndex(rows, completed, 2, 1), 0);
  assert.equal(core.adjacentPendingIndex(rows, completed, 0, -1), 2);
});

test('assistant suggestion is shown but never preselected or auto-submitted', () => {
  const html = fs.readFileSync(path.join(__dirname, '../source/web/index.html'), 'utf8');
  const ui = fs.readFileSync(path.join(__dirname, '../source/web/review-packet.js'), 'utf8');
  assert.match(html, /id="reviewPacketReviewer"/);
  assert.match(html, /id="reviewPacketDecision"/);
  assert.match(html, /id="reviewPacketDecisionSubmit"/);
  assert.match(html, /请选择人工判断（助手不会代选）/);
  assert.match(ui, /decisionSelect\.value = ''/);
  assert.doesNotMatch(ui, /decisionSelect\.value\s*=\s*proposal/);
  assert.match(ui, /expected_assistant_proposal_sha256/);
  assert.match(ui, /human_truth_written !== false/);
  assert.match(ui, /training_data_written !== false/);
  assert.match(ui, /currentDraftIsDirty\(\)/);
});
