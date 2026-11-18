const test = require('node:test');
const assert = require('node:assert/strict');
const core = require('../source/web/review-packet-core.js');

function packet() {
  return {
    mcap_path: 'D:\\data\\review.mcap', topic: '/camera',
    rows: [
      {frame: 7, source_frame_sha256: 'a'.repeat(64)},
      {frame: 9, source_frame_sha256: 'b'.repeat(64)},
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
