const assert = require('node:assert/strict');
const core = require('../source/web/component-review-core.js');

const packet = {
  manifest_sha256: 'a'.repeat(64),
  rows: [
    {component_id: 'c1', policy_queue: 'priority_review', focused_image_sha256: '1'.repeat(64)},
    {component_id: 'c2', policy_queue: 'deferred_manual_review', focused_image_sha256: '2'.repeat(64)},
  ],
};
const key = core.draftKey(packet.manifest_sha256, 'c1');
const drafts = {[key]: {
  status: 'uncertain', local_only: true,
  evidence_loaded: true,
  manifest_sha256: packet.manifest_sha256, image_sha256: '1'.repeat(64),
}};

assert.equal(core.rowDraft(packet, packet.rows[0], drafts).status, 'uncertain');
assert.equal(core.rowDraft(packet, packet.rows[1], drafts), null);
assert.deepEqual(core.filterRows(packet, drafts, 'priority').map(row => row.component_id), ['c1']);
assert.deepEqual(core.filterRows(packet, drafts, 'deferred').map(row => row.component_id), ['c2']);
assert.deepEqual(core.filterRows(packet, drafts, 'pending').map(row => row.component_id), ['c2']);
assert.deepEqual(core.filterRows(packet, drafts, 'reviewed').map(row => row.component_id), ['c1']);
assert.deepEqual(core.summarize(packet, drafts), {
  total: 2, reviewed: 1, pending: 1,
  likely_fill_ok: 0, uncertain: 1, likely_reject: 0,
});

const stalePacket = {...packet, manifest_sha256: 'b'.repeat(64)};
assert.equal(core.rowDraft(stalePacket, stalePacket.rows[0], drafts), null);
assert.equal(core.summarize(stalePacket, drafts).reviewed, 0);
assert.equal(core.validDraft(packet, packet.rows[0], {...drafts[key], local_only: false}), false);
assert.equal(core.validDraft(packet, packet.rows[0], {
  ...drafts[key], local_only: false, session_only: true,
}), true);
assert.equal(core.validDraft(packet, packet.rows[0], {...drafts[key], evidence_loaded: false}), false);
assert.equal(core.validDraft(packet, packet.rows[0], {...drafts[key], image_sha256: 'x'.repeat(64)}), false);

console.log('component review core tests passed');
