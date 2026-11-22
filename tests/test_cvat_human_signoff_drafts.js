const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const vm = require("node:vm");
const { spawnSync } = require("node:child_process");

const root = path.resolve(__dirname, "..");
const temp = fs.mkdtempSync(path.join(os.tmpdir(), "signoff-draft-"));
try {
  const py = spawnSync("python", ["-c", `import hashlib,json,sys; from pathlib import Path; from tools.build_cvat_human_signoff_reviewer import _page_html; rows=[{"candidate_id":f"id-{i}","frame_id":i,"source_image_sha256":hashlib.sha256(f"src{i}".encode()).hexdigest(),"canonical_geometry_sha256":hashlib.sha256(f"geo{i}".encode()).hexdigest(),"focused_image_sha256":hashlib.sha256(f"img{i}".encode()).hexdigest(),"decision":None,"reviewer":None,"reviewed_at":None,"note":None} for i in range(81)]; Path(sys.argv[1]).write_text(_page_html(rows,["data:image/jpeg;base64,AA=="]*81,"a"*64,"b"*64),encoding="utf-8")` , path.join(temp, "page.html")], { cwd: root, encoding: "utf8" });
  assert.equal(py.status, 0, py.stderr);
  const html = fs.readFileSync(path.join(temp, "page.html"), "utf8");
  const script = [...html.matchAll(/<script(?:\s+id="items"[^>]*)?>([\s\S]*?)<\/script>/g)].map(x => x[1]).filter(x => x.includes('"use strict"'))[0];
  assert.ok(script);
  const elements = new Map();
  function element(id) { if (!elements.has(id)) elements.set(id, { id, value: "", textContent: "", disabled: false, listeners: {}, addEventListener(k, fn) { this.listeners[k] = fn; }, click() { this.clicked = true; } }); return elements.get(id); }
  const radios = ["pass", "reject", "uncertain"].map(value => ({ value, checked: false, addEventListener(k, fn) { this.listeners ||= {}; this.listeners[k] = fn; } }));
  let downloads = [];
  const document = { getElementById(id) { if (id === "items") return { textContent: html.match(/<script id="items" type="application\/json">([\s\S]*?)<\/script>/)[1] }; return element(id); }, querySelectorAll(sel) { if (sel.includes('input[name="decision"]')) return radios; if (sel === "button,input,textarea,select") return [...elements.values()]; return []; }, querySelector(sel) { return radios.find(x => x.checked) || null; }, createElement() { return { click() {} }; } };
  class BlobStub { constructor(parts) { this.text = parts.join(""); } }
  const context = { document, Blob: BlobStub, URL: { createObjectURL(blob) { downloads.push(blob); return "blob:test"; }, revokeObjectURL() {} }, TextEncoder, Date, console };
  vm.runInNewContext(script, context);
  const draft = () => JSON.parse(downloads.at(-1).text);
  element("exportDraft").listeners.click();
  const blank = draft(); assert.equal(blank.schema_version, "labelagent.cvat-human-signoff-draft.v1"); assert.equal(blank.rows.length, 81); assert.equal(blank.rows[0].decision, null); assert.deepEqual(Object.keys(blank.rows[0]).sort(), ["candidate_id","frame_id","source_image_sha256","canonical_geometry_sha256","focused_image_sha256","decision","reviewer","reviewed_at","note"].sort());
  const originalGate = element("export").disabled; assert.equal(originalGate, true); assert.equal(/training_authorized:false,automatic_approval:false,approved_for_training:false/.test(script), true);
  const bad = JSON.parse(JSON.stringify(blank)); bad.rows[0].reviewer = "private-value"; bad.rows[0].reviewed_at = "incomplete"; bad.rows[0].note = "partial";
  const fileEvent = text => element("importDraft").listeners.change({ target: { files: [{ size: Buffer.byteLength(text), text: async () => text }], value: "" } });
  (async () => {
    await fileEvent(JSON.stringify(bad)); assert.equal(element("reviewer").value, "private-value"); assert.equal(element("reviewedAt").value, "incomplete"); element("exportDraft").listeners.click(); assert.deepEqual(draft().rows[0], bad.rows[0]); assert.equal(element("export").disabled, true);
    for (let i = 0; i < 81; i++) { bad.rows[i].decision = "pass"; bad.rows[i].reviewer = "reviewer"; bad.rows[i].reviewed_at = "2026-11-22T10:00:00Z"; }
    await fileEvent(JSON.stringify(bad)); assert.equal(element("export").disabled, false); element("export").listeners.click(); const official = JSON.parse(downloads.at(-1).text); assert.equal(official.training_authorized, false); assert.equal(official.automatic_approval, false); assert.equal(official.approved_for_training, false);
    for (const mutate of [x => x.rows[0].focused_image_sha256 = "0".repeat(64), x => x.review_manifest_sha256 = "0".repeat(64), x => x.signoff_template_sha256 = "0".repeat(64), x => x.rows[0].extra = 1, x => x.training_authorized = false, x => x.rows[0].approved_for_training = false, x => x.rows.pop(), x => x.rows[1].candidate_id = x.rows[0].candidate_id, x => x.rows[0].decision = true, x => x.rows[0].note = 4]) { const badDraft = JSON.parse(JSON.stringify(blank)); mutate(badDraft); const before = element("reviewer").value; await fileEvent(JSON.stringify(badDraft)); assert.equal(element("status").textContent, "草稿导入失败"); assert.equal(element("reviewer").value, before); }
    await fileEvent('{"schema_version":"labelagent.cvat-human-signoff-draft.v1","schema_version":"x"}'); assert.equal(element("status").textContent, "草稿导入失败");
    await fileEvent("{" ); assert.equal(element("status").textContent, "草稿导入失败");
    await fileEvent(" ".repeat(1048577)); assert.equal(element("status").textContent, "草稿导入失败");
    await element("importDraft").listeners.change({ target: { files: [{ size: 10, text: async () => { throw Error("private"); } }], value: "" } }); assert.equal(element("status").textContent, "草稿导入失败");
    assert.equal(html.includes('id="exportDraft"'), true); assert.equal(html.includes('id="importDraft"'), true);
    console.log("Node draft tests passed");
  })().catch(e => { console.error(e); process.exitCode = 1; });
} finally {
  process.on("exit", () => fs.rmSync(temp, { recursive: true, force: true }));
}
