"""End-to-end gate: the offline reviewer's official export must satisfy the import validator.

Synthetic fixtures only. The page JavaScript is executed verbatim in a Node VM stub so the
exported JSON is exactly what a human reviewer would download.
"""
import hashlib
import json
from io import BytesIO
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from PIL import Image

from tools.build_cvat_human_signoff_reviewer import build
from tools.validate_cvat_human_signoff_import import validate

NODE = shutil.which("node")

DRIVER = r"""
const fs = require("node:fs"); const vm = require("node:vm");
const [html_path, out_path, mode] = process.argv.slice(2);
const html = fs.readFileSync(html_path, "utf8");
const script = [...html.matchAll(/<script(?:\s+id="items"[^>]*)?>([\s\S]*?)<\/script>/g)].map(x => x[1]).find(x => x.includes('"use strict"'));
const els = new Map();
const el = id => { if (!els.has(id)) els.set(id, { id, value: "", textContent: "", disabled: false, listeners: {}, addEventListener(k, f) { this.listeners[k] = f; }, click() {} }); return els.get(id); };
const radios = ["pass", "reject", "uncertain"].map(value => ({ value, checked: false, addEventListener() {} }));
const downloads = [];
const document = {
  getElementById(id) { if (id === "items") return { textContent: html.match(/<script id="items" type="application\/json">([\s\S]*?)<\/script>/)[1] }; return el(id); },
  querySelectorAll(sel) { if (sel.includes('input[name="decision"]')) return radios; if (sel === "button,input,textarea,select") return [...els.values()]; return []; },
  querySelector() { return radios.find(x => x.checked) || null; },
  createElement() { return { click() {} }; },
};
class Blob { constructor(parts) { this.text = parts.join(""); } }
const ctx = { document, Blob, URL: { createObjectURL(b) { downloads.push(b); return "blob:x"; }, revokeObjectURL() {} }, TextEncoder, Date, console };
vm.runInNewContext(script, ctx);
el("exportDraft").listeners.click();
const draft = JSON.parse(downloads.at(-1).text);
draft.rows.forEach((r, i) => { r.decision = ["pass", "reject", "uncertain"][i % 3]; r.reviewer = "synthetic reviewer"; r.reviewed_at = "2026-11-22T10:00:00+08:00"; r.note = i === 0 ? "note" : null; });
if (mode === "incomplete") { draft.rows[80].decision = null; }
(async () => {
  await el("importDraft").listeners.change({ target: { value: "", files: [{ size: 10, text: async () => JSON.stringify(draft) }] } });
  const before = downloads.length;
  el("export").listeners.click();
  const result = { gate_disabled: el("export").disabled, exported: downloads.length > before, status: el("status").textContent };
  if (result.exported) result.payload = JSON.parse(downloads.at(-1).text);
  fs.writeFileSync(out_path, JSON.stringify(result));
})();
"""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@unittest.skipUnless(NODE, "node is required to execute the reviewer page script")
class ReviewerExportImportGateTests(unittest.TestCase):
    def fixture(self, root: Path):
        review_root = root / "review"
        (review_root / "items").mkdir(parents=True)
        review_items, rows = [], []
        for index in range(81):
            buffer = BytesIO()
            Image.new("RGB", (16, 12), (index, 60, 90)).save(buffer, format="JPEG")
            image = buffer.getvalue()
            name = f"items/evidence-{index:03d}.jpg"
            (review_root / name).write_bytes(image)
            binding = {
                "candidate_id": f"synthetic-{index:03d}", "frame_id": index,
                "source_image_sha256": sha(f"source-{index}".encode()),
                "canonical_geometry_sha256": sha(f"geometry-{index}".encode()),
                "focused_image_sha256": sha(image),
            }
            review_items.append({**binding, "focused_image_file": name})
            rows.append({**binding, "decision": None, "reviewer": None, "reviewed_at": None, "note": None})
        review = root / "review_manifest.json"
        review.write_text(json.dumps({
            "schema_version": "labelagent.cvat-manual-instance-review-pack.v1",
            "status": "pending_independent_review", "test24_intersection_count": 0,
            "training_authorized": False, "candidate_count": 81, "review_items": review_items,
        }), encoding="utf-8")
        template = root / "signoff_template.json"
        template.write_text(json.dumps({
            "schema_version": "labelagent.cvat-human-instance-signoff.v1",
            "status": "pending_human_review", "review_manifest_sha256": sha(review.read_bytes()),
            "candidate_count": 81, "allowed_decisions": ["pass", "reject", "uncertain"],
            "rows": rows, "human_signoff_count": 0, "training_authorized": False,
            "automatic_approval": False,
        }), encoding="utf-8")
        return review, template, review_root

    def run_page(self, root: Path, mode: str) -> dict:
        review, template, review_root = self.fixture(root)
        repository = root / "repo"
        repository.mkdir()
        output = repository / ".repro" / "session"
        build(review, template, review_root, output, sha(review.read_bytes()), sha(template.read_bytes()), repository)
        driver = root / "driver.js"
        driver.write_text(DRIVER, encoding="utf-8")
        result_path = root / "result.json"
        completed = subprocess.run(
            [NODE, str(driver), str(output / "index.html"), str(result_path), mode],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return review, json.loads(result_path.read_text(encoding="utf-8"))

    def test_complete_page_export_passes_import_gate_without_training_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, result = self.run_page(root, "complete")
            self.assertFalse(result["gate_disabled"])
            self.assertTrue(result["exported"])
            payload = result["payload"]
            self.assertEqual(payload["status"], "human_review_complete")
            self.assertEqual(payload["human_signoff_count"], 81)
            self.assertFalse(payload["training_authorized"])
            self.assertFalse(payload["human_pass_is_training_authorization"])
            exported = root / "human-signoff.json"
            exported.write_text(json.dumps(payload), encoding="utf-8")
            normalized, receipt = validate(review, exported)
            self.assertEqual(receipt["validated_row_count"], 81)
            self.assertEqual(receipt["decision_counts"], {"pass": 27, "reject": 27, "uncertain": 27})
            self.assertEqual(normalized["human_pass_count"], 27)
            self.assertFalse(normalized["training_authorized"])
            self.assertFalse(receipt["approved_for_training"])
            self.assertNotIn("human_pass_candidate_ids", receipt)

    def test_incomplete_page_cannot_export_and_tampered_export_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, result = self.run_page(root, "incomplete")
            self.assertTrue(result["gate_disabled"])
            self.assertFalse(result["exported"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, result = self.run_page(root, "complete")
            payload = result["payload"]
            for mutate, message in (
                (lambda p: p.update(training_authorized=True), "must not authorize training"),
                (lambda p: p["rows"][0].update(approved_for_training=True), "must not approve training"),
                (lambda p: p["rows"][0].update(focused_image_sha256="0" * 64), "binding drift"),
                (lambda p: p.update(human_signoff_count=80), "complete candidate count"),
            ):
                tampered = json.loads(json.dumps(payload))
                mutate(tampered)
                exported = root / "tampered.json"
                exported.write_text(json.dumps(tampered), encoding="utf-8")
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    validate(review, exported)


if __name__ == "__main__":
    unittest.main()
