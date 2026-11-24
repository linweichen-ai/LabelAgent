import json
import html
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.build_mixed_hard_adjudication_workspace import (
    LABELS, SUBMISSION_SCHEMA, build_workspace, sha256, validate_packet,
    validate_submission,
)
from tools.audit_second_assistant_slope_triage import _sha256


class MixedHardAdjudicationWorkspaceTest(unittest.TestCase):
    def assert_page_javascript_compiles(self, page):
        node = shutil.which("node")
        if not node:
            self.skipTest("node is unavailable")
        scripts = re.findall(r"<script(?: [^>]*)?>(.*?)</script>", page, re.S)
        self.assertGreaterEqual(len(scripts), 2)
        result = subprocess.run(
            [node, "-e", "new Function(require('fs').readFileSync(0,'utf8'))"],
            input=html.unescape(scripts[-1]), capture_output=True, text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def packet(self, root):
        root = Path(root)
        (root / "raw").mkdir(parents=True)
        (root / "review").mkdir()
        rows, files = [], []
        for index in range(24):
            raw = root / "raw" / f"{index}.jpg"
            review = root / "review" / f"{index}.jpg"
            Image.new("RGB", (8, 8), (index, 0, 0)).save(raw)
            Image.new("RGB", (8, 8), (0, index, 0)).save(review)
            for path in (raw, review):
                files.append({"path": path.relative_to(root).as_posix(),
                              "sha256": sha256(path), "bytes": path.stat().st_size})
            rows.append({"sample_id": f"sample-{index}", "cohort": "slope_negative_candidate",
                         "source_image_sha256": f"{index + 1:064x}",
                         "raw_image": f"raw/{index}.jpg", "review_image": f"review/{index}.jpg",
                         "review_note": "fixture", "candidate_only": True,
                         "human_truth": False, "approved_for_training": False})
        manifest = {"schema_version": "labelagent.mixed-hard-review-pack.v1",
                    "summary": {"test24_intersection_count": 0}, "rows": rows,
                    "candidate_only": True, "human_truth": False,
                    "approved_for_training": False, "training_authorized": False,
                    "test24_used": False, "production_deployed": False}
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (root / "receipt.json").write_text(json.dumps({
            "manifest_sha256": sha256(manifest_path), "files": files}), encoding="utf-8")
        return root

    def submission(self, manifest, digest):
        return {"schema_version": SUBMISSION_SCHEMA,
                "review_manifest_sha256": digest, "reviewer_declared_id": "reviewer-1",
                "rows": [{"sample_id": row["sample_id"],
                          "source_image_sha256": row["source_image_sha256"],
                          "candidate_decision": "uncertain", "slope_presence": "absent",
                          "class_conflict": "none", "pallet_quality": "not_applicable",
                          "class_presence": {label: "absent" for label in LABELS},
                          "pallet_instance_count": 0}
                         for row in manifest["rows"]],
                "candidate_only": True, "human_truth": False,
                "approved_for_training": False, "training_authorized": False,
                "automatic_approval": False}

    def test_builds_hash_bound_local_reviewer_without_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            packet = self.packet(Path(directory) / "packet")
            output = Path(directory) / "workspace"
            receipt = build_workspace(packet, output)
            self.assertFalse(receipt["human_truth"])
            page = (output / "reviewer.html").read_text(encoding="utf-8")
            self.assertIn("localStorage", page)
            self.assertIn("mixed-hard-adjudication.json", page)
            self.assertIn("八类存在性", page)
            self.assertIn("pallet_instance_count", page)
            self.assertIn("已完整填写", page)
            self.assertIn("下一张未完成 G", page)
            self.assertIn("左右对比 B", page)
            self.assertIn("mixed-hard-adjudication-draft.json", page)
            self.assertIn("draftFile", page)
            self.assertIn("文件边界或数据包哈希不匹配", page)
            self.assertIn("JSON.stringify(out,null,2)+'\\n'", page)
            self.assertIn("setImageMode('both')", page)
            self.assertIn("nextIncomplete", page)
            self.assertIn("addEventListener('change',save)", page)
            self.assertIn("key==='j'", page)
            self.assertIn("key==='k'", page)
            self.assertIn("key==='g'", page)
            self.assertIn("human_truth:false", page)
            self.assertNotIn("approved_for_training:true", page)
            self.assertEqual(receipt["copied_image_count"], 48)
            for relative in ("raw/0.jpg", "review/0.jpg", "raw/23.jpg", "review/23.jpg"):
                copied = output / relative
                self.assertTrue(copied.is_file())
                self.assertEqual(sha256(copied), sha256(packet / relative))
                self.assertEqual(receipt["files"][relative], sha256(copied))

    def test_reviewer_image_links_resolve_inside_portable_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            packet = self.packet(Path(directory) / "packet")
            output = Path(directory) / "portable-workspace"
            build_workspace(packet, output)
            page = (output / "reviewer.html").read_text(encoding="utf-8")
            decoded = page.replace("&quot;", '"')
            self.assertIn('"review_image": "review/0.jpg"', decoded)
            self.assertIn('"raw_image": "raw/0.jpg"', decoded)
            self.assertTrue((output / "review" / "0.jpg").is_file())
            self.assertTrue((output / "raw" / "0.jpg").is_file())

    def test_optional_machine_hints_are_hash_bound_hidden_and_never_autofill(self):
        with tempfile.TemporaryDirectory() as directory:
            packet = self.packet(Path(directory) / "packet")
            manifest, digest = validate_packet(packet)
            suggestion = {
                "assessments": {label: "absent" for label in LABELS},
                "overall_confidence": 0.91,
                "pallet_instance_count": 0,
                "pallet_quality": "no_pallet",
                "slope_geometry": "no_visible_slope",
                "class_conflicts": [],
                "difficulty_flags": [],
                "unknown_candidates": [],
                "review_note": "fixture",
            }
            suggestions = Path(directory) / "suggestions.json"
            suggestions.write_text(json.dumps({
                "schema_version": "labelagent.local-qwen-mixed-hard-suggestions.v1",
                "manifest_sha256": digest,
                "status": "completed",
                "samples_passed": len(manifest["rows"]),
                "samples_failed": 0,
                "candidate_only": True,
                "human_truth": False,
                "approved_for_training": False,
                "test24_used": False,
                "rows": [{
                    "sample_id": row["sample_id"],
                    "copied_image_sha256": sha256(
                        packet / row["raw_image"]),
                    "suggestion": suggestion,
                    "candidate_only": True,
                    "human_truth": False,
                    "approved_for_training": False,
                    "test24_used": False,
                } for row in manifest["rows"]],
            }), encoding="utf-8")
            output = Path(directory) / "workspace"
            receipt = build_workspace(packet, output, suggestions)
            page = (output / "reviewer.html").read_text(encoding="utf-8")
            self.assertEqual(receipt["assistant_hint_count"], 24)
            self.assertEqual(receipt["assistant_suggestions_sha256"], sha256(suggestions))
            self.assertIn("显示机器建议 H", page)
            self.assertIn("不会自动填表", page)
            self.assertIn("hintText(item.assistant_hint)", page)
            self.assertIn("$('assistantPanel').open=false", page)
            self.assertIn("完成本帧独立判断后显示机器建议 H", page)
            self.assertIn("!rowComplete(state.rows[items[state.index].sample_id])", page)
            self.assertIn("请先完整填写本帧独立判断", page)
            self.assertTrue(receipt["assistant_hint_blind_gate"])
            self.assertNotIn("candidate_decision:hint", page)
            self.assert_page_javascript_compiles(page)
            payload = json.loads(suggestions.read_text(encoding="utf-8"))
            payload["rows"][0]["copied_image_sha256"] = "0" * 64
            suggestions.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "binding differs"):
                build_workspace(packet, Path(directory) / "drift", suggestions)

    def test_priority_triage_reorders_review_without_autofilling_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet = self.packet(root / "packet")
            manifest, digest = validate_packet(packet)
            suggestion = {
                "assessments": {label: "absent" for label in LABELS},
                "overall_confidence": 0.8,
                "pallet_instance_count": 0,
                "pallet_quality": "no_pallet",
                "slope_geometry": "no_visible_slope",
                "class_conflicts": [], "difficulty_flags": [],
                "unknown_candidates": [], "review_note": "fixture",
            }
            suggestions = root / "suggestions.json"
            suggestions.write_text(json.dumps({
                "schema_version": "labelagent.local-qwen-mixed-hard-suggestions.v1",
                "manifest_sha256": digest, "status": "completed",
                "samples_passed": len(manifest["rows"]), "samples_failed": 0,
                "candidate_only": True, "human_truth": False,
                "approved_for_training": False, "test24_used": False,
                "rows": [{
                    "sample_id": row["sample_id"],
                    "copied_image_sha256": sha256(packet / row["raw_image"]),
                    "suggestion": suggestion, "candidate_only": True,
                    "human_truth": False, "approved_for_training": False,
                    "test24_used": False,
                } for row in manifest["rows"]],
            }), encoding="utf-8")
            selected = [(manifest["rows"][7], "low"),
                        (manifest["rows"][9], "high"),
                        (manifest["rows"][8], "medium")]
            triage = root / "triage.json"
            triage.write_text(json.dumps({
                "schema_version": "labelagent.local-second-assistant-slope-triage.v1",
                "status": "assistant_review_complete_human_confirmation_pending",
                "review_manifest_sha256": digest,
                "local_qwen_suggestions_sha256": _sha256(suggestions),
                "reviewed_sample_count": len(manifest["rows"]),
                "priority_disagreement_count": len(selected),
                "rows": [{
                    "sample_id": row["sample_id"],
                    "source_image_sha256": row["source_image_sha256"],
                    "local_qwen_slope": "absent",
                    "assistant_slope_candidate": "present",
                    "assistant_confidence": confidence,
                    "difficulty_flags": ["edge"],
                    "rationale": "fixture disagreement",
                } for row, confidence in selected],
                "candidate_only": True, "human_truth": False,
                "approved_for_training": False, "training_authorized": False,
                "automatic_approval": False, "test24_used": False,
                "external_api_calls": 0, "factory_data_uploaded": False,
                "production_deployed": False, "human_review_required": True,
            }), encoding="utf-8")
            output = root / "workspace"
            receipt = build_workspace(packet, output, suggestions, triage)
            page = (output / "reviewer.html").read_text(encoding="utf-8")
            decoded = page.replace("&quot;", '"')
            positions = [decoded.index(f'"sample_id": "sample-{index}"')
                         for index in (9, 8, 7, 0)]
            self.assertEqual(positions, sorted(positions))
            self.assertEqual(receipt["priority_review_count"], 3)
            self.assertEqual(receipt["priority_triage_sha256"], _sha256(triage))
            self.assertIn("优先复核", page)
            self.assertIn("priority_review", page)
            self.assertNotIn("candidate_decision:priority", page)
            self.assertNotIn("slope_presence:priority", page)

    def test_priority_triage_requires_bound_assistant_suggestions(self):
        with tempfile.TemporaryDirectory() as directory:
            packet = self.packet(Path(directory) / "packet")
            triage = Path(directory) / "triage.json"
            triage.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires assistant suggestions"):
                build_workspace(packet, Path(directory) / "workspace", None, triage)

    def test_script_entrypoint_can_import_priority_auditor(self):
        script = (Path(__file__).resolve().parents[1] / "tools" /
                  "build_mixed_hard_adjudication_workspace.py")
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=Path(tempfile.gettempdir()), capture_output=True, text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--priority-triage", result.stdout)

    def test_validates_complete_bound_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            packet = self.packet(Path(directory) / "packet")
            manifest, digest = validate_packet(packet)
            result = validate_submission(manifest, digest, self.submission(manifest, digest))
            self.assertEqual(result["sample_count"], 24)
            self.assertTrue(result["benchmark_reference_ready"])
            self.assertFalse(result["human_truth"])

    def test_rejects_training_approval_and_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            packet = self.packet(Path(directory) / "packet")
            manifest, digest = validate_packet(packet)
            submission = self.submission(manifest, digest)
            submission["approved_for_training"] = True
            with self.assertRaisesRegex(ValueError, "approved_for_training"):
                validate_submission(manifest, digest, submission)

    def test_uncertain_presence_is_valid_review_but_not_benchmark_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            packet = self.packet(Path(directory) / "packet")
            manifest, digest = validate_packet(packet)
            submission = self.submission(manifest, digest)
            submission["rows"][0]["class_presence"]["slope"] = "uncertain"
            result = validate_submission(manifest, digest, submission)
            self.assertFalse(result["benchmark_reference_ready"])

    def test_rejects_pallet_presence_count_disagreement(self):
        with tempfile.TemporaryDirectory() as directory:
            packet = self.packet(Path(directory) / "packet")
            manifest, digest = validate_packet(packet)
            submission = self.submission(manifest, digest)
            submission["rows"][0]["class_presence"]["pallet"] = "present"
            with self.assertRaisesRegex(ValueError, "count disagrees"):
                validate_submission(manifest, digest, submission)
            submission = self.submission(manifest, digest)
            submission["rows"][0]["source_image_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "source image binding differs"):
                validate_submission(manifest, digest, submission)


if __name__ == "__main__":
    unittest.main()
