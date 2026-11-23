import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "check_release_doc_refs.py"
SPEC = importlib.util.spec_from_file_location("check_release_doc_refs", MODULE_PATH)
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


class ReleaseDocumentReferenceTests(unittest.TestCase):
    def test_checkout_verification_runs_reference_gate_before_optional_suites(self):
        verify_script = (
            MODULE_PATH.parents[1] / "tools" / "verify_checkout.ps1"
        ).read_text(encoding="utf-8")
        gate = '& $Python tools\\check_release_doc_refs.py --json'
        self.assertIn(gate, verify_script)
        self.assertLess(verify_script.index(gate), verify_script.index("$doctorArgs"))
        self.assertLess(
            verify_script.index(gate),
            verify_script.index("if (-not $SkipPythonTests)"),
        )

    def test_release_documents_have_no_missing_local_references(self):
        report = checker.audit_release_documents()
        self.assertTrue(report["ready"], report["missing"])
        self.assertGreater(len(report["checked"]), 10)
        self.assertEqual(report["missing"], [])

    def test_missing_repository_reference_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text(
                "Run `tools/missing.py` and see [guide](docs/missing.md).",
                encoding="utf-8",
            )
            report = checker.audit_release_documents(root, ("README.md",))
        self.assertFalse(report["ready"])
        self.assertEqual(
            {item["path"] for item in report["missing"]},
            {"tools/missing.py", "docs/missing.md"},
        )

    def test_external_runtime_and_placeholder_references_are_not_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text(
                "\n".join(
                    (
                        "<https://example.invalid/tools/not-local.py>",
                        "`.venv/Scripts/python.exe`",
                        "`source/sam3/weights/sam3.pt`",
                        "`tools/<example>.py`",
                        "`$env:LABELAGENT_WORK_DIR`",
                        "`artifacts/history-only`",
                        "<!-- release-doc-ref: external artifacts/history-only -->",
                    )
                ),
                encoding="utf-8",
            )
            before = (root / "README.md").read_bytes()
            report = checker.audit_release_documents(root, ("README.md",))
            after = (root / "README.md").read_bytes()
        self.assertTrue(report["ready"], report["missing"])
        self.assertEqual(before, after)
        self.assertEqual(
            {item["path"] for item in report["external"]},
            {"artifacts/history-only"},
        )

    def test_markdown_image_is_resolved_relative_to_document(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            image = docs / "assets" / "example.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"png")
            (docs / "guide.md").write_text(
                "![example](assets/example.png)", encoding="utf-8"
            )
            report = checker.audit_release_documents(root, ("docs/guide.md",))
        self.assertTrue(report["ready"], report["missing"])
        self.assertEqual(report["checked"][0]["path"], "assets/example.png")


if __name__ == "__main__":
    unittest.main()
