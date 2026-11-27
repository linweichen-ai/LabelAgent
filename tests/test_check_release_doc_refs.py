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
    def test_release_documents_have_no_missing_local_references(self):
        report = checker.audit_release_documents()
        self.assertTrue(report["ready"], report["missing"])
        self.assertGreater(len(report["checked"]), 10)

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

    def test_external_runtime_and_placeholder_references_are_ignored(self):
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
                    )
                ),
                encoding="utf-8",
            )
            report = checker.audit_release_documents(root, ("README.md",))
        self.assertTrue(report["ready"], report["missing"])


if __name__ == "__main__":
    unittest.main()
