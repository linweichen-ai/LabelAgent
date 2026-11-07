import tempfile,unittest
from pathlib import Path
from handoff_artifact_policy import approved_artifact


class HandoffPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.workspace=self.root/'workspace';self.extra=self.root/'project_data'
    def file(self,relative,extra=False):
        p=(self.extra if extra else self.workspace)/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'local evidence');return p
    def test_existing_workspace_rule_preserved(self):
        p=self.file('outputs/labelagent_trial/gallery.jpg');self.assertEqual(approved_artifact(p,self.workspace),p.resolve());self.assertIsNone(approved_artifact(self.file('other/gallery.jpg'),self.workspace));self.assertIsNone(approved_artifact(self.file('outputs/unrelated/gallery.jpg'),self.workspace))
    def test_extra_experiment_root_explicit_only(self):
        p=self.file('completed_trial/gallery.jpg',True);self.assertIsNone(approved_artifact(p,self.workspace));self.assertEqual(approved_artifact(p,self.workspace,[self.extra]),p.resolve())
    def test_direct_experiment_root_image_allowed(self):
        p=self.file('gallery.jpg',True);self.assertEqual(approved_artifact(p,self.workspace,[self.extra]),p.resolve())
    def test_escape_and_credentials_never_admitted(self):
        p=self.file('auth_flow/image.png',True);self.assertIsNone(approved_artifact(p,self.workspace,[self.extra]));outside=self.root/'outside.png';outside.write_bytes(b'outside');self.assertIsNone(approved_artifact(self.extra/'..'/'outside.png',self.workspace,[self.extra]))
    def test_private_text_excluded_but_local_image_evidence_kept(self):
        p=self.file('review/comparison_PRIVATE.png',True);self.assertEqual(approved_artifact(p,self.workspace,[self.extra]),p.resolve());private=self.file('review/verification_PRIVATE.json',True);self.assertIsNone(approved_artifact(private,self.workspace,[self.extra]))
    def test_browser_and_missing_file_not_admitted(self):
        self.assertIsNone(approved_artifact(self.file('browser_tmp/page.png',True),self.workspace,[self.extra]));self.assertIsNone(approved_artifact(self.extra/'review'/'missing.jpg',self.workspace,[self.extra]))


if __name__=='__main__':unittest.main()
