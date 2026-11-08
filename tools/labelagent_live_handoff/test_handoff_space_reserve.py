"""Verify actual media-copy fallback and automatic recovery without deleting inputs."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import maintain

class SpaceReserve(unittest.TestCase):
    def test_external_manifest_preserves_old_copy_and_recovers(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'task_workspaces') as tmp:
            root=Path(tmp);home=root/'home';overflow=root/'overflow';(home/'06_核验').mkdir(parents=True);(home/'original.txt').write_text('untouched-original');old=home/'06_核验/文件SHA256清单.json';old.write_text('historical-manifest')
            with patch.object(maintain.shutil,'disk_usage',return_value=SimpleNamespace(free=1024)):
                rows=maintain.refresh_manifest(home,overflow)
            pointer=json.loads((home/'06_核验/当前文件清单位置.json').read_text());self.assertTrue(pointer['external_due_to_low_space']);self.assertEqual(old.read_text(),'historical-manifest');self.assertEqual(maintain.digest(pointer['path']),pointer['sha256']);self.assertEqual([r['path'] for r in rows],['original.txt'])
            with patch.object(maintain.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**2)):
                maintain.refresh_manifest(home,overflow)
            pointer=json.loads((home/'06_核验/当前文件清单位置.json').read_text());self.assertFalse(pointer['external_due_to_low_space']);self.assertEqual(Path(pointer['path']),old.resolve());self.assertEqual(json.loads(old.read_text())['files'][0]['sha256'],maintain.digest(home/'original.txt'));self.assertEqual((home/'original.txt').read_text(),'untouched-original')

    def test_external_reference_then_verified_local_copy(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'task_workspaces') as tmp:
            root=Path(tmp);home=root/'handoff';control=root/'control';source=root/'gallery/run';source.mkdir(parents=True);home.mkdir();control.mkdir();media=source/'frame.jpg';media.write_bytes(b'original-media-payload');(source/'index.html').write_text('<img src="frame.jpg">');cfg={'handoff':str(home),'control':str(control),'media_workspace':str(root),'artifact_gallery_roots':[str(source.parent)],'artifact_stable_seconds':0}
            with patch('handoff_artifact_policy.approved_artifact',side_effect=lambda path,*a:path),patch.object(maintain.shutil,'disk_usage',return_value=SimpleNamespace(free=80*1024**2)):
                self.assertEqual(maintain.collect_delivered_artifacts(cfg),[])
            record=json.loads((control/'delivered_artifacts.json').read_text())[str(media)];self.assertFalse(record['copied_into_handoff']);self.assertEqual(record['storage_mode'],'external_reference_low_space');self.assertEqual(Path(record['destination']).read_bytes(),b'original-media-payload');self.assertFalse(any((home/'03_图片').rglob('*.jpg')))
            with patch('handoff_artifact_policy.approved_artifact',side_effect=lambda path,*a:path),patch.object(maintain.shutil,'disk_usage',return_value=SimpleNamespace(free=200*1024**2)):
                added=maintain.collect_delivered_artifacts(cfg)
            self.assertEqual(len(added),1);self.assertEqual((home/added[0]).read_bytes(),media.read_bytes());record=json.loads((control/'delivered_artifacts.json').read_text())[str(media)];self.assertNotIn('storage_mode',record);self.assertNotIn(str(media),(home/'06_核验/空间保护_外部原始资料索引.txt').read_text());self.assertEqual(media.read_bytes(),b'original-media-payload')

if __name__=='__main__':unittest.main()
