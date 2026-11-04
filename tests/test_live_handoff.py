import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('maintain', Path(__file__).resolve().parents[1] / 'tools/maintain_live_handoff.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class HandoffChecks(unittest.TestCase):
    def test_credential_message_is_omitted(self):
        self.assertNotIn('secret-example', m.safe_message('password: secret-example'))
        self.assertNotIn('secret-example', m.safe_message('密码 secret-example'))

    def test_addresses_redacted(self):
        self.assertNotIn('172.16.1.2', m.safe_message('连接 172.16.1.2'))

    def test_missing_or_incomplete_score_not_counted(self):
        self.assertIsNone(m.compact_score({}))
        self.assertIsNone(m.compact_score({'status': 'complete', 'evaluated_images': 32}))

    def test_private_pipeline_fields_never_exported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            m.write_json(root / 'raw_certificate_full_pipeline_v2.json',
                         {'status': 'scoring', 'private_token': 'secret-example', 'command': 'private-path'})
            value = m.public_observation({'evidence': folder, 'video': folder})
            self.assertNotIn('secret-example', json.dumps(value))
            self.assertNotIn('private-path', json.dumps(value))
            self.assertIsNone(value['primary_588_passed'])

    def test_atomic_read_and_idempotent_write(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'x.json'
            self.assertTrue(m.write_json(path, {'x': 1}))
            self.assertFalse(m.write_json(path, {'x': 1}))
            self.assertEqual(m.read_json(path), {'x': 1})

    def test_tampered_media_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source'
            source.write_bytes(b'altered')
            with self.assertRaises(ValueError):
                m.copy_verified(source, Path(folder) / 'target', '0' * 64)
            self.assertFalse((Path(folder) / 'target').exists())

    def test_partial_thread_line_and_tools_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            line = {'type': 'response_item', 'payload': {'type': 'message', 'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': '进度已验证'}]}}
            path = root / 'thread.jsonl'
            path.write_text(json.dumps(line) + '\n' + '{"partial":', encoding='utf-8')
            config = {'threads': [{'id': 'test', 'path': str(path)}], 'handoff': folder}
            state = {}
            self.assertEqual(m.export_threads(config, state)[0]['messages_exported'], 1)
            self.assertIn('进度已验证', (root / '05_对话记录/test.txt').read_text(encoding='utf-8'))

    def test_real_local_git_dual_publication_and_conflicting_remote(self):
        # Actual Git objects and two local bare remotes; no network or credentials.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            producer = root / 'producer'
            repo = root / 'checkout'
            upstreams = [root / 'github.git', root / 'cnb.git']
            def run(*args):
                values = [a.as_posix() if isinstance(a, Path) else str(a) for a in args]
                result = subprocess.run(['git', *values], capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(result.stderr)
            run('init', producer)
            run('-C', producer, 'config', 'user.name', 'Test')
            run('-C', producer, 'config', 'user.email', 'test@example.invalid')
            (producer / 'code.py').write_text('print(1)\n')
            run('-C', producer, 'add', '.')
            run('-C', producer, 'commit', '-m', 'initial')
            run('-C', producer, 'branch', '-M', 'work')
            for p in upstreams:
                run('init', '--bare', p)
            run('-C', producer, 'push', upstreams[1], 'HEAD:refs/heads/progress')
            run('clone', producer, repo)
            run('-C', repo, 'remote', 'set-url', 'origin', upstreams[0])
            run('-C', repo, 'remote', 'add', 'cnb', upstreams[1])
            run('-C', repo, 'remote', 'add', 'producer', producer)
            run('-C', repo, 'config', 'user.name', 'Test')
            run('-C', repo, 'config', 'user.email', 'test@example.invalid')
            config = {'repo': str(repo), 'producer_branch': 'work', 'cnb_progress_branch': 'progress',
                      'publish_branch': 'handoff'}
            observation = {'primary_588_passed': None, 'pipeline': {'status': 'scoring'}, 'scores': {}}
            first = m.sync_git(config, observation)
            self.assertTrue(first['both_verified'])
            second = m.sync_git(config, observation)
            self.assertEqual(first['commit'], second['commit'])
            (producer / 'code.py').write_text('print(2)\n')
            run('-C', producer, 'add', '.')
            run('-C', producer, 'commit', '-m', 'new code')
            third = m.sync_git(config, observation)
            self.assertNotEqual(first['commit'], third['commit'])
            self.assertEqual((repo / 'code.py').read_text(), 'print(2)\n')
            # An unrelated remote commit must be preserved by rejected normal push.
            run('-C', producer, 'push', upstreams[0], 'HEAD:refs/heads/other')
            run('-C', producer, 'fetch', upstreams[0], 'handoff')
            run('-C', producer, 'switch', '--detach', 'FETCH_HEAD')
            (producer / 'remote-only.txt').write_text('keep')
            run('-C', producer, 'add', '.')
            run('-C', producer, 'commit', '-m', 'remote writer')
            run('-C', producer, 'push', upstreams[0], 'HEAD:refs/heads/handoff')
            remote_before = m.git(producer, 'rev-parse', 'HEAD')[0]
            with self.assertRaises(RuntimeError):
                m.sync_git(config, observation)
            self.assertEqual(m.git(repo, 'ls-remote', 'origin', 'refs/heads/handoff')[0].split()[0], remote_before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
