import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('maintain', Path(__file__).resolve().parents[1] / 'tools/maintain_live_handoff.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class HandoffChecks(unittest.TestCase):
    def test_retrospective_discussion_does_not_become_a_new_milestone(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            event = {'type': 'response_item', 'timestamp': '2026-11-04T08:00:00Z', 'payload': {
                'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text',
                'text': '是的，您明确问过。当时讨论了模型验证结果和微调准备。'}]}}
            path = root / 'thread.jsonl'
            path.write_text(json.dumps(event), encoding='utf-8')
            values = m.export_threads({'threads': [{'id': 'test', 'path': str(path)}], 'handoff': folder}, {})
            self.assertEqual(values[0]['messages_exported'], 1)
            self.assertEqual(values[0]['important_updates'], [])

    def test_linked_artifacts_are_scoped_versioned_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            home = root / 'deliverables/handoff'
            control = root / 'control'
            control.mkdir()
            source = root / 'outputs/labelagent_example/picture.png'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'first-image')
            denied = source.with_name('auth.png')
            denied.write_bytes(b'private-auth-screen')
            outside = root / 'other/private.png'
            outside.parent.mkdir()
            outside.write_bytes(b'outside')
            transcript = home / '05_对话记录/chat.txt'
            transcript.parent.mkdir(parents=True)
            transcript.write_text('\n'.join(f'[image]({p.as_posix()})' for p in (source, denied, outside)), encoding='utf-8')
            config = {'handoff': str(home), 'control': str(control), 'media_workspace': str(root),
                      'artifact_stable_seconds': 0}
            self.assertEqual(len(m.collect_delivered_artifacts(config)), 1)
            self.assertEqual(m.collect_delivered_artifacts(config), [])
            source.write_bytes(b'new-version')
            self.assertEqual(len(m.collect_delivered_artifacts(config)), 1)
            files = list((home / '03_图片').rglob('*.png'))
            self.assertEqual(len(files), 2)
            self.assertNotIn(b'private-auth-screen', [p.read_bytes() for p in files])

    def test_important_notifications_quiet_when_unchanged_and_deduplicate_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {'control': folder, 'handoff': str(Path(folder) / 'handoff')}
            sync = {'producer_commit': 'first', 'commit': 'mirror'}
            observation = {'scores': {}, 'primary_588_passed': None}
            threads = [{'id': 'thread', 'latest_message_utc': '2026-11-04T07:00:00Z'}]
            with mock.patch.object(m, 'notify_desktop', return_value={'status': 'submitted'}) as notify:
                m.report_changes(config, sync, observation, threads, [])
                m.report_changes(config, sync, observation, threads, [])
                notify.assert_not_called()
                sync['producer_commit'] = 'second'
                m.report_changes(config, sync, observation, threads, [])
                self.assertEqual(notify.call_count, 1)
                m.report_changes(config, sync, observation, threads, [])
                self.assertEqual(notify.call_count, 1)
                threads[0].update(latest_message_utc='2026-11-04T07:05:00Z', important_updates=[
                    {'time': '2026-11-04T07:05:00Z', 'text': '完整验证完成，未通过原要求。'}])
                m.report_changes(config, sync, observation, threads, [])
                m.report_changes(config, sync, observation, threads, [])
                self.assertEqual(notify.call_count, 2)
                failure = {'error_type': 'RuntimeError', 'message': 'fetch failed'}
                m.report_failure(config, failure)
                m.report_failure(config, failure)
                self.assertEqual(notify.call_count, 3)
                m.report_changes(config, sync, observation, threads, [])
                self.assertEqual(notify.call_count, 4)

    def test_new_gallery_references_collect_images_without_a_chat_link(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            gallery_root = root / 'outputs/labelagent_example'
            gallery = gallery_root / 'new_gallery'
            gallery.mkdir(parents=True)
            (gallery / 'index.html').write_text('<img src="case.jpg"><img src="https://example.invalid/remote.jpg">')
            (gallery / 'case.jpg').write_bytes(b'actual-local-picture')
            control = root / 'control'
            control.mkdir()
            config = {'handoff': str(root / 'handoff'), 'control': str(control), 'media_workspace': str(root),
                      'artifact_stable_seconds': 0, 'artifact_gallery_roots': [str(gallery_root)]}
            added = m.collect_delivered_artifacts(config)
            self.assertEqual(len(added), 1)
            self.assertEqual(m.collect_delivered_artifacts(config), [])

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

    def test_completed_failed_gate_and_private_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            full = root / 'raw_certificate_full_predictions_v1'
            p = {'supported_reference_objects': 3672, 'reference_projected_points': 168738,
                 'candidates': 6052, 'radii': {'3': {'objects_matched_at_support_threshold': 2286,
                  'covered_projected_points': 141174, 'points_covered_by_other_class': 2193}}}
            score = {'status': 'complete', 'evaluated_images': 588, 'metric_version': 2,
                     'missing_prediction_images': 0, 'groups': {'all': {'pallet': p, 'person': p}}}
            for method in m.METHODS:
                m.write_json(full / ('evaluation_' + method) / 'summary.json', score)
            m.write_json(full / 'comparison.json', {'status': 'complete', 'scored_images_per_method': 588,
                'primary_method': 'raw_certificate_priority', 'primary_passed': False,
                'rows': [{'method': 'raw_certificate_priority', 'checks': {
                    'person_off_class_not_above_previous_best': False, 'private_token': 'secret-example'}}]})
            value = m.public_observation({'evidence': folder, 'video': folder})
            self.assertFalse(value['primary_588_passed'])
            self.assertEqual(value['primary_failed_checks'], ['person_off_class_not_above_previous_best'])
            self.assertNotIn('secret-example', json.dumps(value))

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
            progress_path = 'docs/test-results/raw_coverage_validation_progress.json'
            (producer / progress_path).parent.mkdir(parents=True)
            (producer / progress_path).write_text('{"status":"initial"}\n')
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
                      'publish_branch': 'handoff', 'producer_authoritative_reports': [progress_path]}
            observation = {'primary_588_passed': None, 'pipeline': {'status': 'scoring'}, 'scores': {}}
            first = m.sync_git(config, observation)
            self.assertTrue(first['both_verified'])
            second = m.sync_git(config, observation)
            self.assertEqual(first['commit'], second['commit'])
            (producer / 'code.py').write_text('print(2)\n')
            (producer / progress_path).write_text('{"status":"source-final"}\n')
            run('-C', producer, 'add', '.')
            run('-C', producer, 'commit', '-m', 'new code')
            cnb_writer = root / 'cnb-writer'
            run('clone', '--branch', 'progress', upstreams[1], cnb_writer)
            run('-C', cnb_writer, 'config', 'user.name', 'Test')
            run('-C', cnb_writer, 'config', 'user.email', 'test@example.invalid')
            (cnb_writer / progress_path).write_text('{"status":"cnb-final"}\n')
            run('-C', cnb_writer, 'add', '.')
            run('-C', cnb_writer, 'commit', '-m', 'concurrent generated report')
            run('-C', cnb_writer, 'push', 'origin', 'progress')
            third = m.sync_git(config, observation)
            self.assertNotEqual(first['commit'], third['commit'])
            self.assertEqual((repo / 'code.py').read_text(), 'print(2)\n')
            self.assertEqual(json.loads((repo / progress_path).read_text())['status'], 'source-final')
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
