"""Transport/audit tests: no network, model, or original annotation access."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'evaluate_target_first.py'
if not SOURCE.exists():
    SOURCE = HERE.parent / 'tools/evaluate_target_first.py'
SPEC = importlib.util.spec_from_file_location('target_first_audit_under_test', SOURCE)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Opener:
    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def open(self, request, timeout):
        self.calls += 1
        return io.BytesIO(self.raw)


class ReceiptAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=HERE)
        self.output = Path(self.temp.name)
        self.progress = {'attempted_requests': 0, 'http_responses_received': 0, 'responses_received': 0}
        self.body = {'model': 'local', 'messages': [{'role': 'user', 'content': 'pixel-only'}]}
        self.digest = runner.payload_sha(self.body)
        self.identity = {'stage': 'grounding', 'case_id': 'test', 'view': 'source_only'}

    def tearDown(self):
        self.temp.cleanup()

    def test_raw_bad_model_response_persisted_before_downstream_validation(self):
        # Deliberately lacks every valid-model field: transport must still save it.
        raw = {'choices': [{'finish_reason': 'length'}], 'unexpected': True}
        client = runner.ReceiptClient(self.output, self.progress, lambda: None, 1, opener=Opener(json.dumps(raw).encode()))
        returned, path, reused = client.request(self.body, self.digest, self.identity)
        self.assertEqual(runner.read(path)['raw_response'], raw)
        self.assertEqual(returned, raw)
        self.assertFalse(reused)
        self.assertEqual(self.progress['responses_received'], 1)
        self.assertFalse((self.output / 'request_inflight.json').exists())
        self.assertEqual(runner.read(self.output / 'request_001_PRIVATE.json')['body'], self.body)
        with self.assertRaises(KeyError):
            runner.validate_backend(returned, {}, 2)
        self.assertTrue(Path(path).is_file())

    def test_exact_payload_reuse_and_cross_run_recovery_do_not_call_network(self):
        opener = Opener(b'{"ok":true}')
        client = runner.ReceiptClient(self.output, self.progress, lambda: None, 1, opener=opener)
        _, path, _ = client.request(self.body, self.digest, self.identity)
        _, path2, reused = client.request(self.body, self.digest, {**self.identity, 'case_id': 'duplicate'})
        self.assertTrue(reused)
        self.assertEqual(path2, path)
        self.assertEqual(opener.calls, 1)
        # A frozen completion plan can carry the exact payload/receipt binding.
        def must_not_run():
            raise AssertionError('recovery unexpectedly attempted a new model call')
        recovered = runner.ReceiptClient(self.output, self.progress, must_not_run, 1,
            {self.digest: {'path': path, 'sha256': runner.sha(path)}}, opener)
        _, recovered_path, reused = recovered.request(self.body, self.digest, self.identity)
        self.assertTrue(reused)
        self.assertEqual(recovered_path, path)
        self.assertEqual(opener.calls, 1)

    def test_mutated_cached_receipt_is_rejected_without_fallback_call(self):
        opener = Opener(b'{"ok":true}')
        client = runner.ReceiptClient(self.output, self.progress, lambda: None, 2, opener=opener)
        _, path, _ = client.request(self.body, self.digest, self.identity)
        Path(path).write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            client.request(self.body, self.digest, self.identity)
        self.assertEqual(opener.calls, 1)

    def test_non_json_reply_kept_and_budget_still_counts_attempt(self):
        opener = Opener(b'not-json')
        client = runner.ReceiptClient(self.output, self.progress, lambda: None, 1, opener=opener)
        with self.assertRaises(json.JSONDecodeError):
            client.request(self.body, self.digest, self.identity)
        self.assertEqual((self.output / 'http_response_001_PRIVATE.bin').read_bytes(), b'not-json')
        self.assertEqual(self.progress, {'attempted_requests': 1, 'http_responses_received': 1, 'responses_received': 0})
        self.assertTrue((self.output / 'attempt_error_001.json').exists())
        with self.assertRaisesRegex(ValueError, 'budget exhausted'):
            client.request(self.body, self.digest, self.identity)
        self.assertEqual(opener.calls, 1)

    def test_busy_peer_blocks_before_attempt_counter_or_request_write(self):
        def busy():
            raise InterruptedError('Another model client is connected')
        opener = Opener(b'{}')
        client = runner.ReceiptClient(self.output, self.progress, busy, 1, opener=opener)
        with self.assertRaises(InterruptedError):
            client.request(self.body, self.digest, self.identity)
        self.assertEqual(self.progress['attempted_requests'], 0)
        self.assertEqual(opener.calls, 0)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_raw_and_guarded_categories_are_scored_separately(self):
        fixture = {'cases': [{'id': 'target'}]}
        plan = {'evaluation_only': {'target': {'expected': 'pallet', 'role': 'protected_pallet'}}}
        rows = [{'case_id': 'target', 'view': v, 'raw_category': 'pallet', 'guarded_category': 'uncertain'} for v in ('overview', 'linked')]
        progress = {'status': 'stopped', 'rows': rows, 'groundings': [], 'guard': None}
        baseline = {'rows': [{'variant': 'candidate', 'case_id': 'target', 'view': v, 'parsed': {'category': 'pallet'}} for v in ('overview', 'linked')]}
        scores, checks = runner.aggregate(plan, fixture, progress, baseline)
        self.assertEqual(scores['candidate'][0]['correct_views'], 2)
        self.assertEqual(scores['guarded_candidate'][0]['correct_views'], 0)
        self.assertFalse(checks['positive_localization_geometry_consistent'])
        self.assertFalse(checks['complete'])

    def test_diagnostic_answers_cannot_improve_candidate_scores(self):
        fixture = {'cases': [{'id': 'target'}]}
        plan = {'evaluation_only': {'target': {'expected': 'pallet', 'role': 'protected_pallet'}}}
        rows = [{'case_id': 'target', 'view': v, 'raw_category': 'uncertain', 'guarded_category': 'uncertain'} for v in ('overview', 'linked')]
        progress = {'status': 'stopped', 'rows': rows, 'groundings': [], 'guard': None}
        baseline = {'rows': [{'variant': 'candidate', 'case_id': 'target', 'view': v, 'parsed': {'category': 'pallet'}} for v in ('overview', 'linked')]}
        before = runner.aggregate(plan, fixture, progress, baseline)
        progress['diagnostics'] = [{'case_id': 'target', 'view': v, 'clean_clean_parsed': {'category': 'pallet'}} for v in ('overview', 'linked')]
        self.assertEqual(before, runner.aggregate(plan, fixture, progress, baseline))
        self.assertEqual(before[0]['candidate'][0]['correct_views'], 0)

    def test_two_image_encoding_mismatch_is_rejected(self):
        fixture = {'model_config_sha256': 'fixed'}
        meta = {'backend': 'transformers_local_cuda', 'model_config_sha256': 'fixed', 'quantization': 'nf4',
                'image_count': 2, 'actual_image_token_counts': [256, 256], 'effective_image_token_budget': 256}
        self.assertEqual(runner.validate_backend({'local_vlm': meta}, fixture, 2), meta)
        with self.assertRaisesRegex(ValueError, 'encoding differs'):
            runner.validate_backend({'local_vlm': {**meta, 'actual_image_token_counts': [256]}}, fixture, 2)


if __name__ == '__main__':
    unittest.main()
