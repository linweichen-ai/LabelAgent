"""Frozen-panel, gate and pre-request checks; no model or network access."""
import copy
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'evaluate_source_last.py'
if not SOURCE.exists():
    SOURCE = HERE.parent / 'tools/evaluate_source_last.py'
SPEC = importlib.util.spec_from_file_location('source_last_runner_under_test', SOURCE)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fixture():
    evaluation = {}
    for prefix, role, expected, size in (
            ('old', 'protected_pallet', 'pallet', 2),
            ('new', 'confirmed_six', None, 6),
            ('person', 'person_fixture', 'person', 1),
            ('plain', 'blank_control', 'uncertain', 2),
            ('marked', 'marked_blank_control', 'uncertain', 2),
            ('suspect', 'suspected_non_pallet', 'uncertain', 1)):
        for i in range(size):
            evaluation[f'{prefix}{i}'] = dict(role=role, expected=expected or ('pallet' if i % 2 == 0 else 'person'))
    f = {'cases': [{'id': cid} for cid in evaluation]}
    plan = dict(evaluation_only=evaluation, max_new_requests=29, jobs=[])
    progress = dict(status='complete', rows=[], guard={'parsed': {'category': 'uncertain'}})
    baseline = dict(status='complete', rows=[])
    for variant in ('baseline', 'candidate'):
        for cid, ev in evaluation.items():
            for view in ('overview', 'linked'):
                label = ev['expected']
                if variant == 'baseline' and ev['role'] in ('marked_blank_control', 'suspected_non_pallet'):
                    label = 'uncertain' if (cid, view) == ('marked1', 'linked') else 'pallet'
                row = dict(variant=variant, case_id=cid, view=view, parsed={'category': label})
                progress['rows'].append(row)
                digest = f'{variant}-{cid}-{view}'
                job = dict(variant=variant, case_id=cid, view=view, payload_sha256=digest)
                if variant == 'baseline':
                    path = str(HERE / f'{cid}_{view}_cached.json')
                    baseline['rows'].append(dict(case_id=cid, view=view, source_receipt=path, full_payload_sha256=digest))
                    job['cached_response'] = dict(path=path, sha256='unused-file-digest')
                plan['jobs'].append(job)
    plan['jobs'].append(dict(variant='candidate', case_id='default_budget_guard', view='guard', guard=True, payload_sha256='guard'))
    return plan, {'baseline': f, 'candidate': copy.deepcopy(f)}, progress, baseline


class SourceLastRunnerTests(unittest.TestCase):
    def test_full_panel_with_negative_gain_passes_only_automatic_conditions(self):
        plan, fixtures, progress, baseline = fixture()
        runner.checked_layout(plan, fixtures, baseline)
        scores, checks = runner.aggregate(plan, progress)
        self.assertTrue(all(checks.values()))
        self.assertEqual(len(scores['candidate']), 14)
        self.assertNotIn('guarded_candidate', scores)
        self.assertNotIn('positive_localization_geometry_consistent', checks)

    def test_blank_improvement_cannot_hide_an_old_pallet_regression(self):
        plan, _, progress, _ = fixture()
        row = next(r for r in progress['rows'] if r['variant'] == 'candidate' and r['case_id'] == 'old0')
        row['parsed']['category'] = 'person'
        _, checks = runner.aggregate(plan, progress)
        self.assertTrue(checks['strict_marked_blank_gain'])
        self.assertFalse(checks['two_original_pallets_all_four_views_correct'])

    def test_one_wrong_confirmed_view_blocks_six_target_protection(self):
        plan, _, progress, _ = fixture()
        row = next(r for r in progress['rows'] if r['variant'] == 'candidate' and r['case_id'] == 'new2')
        row['parsed']['category'] = 'uncertain'
        _, checks = runner.aggregate(plan, progress)
        self.assertFalse(checks['confirmed_six_all_twelve_views_correct'])
        self.assertTrue(checks['two_original_pallets_all_four_views_correct'])

    def test_more_blank_hits_cannot_hide_loss_of_previously_correct_negative(self):
        plan, _, progress, _ = fixture()
        row = next(r for r in progress['rows'] if r['variant'] == 'candidate' and (r['case_id'], r['view']) == ('marked1', 'linked'))
        row['parsed']['category'] = 'pallet'
        _, checks = runner.aggregate(plan, progress)
        self.assertTrue(checks['strict_marked_blank_gain'])
        self.assertFalse(checks['no_previously_correct_negative_view_lost'])
        self.assertFalse(checks['all_marked_blank_views_abstained'])

    def test_incomplete_or_duplicate_rows_never_count_as_completed(self):
        for mode in ('missing', 'duplicate'):
            plan, _, progress, _ = fixture()
            if mode == 'missing':
                progress['rows'].pop()
            else:
                progress['rows'].append(copy.deepcopy(progress['rows'][0]))
            _, checks = runner.aggregate(plan, progress)
            self.assertFalse(checks['complete'])
            self.assertFalse(checks['strict_marked_blank_gain'])

    def test_equal_marked_blank_score_is_not_a_gain(self):
        plan, _, progress, _ = fixture()
        for row in progress['rows']:
            if row['variant'] == 'candidate' and row['case_id'].startswith('marked'):
                row['parsed']['category'] = 'uncertain' if (row['case_id'], row['view']) == ('marked1', 'linked') else 'pallet'
        _, checks = runner.aggregate(plan, progress)
        self.assertFalse(checks['strict_marked_blank_gain'])

    def test_layout_rejects_baseline_rebinding_and_missing_positive_panel(self):
        plan, fixtures, _, baseline = fixture()
        plan['jobs'][0]['cached_response']['path'] = str(HERE / 'wrong_receipt.json')
        with self.assertRaisesRegex(ValueError, 'Baseline must reuse'):
            runner.checked_layout(plan, fixtures, baseline)
        plan, fixtures, _, baseline = fixture()
        for f in fixtures.values():
            f['cases'] = f['cases'][1:]
        with self.assertRaisesRegex(ValueError, 'fourteen'):
            runner.checked_layout(plan, fixtures, baseline)

    def test_guard_must_be_fresh_and_final(self):
        plan, fixtures, _, baseline = fixture()
        plan['jobs'][-1]['cached_response'] = {'path': 'old', 'sha256': 'old'}
        with self.assertRaisesRegex(ValueError, 'fresh default guard'):
            runner.checked_layout(plan, fixtures, baseline)
        plan, fixtures, _, baseline = fixture()
        plan['jobs'] = plan['jobs'][-1:] + plan['jobs'][:-1]
        with self.assertRaisesRegex(ValueError, 'fresh default guard'):
            runner.checked_layout(plan, fixtures, baseline)

    def test_changed_body_rejected_before_request_dispatch(self):
        body = {'model': 'test', 'messages': [{'role': 'user', 'content': 'unchanged source'}]}
        digest = runner.body_sha(body)
        job = {'payload_sha256': digest}
        self.assertEqual(runner.checked_body(body, digest, job), (body, digest))
        changed = copy.deepcopy(body)
        changed['messages'][0]['content'] = 'different target'
        with self.assertRaisesRegex(ValueError, 'exact frozen payload'):
            runner.checked_body(changed, digest, job)
        with self.assertRaisesRegex(ValueError, 'exact frozen payload'):
            runner.checked_body(changed, runner.body_sha(changed), job)


if __name__ == '__main__':
    unittest.main()
