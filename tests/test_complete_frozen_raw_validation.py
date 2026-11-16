import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
SPEC=importlib.util.spec_from_file_location('full_pipeline',ROOT/'tools/complete_frozen_raw_validation.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


class FrozenPipelineReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.full={'new_remaining_SAM_requests':5004,'remaining_images':556}
        self.write('raw_certificate_full_validation_plan_v1.json',self.full)
        self.write('raw_certificate_validation_remaining_dense_v1/study_plan.json',{'frames':[1,2],'grid':3})
        self.leaf={'status':'running','completed_requests':123,'completed_images':13,'plan_unchanged':True,
            'baseline_cache_unchanged':True,'plan_sha256':M.sha(self.root/'raw_certificate_validation_remaining_dense_v1/study_plan.json')}
        self.controller={'status':'collecting_real_SAM3','controller_pid':1234,'created_utc':'2026-11-04T05:04:16+00:00',
            'plan_sha256':M.sha(self.root/'raw_certificate_full_validation_plan_v1.json')}
        self.flush()
        created=1793768656.
        self.proc=SimpleNamespace(pid=1234,is_running=lambda:True,create_time=lambda:created,
            cmdline=lambda:['python','collect_raw_certificate_remaining.py'])

    def write(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(json.dumps(value).encode())

    def flush(self):
        self.write('raw_certificate_remaining_collection_v1.json',self.controller)
        self.write('raw_certificate_validation_remaining_dense_v1/progress.json',self.leaf)

    def check(self):
        with patch.object(M.psutil,'Process',return_value=self.proc):return M.collection_readiness(self.root,self.full)

    def test_live_identity_is_required_and_check_never_starts_another_collector(self):
        before={p:p.read_bytes() for p in self.root.rglob('*.json')}
        with patch.object(M.subprocess,'Popen') as popen:
            result=self.check();self.assertEqual(result['status'],'waiting_for_existing_collection')
            self.assertEqual(result['completed_requests'],123);popen.assert_not_called()
            self.controller.update(already_cached_responses=2169,already_cached_images=241);self.flush()
            resumed=self.check()
            self.assertEqual(resumed['completed_requests'],2169)
            self.assertEqual(resumed['verified_requests_this_process'],123)
            self.assertEqual(resumed['new_requests_this_process'],0)
            self.proc.cmdline=lambda:['python','foreign_work.py']
            with self.assertRaises(ValueError):self.check()
        self.controller.pop('already_cached_responses');self.controller.pop('already_cached_images');self.flush()
        for path,raw in before.items():self.assertEqual(path.read_bytes(),raw)

    def test_completed_collection_requires_all_images_calls_and_matching_plan(self):
        self.controller['status']='complete';self.leaf.update(status='complete',completed_requests=5004,completed_images=556)
        self.flush();self.assertEqual(self.check()['status'],'ready')
        for field,value in [('completed_requests',5003),('completed_images',555),('plan_unchanged',False),('plan_sha256','0'*64)]:
            with self.subTest(field=field):
                saved=self.leaf[field];self.leaf[field]=value;self.flush()
                with self.assertRaises(ValueError):self.check()
                self.leaf[field]=saved

    def test_terminal_collection_and_changed_frozen_binding_do_not_restart(self):
        for kind in ('failed','cancelled','different_plan'):
            with self.subTest(kind=kind),patch.object(M.subprocess,'Popen') as popen:
                self.controller.update(status='collecting_real_SAM3',plan_sha256=M.sha(self.root/'raw_certificate_full_validation_plan_v1.json'))
                if kind=='different_plan':self.controller['plan_sha256']='b'*64
                else:self.controller['status']=kind
                self.flush()
                with self.assertRaises(ValueError):self.check()
                popen.assert_not_called()

    def test_user_cancel_marker_prevents_further_work(self):
        (self.root/'raw_certificate_remaining_CANCEL').write_text('stop')
        with patch.object(M.subprocess,'Popen') as popen:
            self.assertEqual(self.check()['status'],'cancelled');popen.assert_not_called()

    def test_partial_progress_write_is_a_verified_wait_instead_of_terminal_failure(self):
        (self.root/'raw_certificate_validation_remaining_dense_v1/progress.json').write_bytes(b'{"status":')
        result=self.check()
        self.assertEqual(result['status'],'waiting_for_existing_collection')
        self.assertEqual(result['progress_observation_error'],'JSONDecodeError')
        self.assertEqual(result['controller_pid'],1234)

    def test_frozen_code_guard_rejects_source_changes(self):
        repo=self.root/'repo';tools=repo/'tools';tools.mkdir(parents=True)
        files={'select_raw_coverage_candidates.py':b'predictor','collect_tiled_candidates.py':b'collector',
               'evaluate_projected_support.py':b'metric','helper.py':b'helper'}
        for name,raw in files.items():(tools/name).write_bytes(raw)
        hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
        frozen={'predictor_sha256':hashes['select_raw_coverage_candidates.py'],'collector_sha256':hashes['collect_tiled_candidates.py'],
                'metric_sha256':hashes['evaluate_projected_support.py'],'helper_sha256':{'helper.py':hashes['helper.py']}}
        M.frozen_code(repo,frozen)
        (tools/'helper.py').write_bytes(b'changed')
        with self.assertRaises(ValueError):M.frozen_code(repo,frozen)

    def test_state_save_keeps_finite_json_contract_and_uses_shared_atomic_retry(self):
        with patch.object(M,'atomic_save') as save:
            target=self.root/'state.json';value={'status':'waiting'}
            M.save(target,value);save.assert_called_once_with(target,value)
            save.reset_mock()
            with self.assertRaises(ValueError):M.save(target,{'value':float('nan')})
            save.assert_not_called()

    def test_new_scheduler_generation_keeps_previous_state_and_cannot_escape_root(self):
        first,logs=M.pipeline_paths(self.root,'v1');second,new_logs=M.pipeline_paths(self.root,'v2')
        self.assertNotEqual(first,second);self.assertNotEqual(logs,new_logs)
        self.assertEqual(second.parent,self.root)
        for value in ('../v2','v0','/absolute','v2/other',''):
            with self.subTest(value=value),self.assertRaises(ValueError):M.pipeline_paths(self.root,value)


if __name__=='__main__':unittest.main(verbosity=2)
