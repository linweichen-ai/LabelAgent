import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('frozen_merge',ROOT/'tools/merge_frozen_prediction_exports.py')
MERGE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGE)
TOPIC='/capture/camera_0/image'
PRIMARY='raw_certificate_priority'
METHODS={'four_crop_control':{'type':'legacy'},PRIMARY:{'type':'raw_cover','objective':'fixed_priority'}}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes((json.dumps(value,indent=2,allow_nan=False)+'\n').encode())


class FrozenMergeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.serial=0

    def fixture(self):
        self.serial+=1
        root=Path(self.temp.name)/str(self.serial)
        dataset=root/'dataset'
        baseline=root/'baseline.json'
        receipt=root/'baseline_receipt.json'
        frozen=root/'frozen.json'
        shape={'id':1,'label':'pallet','points':[[1,1],[6,1],[6,6]],'holes':[],
               'parts':[{'points':[[1,1],[6,1],[6,6]],'holes':[]}],'confidence':0.8}
        predictions={TOPIC:{str(f):[copy.deepcopy(shape)] for f in (0,1)}}
        image_sha={TOPIC:{str(f):str(f+1)*64 for f in (0,1)}}
        manifest={'archive_sha256':'a'*64,'records':[{'topic':TOPIC,'frame':f,'sha256':image_sha[TOPIC][str(f)]} for f in (0,1)]}
        write(dataset/'import_manifest.json',manifest)
        write(baseline,predictions)
        write(receipt,{'archive_sha256':'a'*64,'prediction_json_sha256':MERGE.digest(baseline),'image_sha256':image_sha})
        policy={'strict_instance_gain_over_previous_best':True,'all_original_candidates_preserved':True}
        full={'full_validation_frames':[0,1],'cameras':[0],'full_validation_images':2,'primary_method':PRIMARY,
              'predictor_sha256':'c'*64,'helper_sha256':{'raw.py':'d'*64},
              'baseline_prediction_sha256':MERGE.digest(baseline),'comparison_requires':policy}
        write(frozen,full)
        studies=[]
        for f in (0,1):
            directory=root/('study'+str(f));studies.append(directory)
            plan={'version':1,'methods':METHODS,'fixed_parameters':{'minimum_score':0.5},
                'solver_seconds_per_image':10.,'raw_projection':{'cell_size':8,'kernel':'ellipse'},
                'coverage_certificate':'every anchor point retained','person_prediction_guard':'raw point guard',
                'component_objective':'unique foreground support','component_novelty':'outside baseline',
                'prediction_uses_3D_reference':False,'no_reference_oracle_inputs':True,'comparison_requires':policy,
                'helper_sha256':full['helper_sha256'],'code_sha256':full['predictor_sha256'],
                'frames':[f],'cameras':[0],'planned_images':1,
                'input_bindings':{'baseline':{'prediction_sha256':MERGE.digest(baseline),'receipt_sha256':MERGE.digest(receipt)}}}
            write(directory/'study_plan.json',plan)
            plan_sha=MERGE.digest(directory/'study_plan.json')
            rows=[{'topic':TOPIC,'frame':f,'method':m,'baseline_candidates_preserved':True,
                  'raw_certificate_verified':True,'solver':{'certificate_satisfied':True}} for m in METHODS]
            progress={'status':'complete','completed_images':1,'plan_sha256':plan_sha,'plan_unchanged':True,
                'reference_files_opened':0,'baseline_cache_unchanged':True,'frozen_pool_and_anchors_unchanged':True,
                'code_unchanged':True,'workbench_modified':False,'postprocessing':rows}
            write(directory/'progress.json',progress)
            for m in METHODS:
                values=copy.deepcopy(predictions[TOPIC][str(f)])+[{'id':2,'label':'pallet','points':[[9,9],[11,9],[11,11]],'confidence':0.7}]
                path=directory/(m+'_candidates.json')
                write(path,{TOPIC:{str(f):values}})
                write(directory/(m+'_receipt.json'),{'prediction_json_sha256':MERGE.digest(path),'plan_sha256':plan_sha,
                    'archive_sha256':'a'*64,'method':m,'candidates_confirmed_as_truth':False,
                    'source':'reference_free_frozen_SAM_proposals_and_raw_point_coverage',
                    'image_sha256':{TOPIC:{str(f):image_sha[TOPIC][str(f)]}}})
        return {'root':root,'dataset':dataset,'baseline':baseline,'receipt':receipt,'frozen':frozen,
                'studies':studies,'output':root/'merged'}

    def run_merge(self,case,**kwargs):
        return MERGE.merge_studies(case['dataset'],kwargs.pop('frozen',case['frozen']),case['baseline'],case['receipt'],
            kwargs.pop('studies',case['studies']),kwargs.pop('output',case['output']),**kwargs)

    def rejected(self,case,**kwargs):
        with self.assertRaises((ValueError,KeyError,TypeError)):
            self.run_merge(case,**kwargs)
        self.assertFalse(case['output'].exists())

    def rebind(self,study):
        plan_sha=MERGE.digest(study/'study_plan.json')
        progress=json.loads((study/'progress.json').read_bytes());progress['plan_sha256']=plan_sha
        write(study/'progress.json',progress)
        for method in METHODS:
            path=study/(method+'_receipt.json');receipt=json.loads(path.read_bytes())
            receipt.update(plan_sha256=plan_sha,prediction_json_sha256=MERGE.digest(study/(method+'_candidates.json')))
            write(path,receipt)

    def mutate(self,path,change):
        value=json.loads(path.read_bytes());change(value);write(path,value)

    def test_disjoint_exports_merge_exactly_and_all_inputs_remain_unchanged(self):
        case=self.fixture()
        before={p:p.read_bytes() for p in case['root'].rglob('*.json')}
        result=self.run_merge(case)
        self.assertEqual(result['images'],2)
        self.assertTrue(result['export_readback_verified'])
        self.assertEqual(result['new_model_requests'],0)
        self.assertEqual(result['reference_files_opened'],0)
        self.assertFalse(result['candidates_confirmed_as_truth'])
        for method in METHODS:
            expected={TOPIC:{}}
            for study in case['studies']:expected[TOPIC].update(json.loads((study/(method+'_candidates.json')).read_bytes())[TOPIC])
            actual=json.loads((case['output']/(method+'_candidates.json')).read_bytes())
            self.assertEqual(actual,expected)
            receipt=json.loads((case['output']/(method+'_receipt.json')).read_bytes())
            self.assertEqual(receipt['prediction_json_sha256'],MERGE.digest(case['output']/(method+'_candidates.json')))
            self.assertEqual(len(receipt['parent_prediction_receipt_sha256']),2)
            self.assertNotIn(receipt['plan_sha256'],[MERGE.digest(p/'study_plan.json') for p in case['studies']])
        for path,raw in before.items():self.assertEqual(path.read_bytes(),raw)

    def test_stale_hashes_wrong_images_archive_or_baseline_receipt_are_rejected(self):
        for kind in ('prediction','plan','image','archive','baseline_receipt'):
            with self.subTest(kind=kind):
                c=self.fixture();d=c['studies'][1]
                if kind=='prediction':self.mutate(d/(PRIMARY+'_candidates.json'),lambda v:v[TOPIC]['1'][1].update(confidence=.9))
                elif kind=='plan':self.mutate(d/'study_plan.json',lambda v:v.update(version=2))
                elif kind=='image':self.mutate(d/(PRIMARY+'_receipt.json'),lambda v:v['image_sha256'][TOPIC].update({'1':'b'*64}))
                elif kind=='archive':self.mutate(d/(PRIMARY+'_receipt.json'),lambda v:v.update(archive_sha256='b'*64))
                else:self.mutate(c['receipt'],lambda v:v.update(extra='changed'))
                self.rejected(c)

    def test_missing_overlap_extra_and_invalid_dimensions_are_rejected(self):
        c=self.fixture();self.rejected(c,studies=c['studies'][:1])
        c=self.fixture();self.rejected(c,studies=[c['studies'][0],c['studies'][0]])
        for frames in ([0],[2],[1,1],[True],['1']):
            with self.subTest(frames=frames):
                c=self.fixture();d=c['studies'][1]
                self.mutate(d/'study_plan.json',lambda v:v.update(frames=frames))
                self.rebind(d);self.rejected(c)

    def test_mixed_parameters_implementation_and_primary_replacement_are_rejected(self):
        for key,value in [('fixed_parameters',{'minimum_score':.4}),('helper_sha256',{'raw.py':'e'*64}),
                          ('code_sha256','e'*64),('comparison_requires',{'all_original_candidates_preserved':False})]:
            with self.subTest(key=key):
                c=self.fixture();d=c['studies'][1]
                self.mutate(d/'study_plan.json',lambda v:v.update({key:value}));self.rebind(d);self.rejected(c)
        c=self.fixture();self.rejected(c,methods=['four_crop_control'])

    def test_changed_original_geometry_confidence_or_duplicate_ids_rejected_after_rebinding(self):
        for kind in ('geometry','confidence','class','duplicate_id','boolean_id'):
            with self.subTest(kind=kind):
                c=self.fixture();d=c['studies'][1];path=d/(PRIMARY+'_candidates.json')
                def change(v):
                    shapes=v[TOPIC]['1']
                    if kind=='geometry':shapes[0]['points'][0][0]=2
                    elif kind=='confidence':shapes[0]['confidence']=.81
                    elif kind=='class':shapes[0]['label']='person'
                    elif kind=='duplicate_id':shapes[1]['id']=1
                    else:shapes[1]['id']=True
                self.mutate(path,change);self.rebind(d);self.rejected(c)

    def test_incomplete_reference_read_uncertified_and_missing_processing_rows_rejected(self):
        for kind in ('incomplete','reference','certificate','missing_row','extra_row','duplicate_row'):
            with self.subTest(kind=kind):
                c=self.fixture();d=c['studies'][1]
                def change(v):
                    if kind=='incomplete':v['status']='running'
                    elif kind=='reference':v['reference_files_opened']=1
                    elif kind=='certificate':v['postprocessing'][1]['solver']['certificate_satisfied']=False
                    elif kind=='missing_row':v['postprocessing'].pop()
                    elif kind=='extra_row':v['postprocessing'].append({**v['postprocessing'][1],'frame':99})
                    else:v['postprocessing'].append(copy.deepcopy(v['postprocessing'][1]))
                self.mutate(d/'progress.json',change);self.rejected(c)

    def test_receipt_semantics_method_and_export_frame_coverage_are_verified(self):
        for kind in ('confirmed','source','method','noncanonical','missing_image'):
            with self.subTest(kind=kind):
                c=self.fixture();d=c['studies'][1];r=d/(PRIMARY+'_receipt.json')
                if kind=='confirmed':self.mutate(r,lambda v:v.update(candidates_confirmed_as_truth=True))
                elif kind=='source':self.mutate(r,lambda v:v.update(source='other_predictor'))
                elif kind=='method':self.mutate(r,lambda v:v.update(method='other'))
                else:
                    p=d/(PRIMARY+'_candidates.json')
                    self.mutate(p,lambda v:v[TOPIC].update({'01':v[TOPIC].pop('1')}) if kind=='noncanonical' else v[TOPIC].clear())
                    self.rebind(d)
                self.rejected(c)

    def test_duplicate_json_keys_and_nonfinite_json_values_are_rejected(self):
        for raw in (b'{"status":"complete","status":"running"}',b'{"value":NaN}'):
            with self.subTest(raw=raw):
                c=self.fixture();(c['studies'][1]/'progress.json').write_bytes(raw);self.rejected(c)

    def test_global_inputs_changed_during_merge_are_detected_before_output_creation(self):
        for name in ('baseline','receipt','frozen','manifest'):
            with self.subTest(name=name):
                c=self.fixture();original_canonical=MERGE.canonical;changed=False
                target=c['dataset']/'import_manifest.json' if name=='manifest' else c[name]
                def canonical(value):
                    nonlocal changed
                    if not changed:
                        changed=True;self.mutate(target,lambda v:v.update(concurrent_change=True))
                    return original_canonical(value)
                with patch.object(MERGE,'canonical',side_effect=canonical):self.rejected(c)
                self.assertTrue(changed)

    def test_reference_inputs_and_reusing_output_or_writing_into_source_are_rejected(self):
        c=self.fixture();forbidden=c['dataset']/'source/labels/frozen.json'
        write(forbidden,json.loads(c['frozen'].read_bytes()));self.rejected(c,frozen=forbidden)
        c=self.fixture();self.rejected(c,output=c['studies'][0]/'merged')
        c=self.fixture();c['output'].mkdir();marker=c['output']/'keep.txt';marker.write_text('preserve')
        with self.assertRaises(ValueError):self.run_merge(c)
        self.assertEqual(marker.read_text(),'preserve')


if __name__=='__main__':unittest.main(verbosity=2)
