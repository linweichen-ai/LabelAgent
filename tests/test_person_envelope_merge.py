from copy import deepcopy
import json
from pathlib import Path
import tempfile,unittest
from tools.merge_person_envelope_predictions import merge_exports,sha


class PersonEnvelopeMergeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.dataset=self.root/'dataset';self.dataset.mkdir()
        self.topic='/capture/camera_0/image';self.methods={'original_point_guard_control':'mask','convex_hull_guard':'convex_hull'}
        self.baseline={self.topic:{str(f):[{'id':1,'label':'pallet'}] for f in (0,1)}}
        self.current=deepcopy(self.baseline)
        for f in ('0','1'):self.current[self.topic][f].append({'id':'extra','label':'pallet'})
        self.images={self.topic:{'0':'image0','1':'image1'}}
        self.dump(self.dataset/'import_manifest.json',{'archive_sha256':'archive','records':[
            {'topic':self.topic,'frame':f,'sha256':f'image{f}'} for f in (0,1)]})
        self.base=self.dump(self.root/'base.json',self.baseline);self.cur=self.dump(self.root/'current.json',self.current)
        common={'archive_sha256':'archive','image_sha256':self.images}
        self.br=self.dump(self.root/'base_receipt.json',{**common,'prediction_json_sha256':sha(self.base)})
        self.cr=self.dump(self.root/'current_receipt.json',{**common,'prediction_json_sha256':sha(self.cur)})
        self.frozen=self.dump(self.root/'frozen.json',{'full_validation_frames':[0,1],'cameras':[0],
            'full_validation_images':2,'primary_method':'convex_hull_guard','predictor_sha256':'code','helper_sha256':{},
            'baseline_prediction_sha256':sha(self.base),'current_prediction_sha256':sha(self.cur)})
        self.studies=[]
        for frame in (0,1):
            directory=self.root/f'study{frame}';directory.mkdir();self.studies.append(directory)
            plan={k:None for k in ('version','minimum_score','addition_ratio','nms_iou','solver_seconds','raw_projection',
                'person_envelopes','guard','raw_coverage_certificate')}
            plan.update(methods=self.methods,code_sha256='code',helper_sha256={},prediction_uses_reference_labels=False,
                        frames=[frame],cameras=[0],planned_images=1)
            p=self.dump(directory/'study_plan.json',plan)
            rows=[{'topic':self.topic,'frame':frame,'method':m,'original_prefix_preserved':True,
                   'raw_certificate_verified':True,'person_guard_verified':True,'solver':{'certificate_satisfied':True}} for m in self.methods]
            self.dump(directory/'progress.json',{'status':'complete','reference_files_opened':0,'workbench_writes':False,
                'input_bindings_unchanged':True,'plan_unchanged':True,'code_unchanged':True,
                'all_predictions_frozen_before_scoring':True,'completed_images':1,'rows':rows})
            for method in self.methods:
                path=self.dump(directory/(method+'_candidates.json'),{self.topic:{str(frame):self.current[self.topic][str(frame)]}})
                self.dump(directory/(method+'_receipt.json'),{'archive_sha256':'archive','method':method,
                    'prediction_json_sha256':sha(path),'plan_sha256':sha(p),
                    'image_sha256':{self.topic:{str(frame):f'image{frame}'}},'candidates_confirmed_as_truth':False})
        self.output=self.root/'output'

    def dump(self,path,value):
        path.write_text(json.dumps(value),encoding='utf-8');return path

    def merge(self,studies=None):
        return merge_exports(self.dataset,self.frozen,self.base,self.br,self.cur,self.cr,
                             studies if studies is not None else self.studies,self.output)

    def test_complete_disjoint_exports_round_trip(self):
        result=self.merge();self.assertEqual(result['images'],2)
        values=json.loads((self.output/'convex_hull_guard_candidates.json').read_bytes())
        self.assertEqual(values,self.current);self.assertTrue(result['original_control_replayed_exact'])

    def test_missing_and_duplicate_cohorts_fail_before_output(self):
        for sources in (self.studies[:1],[self.studies[0],self.studies[0]]):
            with self.assertRaises(ValueError):self.merge(sources)
            self.assertFalse(self.output.exists())

    def test_reference_read_marker_is_rejected(self):
        path=self.studies[1]/'progress.json';value=json.loads(path.read_bytes());value['reference_files_opened']=1;self.dump(path,value)
        with self.assertRaises(ValueError):self.merge()
        self.assertFalse(self.output.exists())

    def test_recipe_mismatch_is_rejected_before_output(self):
        path=self.studies[1]/'study_plan.json';value=json.loads(path.read_bytes());value['code_sha256']='changed';self.dump(path,value)
        with self.assertRaises(ValueError):self.merge()
        self.assertFalse(self.output.exists())

    def test_changed_original_prefix_is_rejected_even_with_updated_receipt(self):
        directory=self.studies[1];path=directory/'convex_hull_guard_candidates.json'
        value=json.loads(path.read_bytes());value[self.topic]['1'][0]['label']='person';self.dump(path,value)
        receipt=directory/'convex_hull_guard_receipt.json';r=json.loads(receipt.read_bytes());r['prediction_json_sha256']=sha(path);self.dump(receipt,r)
        with self.assertRaises(ValueError):self.merge()
        self.assertFalse(self.output.exists())


if __name__=='__main__':unittest.main()
