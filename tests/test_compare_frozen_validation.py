import copy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('frozen_comparison',ROOT/'tools/compare_frozen_validation.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)
PRIMARY='raw_certificate_priority';SECONDARY='raw_certificate_components'
TOPIC='/capture/camera_0/image'


class FrozenComparisonTests(unittest.TestCase):
    def put(self,c,name,**changes):
        metrics=dict(matched=62,points=820,candidates=150,area=1250,person_cross=105,reference_objects=100,
                     reference_points=1000,person_points=1000)
        metrics.update(changes)
        rows=[];mapping={TOPIC:{}}
        for frame in (0,1):
            half=lambda n:n//2+(n%2 if frame else 0)
            classes={}
            for label in ('pallet','person'):
                fields=dict(reference_projected_points=half(metrics['reference_points'] if label=='pallet' else metrics['person_points']),
                    supported_reference_objects=half(metrics['reference_objects'] if label=='pallet' else 10),
                    candidates=half(metrics['candidates']) if label=='pallet' else 0,
                    raw_mask_union_pixels=half(metrics['area']) if label=='pallet' else 0)
                fields['radii']={radius:dict(covered_projected_points=half(metrics['points']) if label=='pallet' else 0,
                    points_covered_by_other_class=half(metrics['person_cross']) if label=='person' else 0,
                    objects_matched_at_support_threshold=half(metrics['matched']) if label=='pallet' else 0) for radius in M.RADII}
                classes[label]=fields
            rows.append(dict(frame=frame,camera=0,classes=classes))
            mapping[TOPIC][str(frame)]=[dict(id=i,label='pallet',points=[[i,1],[i+1,1],[i+1,2]],confidence=.8)
                                      for i in range(half(metrics['candidates']))]
        groups={}
        for label in ('pallet','person'):
            groups[label]={field:sum(r['classes'][label][field] for r in rows) for field in M.COUNT_FIELDS}
            groups[label]['images']=2
            groups[label]['radii']={radius:{field:sum(r['classes'][label]['radii'][radius][field] for r in rows)
                                           for field in M.RADIUS_COUNTS} for radius in M.RADII}
        c['reports'][name]=dict(status='complete',metric_version=2,planned_images=2,evaluated_images=2,
            missing_prediction_images=0,archive_sha256='a'*64,parameters={'radii_pixels':[0,3,6],'minimum_object_points':5},
            reference_source='labels',reference_is_independent_ground_truth=False,groups={'all':groups})
        c['per_image'][name]=rows;c['candidate_maps'][name]=mapping

    def fixture(self):
        policy=dict(strict_instance_gain_over_previous_best=True,point_coverage_not_below_previous_best=True,
            person_off_class_not_above_previous_best=True,all_original_candidates_preserved=True,
            original_minimum_absolute_gain=.02,original_maximum_candidate_count_ratio=1.5,
            original_maximum_union_area_ratio=1.25,original_person_off_class_increase_limit=.005)
        c=dict(frozen={'full_validation_frames':[0,1],'cameras':[0],'full_validation_images':2,'primary_method':PRIMARY,
                     'secondary_can_replace_primary':False,'comparison_requires':policy},reports={},per_image={},candidate_maps={})
        self.put(c,M.BASELINE,matched=60,points=800,candidates=100,area=1000,person_cross=100)
        self.put(c,M.PREVIOUS,matched=61,points=810,candidates=130,area=1100,person_cross=105)
        self.put(c,PRIMARY)
        self.put(c,SECONDARY,matched=65,points=825)
        self.history(c)
        return c

    def history(self,c):
        c['historical_summary']=copy.deepcopy(c['reports'][M.PREVIOUS])
        c['historical_rows']=copy.deepcopy(c['per_image'][M.PREVIOUS])

    def result(self,c):return M.build_comparison(**c)

    def primary(self,result):return next(r for r in result['rows'] if r['method']==PRIMARY)

    def test_exact_original_thresholds_pass_without_rounding_or_primary_replacement(self):
        r=self.result(self.fixture());row=self.primary(r)
        self.assertTrue(all(row['checks'].values()))
        self.assertEqual(r['winner'],PRIMARY)
        self.assertTrue(r['primary_passed'])
        self.assertFalse(r['adopted'])
        self.assertEqual(r['scored_images_per_method'],2)
        secondary=next(v for v in r['rows'] if v['method']==SECONDARY)
        self.assertGreater(secondary['pallet_objects_matched'],row['pallet_objects_matched'])
        self.assertFalse(secondary['eligible'])

    def test_each_original_gate_rejects_its_boundary_failure(self):
        cases=[('strict_instance_gain_over_previous_best',dict(matched=61)),
               ('point_coverage_not_below_previous_best',dict(points=809)),
               ('person_off_class_not_above_previous_best',dict(person_cross=106)),
               ('original_minimum_two_percentage_point_gain',dict(matched=61)),
               ('original_count_ratio_limit',dict(candidates=151)),
               ('original_union_area_limit',dict(area=1251)),
               ('original_person_cross_class_limit',dict(person_cross=106))]
        for check,changes in cases:
            with self.subTest(check=check):
                c=self.fixture()
                if check=='original_minimum_two_percentage_point_gain':
                    self.put(c,M.PREVIOUS,matched=60,points=810,candidates=130,area=1100,person_cross=105);self.history(c)
                if check=='original_person_cross_class_limit':
                    self.put(c,M.PREVIOUS,matched=61,points=810,candidates=130,area=1100,person_cross=110);self.history(c)
                self.put(c,PRIMARY,**changes);r=self.result(c)
                self.assertFalse(self.primary(r)['checks'][check])
                self.assertFalse(r['primary_passed']);self.assertIsNone(r['winner'])

    def test_missing_duplicate_wrong_dimensions_or_metric_scope_cannot_be_scored(self):
        for kind in ('missing','duplicate','extra','boolean_frame','denominator','parameters','archive','version','incomplete','reference_scope'):
            with self.subTest(kind=kind):
                c=self.fixture();s=c['reports'][PRIMARY];rows=c['per_image'][PRIMARY]
                if kind=='missing':rows.pop()
                elif kind=='duplicate':rows[1]=copy.deepcopy(rows[0])
                elif kind=='extra':rows[1]['frame']=2
                elif kind=='boolean_frame':rows[1]['frame']=True
                elif kind=='denominator':self.put(c,PRIMARY,reference_objects=101)
                elif kind=='parameters':s['parameters']['minimum_object_points']=3
                elif kind=='archive':s['archive_sha256']='b'*64
                elif kind=='version':s['metric_version']=1
                elif kind=='incomplete':s['missing_prediction_images']=1
                else:s['reference_is_independent_ground_truth']=True
                with self.assertRaises(ValueError):self.result(c)

    def test_better_secondary_cannot_rescue_failed_frozen_primary(self):
        c=self.fixture();self.put(c,PRIMARY,candidates=151)
        r=self.result(c)
        self.assertFalse(r['primary_passed']);self.assertIsNone(r['winner'])
        secondary=next(v for v in r['rows'] if v['method']==SECONDARY)
        self.assertTrue(secondary['passed_numeric_and_preservation_gates']);self.assertFalse(secondary['eligible'])

    def test_summary_inflation_and_candidate_export_mismatch_are_rejected(self):
        for kind in ('summary','candidate_count','candidate_frames'):
            with self.subTest(kind=kind):
                c=self.fixture()
                if kind=='summary':c['reports'][PRIMARY]['groups']['all']['pallet']['radii']['3']['covered_projected_points']+=1
                elif kind=='candidate_count':c['candidate_maps'][PRIMARY][TOPIC]['0'].append(dict(id=999,label='pallet',points=[]))
                else:c['candidate_maps'][PRIMARY][TOPIC]['2']=[]
                with self.assertRaises(ValueError):self.result(c)

    def test_historical_control_and_every_image_point_preservation_are_checked(self):
        c=self.fixture();c['historical_rows'][0]['classes']['pallet']['radii']['0']['covered_projected_points']+=1
        c['historical_summary']['groups']['all']['pallet']['radii']['0']['covered_projected_points']+=1
        with self.assertRaises(ValueError):self.result(c)
        c=self.fixture();rows=c['per_image'][PRIMARY]
        rows[0]['classes']['pallet']['radii']['3']['covered_projected_points']-=10
        rows[1]['classes']['pallet']['radii']['3']['covered_projected_points']+=10
        r=self.result(c)
        self.assertTrue(self.primary(r)['checks']['point_coverage_not_below_previous_best'])
        self.assertFalse(self.primary(r)['raw_point_coverage_nonregression_every_image'])
        self.assertFalse(r['primary_passed'])

    def test_original_geometry_and_confidence_must_survive_in_actual_candidate_prefix(self):
        for kind in ('geometry','confidence'):
            with self.subTest(kind=kind):
                c=self.fixture();shape=c['candidate_maps'][PRIMARY][TOPIC]['0'][0]
                if kind=='geometry':shape['points'][0][0]+=1
                else:shape['confidence']=.9
                r=self.result(c)
                self.assertFalse(self.primary(r)['checks']['all_original_candidates_preserved'])
                self.assertFalse(r['primary_passed'])

    def test_original_policy_cannot_be_weakened(self):
        for key,value in [('strict_instance_gain_over_previous_best',False),('original_minimum_absolute_gain',0),
                          ('original_maximum_candidate_count_ratio',2),('original_maximum_union_area_ratio',2),
                          ('original_person_off_class_increase_limit',.1)]:
            with self.subTest(key=key):
                c=self.fixture();c['frozen']['comparison_requires'][key]=value
                with self.assertRaises(ValueError):self.result(c)
        c=self.fixture();c['frozen']['secondary_can_replace_primary']=True
        with self.assertRaises(ValueError):self.result(c)


if __name__=='__main__':unittest.main(verbosity=2)
