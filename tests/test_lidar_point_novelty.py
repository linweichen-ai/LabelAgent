"""Distinct raw point support must matter without rewriting prediction facts."""
from copy import deepcopy
import unittest
import numpy as np
from tools.lidar_candidate_evidence import RawPointNovelty
from tools.tiled_candidate_utils import augment_candidates


def shape(identifier,x0,x1,y0=6,y1=14,score=1.):
    return {'id':identifier,'label':'pallet','points':[[x0,y0],[x1,y0],[x1,y1],[x0,y1]],'holes':[],'score':score}


class RawPointNoveltyTests(unittest.TestCase):
    def sensor(self):
        pixels=np.array([[x,10] for x in range(10,16)]+[[x,10] for x in range(40,46)])
        return {'plane_usable':True,'pixels':pixels,'heights':np.full(12,.1)}

    def test_originally_covered_points_do_not_count_as_new_support(self):
        original=[shape(1,6,18)];proposals=[shape(2,7,19),shape(3,38,48)]
        before=deepcopy(original+proposals)
        context=RawPointNovelty(original,proposals,80,20,self.sensor())
        values=context.priorities('mass')
        self.assertGreater(values[3],values[2])
        self.assertEqual(original+proposals,before)

    def test_selected_point_set_reduces_redundant_priority(self):
        a,b,c=shape(1,6,16),shape(2,8,18,5,15),shape(3,38,48)
        context=RawPointNovelty([], [a,b,c],80,20,self.sensor())
        self.assertEqual(context.dynamic_priority(b,()),context.dynamic_priority(c,()))
        self.assertLess(context.dynamic_priority(b,(a,)),context.dynamic_priority(c,(a,)))

    def test_unknown_geometry_preserves_a_neutral_order_and_rejects_id_collisions(self):
        proposals=[shape(1,6,16,score=.8),shape(2,38,48,score=.6)]
        sensor=self.sensor();sensor['plane_usable']=False;sensor['heights']=None
        context=RawPointNovelty([],proposals,80,20,sensor)
        self.assertEqual(context.priorities('mass'),{1:.4,2:.3})
        with self.assertRaises(ValueError):RawPointNovelty([proposals[0]],proposals,80,20,self.sensor())

    def test_dynamic_selection_covers_two_point_groups_with_same_unchanged_budget(self):
        baseline=[shape(i,x,x+2,0,2) for i,x in enumerate((0,25,55,72),1)]
        a,b,c=shape(5,6,16),shape(6,8,18,5,15),shape(7,38,48,score=.9)
        proposals=[a,b,c];before=deepcopy(baseline+proposals)
        ordinary,_=augment_candidates(baseline,proposals,80,20,minimum_score=.5,addition_ratio=.5)
        context=RawPointNovelty(baseline,proposals,80,20,self.sensor())
        selected,info=augment_candidates(baseline,proposals,80,20,minimum_score=.5,addition_ratio=.5,
            priority_callback=context.dynamic_priority)
        self.assertEqual([s['id'] for s in ordinary[4:]],[5,6])
        self.assertEqual([s['id'] for s in selected[4:]],[5,7])
        self.assertEqual(info['added_pallets'],2)
        self.assertEqual(selected[:4],baseline)
        self.assertEqual(baseline+proposals,before)

    def test_novel_raw_points_can_outweigh_pixel_overlap_without_changing_masks(self):
        baseline=[shape(1,6,18)]
        overlap,unsupported=shape(2,7,28,score=.9),shape(3,50,60,score=.8)
        proposals=[overlap,unsupported];before=deepcopy(baseline+proposals)
        sensor={'plane_usable':True,'pixels':np.array([[x,10] for x in range(23,29)]),'heights':np.full(6,.1)}
        hybrid=RawPointNovelty(baseline,proposals,80,20,sensor)
        point_only=RawPointNovelty(baseline,proposals,80,20,sensor,base_mode='point_only')
        self.assertLess(hybrid.priorities('fraction')[2],hybrid.priorities('fraction')[3])
        self.assertGreater(point_only.priorities('fraction')[2],point_only.priorities('fraction')[3])
        old,_=augment_candidates(baseline,proposals,80,20,ranking_scores=hybrid.priorities('fraction'))
        new,info=augment_candidates(baseline,proposals,80,20,ranking_scores=point_only.priorities('fraction'))
        self.assertEqual([s['id'] for s in old],[1,3])
        self.assertEqual([s['id'] for s in new],[1,2])
        self.assertEqual(info['added_pallets'],1)
        self.assertEqual(baseline+proposals,before)

    def test_unknown_point_only_evidence_remains_soft_and_bad_base_is_rejected(self):
        proposal=shape(2,6,16,score=.8)
        sensor=self.sensor();sensor['plane_usable']=False;sensor['heights']=None
        context=RawPointNovelty([], [proposal],80,20,sensor,base_mode='point_only')
        self.assertEqual(context.priorities('fraction'),{2:.4})
        with self.assertRaises(ValueError):RawPointNovelty([], [proposal],80,20,sensor,base_mode='oracle')

    def test_invalid_dynamic_values_and_mixed_policies_fail_closed(self):
        proposals=[shape(1,6,16)]
        with self.assertRaises(ValueError):augment_candidates([],proposals,80,20,priority_callback=lambda *_:float('nan'))
        with self.assertRaises(ValueError):augment_candidates([],proposals,80,20,priority_callback=lambda *_:1.1)
        with self.assertRaises(ValueError):augment_candidates([],proposals,80,20,priority_callback=lambda *_:.5,ranking_scores={1:.5})


if __name__=='__main__':unittest.main()
