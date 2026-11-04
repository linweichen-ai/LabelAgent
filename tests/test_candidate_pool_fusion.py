from copy import deepcopy
import unittest
from tools.candidate_pool_fusion import fuse_image,raster_key

def shape(identifier,left=1,top=1,right=8,bottom=8,label='pallet',score=.8):
    return {'id':identifier,'label':label,'score':score,'points':[[left,top],[right,top],[right,bottom],[left,bottom]],'holes':[]}

class PoolFusionTests(unittest.TestCase):
    def test_prefix_and_inputs_preserved_and_new_shape_changes_only_id(self):
        old=[shape('original')];new=shape(1,12,1,18,8);snapshot=deepcopy((old,new))
        values,provenance,counts=fuse_image(old,[('wooden',[new])],32,24)
        self.assertEqual((old,new),snapshot);self.assertEqual(values[:1],old)
        copy=deepcopy(values[1]);copy['id']=new['id'];self.assertEqual(copy,new)
        self.assertEqual(provenance[0]['source_id'],1);self.assertEqual(counts['wooden']['added'],1)

    def test_raster_dedup_preserves_original_even_if_duplicate_score_higher(self):
        old=shape('old',score=.5);duplicate=shape(99,score=.99)
        duplicate['points']=duplicate['points'][2:]+duplicate['points'][:2]
        values,origin,counts=fuse_image([old],[('wooden',[duplicate])],32,24)
        self.assertEqual(values,[old]);self.assertEqual(origin,[]);self.assertEqual(counts['wooden']['duplicate'],1)

    def test_holes_and_position_are_part_of_raster_identity(self):
        hollow=shape('old');hollow['holes']=[[[3,3],[5,3],[5,5],[3,5]]]
        filled=shape(1);translated=shape(2,12,1,19,8)
        self.assertNotEqual(raster_key(hollow,32,24),raster_key(filled,32,24))
        values,origin,_=fuse_image([hollow],[('wooden',[filled,translated])],32,24)
        self.assertEqual(len(values),3);self.assertEqual(len(origin),2)

    def test_pallet_only_and_threshold_keep_original_person(self):
        person=shape('person',label='person')
        values,_,counts=fuse_image([person],[('wooden',[shape(1,score=.49),shape(2,score=.5),shape(3,label='person')])],32,24)
        self.assertEqual(values[0],person);self.assertEqual(len(values),2)
        self.assertEqual(counts['wooden'],{'added':1,'duplicate':0,'low_score':1,'non_pallet':1})

    def test_source_id_reuse_is_safe_and_new_raster_duplicates_keep_first(self):
        a=shape(1);b=shape(1,12,1,19,8);duplicate=deepcopy(a);duplicate['id']=2
        values,origin,counts=fuse_image([],[('wooden',[a]),('plastic',[b,duplicate])],32,24)
        self.assertEqual(len({v['id'] for v in values}),2);self.assertEqual(len(origin),2)
        self.assertEqual(counts['plastic']['duplicate'],1)

    def test_invalid_bindings_fail_instead_of_overwriting(self):
        with self.assertRaises(ValueError):fuse_image([shape('same'),shape('same')],[],32,24)
        with self.assertRaises(ValueError):fuse_image([],[('same',[]),('same',[])],32,24)
        with self.assertRaises(ValueError):fuse_image([shape('material_raw:wooden:0')],[('wooden',[shape(1,12,1,19,8)])],32,24)
        with self.assertRaises(ValueError):fuse_image([],[('wooden',[shape(1,score=float('nan'))])],32,24)

if __name__=='__main__':unittest.main()
