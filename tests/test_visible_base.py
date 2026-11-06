import sys,unittest
from pathlib import Path
sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).resolve().parents[1]/'tools/labelagent_experiments')]
import numpy as np
from PIL import Image
from visible_base_policy import planning_views,bind_points,touches_crop_border
class VisibleBaseTests(unittest.TestCase):
 def test_native_overview_has_no_padding_and_keeps_whole_seed(self):
  from visible_base_native_policy import planning_views as native
  m=np.zeros((150,220),np.uint8);m[68:75,70:140]=1;v,_,b=native(Image.new('RGB',(220,150),'white'),m);self.assertTrue(b['no_padding_in_overview']);self.assertEqual(b['retained_seed_pixels'],int(m.sum()));self.assertEqual(v['pointer'].size,(512,512))
 def test_native_normalized_coordinates_are_mapped_not_snapped(self):
  from visible_base_native_policy import bind_points as native
  m=np.zeros((100,100),np.uint8);m[45:55,45:55]=1;b={'source_box':[0,0,100,100],'offset':[0,0],'scale_xy':[5.12,5.12]};d={'instance_scope':'single_visible_object','positive_points':[{'x':460,'y':460},{'x':700,'y':700}],'negative_points':[{'x':100,'y':100}]};p,why=native(d,b,m);self.assertIsNone(why);self.assertEqual(p[0]['x'],46);self.assertEqual(p[1]['x'],70)
 def test_native_crop_cannot_clip_a_seed_too_wide_for_real_square(self):
  from visible_base_native_policy import planning_views as native
  m=np.zeros((100,220),np.uint8);m[45:55,:]=1
  with self.assertRaises(ValueError):native(Image.new('RGB',(220,100)),m)
 def test_overview_contains_entire_seed_and_details_are_bound_to_original_source(self):
  a=np.random.default_rng(4).integers(0,256,(160,220,3),dtype=np.uint8);m=np.zeros((160,220),np.uint8);m[68:75,40:180]=1;before=m.copy();views,details,b=planning_views(Image.fromarray(a),m);self.assertEqual(b['retained_seed_pixels'],int(m.sum()));self.assertEqual(len(views),2);self.assertEqual(len(details),2);x,y=b['anchor_source'];self.assertTrue(m[y,x]);np.testing.assert_array_equal(m,before)
 def test_padding_points_are_rejected_instead_of_clamped_or_snapped(self):
  m=np.zeros((150,300),np.uint8);m[70:74,20:280]=1;_,_,b=planning_views(Image.new('RGB',(300,150)),m);d={'instance_scope':'single_visible_object','positive_points':[{'x':256,'y':256},{'x':0,'y':0}],'negative_points':[]};points,why=bind_points(d,b,m);self.assertIsNone(points);self.assertIn('padding',why)
 def test_points_must_bind_to_seed_and_negative_cannot_erase_seed(self):
  m=np.zeros((100,100),np.uint8);m[45:55,45:55]=1;b={'source_box':[0,0,100,100],'offset':[0,0],'scale_xy':[1,1]};d={'instance_scope':'single_visible_object','positive_points':[{'x':46,'y':46},{'x':70,'y':70}],'negative_points':[{'x':50,'y':50}]};p,why=bind_points(d,b,m);self.assertIsNone(p);self.assertIn('negative',why);d['negative_points']=[{'x':10,'y':10}];p,why=bind_points(d,b,m);self.assertIsNotNone(p);self.assertIsNone(why)
 def test_complete_border_touching_output_is_rejected_not_cropped(self):
  m=np.zeros((40,60),np.uint8);m[5:25,0:35]=1;before=m.copy();self.assertTrue(touches_crop_border(m));np.testing.assert_array_equal(m,before);m[:,0]=0;self.assertFalse(touches_crop_border(m))
if __name__=='__main__':unittest.main()
